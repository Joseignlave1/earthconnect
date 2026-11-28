#!/usr/bin/env python3
"""Streaming, sharded, low-RAM, resumable ECV3 packer for Earth at ~61 m global land resolution.

Sources (100% public, no-auth, high-bandwidth cloud-native GeoTIFFs):
  - Visual RGB (L0..L5): ESA Sentinel-2 Level 3 Quarterly Cloudless Mosaics (2025 Q3 North / Q1 South,
    opposite quarter fallback for cloudy/nodata pixels, Blue Marble fallback for >84°N / <80°S and
    glare-free open ocean beyond ~1.5 km from land).
  - Elevation (L0..L5): AWS Open Data Copernicus DEM GLO-30 (overview 2x = ~60 m native).
  - Coastline / Water Mask: Copernicus DEM 30 m Water Body Mask (AUXFILES/*_WBM.tif) antialiased
    into tile alpha [128..255] (128 = water, 255 = land). Inland tiles (alpha == 255) encode as
    compact stripped TPL_RGB8; coastal/water tiles encode as self-contained TPL_RGBA8; pure-ocean
    tiles (alpha <= 130) are skipped completely (0 bytes).

Grid & Sharding:
  - BASE_COLS = 32, BASE_ROWS = 16, LEVEL_COUNT = 6 (L0: 32x16 .. L5: 1024x512).
  - 512 square regional packs (16 latitude bands x 32 longitude blocks, 11.25° x 11.25°):
      band    = ty >> level  (0..15)
      lon_blk = tx >> level  (0..31)
      pack_id = band * 32 + lon_blk (0..511)
  - 64-pixel L5 border padding halves cleanly through every mip (64->32->16->8->4->2 = GUTTER)
    so every tile at L0..L5 has exact cross-pack seamless gutters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import json
import math
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from ecv3_format import (
    BODY_IDS,
    GUTTER,
    INNER,
    TILE_SIZE,
    TPL_HEIGHT10,
    TPL_RGBA8,
    Ecv3Record,
    ecv3_tile_key,
    encode_height10_tile,
    encode_visual_tile_ecv32,
    strip_avif_header,
    verify_ecv3_pack,
    write_ecv3_pack,
)

Image.MAX_IMAGE_PIXELS = None

BASE_COLS = 32
BASE_ROWS = 16
LEVEL_COUNT = 6
MAX_LEVEL = LEVEL_COUNT - 1  # 5
BANDS = 16
LON_BLKS = 32
LAT_STEP = 180.0 / BANDS  # 11.25°
LON_STEP = 360.0 / LON_BLKS  # 11.25°
EARTH_RADIUS_M = 6_371_008.8

FULL_COLS = BASE_COLS << MAX_LEVEL  # 1024
FULL_ROWS = BASE_ROWS << MAX_LEVEL  # 512
FULL_W = FULL_COLS * INNER  # 520,192
FULL_H = FULL_ROWS * INNER  # 260,096

PACK_TILES_X_L5 = FULL_COLS // LON_BLKS  # 32
PACK_TILES_Y_L5 = FULL_ROWS // BANDS  # 32
PACK_W_L5 = PACK_TILES_X_L5 * INNER  # 16,256
PACK_H_L5 = PACK_TILES_Y_L5 * INNER  # 16,256
PAD_L5 = GUTTER << MAX_LEVEL  # 64 (halves cleanly to 2 at L0!)

COP_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
S2_BASE = "https://data.source.coop/tge-labs/sentinel-2-quarterly-cloudless-mosaics"
WGS84 = CRS.from_epsg(4326)

GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TIMEOUT": "45",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",
}

# Read-only fork COW globals for zero-IPC tile encoding
_G_RGB: np.ndarray | None = None
_G_ALPHA: np.ndarray | None = None
_G_UP_RGB: np.ndarray | None = None
_G_UP_ALPHA: np.ndarray | None = None
_G_DEM: np.ndarray | None = None
_G_UP_DEM: np.ndarray | None = None
_G_PAD: int = 0
_G_ALLOW_PRUNE: bool = False
_G_SPEED: int = 8
_G_DEM_Q: int = 88
_G_DEM_PX_M: float = 70.0


def pack_canvas_bounds(band: int, lon_blk: int) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) of the padded L5 canvas in degrees."""
    dlon = 360.0 / FULL_W
    dlat = 180.0 / FULL_H
    lat_top = 90.0 - band * LAT_STEP
    lat_bot = lat_top - LAT_STEP
    lon_left = -180.0 + lon_blk * LON_STEP
    lon_right = lon_left + LON_STEP
    return (
        lon_left - PAD_L5 * dlon,
        lat_bot - PAD_L5 * dlat,
        lon_right + PAD_L5 * dlon,
        lat_top + PAD_L5 * dlat,
    )


def init_blue_marble_canvas(
    bm_img: Image.Image, band: int, lon_blk: int, out_w: int, out_h: int
) -> np.ndarray:
    """Fast C/Pillow Blue Marble fallback canvas (handles ±180° wrap & polar clamp in <1s)."""
    bw, bh = bm_img.size
    lon0, lat0, lon1, lat1 = pack_canvas_bounds(band, lon_blk)
    y0 = max(0, int(math.floor(((90.0 - min(90.0, lat1)) / 180.0) * bh)) - 2)
    y1 = min(bh, int(math.ceil(((90.0 - max(-90.0, lat0)) / 180.0) * bh)) + 2)
    x0_f = ((lon0 + 180.0) / 360.0) * bw
    x1_f = ((lon1 + 180.0) / 360.0) * bw
    ix0 = int(math.floor(x0_f)) - 2
    ix1 = int(math.ceil(x1_f)) + 2
    if ix0 < 0 or ix1 > bw:
        bm = np.asarray(bm_img, dtype=np.uint8)
        xs = np.arange(ix0, ix1) % bw
        crop_img = Image.fromarray(bm[y0:y1, xs], mode="RGB")
    else:
        crop_img = bm_img.crop((ix0, y0, ix1, y1))
    yt_f = float(np.clip(((90.0 - lat1) / 180.0) * bh, 0.0, float(bh)))
    yb_f = float(np.clip(((90.0 - lat0) / 180.0) * bh, 0.0, float(bh)))
    box = (x0_f - ix0, yt_f - y0, x1_f - ix0, yb_f - y0)
    resized = crop_img.resize((out_w, out_h), Image.Resampling.BILINEAR, box=box)
    return np.asarray(resized, dtype=np.uint8).copy()


def fetch_copernicus_tile(
    name: str,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    out_w: int,
    out_h: int,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray] | None:
    """Stream one 1°x1° Copernicus DEM GLO-30 (ov=2, ~60m) + 30m WBM and return canvas window.
    Returns (x0, y0, x1, y1, dem_i16, alpha_u8, non_ocean_u8) where non_ocean_u8=255 on land/lake/river."""
    dem_url = f"/vsicurl/{COP_BASE}/{name}/{name}.tif"
    wbm_url = f"/vsicurl/{COP_BASE}/{name}/AUXFILES/{name}_WBM.tif"

    for attempt in range(4):
        try:
            with rasterio.Env(**GDAL_ENV):
                with rasterio.open(dem_url) as src:
                    tb = src.bounds
                    shift = None
                    for s in (0.0, -360.0, 360.0):
                        if tb.left + s < lon1 and tb.right + s > lon0:
                            shift = s
                            break
                    if shift is None or tb.bottom >= lat1 or tb.top <= lat0:
                        return None

                    ileft = max(lon0, tb.left + shift)
                    iright = min(lon1, tb.right + shift)
                    ibot = max(lat0, tb.bottom)
                    itop = min(lat1, tb.top)

                    dlon = (lon1 - lon0) / out_w
                    dlat = (lat1 - lat0) / out_h
                    x0 = max(0, int(math.floor((ileft - lon0) / dlon)))
                    x1 = min(out_w, int(math.ceil((iright - lon0) / dlon)))
                    y0 = max(0, int(math.floor((lat1 - itop) / dlat)))
                    y1 = min(out_h, int(math.ceil((lat1 - ibot) / dlat)))
                    if x1 <= x0 or y1 <= y0:
                        return None

                    win_w = x1 - x0
                    win_h = y1 - y0
                    w_left = lon0 + x0 * dlon - shift
                    w_right = lon0 + x1 * dlon - shift
                    w_top = lat1 - y0 * dlat
                    w_bot = lat1 - y1 * dlat

                    ovs = src.overviews(1)
                    if ovs and ovs[0] == 2:
                        with rasterio.open(dem_url, overview_level=0) as ov_src:
                            win = ov_src.window(w_left, w_bot, w_right, w_top)
                            dem = ov_src.read(
                                1,
                                window=win,
                                out_shape=(win_h, win_w),
                                resampling=Resampling.bilinear,
                                boundless=True,
                                fill_value=0.0,
                            )
                    else:
                        win = src.window(w_left, w_bot, w_right, w_top)
                        dem = src.read(
                            1,
                            window=win,
                            out_shape=(win_h, win_w),
                            resampling=Resampling.bilinear,
                            boundless=True,
                            fill_value=0.0,
                        )

                dem = np.nan_to_num(dem.astype(np.float32), nan=0.0)
                dem_i16 = np.clip(np.rint(dem * 2.0), -32767, 32767).astype(np.int16)

                # Read 30 m Water Body Mask (0=land, 1=ocean, 2=lake, 3=river) with box averaging for crisp AA coastlines
                try:
                    with rasterio.open(wbm_url) as wsrc:
                        wwin = wsrc.window(w_left, w_bot, w_right, w_top)
                        wraw = wsrc.read(1, window=wwin, boundless=True, fill_value=1)
                        land_u8 = np.where(wraw == 0, 255, 0).astype(np.uint8)
                        land_res = np.asarray(
                            Image.fromarray(land_u8, mode="L").resize(
                                (win_w, win_h), Image.Resampling.BOX
                            ),
                            dtype=np.float32,
                        ) * (1.0 / 255.0)
                        non_ocean_raw = np.where(wraw != 1, 255, 0).astype(np.uint8)
                        non_ocean_u8 = np.asarray(
                            Image.fromarray(non_ocean_raw, mode="L").resize(
                                (win_w, win_h), Image.Resampling.BOX
                            ),
                            dtype=np.uint8,
                        )
                except Exception:
                    land_res = (dem > 0.25).astype(np.float32)
                    non_ocean_u8 = np.where(dem > 0.25, 255, 0).astype(np.uint8)

                alpha_u8 = np.clip(np.rint(128.0 + land_res * 127.0), 128, 255).astype(
                    np.uint8
                )
                return x0, y0, x1, y1, dem_i16, alpha_u8, non_ocean_u8
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Copernicus DEM fetch failed after 4 attempts: {dem_url}"
                ) from exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Copernicus DEM fetch exhausted: {dem_url}")


def s2_reflectance_to_srgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calibrated Sentinel-2 L3 BOA reflectance -> natural true-color sRGB uint8."""
    rf = np.maximum(0.0, r.astype(np.float32) * (1.0 / 2800.0) - 0.008)
    gf = np.maximum(0.0, g.astype(np.float32) * (1.0 / 2800.0) - 0.008)
    bf = np.maximum(0.0, b.astype(np.float32) * (1.0 / 2800.0) - 0.008)
    rf = rf / (1.0 + 0.16 * rf)
    gf = gf / (1.0 + 0.16 * gf)
    bf = bf / (1.0 + 0.16 * bf)
    lin = np.stack([rf, gf, bf], axis=-1)
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(
        lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055
    )
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def fetch_sentinel2_tile(
    quarter: str,
    folder: str,
    ov_level: int,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    out_w: int,
    out_h: int,
) -> list[tuple[int, int, int, int, np.ndarray, np.ndarray]] | None:
    """Fetch B04/B03/B02 COGs at chosen overview level and warp to pack EPSG:4326 sub-window(s)."""
    base = f"/vsicurl/{S2_BASE}/{quarter}/{folder}"
    urls = [f"{base}/B04.tif", f"{base}/B03.tif", f"{base}/B02.tif"]

    for attempt in range(4):
        try:
            with rasterio.Env(**GDAL_ENV):
                bands_arr = []
                src_crs = None
                src_transform = None
                src_bounds = None
                for u in urls:
                    with rasterio.open(u, overview_level=ov_level) as src:
                        if src_crs is None:
                            src_crs = src.crs
                            src_transform = src.transform
                            src_bounds = src.bounds
                        bands_arr.append(src.read(1))

            w_left, w_bot, w_right, w_top = transform_bounds(
                src_crs, WGS84, *src_bounds, densify_pts=21
            )
            if w_right - w_left > 180.0:
                lon_ranges = [(-180.0, -165.0), (165.0, 180.0)]
            else:
                lon_ranges = [(w_left, w_right)]

            dlon = (lon1 - lon0) / out_w
            dlat = (lat1 - lat0) / out_h
            src_stack = np.stack(bands_arr, axis=0)
            results: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []

            for rl, rr in lon_ranges:
                for shift in (0.0, -360.0, 360.0):
                    sl, sr = rl + shift, rr + shift
                    if sl >= lon1 or sr <= lon0 or w_bot >= lat1 or w_top <= lat0:
                        continue
                    ileft = max(lon0, sl)
                    iright = min(lon1, sr)
                    ibot = max(lat0, w_bot)
                    itop = min(lat1, w_top)
                    x0 = max(0, int(math.floor((ileft - lon0) / dlon)))
                    x1 = min(out_w, int(math.ceil((iright - lon0) / dlon)))
                    y0 = max(0, int(math.floor((lat1 - itop) / dlat)))
                    y1 = min(out_h, int(math.ceil((lat1 - ibot) / dlat)))
                    if x1 <= x0 or y1 <= y0:
                        continue

                    win_w = x1 - x0
                    win_h = y1 - y0
                    dst_left = lon0 + x0 * dlon - shift
                    dst_right = lon0 + x1 * dlon - shift
                    dst_top = lat1 - y0 * dlat
                    dst_bot = lat1 - y1 * dlat
                    dst_tf = from_bounds(
                        dst_left, dst_bot, dst_right, dst_top, win_w, win_h
                    )

                    dst_i16 = np.full((3, win_h, win_w), -32768, dtype=np.int16)
                    reproject(
                        source=src_stack,
                        destination=dst_i16,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        src_nodata=-32768,
                        dst_transform=dst_tf,
                        dst_crs=WGS84,
                        dst_nodata=-32768,
                        resampling=Resampling.bilinear,
                    )
                    valid = (
                        (dst_i16[0] != -32768)
                        & (dst_i16[1] != -32768)
                        & (dst_i16[2] != -32768)
                    )
                    if not np.any(valid):
                        continue
                    rgb = s2_reflectance_to_srgb(dst_i16[0], dst_i16[1], dst_i16[2])
                    results.append((x0, y0, x1, y1, rgb, valid))
            return results if results else None
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Sentinel-2 fetch failed after 4 attempts: {quarter}/{folder}"
                ) from exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Sentinel-2 fetch exhausted: {quarter}/{folder}")


def blend_open_ocean_to_blue_marble(
    rgb_l5: np.ndarray,
    bm_l5: np.ndarray,
    non_ocean_l5: np.ndarray,
) -> None:
    """Keep 100% crisp Sentinel-2 over land, lakes, and rivers (subpixel-antialiased right at the
    30 m Copernicus shoreline), while using clean glare-free Blue Marble over open ocean (WBM == 1)
    so coastal tiles have zero Sentinel-2 sun-glint/swath halos and match skipped ocean tiles seamlessly."""
    if int(non_ocean_l5.min()) >= 254:
        return  # 100% inland pack — nothing to blend!
    h = non_ocean_l5.shape[0]
    chunk = 512
    for ys in range(0, h, chunk):
        ye = min(h, ys + chunk)
        w_blk = non_ocean_l5[ys:ye]
        if int(w_blk.min()) == 255:
            continue
        if int(w_blk.max()) == 0:
            rgb_l5[ys:ye] = bm_l5[ys:ye]
            continue
        w16 = w_blk.astype(np.uint16)[:, :, None]
        inv16 = np.uint16(255) - w16
        rgb_l5[ys:ye] = (
            (
                bm_l5[ys:ye].astype(np.uint16) * inv16
                + rgb_l5[ys:ye].astype(np.uint16) * w16
                + 127
            )
            // 255
        ).astype(np.uint8)


def downsample2x_rgb_alpha(
    rgb: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fast C/Pillow exact 2x2 box downsample for RGB + Alpha (~15x faster than NumPy loop)."""
    h, w, _ = rgb.shape
    h2, w2 = h // 2, w // 2
    out_rgb = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((w2, h2), Image.Resampling.BOX),
        dtype=np.uint8,
    )
    out_a = np.asarray(
        Image.fromarray(alpha, mode="L").resize((w2, h2), Image.Resampling.BOX),
        dtype=np.uint8,
    )
    return out_rgb, out_a


def downsample2x_i16(arr: np.ndarray) -> np.ndarray:
    """Fast chunked 2x2 box downsample for int16 DEM."""
    h, w = arr.shape
    h2, w2 = h // 2, w // 2
    out = np.empty((h2, w2), dtype=np.int16)
    chunk = 512
    for y in range(0, h2, chunk):
        ye = min(h2, y + chunk)
        y0, y1 = y * 2, ye * 2
        s = (
            arr[y0:y1:2, 0 : w2 * 2 : 2].astype(np.int32)
            + arr[y0:y1:2, 1 : w2 * 2 : 2].astype(np.int32)
            + arr[y0 + 1 : y1 : 2, 0 : w2 * 2 : 2].astype(np.int32)
            + arr[y0 + 1 : y1 : 2, 1 : w2 * 2 : 2].astype(np.int32)
        )
        out[y:ye] = ((s + np.where(s >= 0, 2, -2)) // 4).astype(np.int16)
    return out


def _extract_patch(arr: np.ndarray, ltx: int, lty: int, pad: int) -> np.ndarray:
    x0 = pad + ltx * INNER - GUTTER
    y0 = pad + lty * INNER - GUTTER
    return arr[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]


def _extract_up_u8(up: np.ndarray, ltx: int, lty: int, up_pad: int) -> np.ndarray:
    px0 = up_pad + (ltx // 2) * INNER - GUTTER
    py0 = up_pad + (lty // 2) * INNER - GUTTER
    parent = up[py0 : py0 + TILE_SIZE, px0 : px0 + TILE_SIZE]
    ox = GUTTER + (ltx % 2) * (INNER // 2)
    oy = GUTTER + (lty % 2) * (INNER // 2)
    crop = parent[oy - 1 : oy + INNER // 2 + 1, ox - 1 : ox + INNER // 2 + 1]
    mode = "RGB" if crop.ndim == 3 else "L"
    return np.asarray(
        Image.fromarray(crop, mode=mode).resize(
            (TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def _extract_up_dem_m(up: np.ndarray, ltx: int, lty: int, up_pad: int) -> np.ndarray:
    px0 = up_pad + (ltx // 2) * INNER - GUTTER
    py0 = up_pad + (lty // 2) * INNER - GUTTER
    parent = up[py0 : py0 + TILE_SIZE, px0 : px0 + TILE_SIZE].astype(np.float32) * 0.5
    ox = GUTTER + (ltx % 2) * (INNER // 2)
    oy = GUTTER + (lty % 2) * (INNER // 2)
    crop = parent[oy - 1 : oy + INNER // 2 + 1, ox - 1 : ox + INNER // 2 + 1]
    return np.asarray(
        Image.fromarray(crop, mode="F").resize(
            (TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def _worker_visual(
    args: tuple[int, int],
) -> tuple[int, int, int, bytes, int, int] | None:
    ltx, lty = args
    assert _G_RGB is not None and _G_ALPHA is not None
    a_tile = _extract_patch(_G_ALPHA, ltx, lty, _G_PAD)
    if int(a_tile.max()) <= 130:
        return None
    rgb_tile = _extract_patch(_G_RGB, ltx, lty, _G_PAD)
    is_inland = int(a_tile.min()) >= 254

    if is_inland:
        tile = rgb_tile
        up_tile = (
            _extract_up_u8(_G_UP_RGB, ltx, lty, _G_PAD // 2)
            if _G_UP_RGB is not None
            else None
        )
    else:
        tile = np.dstack([rgb_tile, a_tile])
        if _G_UP_RGB is not None and _G_UP_ALPHA is not None:
            up_rgb = _extract_up_u8(_G_UP_RGB, ltx, lty, _G_PAD // 2)
            up_a = _extract_up_u8(_G_UP_ALPHA, ltx, lty, _G_PAD // 2)
            up_tile = np.dstack([up_rgb, up_a])
        else:
            up_tile = None

    res = encode_visual_tile_ecv32(
        tile, up_tile, allow_prune=_G_ALLOW_PRUNE, speed=_G_SPEED
    )
    if res is None:
        return None
    tpl_id, avif_b, grain_u8, q = res
    return ltx, lty, tpl_id, avif_b, grain_u8, q


def _worker_dem(args: tuple[int, int]) -> tuple[int, int, bytes, int, int] | None:
    ltx, lty = args
    assert _G_DEM is not None and _G_ALPHA is not None
    a_tile = _extract_patch(_G_ALPHA, ltx, lty, _G_PAD)
    if int(a_tile.max()) <= 130:
        return None
    dem_m = _extract_patch(_G_DEM, ltx, lty, _G_PAD).astype(np.float32) * 0.5
    up_m = (
        _extract_up_dem_m(_G_UP_DEM, ltx, lty, _G_PAD // 2)
        if _G_UP_DEM is not None
        else None
    )
    res = encode_height10_tile(
        dem_m,
        up_m,
        allow_prune=_G_ALLOW_PRUNE,
        q=_G_DEM_Q,
        speed=6,
        px_m=_G_DEM_PX_M,
    )
    if res is None:
        return None
    avif_b, aux, q = res
    return ltx, lty, avif_b, aux, q


def remote_exists(upload_url: str, token: str, key: str) -> bool:
    req = urllib.request.Request(
        f"{upload_url.rstrip('/')}/{key}",
        method="HEAD",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "EarthConnect-ECV3-Packer/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200 and int(r.headers.get("Content-Length", "0")) > 1024
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def upload_pack(upload_url: str, token: str, key: str, path: Path) -> None:
    data = path.read_bytes()
    url = f"{upload_url.rstrip('/')}/{key}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method="PUT",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                    "User-Agent": "EarthConnect-ECV3-Packer/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status not in (200, 201):
                    raise RuntimeError(f"Upload failed HTTP {r.status}")
                return
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  [upload retry {attempt + 1}] {key}: {e}", flush=True)
            time.sleep(2.0 * (attempt + 1))


def process_pack(
    pack_id: int,
    pinfo: dict,
    bm_img: Image.Image,
    out_dir: Path,
    workers: int,
    io_workers: int,
    dem_q: int,
    speed: int,
    upload_url: str | None,
    upload_token: str | None,
    force: bool,
    keep_local: bool,
) -> dict | None:
    global _G_RGB, _G_ALPHA, _G_UP_RGB, _G_UP_ALPHA, _G_DEM, _G_UP_DEM
    global _G_PAD, _G_ALLOW_PRUNE, _G_SPEED, _G_DEM_Q, _G_DEM_PX_M

    r2_key = f"ecv3/earth/earth-ecv3-p{pack_id:03d}.ecv3"
    out_path = out_dir / f"earth-ecv3-p{pack_id:03d}.ecv3"

    if not force:
        if (
            upload_url
            and upload_token
            and remote_exists(upload_url, upload_token, r2_key)
        ):
            print(
                f"[earth p{pack_id:03d}] Already uploaded to R2 — skipping.", flush=True
            )
            return None
        if not upload_url and out_path.exists() and out_path.stat().st_size > 1024:
            print(
                f"[earth p{pack_id:03d}] Already exists locally — skipping.", flush=True
            )
            return None

    t0 = time.time()
    band = pinfo["band"]
    lon_blk = pinfo["lon_blk"]
    lat_mid = 90.0 - (band + 0.5) * LAT_STEP
    lon0, lat0, lon1, lat1 = pack_canvas_bounds(band, lon_blk)
    out_w = PACK_W_L5 + 2 * PAD_L5  # 16,384
    out_h = PACK_H_L5 + 2 * PAD_L5  # 16,384

    cop_list: list[str] = pinfo["cop"]
    s2_pri: list[str] = pinfo["s2_pri"]
    s2_fb: list[str] = pinfo["s2_fb"]
    pri_q: str = pinfo["pri_q"]
    fb_q: str = pinfo["fb_q"]

    print(
        f"\n=== [earth p{pack_id:03d}] band={band:02d} lon_blk={lon_blk:02d} "
        f"lat=[{90 - (band + 1) * LAT_STEP:+.2f},{90 - band * LAT_STEP:+.2f}] "
        f"lon=[{-180 + lon_blk * LON_STEP:+.2f},{-180 + (lon_blk + 1) * LON_STEP:+.2f}] "
        f"| cop={len(cop_list)} s2_pri={len(s2_pri)} ({pri_q}) ===",
        flush=True,
    )

    # 1. Allocate square L5 pack canvases (Blue Marble base RGB, ocean alpha=128, non_ocean=0, 0m DEM)
    t_step = time.time()
    bm_l5 = init_blue_marble_canvas(bm_img, band, lon_blk, out_w, out_h)
    rgb_l5 = bm_l5.copy()
    alpha_l5 = np.full((out_h, out_w), 128, dtype=np.uint8)
    non_ocean_l5 = np.zeros((out_h, out_w), dtype=np.uint8)
    dem_l5 = np.zeros((out_h, out_w), dtype=np.int16)
    print(
        f"  [init] Blue Marble canvas {out_w}x{out_h} in {time.time() - t_step:.1f}s",
        flush=True,
    )

    # 2. Stream Copernicus DEM GLO-30 (~60 m) + 30 m Water Body Mask
    t_step = time.time()
    cop_ok = 0
    if cop_list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
            futs = [
                ex.submit(
                    fetch_copernicus_tile, name, lon0, lat0, lon1, lat1, out_w, out_h
                )
                for name in cop_list
            ]
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                x0, y0, x1, y1, d_tile, a_tile, no_tile = res
                mask = (a_tile > 128) | (d_tile != 0) | (no_tile > 0)
                if np.any(mask):
                    dst_a = alpha_l5[y0:y1, x0:x1]
                    dst_d = dem_l5[y0:y1, x0:x1]
                    dst_no = non_ocean_l5[y0:y1, x0:x1]
                    dst_a[mask] = np.maximum(dst_a[mask], a_tile[mask])
                    dst_no[mask] = np.maximum(dst_no[mask], no_tile[mask])
                    dst_d[mask] = d_tile[mask]
                cop_ok += 1
    land_frac = float((alpha_l5 > 130).mean())
    print(
        f"  [copernicus] {cop_ok}/{len(cop_list)} DEM+WBM tiles in {time.time() - t_step:.1f}s | land={land_frac * 100:.1f}%",
        flush=True,
    )

    if land_frac < 1e-6:
        print(
            f"  [earth p{pack_id:03d}] 0.0% land after Copernicus mask — skipping pure-ocean pack.",
            flush=True,
        )
        return None

    # 3. Stream Sentinel-2 Cloudless Mosaics (ov_level=1 -> 40m native for sharp antialiased ~61m L5)
    t_step = time.time()
    ov_level = 1
    covered = np.zeros((out_h, out_w), dtype=bool)
    s2_ok = 0

    if s2_pri:
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
            futs = [
                ex.submit(
                    fetch_sentinel2_tile,
                    pri_q,
                    folder,
                    ov_level,
                    lon0,
                    lat0,
                    lon1,
                    lat1,
                    out_w,
                    out_h,
                )
                for folder in s2_pri
            ]
            for fut in concurrent.futures.as_completed(futs):
                patches = fut.result()
                if not patches:
                    continue
                for x0, y0, x1, y1, rgb_p, valid in patches:
                    dst_c = covered[y0:y1, x0:x1]
                    write_m = valid & (~dst_c)
                    if np.any(write_m):
                        rgb_l5[y0:y1, x0:x1][write_m] = rgb_p[write_m]
                        dst_c[write_m] = True
                s2_ok += 1

    land_mask = alpha_l5 > 130
    uncovered_land = float((land_mask & (~covered)).sum()) / max(
        1, int(land_mask.sum())
    )
    print(
        f"  [sentinel-2 {pri_q}] {s2_ok}/{len(s2_pri)} MGRS tiles in {time.time() - t_step:.1f}s | "
        f"land covered={(1.0 - uncovered_land) * 100:.2f}%",
        flush=True,
    )

    # 3b. Opposite-quarter fallback if >0.25% of land is still uncovered
    if uncovered_land > 0.0025 and s2_fb:
        t_fb = time.time()
        fb_ok = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
            futs = [
                ex.submit(
                    fetch_sentinel2_tile,
                    fb_q,
                    folder,
                    ov_level,
                    lon0,
                    lat0,
                    lon1,
                    lat1,
                    out_w,
                    out_h,
                )
                for folder in s2_fb
            ]
            for fut in concurrent.futures.as_completed(futs):
                patches = fut.result()
                if not patches:
                    continue
                for x0, y0, x1, y1, rgb_p, valid in patches:
                    dst_c = covered[y0:y1, x0:x1]
                    write_m = valid & (~dst_c)
                    if np.any(write_m):
                        rgb_l5[y0:y1, x0:x1][write_m] = rgb_p[write_m]
                        dst_c[write_m] = True
                fb_ok += 1
        uncovered_after = float((land_mask & (~covered)).sum()) / max(
            1, int(land_mask.sum())
        )
        print(
            f"  [sentinel-2 fallback {fb_q}] {fb_ok}/{len(s2_fb)} tiles in {time.time() - t_fb:.1f}s | "
            f"land covered={(1.0 - uncovered_after) * 100:.2f}%",
            flush=True,
        )
    del covered, land_mask

    # 3c. Smoothly crossfade open ocean (WBM == 1 beyond ~1 km coastal shelf) to clean Blue Marble ocean
    t_oc = time.time()
    blend_open_ocean_to_blue_marble(rgb_l5, bm_l5, non_ocean_l5)
    del bm_l5, non_ocean_l5
    gc.collect()
    print(
        f"  [ocean-blend] Blended open ocean to clean Blue Marble in {time.time() - t_oc:.1f}s",
        flush=True,
    )

    # Clamp polar vertical padding on band 0 (North Pole) and band 15 (South Pole)
    if band == 0:
        rgb_l5[:PAD_L5] = rgb_l5[PAD_L5 : PAD_L5 + 1]
        alpha_l5[:PAD_L5] = alpha_l5[PAD_L5 : PAD_L5 + 1]
        dem_l5[:PAD_L5] = dem_l5[PAD_L5 : PAD_L5 + 1]
    elif band == BANDS - 1:
        rgb_l5[-PAD_L5:] = rgb_l5[-PAD_L5 - 1 : -PAD_L5]
        alpha_l5[-PAD_L5:] = alpha_l5[-PAD_L5 - 1 : -PAD_L5]
        dem_l5[-PAD_L5:] = dem_l5[-PAD_L5 - 1 : -PAD_L5]

    # 4. Build L5 -> L0 mip pyramids
    t_mip = time.time()
    rgb_mips: list[np.ndarray | None] = [None] * LEVEL_COUNT
    alpha_mips: list[np.ndarray | None] = [None] * LEVEL_COUNT
    dem_mips: list[np.ndarray | None] = [None] * LEVEL_COUNT

    rgb_mips[MAX_LEVEL] = rgb_l5
    alpha_mips[MAX_LEVEL] = alpha_l5
    dem_mips[MAX_LEVEL] = dem_l5
    for lv in range(MAX_LEVEL - 1, -1, -1):
        rgb_mips[lv], alpha_mips[lv] = downsample2x_rgb_alpha(
            rgb_mips[lv + 1], alpha_mips[lv + 1]
        )  # type: ignore[arg-type]
        dem_mips[lv] = downsample2x_i16(dem_mips[lv + 1])  # type: ignore[arg-type]
    print(
        f"  [mips] Built L5..L0 RGB+Alpha+DEM pyramids in {time.time() - t_mip:.1f}s",
        flush=True,
    )

    # 5. Encode L0..L5 tiles using zero-copy Linux fork pools
    templates: dict[int, tuple[bytes, int, int]] = {}
    records: list[Ecv3Record] = []
    ctx = mp.get_context("fork")

    cos_mid = max(math.cos(math.radians(lat_mid)), 0.15)
    t_enc = time.time()
    for lv in range(LEVEL_COUNT):
        tx_base = lon_blk * (1 << lv)
        ty_base = band * (1 << lv)
        ntx = 1 << lv
        nty = 1 << lv
        pad = GUTTER << lv

        _G_RGB = rgb_mips[lv]
        _G_ALPHA = alpha_mips[lv]
        _G_UP_RGB = rgb_mips[lv - 1] if lv > 0 else None
        _G_UP_ALPHA = alpha_mips[lv - 1] if lv > 0 else None
        _G_PAD = pad
        _G_ALLOW_PRUNE = lv > 0
        _G_SPEED = speed

        coords = [(ltx, lty) for lty in range(nty) for ltx in range(ntx)]
        vis_kept = 0
        vis_bytes = 0
        with ctx.Pool(processes=workers) as pool:
            for item in pool.imap_unordered(
                _worker_visual, coords, chunksize=max(1, len(coords) // (workers * 4))
            ):
                if item is None:
                    continue
                ltx, lty, tpl_id, avif_b, grain_u8, q = item
                if tpl_id == TPL_RGBA8:
                    if tpl_id not in templates:
                        templates[tpl_id] = (b"", 0, 0)
                    obu = avif_b
                else:
                    hdr, obu, iloc_off, mdat_off = strip_avif_header(avif_b)
                    if tpl_id not in templates:
                        templates[tpl_id] = (hdr, iloc_off, mdat_off)
                records.append(
                    Ecv3Record(
                        key=ecv3_tile_key(0, lv, tx_base + ltx, ty_base + lty),
                        tpl_id=tpl_id,
                        obu=obu,
                        grain_u8=grain_u8,
                        quality=q,
                    )
                )
                vis_kept += 1
                vis_bytes += len(obu)

        _G_DEM = dem_mips[lv]
        _G_UP_DEM = dem_mips[lv - 1] if lv > 0 else None
        _G_DEM_Q = dem_q
        cols_lv = BASE_COLS << lv
        dy_m = (math.pi * EARTH_RADIUS_M) / ((BASE_ROWS << lv) * INNER)
        dx_m = (2.0 * math.pi * EARTH_RADIUS_M * cos_mid) / (cols_lv * INNER)
        _G_DEM_PX_M = math.sqrt(dx_m * dy_m)

        dem_kept = 0
        dem_bytes = 0
        with ctx.Pool(processes=workers) as pool:
            for item in pool.imap_unordered(
                _worker_dem, coords, chunksize=max(1, len(coords) // (workers * 4))
            ):
                if item is None:
                    continue
                ltx, lty, avif_b, aux, q = item
                hdr, obu, iloc_off, mdat_off = strip_avif_header(avif_b)
                if TPL_HEIGHT10 not in templates:
                    templates[TPL_HEIGHT10] = (hdr, iloc_off, mdat_off)
                records.append(
                    Ecv3Record(
                        key=ecv3_tile_key(1, lv, tx_base + ltx, ty_base + lty),
                        tpl_id=TPL_HEIGHT10,
                        obu=obu,
                        quality=q,
                        aux=aux,
                    )
                )
                dem_kept += 1
                dem_bytes += len(obu)

        print(
            f"  [L{lv}] grid={ntx}x{nty} ({len(coords):4d}) | "
            f"vis={vis_kept:4d} ({vis_bytes / 1e6:5.2f} MB) | "
            f"dem={dem_kept:4d} ({dem_bytes / 1e6:5.2f} MB, px={_G_DEM_PX_M:.0f}m)",
            flush=True,
        )

        if lv > 0:
            rgb_mips[lv - 1] = None
            alpha_mips[lv - 1] = None
            dem_mips[lv - 1] = None

    _G_RGB = _G_ALPHA = _G_UP_RGB = _G_UP_ALPHA = _G_DEM = _G_UP_DEM = None
    rgb_mips.clear()
    alpha_mips.clear()
    dem_mips.clear()
    gc.collect()

    if not records:
        print(f"  [earth p{pack_id:03d}] 0 tiles kept — skipping write.", flush=True)
        return None

    sz = write_ecv3_pack(
        out_path,
        BODY_IDS["earth"],
        pack_id,
        BASE_COLS,
        BASE_ROWS,
        LEVEL_COUNT,
        templates,
        records,
    )
    info = verify_ecv3_pack(out_path, sample_n=12)
    print(
        f"  [OK p{pack_id:03d}] {sz / 1e6:.2f} MB | {info['tiles']} tiles | tpls={info['templates']} | "
        f"verified={info['verified_decoded']} | enc={time.time() - t_enc:.1f}s | total={time.time() - t0:.1f}s",
        flush=True,
    )

    if upload_url and upload_token:
        t_up = time.time()
        upload_pack(upload_url, upload_token, r2_key, out_path)
        print(
            f"  [UPLOADED] {r2_key} ({sz / 1e6:.2f} MB) in {time.time() - t_up:.1f}s",
            flush=True,
        )
        if not keep_local:
            out_path.unlink(missing_ok=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Earth ~61m ECV3 regional packs.")
    ap.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).parent / "earth_sources_index.json.gz",
    )
    ap.add_argument("--blue-marble", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/ecv3_earth"))
    ap.add_argument(
        "--packs", type=str, default="", help="Comma-separated pack IDs (e.g. 133)"
    )
    ap.add_argument(
        "--shard",
        type=str,
        default="",
        help="Shard i/N across all land packs (0-indexed)",
    )
    ap.add_argument("--workers", type=int, default=max(4, os.cpu_count() or 8))
    ap.add_argument("--io-workers", type=int, default=20)
    ap.add_argument("--dem-q", type=int, default=88)
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--upload-url", type=str, default=os.environ.get("UPLOAD_URL", ""))
    ap.add_argument(
        "--upload-token", type=str, default=os.environ.get("UPLOAD_TOKEN", "")
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-local", action="store_true")
    args = ap.parse_args()

    with gzip.open(args.index, "rt", encoding="utf-8") as f:
        idx = json.load(f)
    packs_map: dict[str, dict] = idx["packs"]
    all_land_pids = sorted(int(k) for k in packs_map.keys())

    if args.packs:
        target_pids = [int(x.strip()) for x in args.packs.split(",") if x.strip()]
    elif args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        target_pids = [pid for i, pid in enumerate(all_land_pids) if i % sn == si]
    else:
        target_pids = all_land_pids

    bm_path = args.blue_marble
    if bm_path is None:
        p = Path(__file__).resolve()
        repo_bm = (
            (p.parents[2] / "public/textures/earth-blue-marble.jpg")
            if len(p.parents) > 2
            else Path("/app/earth-blue-marble.jpg")
        )
        bm_path = repo_bm if repo_bm.exists() else Path("/app/earth-blue-marble.jpg")
    elif not bm_path.exists():
        bm_path = Path("/app/earth-blue-marble.jpg")
    bm_img = Image.open(bm_path).convert("RGB")

    print(
        f"Earth ECV3 packer starting: {len(target_pids)}/{len(all_land_pids)} land packs | "
        f"workers={args.workers} io={args.io_workers} dem_q={args.dem_q} speed={args.speed}",
        flush=True,
    )

    done = 0
    failed = []
    for pid in target_pids:
        pinfo = packs_map.get(str(pid))
        if not pinfo:
            continue
        try:
            process_pack(
                pack_id=pid,
                pinfo=pinfo,
                bm_img=bm_img,
                out_dir=args.out_dir,
                workers=args.workers,
                io_workers=args.io_workers,
                dem_q=args.dem_q,
                speed=args.speed,
                upload_url=args.upload_url or None,
                upload_token=args.upload_token or None,
                force=args.force,
                keep_local=args.keep_local,
            )
            done += 1
        except Exception as e:
            print(f"[ERROR p{pid:03d}] {e}", flush=True)
            failed.append(pid)

    print(
        f"\nFinished shard: {done}/{len(target_pids)} packs processed, {len(failed)} failed: {failed}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
