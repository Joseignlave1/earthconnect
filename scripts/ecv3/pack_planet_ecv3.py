#!/usr/bin/env python3
"""Streaming, sharded, low-RAM, resumable ECV3 packer for Moon (100m WAC + 60m SLDEM2015)
and Mars (100m THEMIS + 200m HRSC/MOLA).

Design:
  - 12 horizontal latitude bands (each 15° tall -> aligns cleanly to BASE_ROWS=12 across all mips).
  - Each band is a single contiguous HTTP Range read on uncompressed row-major USGS S3 GeoTIFFs.
  - Zero-copy Linux `fork` worker pools share the band array via COW (passing only (tx, lty) ints
    instead of pickling 512x512 arrays -> < 4 GB peak RAM even on 195K x 8K Moon SLDEM2015!).
  - Emits 2 regional .ecv3 packs per band (West / East = 24 packs per planet, ~20-60 MB each).
  - Remote-resumable via HTTP HEAD against the upload Worker.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import math
import multiprocessing as mp
import os
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from ecv3_format import (
    BODY_IDS,
    GUTTER,
    INNER,
    TILE_SIZE,
    TPL_HEIGHT10,
    Ecv3Record,
    ecv3_tile_key,
    encode_height10_tile,
    encode_visual_tile_ecv32,
    strip_avif_header,
    verify_ecv3_pack,
    write_ecv3_pack,
)

Image.MAX_IMAGE_PIXELS = None

BASE_COLS = 24
BASE_ROWS = 12

PLANET_SPECS = {
    "moon": {
        "radius_m": 1_737_400.0,
        "albedo_url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/LRO_WAC_Mosaic_Global_303ppd.tif",
        "albedo_lat_range": (90.0, -90.0),
        "albedo_levels": 4,  # L0:24x12 .. L3:192x96 (~112 m/px)
        "dem_url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/LolaKaguya_Topo/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif",
        "dem_lat_range": (60.0, -60.0),
        "dem_scale_m": 0.5,
        "dem_levels": 5,  # L0..L4: 384x192 (~56 m/px within ±60°)
        "level_count": 5,
    },
    "mars": {
        "radius_m": 3_389_500.0,
        "albedo_url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/THEMIS_controlled_mosaics/AA_mosaics/THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018.tif",
        "albedo_lat_range": (65.0, -65.0),
        "albedo_levels": 5,  # L0..L4: 384x192 (~109 m/px within ±65°)
        "dem_url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
        "dem_lat_range": (90.0, -90.0),
        "dem_scale_m": 1.0,
        "dem_levels": 4,  # L0..L3: 192x96 (~219 m/px)
        "level_count": 5,
    },
}

# Read-only globals shared via Linux fork COW (zero IPC copy!)
_G_BAND: np.ndarray | None = None
_G_UP: np.ndarray | None = None
_G_PAD_TOP: int = 0
_G_DEM_SCALE: float = 1.0
_G_DEM_PX_M: float = 200.0
_G_ALLOW_PRUNE: bool = False
_G_SPEED: int = 8
_G_DEM_Q: int = 74


def http_range(url: str, start: int, length: int) -> bytes:
    end = start + length - 1
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != length:
                raise IOError(f"Short range read: got {len(data)} != {length}")
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


class RemoteStripTiff:
    TYPE_SIZES = {1: 1, 3: 2, 4: 4, 16: 8}
    TYPE_FMTS = {1: "B", 3: "H", 4: "I", 16: "Q"}

    def __init__(self, url: str):
        self.url = url
        hdr = http_range(url, 0, 16)
        self.order = "<" if hdr[:2] == b"II" else ">"
        self.big = struct.unpack(self.order + "H", hdr[2:4])[0] == 43
        ifd_off = struct.unpack(
            self.order + ("Q" if self.big else "I"), hdr[8:16] if self.big else hdr[4:8]
        )[0]
        ifd_buf = http_range(url, ifd_off, 65536)
        n = struct.unpack(
            self.order + ("Q" if self.big else "H"), ifd_buf[: 8 if self.big else 2]
        )[0]
        ent_sz, off0 = (20, 8) if self.big else (12, 2)
        self.tags: dict[int, tuple[int, int, bytes]] = {}
        for i in range(n):
            o = off0 + i * ent_sz
            if self.big:
                tag, typ, cnt = struct.unpack(self.order + "HHQ", ifd_buf[o : o + 12])
                raw = ifd_buf[o + 12 : o + 20]
            else:
                tag, typ, cnt = struct.unpack(self.order + "HHI", ifd_buf[o : o + 8])
                raw = ifd_buf[o + 8 : o + 12]
            self.tags[tag] = (typ, cnt, raw)

        self.w = self._tag_int(256)
        self.h = self._tag_int(257)
        self.bps = self._tag_int(258, 8)
        self.comp = self._tag_int(259, 1)
        self.sf = self._tag_int(339, 1)
        self.rps = self._tag_int(278, 1)
        assert self.comp == 1 and self.rps == 1, (
            f"Expected uncompressed rps=1, got comp={self.comp} rps={self.rps}"
        )
        self.dtype = (
            np.int16
            if (self.bps == 16 and self.sf == 2)
            else (np.uint16 if self.bps == 16 else np.uint8)
        )
        self.row_bytes = self.w * (self.bps // 8)
        offs = self._strip_offsets(0, 2)
        assert offs[1] == offs[0] + self.row_bytes, "Non-contiguous strips!"
        self.data_base = offs[0]

    def _tag_int(self, tag: int, default: int = 0) -> int:
        if tag not in self.tags:
            return default
        typ, cnt, raw = self.tags[tag]
        sz = self.TYPE_SIZES[typ]
        max_inline = 8 if self.big else 4
        if cnt * sz <= max_inline:
            return struct.unpack(self.order + self.TYPE_FMTS[typ], raw[:sz])[0]
        off = struct.unpack(self.order + ("Q" if self.big else "I"), raw)[0]
        return struct.unpack(
            self.order + self.TYPE_FMTS[typ], http_range(self.url, off, sz)
        )[0]

    def _strip_offsets(self, start: int, count: int) -> list[int]:
        typ, _, raw = self.tags[273]
        sz = self.TYPE_SIZES[typ]
        base = struct.unpack(self.order + ("Q" if self.big else "I"), raw)[0]
        buf = http_range(self.url, base + start * sz, count * sz)
        return list(struct.unpack(self.order + self.TYPE_FMTS[typ] * count, buf))

    def read_rows(self, y0: int, y1: int, workers: int = 16) -> np.ndarray:
        y0 = max(0, min(self.h, y0))
        y1 = max(y0, min(self.h, y1))
        rows = y1 - y0
        if rows <= 0:
            return np.zeros((0, self.w), dtype=self.dtype)
        start = self.data_base + y0 * self.row_bytes
        total = rows * self.row_bytes
        buf = bytearray(total)
        mv = memoryview(buf)
        chunk = 8 * 1024 * 1024
        ranges = [(s, min(chunk, total - s)) for s in range(0, total, chunk)]

        def fetch_into(seg: tuple[int, int]) -> None:
            s, ln = seg
            mv[s : s + ln] = http_range(self.url, start + s, ln)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fetch_into, ranges))
        mv.release()
        return np.frombuffer(buf, dtype=self.dtype).reshape(rows, self.w)


def parallel_resize_u8(
    src: np.ndarray, dst_h: int, dst_w: int, threads: int = 8
) -> np.ndarray:
    """Fast low-RAM multi-threaded exact-box strip resize for huge uint8 latitude bands."""
    sh, sw = src.shape
    out = np.empty((dst_h, dst_w), dtype=np.uint8)
    step = 128

    def do_strip(y0: int) -> None:
        y1 = min(dst_h, y0 + step)
        fy0 = y0 * sh / dst_h
        fy1 = y1 * sh / dst_h
        sy0 = max(0, int(np.floor(fy0)) - 1)
        sy1 = min(sh, int(np.ceil(fy1)) + 1)
        sub = src[sy0:sy1]
        box = (0.0, fy0 - sy0, float(sw), fy1 - sy0)
        im = Image.fromarray(sub, mode="L").resize(
            (dst_w, y1 - y0), Image.Resampling.BILINEAR, box=box
        )
        out[y0:y1] = np.asarray(im, dtype=np.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(do_strip, range(0, dst_h, step)))
    return out


def parallel_resize_i16(
    src_i16: np.ndarray, dst_h: int, dst_w: int, threads: int = 8
) -> np.ndarray:
    """Fast low-RAM multi-threaded exact-box strip resize keeping output in int16."""
    sh, sw = src_i16.shape
    out = np.empty((dst_h, dst_w), dtype=np.int16)
    step = 64

    def do_strip(y0: int) -> None:
        y1 = min(dst_h, y0 + step)
        fy0 = y0 * sh / dst_h
        fy1 = y1 * sh / dst_h
        sy0 = max(0, int(np.floor(fy0)) - 1)
        sy1 = min(sh, int(np.ceil(fy1)) + 1)
        sub = src_i16[sy0:sy1].astype(np.float32)
        box = (0.0, fy0 - sy0, float(sw), fy1 - sy0)
        im = Image.fromarray(sub, mode="F").resize(
            (dst_w, y1 - y0), Image.Resampling.BILINEAR, box=box
        )
        arr = np.array(im, dtype=np.float32, copy=True)
        np.rint(arr, out=arr)
        np.clip(arr, -32760.0, 32767.0, out=arr)
        out[y0:y1] = arr.astype(np.int16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(do_strip, range(0, dst_h, step)))
    return out


def downsample_2x_u8(core: np.ndarray, th: int, tw: int) -> np.ndarray:
    out = np.empty((th, tw), dtype=np.uint8)
    for r0 in range(0, th, 256):
        r1 = min(th, r0 + 256)
        blk = core[r0 * 2 : r1 * 2].reshape(r1 - r0, 2, tw, 2).astype(np.uint16)
        out[r0:r1] = ((blk.sum(axis=(1, 3)) + 2) >> 2).astype(np.uint8)
    return out


def downsample_2x_i16(core: np.ndarray, th: int, tw: int) -> np.ndarray:
    out = np.empty((th, tw), dtype=np.int16)
    for r0 in range(0, th, 256):
        r1 = min(th, r0 + 256)
        blk = core[r0 * 2 : r1 * 2].reshape(r1 - r0, 2, tw, 2).astype(np.int32)
        out[r0:r1] = (blk.sum(axis=(1, 3)) // 4).astype(np.int16)
    return out


def _destripe_block_f32(h: np.ndarray, rx: int, ry: int, clamp_m: float) -> np.ndarray:
    H, W = h.shape
    pad_x = np.pad(h, ((0, 0), (rx, rx)), mode="edge")
    cs_x = np.cumsum(np.pad(pad_x, ((0, 0), (1, 0))), axis=1, dtype=np.float64)
    hp_x = h - ((cs_x[:, 2 * rx + 1 :] - cs_x[:, :W]) / (2 * rx + 1)).astype(np.float32)
    del pad_x, cs_x
    hp_c = clamp_m * np.tanh(hp_x / clamp_m)
    del hp_x
    pad_y = np.pad(hp_c, ((ry, ry), (0, 0)), mode="edge")
    del hp_c
    cs_y = np.cumsum(np.pad(pad_y, ((1, 0), (0, 0))), axis=0, dtype=np.float64)
    stripe = ((cs_y[2 * ry + 1 :, :] - cs_y[:H, :]) / (2 * ry + 1)).astype(np.float32)
    del pad_y, cs_y
    return h - stripe


def destripe_dem_band_i16(
    band_i16: np.ndarray, scale_m: float, threads: int = 4
) -> np.ndarray:
    """Two-scale directional notch destriper on a continuous latitude band (removes LOLA N-S laser tracks
    with zero tile seams and full longitude wrap, using <1.5 GB scratch RAM)."""
    H, W = band_i16.shape
    out = np.empty_like(band_i16)
    hx = 12
    col_step = 4096
    inv_scale = 1.0 / scale_m

    def do_chunk(x0: int) -> None:
        x1 = min(W, x0 + col_step)
        xs = np.arange(x0 - hx, x1 + hx) % W
        sub = band_i16[:, xs].astype(np.float32) * scale_m
        sub = _destripe_block_f32(sub, rx=6, ry=20, clamp_m=18.0)
        sub = _destripe_block_f32(sub, rx=3, ry=12, clamp_m=10.0)
        core = sub[:, hx : hx + (x1 - x0)] * inv_scale
        np.rint(core, out=core)
        np.clip(core, -32760.0, 32767.0, out=core)
        out[:, x0:x1] = core.astype(np.int16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(do_chunk, range(0, W, col_step)))
    return out


def fetch_lat_window(
    rt: RemoteStripTiff,
    src_lat_top: float,
    src_lat_bot: float,
    req_lat_top: float,
    req_lat_bot: float,
    dst_h: int,
    dst_w: int,
    is_dem: bool = False,
) -> np.ndarray | None:
    if req_lat_bot >= src_lat_top or req_lat_top <= src_lat_bot:
        return None
    c_top = min(req_lat_top, src_lat_top)
    c_bot = max(req_lat_bot, src_lat_bot)
    lat_span = src_lat_top - src_lat_bot
    sy0 = max(0, int(np.floor((src_lat_top - c_top) / lat_span * rt.h)) - 2)
    sy1 = min(rt.h, int(np.ceil((src_lat_top - c_bot) / lat_span * rt.h)) + 2)
    raw = rt.read_rows(sy0, sy1)
    if raw.size == 0:
        return None

    if is_dem:
        # Low-RAM chunked nodata replacement (never allocate full-band boolean masks or float64 median!)
        sample = raw[::16, ::16]
        valid_s = sample[sample > -32760]
        fill = int(np.median(valid_s)) if valid_s.size > 0 else 0
        del sample, valid_s
        for r0 in range(0, raw.shape[0], 512):
            blk = raw[r0 : r0 + 512]
            m = blk <= -32760
            if m.any():
                blk[m] = fill
        if c_top == req_lat_top and c_bot == req_lat_bot:
            res = parallel_resize_i16(raw, dst_h, dst_w)
            del raw
            gc.collect()
            return res
        out = np.zeros((dst_h, dst_w), dtype=np.int16)
        req_span = req_lat_top - req_lat_bot
        dy0 = int(round((req_lat_top - c_top) / req_span * dst_h))
        dy1 = max(dy0 + 1, int(round((req_lat_top - c_bot) / req_span * dst_h)))
        out[dy0:dy1] = parallel_resize_i16(raw, dy1 - dy0, dst_w)
        del raw
        gc.collect()
        if dy0 > 0:
            out[:dy0] = out[dy0 : dy0 + 1]
        if dy1 < dst_h:
            out[dy1:] = out[dy1 - 1 : dy1]
        return out
    else:
        if c_top == req_lat_top and c_bot == req_lat_bot:
            res = parallel_resize_u8(raw, dst_h, dst_w)
            del raw
            gc.collect()
            return res
        out = np.zeros((dst_h, dst_w), dtype=np.uint8)
        req_span = req_lat_top - req_lat_bot
        dy0 = int(round((req_lat_top - c_top) / req_span * dst_h))
        dy1 = max(dy0 + 1, int(round((req_lat_top - c_bot) / req_span * dst_h)))
        out[dy0:dy1] = parallel_resize_u8(raw, dy1 - dy0, dst_w)
        del raw
        gc.collect()
        return out


def _extract_tile(band: np.ndarray, tx: int, local_ty: int, pad_top: int) -> np.ndarray:
    bh, bw = band.shape
    x0 = tx * INNER - GUTTER
    y0 = pad_top + local_ty * INNER - GUTTER
    xs = np.arange(x0, x0 + TILE_SIZE) % bw
    ys = np.clip(np.arange(y0, y0 + TILE_SIZE), 0, bh - 1)
    return band[np.ix_(ys, xs)]


def _worker_vis(job):
    key, half, tx, lty = job
    assert _G_BAND is not None
    tile = _extract_tile(_G_BAND, tx, lty, _G_PAD_TOP)
    if tile.max() == 0:
        return key, half, None
    up_tile = _extract_tile(_G_UP, tx, lty, GUTTER) if _G_UP is not None else None
    res = encode_visual_tile_ecv32(
        tile, up_tile, allow_prune=_G_ALLOW_PRUNE, speed=_G_SPEED
    )
    return key, half, res


def _worker_dem(job):
    key, half, tx, lty = job
    assert _G_BAND is not None
    tile = _extract_tile(_G_BAND, tx, lty, _G_PAD_TOP).astype(np.float32) * _G_DEM_SCALE
    up_tile = (
        (_extract_tile(_G_UP, tx, lty, GUTTER).astype(np.float32) * _G_DEM_SCALE)
        if _G_UP is not None
        else None
    )
    res = encode_height10_tile(
        tile, up_tile, allow_prune=_G_ALLOW_PRUNE, q=_G_DEM_Q, speed=6, px_m=_G_DEM_PX_M
    )
    return key, half, res


def remote_exists(upload_url: str, token: str, key: str) -> bool:
    url = f"{upload_url.rstrip('/')}/{key.lstrip('/')}"
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "EarthConnect-ECV3-Packer/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def upload_file(path: Path, upload_url: str, token: str, key: str) -> None:
    url = f"{upload_url.rstrip('/')}/{key.lstrip('/')}"
    data = path.read_bytes()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method="PUT",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "EarthConnect-ECV3-Packer/1.0",
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(data)),
                },
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                if 200 <= r.status < 300:
                    return
                raise IOError(f"Upload HTTP {r.status}")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))


def pack_band(
    body: str,
    band_idx: int,
    out_dir: Path,
    speed: int = 8,
    dem_q: int = 74,
    workers: int = 16,
    upload_url: str | None = None,
    upload_token: str | None = None,
    keep_local: bool = True,
    force: bool = False,
) -> list[dict]:
    global \
        _G_BAND, \
        _G_UP, \
        _G_PAD_TOP, \
        _G_DEM_SCALE, \
        _G_DEM_PX_M, \
        _G_ALLOW_PRUNE, \
        _G_SPEED, \
        _G_DEM_Q
    assert 0 <= band_idx < BASE_ROWS
    spec = PLANET_SPECS[body]
    body_id = BODY_IDS[body]
    PACKS_PER_BAND = (
        4  # 4 x 90° quarters per 15° latitude band -> 48 packs/planet (~20-45 MB each)
    )
    pack_ids = [band_idx * PACKS_PER_BAND + q for q in range(PACKS_PER_BAND)]
    out_paths = [out_dir / f"{body}-ecv3-p{pid:02d}.ecv3" for pid in pack_ids]
    remote_keys = [f"ecv3/{body}/{p.name}" for p in out_paths]

    if not force:
        if upload_url and upload_token:
            if all(remote_exists(upload_url, upload_token, k) for k in remote_keys):
                print(
                    f"[{body} band {band_idx:02d}] Already uploaded to R2, skipping.",
                    flush=True,
                )
                return []
        elif all(p.exists() for p in out_paths):
            print(
                f"[{body} band {band_idx:02d}] Already exists locally, skipping.",
                flush=True,
            )
            return [verify_ecv3_pack(p, sample_n=2) for p in out_paths]

    t_band = time.time()
    lat_top = 90.0 - band_idx * (180.0 / BASE_ROWS)
    lat_bot = 90.0 - (band_idx + 1) * (180.0 / BASE_ROWS)
    print(
        f"\n=== [{body} band {band_idx:02d}/11] lat {lat_top:+.1f}° .. {lat_bot:+.1f}° ===",
        flush=True,
    )

    templates_by_half: list[dict[int, tuple[bytes, int, int]]] = [
        {} for _ in range(PACKS_PER_BAND)
    ]
    records_by_half: list[list[Ecv3Record]] = [[] for _ in range(PACKS_PER_BAND)]
    ctx = mp.get_context("fork")

    # ---- STAGE 1: ALBEDO (fetch -> build mips -> encode level-by-level -> free RAM!) ----
    rt_alb = RemoteStripTiff(spec["albedo_url"])
    alb_max_lv = spec["albedo_levels"] - 1
    alb_rows_b = 1 << alb_max_lv
    alb_cols = BASE_COLS << alb_max_lv
    alb_h, alb_w = alb_rows_b * INNER, alb_cols * INNER
    pad_px = GUTTER * (1 << alb_max_lv)
    deg_px = (lat_top - lat_bot) / alb_h
    req_top = min(90.0, lat_top + pad_px * deg_px)
    req_bot = max(-90.0, lat_bot - pad_px * deg_px)
    pt_real = int(round((req_top - lat_top) / deg_px))
    pb_real = int(round((lat_bot - req_bot) / deg_px))

    t0 = time.time()
    alb_fine = fetch_lat_window(
        rt_alb,
        *spec["albedo_lat_range"],
        req_top,
        req_bot,
        alb_h + pt_real + pb_real,
        alb_w,
        is_dem=False,
    )
    print(f"  Albedo fetched ({alb_w}x{alb_h}) in {time.time() - t0:.1f}s", flush=True)

    if alb_fine is not None:
        alb_mips: dict[int, tuple[np.ndarray, int]] = {alb_max_lv: (alb_fine, pt_real)}
        cur, cur_pt = alb_fine, pt_real
        for lv in range(alb_max_lv - 1, -1, -1):
            th, tw = (1 << lv) * INNER, (BASE_COLS << lv) * INNER
            core = cur[cur_pt : cur_pt + (1 << (lv + 1)) * INNER]
            ds = downsample_2x_u8(core, th, tw)
            ds_pad = np.pad(ds, ((GUTTER, GUTTER), (0, 0)), mode="edge")
            del ds
            alb_mips[lv] = (ds_pad, GUTTER)
            cur, cur_pt = ds_pad, GUTTER
        del cur, alb_fine

        t_enc = time.time()
        kept_v, pruned_v = 0, 0
        for lv in range(alb_max_lv + 1):
            band_arr, pt = alb_mips[lv]
            rows_b, cols_g = 1 << lv, BASE_COLS << lv
            q_cols = cols_g // PACKS_PER_BAND
            up_arr = None
            if lv > 0:
                p_arr, p_pt = alb_mips.pop(lv - 1)
                p_core = p_arr[p_pt : p_pt + (1 << (lv - 1)) * INNER]
                up_core = parallel_resize_u8(p_core, rows_b * INNER, cols_g * INNER)
                del p_arr, p_core
                up_arr = np.pad(up_core, ((GUTTER, GUTTER), (0, 0)), mode="edge")
                del up_core

            _G_BAND, _G_UP, _G_PAD_TOP = band_arr, up_arr, pt
            _G_ALLOW_PRUNE = lv >= (alb_max_lv - 1)
            _G_SPEED = speed
            gc.collect()
            jobs = [
                (
                    ecv3_tile_key(0, lv, tx, band_idx * rows_b + lty),
                    min(PACKS_PER_BAND - 1, tx // q_cols),
                    tx,
                    lty,
                )
                for lty in range(rows_b)
                for tx in range(cols_g)
            ]
            with ctx.Pool(processes=workers) as pool:
                for key, half, res in pool.imap_unordered(
                    _worker_vis, jobs, chunksize=8
                ):
                    if res is None:
                        pruned_v += 1
                        continue
                    kept_v += 1
                    tpl_id, avif_b, grain_u8, used_q = res
                    hdr, obu, ioff, moff = strip_avif_header(avif_b)
                    if tpl_id not in templates_by_half[half]:
                        templates_by_half[half][tpl_id] = (hdr, ioff, moff)
                    records_by_half[half].append(
                        Ecv3Record(
                            key=key,
                            tpl_id=tpl_id,
                            obu=obu,
                            grain_u8=grain_u8,
                            quality=used_q,
                        )
                    )
            _G_BAND = _G_UP = None
            del up_arr, band_arr
        alb_mips.clear()
        gc.collect()
        print(
            f"  Albedo encoded: {kept_v} kept, {pruned_v} pruned in {time.time() - t_enc:.1f}s",
            flush=True,
        )

    # ---- STAGE 2: 10-BIT HEIGHT DEM (fetch -> build int16 mips -> encode -> free RAM!) ----
    rt_dem = RemoteStripTiff(spec["dem_url"])
    dem_max_lv = spec["dem_levels"] - 1
    dem_rows_b = 1 << dem_max_lv
    dem_cols = BASE_COLS << dem_max_lv
    dem_h, dem_w = dem_rows_b * INNER, dem_cols * INNER
    d_pad_px = GUTTER * (1 << dem_max_lv)
    d_deg_px = (lat_top - lat_bot) / dem_h
    d_req_top = min(90.0, lat_top + d_pad_px * d_deg_px)
    d_req_bot = max(-90.0, lat_bot - d_pad_px * d_deg_px)
    d_pt = int(round((d_req_top - lat_top) / d_deg_px))
    d_pb = int(round((lat_bot - d_req_bot) / d_deg_px))

    t0 = time.time()
    dem_fine = fetch_lat_window(
        rt_dem,
        *spec["dem_lat_range"],
        d_req_top,
        d_req_bot,
        dem_h + d_pt + d_pb,
        dem_w,
        is_dem=True,
    )
    print(
        f"  DEM fetched    ({dem_w}x{dem_h if dem_fine is not None else 0}) in {time.time() - t0:.1f}s",
        flush=True,
    )

    if dem_fine is not None:
        if body == "moon":
            t_ds = time.time()
            dem_clean = destripe_dem_band_i16(dem_fine, spec["dem_scale_m"])
            del dem_fine
            dem_fine = dem_clean
            gc.collect()
            print(
                f"  DEM destriped  (2-scale LOLA notch) in {time.time() - t_ds:.1f}s",
                flush=True,
            )
        dem_mips: dict[int, tuple[np.ndarray, int]] = {dem_max_lv: (dem_fine, d_pt)}
        cur, cur_pt = dem_fine, d_pt
        for lv in range(dem_max_lv - 1, -1, -1):
            th, tw = (1 << lv) * INNER, (BASE_COLS << lv) * INNER
            core = cur[cur_pt : cur_pt + (1 << (lv + 1)) * INNER]
            ds = downsample_2x_i16(core, th, tw)
            ds_pad = np.pad(ds, ((GUTTER, GUTTER), (0, 0)), mode="edge")
            del ds
            dem_mips[lv] = (ds_pad, GUTTER)
            cur, cur_pt = ds_pad, GUTTER
        del cur, dem_fine

        t_enc = time.time()
        kept_d, pruned_d = 0, 0
        for lv in range(dem_max_lv + 1):
            band_arr, pt = dem_mips[lv]
            rows_b, cols_g = 1 << lv, BASE_COLS << lv
            q_cols = cols_g // PACKS_PER_BAND
            up_arr = None
            if lv > 0:
                p_arr, p_pt = dem_mips.pop(lv - 1)
                p_core = p_arr[p_pt : p_pt + (1 << (lv - 1)) * INNER]
                up_core = parallel_resize_i16(p_core, rows_b * INNER, cols_g * INNER)
                del p_arr, p_core
                up_arr = np.pad(up_core, ((GUTTER, GUTTER), (0, 0)), mode="edge")
                del up_core

            _G_BAND, _G_UP, _G_PAD_TOP = band_arr, up_arr, pt
            _G_DEM_SCALE = spec["dem_scale_m"]
            _G_DEM_PX_M = (math.pi * spec["radius_m"]) / (BASE_ROWS * rows_b * INNER)
            _G_ALLOW_PRUNE = lv >= (dem_max_lv - 1)
            _G_DEM_Q = dem_q
            gc.collect()
            jobs = [
                (
                    ecv3_tile_key(1, lv, tx, band_idx * rows_b + lty),
                    min(PACKS_PER_BAND - 1, tx // q_cols),
                    tx,
                    lty,
                )
                for lty in range(rows_b)
                for tx in range(cols_g)
            ]
            with ctx.Pool(processes=workers) as pool:
                for key, half, res in pool.imap_unordered(
                    _worker_dem, jobs, chunksize=8
                ):
                    if res is None:
                        pruned_d += 1
                        continue
                    kept_d += 1
                    avif_b, aux, used_q = res
                    hdr, obu, ioff, moff = strip_avif_header(avif_b)
                    if TPL_HEIGHT10 not in templates_by_half[half]:
                        templates_by_half[half][TPL_HEIGHT10] = (hdr, ioff, moff)
                    records_by_half[half].append(
                        Ecv3Record(
                            key=key,
                            tpl_id=TPL_HEIGHT10,
                            obu=obu,
                            quality=used_q,
                            aux=aux,
                        )
                    )
            _G_BAND = _G_UP = None
            del up_arr, band_arr
            print(
                f"    DEM L{lv}: {kept_d} kept, {pruned_d} pruned ({time.time() - t_enc:.1f}s)",
                flush=True,
            )
        dem_mips.clear()
        gc.collect()

    summaries = []
    for half in range(PACKS_PER_BAND):
        if not records_by_half[half]:
            continue
        pid = pack_ids[half]
        out_p = out_paths[half]
        sz = write_ecv3_pack(
            out_p,
            body_id,
            pid,
            BASE_COLS,
            BASE_ROWS,
            spec["level_count"],
            templates_by_half[half],
            records_by_half[half],
        )
        ver = verify_ecv3_pack(out_p, sample_n=6)
        print(
            f"  Wrote & verified {out_p.name}: {sz / 1e6:.2f} MB ({ver['tiles']} tiles, tpls={ver['templates']}, ok={ver['verified_decoded']})",
            flush=True,
        )
        if upload_url and upload_token:
            t_up = time.time()
            upload_file(out_p, upload_url, upload_token, remote_keys[half])
            print(
                f"  Uploaded {remote_keys[half]} in {time.time() - t_up:.1f}s",
                flush=True,
            )
            if not keep_local:
                out_p.unlink()
        summaries.append(ver)

    print(
        f"=== [{body} band {band_idx:02d}] Total wall time: {time.time() - t_band:.1f}s ===",
        flush=True,
    )
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", choices=["moon", "mars"], required=True)
    ap.add_argument(
        "--bands", type=str, default=None, help="Comma-separated band indices 0..11"
    )
    ap.add_argument("--shard", type=str, default=None, help="Shard spec i/N (e.g. 0/6)")
    ap.add_argument("--out-dir", type=Path, default=Path("temp/ecv3-out"))
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--dem-q", type=int, default=92)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 16))
    ap.add_argument("--upload-url", type=str, default=os.environ.get("ECV3_UPLOAD_URL"))
    ap.add_argument(
        "--upload-token", type=str, default=os.environ.get("ECV3_UPLOAD_TOKEN")
    )
    ap.add_argument("--delete-after-upload", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-encode and overwrite existing remote/local packs",
    )
    args = ap.parse_args()

    if args.bands is not None:
        band_list = [int(x) for x in args.bands.split(",") if x.strip()]
    elif args.shard is not None:
        idx, total = (int(x) for x in args.shard.split("/"))
        band_list = [b for b in range(BASE_ROWS) if b % total == idx]
    else:
        band_list = list(range(BASE_ROWS))

    print(
        f"Starting {args.body} ECV3 packer: bands={band_list}, workers={args.workers}, speed={args.speed}",
        flush=True,
    )
    all_res = []
    for b in band_list:
        all_res.extend(
            pack_band(
                args.body,
                b,
                args.out_dir,
                speed=args.speed,
                dem_q=args.dem_q,
                workers=args.workers,
                upload_url=args.upload_url,
                upload_token=args.upload_token,
                keep_local=not args.delete_after_upload,
                force=args.force,
            )
        )
    print(
        f"\nDONE {args.body}: {len(all_res)} packs, {sum(r['bytes'] for r in all_res) / 1e6:.2f} MB total."
    )


if __name__ == "__main__":
    main()
