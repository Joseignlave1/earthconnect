#!/usr/bin/env python3
"""Bake the unified 8192x4096 orbital Earth texture from the 384 published ECV3 Sentinel-2 L0 tiles
composited over NASA Blue Marble oceans so orbit -> 61 m uses one consistent dataset, season, and palette."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

CDN_BASE = "https://earth-connect-assets.pages.dev"
PACK_VERSION = "e1h"
BASE_COLS = 32
BASE_ROWS = 16
OUT_W = 8192
OUT_H = 4096
TILE_OUT = OUT_W // BASE_COLS  # 256

# Exact 512-bit mask of all 384 Earth land packs on R2 (matches src/shader-test/earth-tiles.ts).
EARTH_LAND_PACK_MASK = [
    0x03FF7FC0,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFF8FF7,
    0x3FFFA7F0,
    0x3FFFC7E3,
    0xDFFFE7E7,
    0xFFDFCFC7,
    0xFF7F7F8F,
    0xFD3EBE0F,
    0xFC1E1F5D,
    0xFC46CF01,
    0xC0593600,
    0x5FB03E00,
    0xFFFFFFFF,
    0xFFFFFFFF,
]


def has_earth_land_pack(pack_id: int) -> bool:
    return ((EARTH_LAND_PACK_MASK[pack_id >> 5] >> (pack_id & 31)) & 1) == 1


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c >= 0.04045, np.power((c + 0.055) / 1.055, 2.4), c / 12.92)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c >= 0.0031308, 1.055 * np.power(c, 1.0 / 2.4) - 0.055, c * 12.92)


def calibrate_s2_linear(raw_lin: np.ndarray) -> np.ndarray:
    """Exact match to calibrateS2() in src/shader-test/shaders/surface.wgsl."""
    sub = np.maximum(raw_lin - np.array([0.030, 0.030, 0.040], dtype=np.float32), 0.0)
    min_ch = np.minimum(raw_lin[..., 0], np.minimum(raw_lin[..., 1], raw_lin[..., 2]))
    t = np.clip((min_ch - 0.45) / (0.80 - 0.45), 0.0, 1.0)
    snow = (t * t * (3.0 - 2.0 * t))[..., None]
    scale = (1.0 - snow) * np.array(
        [0.593, 0.507, 0.516], dtype=np.float32
    ) + snow * np.array([0.57, 0.57, 0.59], dtype=np.float32)
    gamma = np.array([1.50, 1.45, 1.55], dtype=np.float32)
    return np.clip(scale * np.power(sub, gamma), 0.0, 1.0)


def fetch_range(pack_id: int, offset: int, length: int) -> bytes:
    url = f"{CDN_BASE}/ecv3/earth/earth-ecv3-p{pack_id:03d}.ecv3?v={PACK_VERSION}&r={offset}-{length}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "EarthConnect-OrbitalBake/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed fetching {url}")


def fetch_l0_tile(pack_id: int) -> tuple[int, np.ndarray]:
    hdr = fetch_range(pack_id, 0, 65536)
    magic, ver, _body, _pid, _tsz, _gut, _bc, _br, _lc, tpl_count, tile_count, _res = (
        struct.unpack_from("<4sIHHHHHHHHII", hdr, 0)
    )
    if magic != b"ECV3" or ver != 3:
        raise ValueError(f"Bad ECV3 header in p{pack_id:03d}")

    pos = 32
    templates: dict[int, tuple[bytes, int, int]] = {}
    for _ in range(tpl_count):
        tid, tlen, iloc_off, mdat_off = struct.unpack_from("<HHHH", hdr, pos)
        pos += 8
        tbytes = hdr[pos : pos + tlen]
        pos += (tlen + 3) & ~3
        templates[tid] = (tbytes, iloc_off, mdat_off)

    index_end = pos + tile_count * 20
    if index_end > len(hdr):
        hdr = hdr + fetch_range(pack_id, len(hdr), index_end - len(hdr))

    for i in range(tile_count):
        key, off, length, meta, _aux = struct.unpack_from("<IIIII", hdr, pos + i * 20)
        kind = (key >> 24) & 0xFF
        level = (key >> 20) & 0xF
        if kind == 0 and level == 0:
            tid = (meta >> 24) & 0xFF
            tbytes, iloc_off, mdat_off = templates[tid]
            if off + length <= len(hdr):
                obu = hdr[off : off + length]
            else:
                obu = fetch_range(pack_id, off, length)
            if len(tbytes) == 0:
                avif = obu
            else:
                tb = bytearray(tbytes)
                struct.pack_into(">I", tb, iloc_off, len(obu))
                struct.pack_into(">I", tb, mdat_off, len(obu) + 8)
                avif = bytes(tb) + obu
            img = Image.open(io.BytesIO(avif)).convert("RGBA")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # Strip 2 px gutter -> 508x508 exact inner tile.
            inner = arr[2:510, 2:510, :]
            return pack_id, inner

    raise ValueError(f"No L0 albedo tile found in p{pack_id:03d}")


def downsample_linear_tile(lin_508: np.ndarray) -> np.ndarray:
    """Downsample 508x508 linear float32 RGB -> 256x256 in linear space using Lanczos per channel."""
    out = np.empty((TILE_OUT, TILE_OUT, 3), dtype=np.float32)
    for ch in range(3):
        pil_ch = Image.fromarray(lin_508[:, :, ch], mode="F")
        out[:, :, ch] = np.asarray(
            pil_ch.resize((TILE_OUT, TILE_OUT), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blue-marble",
        type=Path,
        default=Path("public/textures/earth-blue-marble.jpg"),
    )
    parser.add_argument(
        "--sources-index",
        type=Path,
        default=Path("scripts/ecv3/earth_sources_index.json.gz"),
    )
    parser.add_argument(
        "--out-jpg", type=Path, default=Path("temp/earth-sentinel2-8k.jpg")
    )
    parser.add_argument(
        "--out-webp", type=Path, default=Path("temp/earth-sentinel2-8k.webp")
    )
    parser.add_argument("--preview-dir", type=Path, default=Path("temp"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()
    pack_ids = [pid for pid in range(BASE_COLS * BASE_ROWS) if has_earth_land_pack(pid)]
    print(f"Fetching {len(pack_ids)} L0 tiles...", flush=True)

    base_img = Image.open(args.blue_marble).convert("RGB")
    if base_img.size != (OUT_W, OUT_H):
        base_img = base_img.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
    canvas = np.asarray(base_img, dtype=np.uint8).copy()

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_l0_tile, pid) for pid in pack_ids]
        for fut in concurrent.futures.as_completed(futures):
            pid, inner = fut.result()
            band = pid // BASE_COLS
            lon_blk = pid % BASE_COLS
            rgb_lin = srgb_to_linear(inner[:, :, :3])
            land_mask = np.clip((inner[:, :, 3] - 0.5) * 2.0, 0.0, 1.0)[:, :, None]
            # Exact match to surface.wgsl: apply calibrateS2() on land pixels while keeping
            # open ocean pixels (land_mask == 0) untouched so skipped ocean packs match seamlessly.
            cal_lin = calibrate_s2_linear(rgb_lin)
            mixed_lin = rgb_lin * (1.0 - land_mask) + cal_lin * land_mask
            down_lin = downsample_linear_tile(mixed_lin)
            tile_u8 = np.clip(np.rint(linear_to_srgb(down_lin) * 255.0), 0, 255).astype(
                np.uint8
            )
            y0 = band * TILE_OUT
            x0 = lon_blk * TILE_OUT
            canvas[y0 : y0 + TILE_OUT, x0 : x0 + TILE_OUT, :] = tile_u8
            done += 1
            if done % 64 == 0 or done == len(pack_ids):
                print(
                    f"  {done}/{len(pack_ids)} tiles stitched ({time.time() - t0:.1f}s)",
                    flush=True,
                )

    out_img = Image.fromarray(canvas, mode="RGB")
    args.out_jpg.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.out_jpg, format="JPEG", quality=92, subsampling=1, optimize=True)
    out_img.save(args.out_webp, format="WEBP", quality=90, method=5)

    for p in (args.out_jpg, args.out_webp):
        data = p.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        print(f"Wrote {p} ({len(data) / 1e6:.2f} MB, sha256={h})")

    # Save visual inspection previews (global + South America / Patagonia crop)
    out_img.resize((2048, 1024), Image.Resampling.LANCZOS).save(
        args.preview_dir / "preview-earth-s2-global.jpg", quality=90
    )
    # South America crop: lon -85..-50 (u ~ 0.264..0.361), lat 0..-60 (v ~ 0.50..0.833)
    sa_box = (
        int(0.25 * OUT_W),
        int(0.48 * OUT_H),
        int(0.40 * OUT_W),
        int(0.85 * OUT_H),
    )
    out_img.crop(sa_box).save(args.preview_dir / "preview-earth-s2-sa.jpg", quality=92)
    base_img.crop(sa_box).save(args.preview_dir / "preview-earth-bm-sa.jpg", quality=92)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
