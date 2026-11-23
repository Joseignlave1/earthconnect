#!/usr/bin/env python3
"""Slice NASA Blue Marble Next Generation 500m (8-panel 86400x43200) into
8 regional ECVT1 tile packs (public/textures/earth-tiles-0..7.bin).

Each pack covers one 90°x90° octant across 3 dyadic zoom levels:
  - Level 0:  40 x 20 global tiles (20,320 x 10,160 logical, ~2.0 km/px)
  - Level 1:  80 x 40 global tiles (40,640 x 20,320 logical, ~986 m/px)
  - Level 2: 160 x 80 global tiles (81,280 x 40,640 logical, ~493 m/px)

Every stored tile is 512x512 WebP containing 508x508 logical pixels + a 2px
neighbor gutter on all 4 sides (wrapping seamlessly across ±180° and octant
boundaries) for seam-free hardware bilinear/anisotropic mipmap filtering.
Pure open-ocean tiles are omitted so the shader seamlessly uses the 8K base +
GEBCO bathymetry over open water.
"""
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ect_terrain import load_earth_coast

Image.MAX_IMAGE_PIXELS = None

PANELS = [
    ("A1", 0, 0), ("B1", 1, 0), ("C1", 2, 0), ("D1", 3, 0),
    ("A2", 0, 1), ("B2", 1, 1), ("C2", 2, 1), ("D2", 3, 1),
]
TILE_INNER = 508
GUTTER = 2
TILE_SIZE = TILE_INNER + 2 * GUTTER  # 512
LEVEL_SPECS = [
    # (level, tiles_per_octant, global_cols, global_rows, octant_px)
    (0, 10, 40, 20, 10 * TILE_INNER),   # 5080
    (1, 20, 80, 40, 20 * TILE_INNER),   # 10160
    (2, 40, 160, 80, 40 * TILE_INNER),  # 20320
]
WEBP_QUALITY = 74
WEBP_METHOD = 5


def resize_panel(args: tuple[str, Path, Path]) -> tuple[str, dict[int, Path]]:
    name, src_dir, work_dir = args
    src = src_dir / f"world.topo.200407.3x21600x21600.{name}.jpg"
    outputs: dict[int, Path] = {}
    # Resize 21600 -> 20320 (L2) via libvips (fast, low RAM, Lanczos3), then 2x downsample for L1 & L0.
    l2 = work_dir / f"{name}_L2.v"
    l1 = work_dir / f"{name}_L1.v"
    l0 = work_dir / f"{name}_L0.v"
    if not l2.exists():
        subprocess.run(
            ["vips", "resize", str(src), str(l2), str(20320 / 21600), "--kernel", "lanczos3"],
            check=True,
        )
    if not l1.exists():
        subprocess.run(["vips", "resize", str(l2), str(l1), "0.5", "--kernel", "lanczos3"], check=True)
    if not l0.exists():
        subprocess.run(["vips", "resize", str(l1), str(l0), "0.5", "--kernel", "lanczos3"], check=True)
    outputs[2] = l2
    outputs[1] = l1
    outputs[0] = l0
    return name, outputs


def load_vips_rgb(path: Path, size: int) -> np.ndarray:
    """Read a .v file into an (H, W, 3) uint8 numpy array using raw buffer extraction."""
    raw_path = path.with_suffix(".raw")
    if not raw_path.exists():
        subprocess.run(["vips", "rawsave", str(path), str(raw_path)], check=True)
    arr = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(size, size, 3))
    return arr


def is_open_ocean(tile_rgb: np.ndarray, coast_mask: np.ndarray, tx: int, ty: int, cols: int, rows: int) -> bool:
    """Return True only if the tile is guaranteed pure open ocean (no land/islands)."""
    ch, cw = coast_mask.shape
    x0 = max(0, int((tx - 0.15) * cw / cols))
    x1 = min(cw, int((tx + 1.15) * cw / cols) + 1)
    y0 = max(0, int((ty - 0.15) * ch / rows))
    y1 = min(ch, int((ty + 1.15) * ch / rows) + 1)
    if float(coast_mask[y0:y1, x0:x1].max()) >= -2.0:
        return False
    # Also inspect the actual 512x512 Blue Marble pixels so tiny oceanic islands/atolls
    # smaller than the 10 km GEBCO mask are never dropped.
    r = tile_rgb[:, :, 0]
    g = tile_rgb[:, :, 1]
    b = tile_rgb[:, :, 2]
    # NASA BMNG open ocean is very dark blue (R <= 28, G <= 52, B <= 98, B > R + 6)
    non_ocean = (r > 32) | (g > 58) | ((r > 18) & (r.astype(np.int16) >= b.astype(np.int16) - 4))
    if int(np.count_nonzero(non_ocean)) > 12:
        return False
    return True


def encode_tile_job(args: tuple[int, int, int, bytes]) -> tuple[int, bytes]:
    key, _, _, raw_rgb = args
    img = Image.frombytes("RGB", (TILE_SIZE, TILE_SIZE), raw_rgb)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
    return key, buf.getvalue()


def pad_octant(
    arrays: dict[tuple[int, int], np.ndarray],
    ox: int,
    oy: int,
    size: int,
) -> np.ndarray:
    """Assemble (size + 4, size + 4, 3) uint8 padded array with seamless 2px neighbor gutters."""
    padded = np.empty((size + 2 * GUTTER, size + 2 * GUTTER, 3), dtype=np.uint8)
    center = arrays[(ox, oy)]
    padded[GUTTER:GUTTER + size, GUTTER:GUTTER + size] = center

    left = arrays[((ox - 1) % 4, oy)]
    right = arrays[((ox + 1) % 4, oy)]
    padded[GUTTER:GUTTER + size, :GUTTER] = left[:, -GUTTER:]
    padded[GUTTER:GUTTER + size, GUTTER + size:] = right[:, :GUTTER]

    if oy == 0:
        # North pole: replicate row 0
        padded[:GUTTER, :] = padded[GUTTER:GUTTER + 1, :]
        south = arrays[(ox, 1)]
        padded[GUTTER + size:, GUTTER:GUTTER + size] = south[:GUTTER, :]
        padded[GUTTER + size:, :GUTTER] = arrays[((ox - 1) % 4, 1)][:GUTTER, -GUTTER:]
        padded[GUTTER + size:, GUTTER + size:] = arrays[((ox + 1) % 4, 1)][:GUTTER, :GUTTER]
    else:
        # South pole: replicate last row
        north = arrays[(ox, 0)]
        padded[:GUTTER, GUTTER:GUTTER + size] = north[-GUTTER:, :]
        padded[:GUTTER, :GUTTER] = arrays[((ox - 1) % 4, 0)][-GUTTER:, -GUTTER:]
        padded[:GUTTER, GUTTER + size:] = arrays[((ox + 1) % 4, 0)][-GUTTER:, :GUTTER]
        padded[GUTTER + size:, :] = padded[GUTTER + size - 1:GUTTER + size, :]

    return padded


def main() -> None:
    t0 = time.time()
    src_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = src_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    coast_mask = load_earth_coast(root / "public" / "textures" / "earth-terrain.bin")
    print(f"Loaded GEBCO coast mask {coast_mask.shape} in {time.time() - t0:.1f}s", flush=True)

    # 1. Resize all 8 panels in parallel using libvips
    t1 = time.time()
    panel_paths: dict[tuple[int, int], dict[int, Path]] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(resize_panel, [(name, src_dir, work_dir) for name, _, _ in PANELS]))
    by_name = dict(results)
    for name, ox, oy in PANELS:
        panel_paths[(ox, oy)] = by_name[name]
    print(f"Resized 8 panels (L0/L1/L2) in {time.time() - t1:.1f}s", flush=True)

    # 2. Extract raw mmap buffers for each level & panel
    level_arrays: dict[int, dict[tuple[int, int], np.ndarray]] = {0: {}, 1: {}, 2: {}}
    for level, _, _, _, octant_px in LEVEL_SPECS:
        for _, ox, oy in PANELS:
            level_arrays[level][(ox, oy)] = load_vips_rgb(panel_paths[(ox, oy)][level], octant_px)

    # 3. Slice and WebP-encode each of the 8 regional packs
    total_tiles = 0
    total_bytes = 0
    workers = min(16, os.cpu_count() or 8)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pack_id, (name, ox, oy) in enumerate(PANELS):
            tp = time.time()
            jobs: list[tuple[int, int, int, bytes]] = []
            skipped = 0
            for level, n_oct, cols, rows, octant_px in LEVEL_SPECS:
                padded = pad_octant(level_arrays[level], ox, oy, octant_px)
                base_tx = ox * n_oct
                base_ty = oy * n_oct
                for ly in range(n_oct):
                    py = ly * TILE_INNER
                    ty = base_ty + ly
                    for lx in range(n_oct):
                        px = lx * TILE_INNER
                        tx = base_tx + lx
                        tile = padded[py:py + TILE_SIZE, px:px + TILE_SIZE]
                        if is_open_ocean(tile, coast_mask, tx, ty, cols, rows):
                            skipped += 1
                            continue
                        key = (level << 16) | (ty << 8) | tx
                        jobs.append((key, tx, ty, np.ascontiguousarray(tile).tobytes()))
                del padded

            encoded = list(pool.map(encode_tile_job, jobs, chunksize=8))
            encoded.sort(key=lambda item: item[0])

            # Write ECVT1 binary pack
            header_size = 16 + len(encoded) * 12
            offset = header_size
            index_buf = bytearray(header_size)
            struct.pack_into("<4sIII", index_buf, 0, b"ECVT", 1, pack_id, len(encoded))
            for i, (key, data) in enumerate(encoded):
                struct.pack_into("<III", index_buf, 16 + i * 12, key, offset, len(data))
                offset += len(data)

            out_path = out_dir / f"earth-tiles-{pack_id}.bin"
            with out_path.open("wb") as f:
                f.write(index_buf)
                for _, data in encoded:
                    f.write(data)

            size_mb = out_path.stat().st_size / (1024 * 1024)
            total_tiles += len(encoded)
            total_bytes += out_path.stat().st_size
            print(
                f"Pack {pack_id} ({name}): {len(encoded)} land tiles ({skipped} ocean skipped) -> "
                f"{size_mb:.2f} MiB in {time.time() - tp:.1f}s",
                flush=True,
            )

    print(
        f"DONE: {total_tiles} tiles across 8 packs, total {total_bytes / (1024 * 1024):.2f} MiB "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
