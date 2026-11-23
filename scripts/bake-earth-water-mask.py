"""Bake exact ~463 m GEBCO_2026 + inland-lake water masks into earth-tiles-0..7.bin Alpha.

Encoding in each RGBA WebP tile (512x512, 508x508 logical + 2 px seamless gutter):
  - RGB: unchanged NASA 500 m Blue Marble pixels (exact=True preserves RGB everywhere)
  - Alpha: [128..255] where 128 = 100% water, 255 = 100% land, smoothly anti-aliased
           across shorelines. Keeping A >= 128 guarantees neither libwebp nor browser
           decoders ever zero-out or quantize-crush RGB on coastal reefs/shallows.

Streams only the needed GEBCO_2026 COG tiles (16-bit signed metres, 86400 x 43200)
via parallel HTTP 206 Range requests, one octant at a time (~500 MB peak RAM).
"""

from __future__ import annotations

import concurrent.futures
import io
import os
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ect_terrain import load_earth_coast

GEBCO_URL = "https://data.source.coop/ausantarctic/gebco/GEBCO_2026.tif"
GEBCO_W = 86400
GEBCO_H = 43200
COG_TILE = 512
COG_COLS = (GEBCO_W + COG_TILE - 1) // COG_TILE  # 169
COG_ROWS = (GEBCO_H + COG_TILE - 1) // COG_TILE  # 85
COG_COUNT = COG_COLS * COG_ROWS  # 14365

TILE_SIZE = 512
GUTTER = 2
INNER = 508
BASE_COLS = 40
BASE_ROWS = 20
FULL_W = BASE_COLS * 4 * INNER  # 81280
FULL_H = BASE_ROWS * 4 * INNER  # 40640
ECVT_MAGIC = 0x54564345


def fetch_gebco_index() -> tuple[tuple[int, ...], tuple[int, ...]]:
    req = urllib.request.Request(GEBCO_URL, headers={"Range": f"bytes=9148-{124068 + COG_COUNT * 4 - 1}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tbl = r.read()
    offsets = struct.unpack_from(f"<{COG_COUNT}Q", tbl, 0)
    counts = struct.unpack_from(f"<{COG_COUNT}I", tbl, 124068 - 9148)
    return offsets, counts


def fetch_cog_tile(args: tuple[int, int, int]) -> tuple[int, np.ndarray]:
    tid, off, cnt = args
    for attempt in range(5):
        try:
            req = urllib.request.Request(GEBCO_URL, headers={"Range": f"bytes={off}-{off + cnt - 1}"})
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = zlib.decompress(r.read())
            arr = np.frombuffer(raw, dtype="<i2").reshape(COG_TILE, COG_TILE).copy()
            u16 = np.cumsum(arr.view(np.uint16), axis=1, dtype=np.uint16)
            return tid, u16.view(np.int16)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def sample_gebco_window(
    tile_cache: dict[int, np.ndarray],
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> np.ndarray:
    """Gather GEBCO int16 elevations for arbitrary wrapped x_idx (W,) and clamped y_idx (H,)."""
    out = np.empty((len(y_idx), len(x_idx)), dtype=np.int16)
    cx = x_idx // COG_TILE
    lx = x_idx % COG_TILE
    cy = y_idx // COG_TILE
    ly = y_idx % COG_TILE
    for ucy in np.unique(cy):
        ymask = cy == ucy
        sub_ly = ly[ymask]
        for ucx in np.unique(cx):
            xmask = cx == ucx
            tid = int(ucy) * COG_COLS + int(ucx)
            tile = tile_cache.get(tid)
            if tile is None:
                out[np.ix_(ymask, xmask)] = -1000
            else:
                out[np.ix_(ymask, xmask)] = tile[np.ix_(sub_ly, lx[xmask])]
    return out


def compute_tile_land_alpha(
    rgb: np.ndarray,
    level: int,
    tx: int,
    ty: int,
    tile_cache: dict[int, np.ndarray],
    coast_4k: np.ndarray,
) -> np.ndarray:
    """Compute [128..255] uint8 land alpha for one 512x512 ECVT1 tile (128=water, 255=land)."""
    cols = BASE_COLS << level
    rows = BASE_ROWS << level
    lvl_w = cols * INNER
    lvl_h = rows * INNER

    # Pixel centers in [0, 1) geographic coordinates (with 2 px gutter wrapping/clamping)
    xs = ((np.arange(tx * INNER - GUTTER, tx * INNER - GUTTER + TILE_SIZE) % lvl_w) + 0.5) / lvl_w
    ys = (np.clip(np.arange(ty * INNER - GUTTER, ty * INNER - GUTTER + TILE_SIZE), 0, lvl_h - 1) + 0.5) / lvl_h

    # Sample 2x2 sub-grid at GEBCO native 86400x43200 for crisp anti-aliased coastlines
    # (for L2, pixel ratio is 86400/81280 = 1.063; for L1 it's 2.13; for L0 it's 4.25).
    sub_n = 2 if level == 2 else (3 if level == 1 else 4)
    dx_step = (1.0 / lvl_w) / sub_n
    dy_step = (1.0 / lvl_h) / sub_n

    # Macro 4K coast distance at tile pixels (connected ocean prior so Death Valley/Qattara stay land)
    ch, cw = coast_4k.shape
    c4_x = np.floor(xs * cw).astype(np.int32) % cw
    c4_y = np.clip(np.floor(ys * ch).astype(np.int32), 0, ch - 1)
    coast_macro = coast_4k[np.ix_(c4_y, c4_x)]

    # NASA BMNG RGB cues for polder protection, coastal intertidal shoals, and inland seas/lakes
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Bright green farmland polders below sea level (e.g. Netherlands) stay land
    green_polder = (g > r + 4) & (g > b + 12) & (g > 28)
    # NASA BMNG 500 m water signature (inland lakes & enclosed seas like Caspian, Great Lakes, Baikal, Victoria)
    navy_water = (r <= 10) & (g <= 18) & (b >= 6) & (b <= 34) & ((b - r) >= 4) & ((b - g) >= 2)
    dark_lake = (r <= 6) & (g <= 12) & (b <= 28) & (b >= r + 1)
    # Coastal turquoise intertidal shoals where BMNG high-tide water sits on GEBCO 0..+3m flats
    teal_shallow = (b > r + 10) & (g > r + 6) & (r < 55) & (b < 140)

    land_accum = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

    for sy in range(sub_n):
        oy = (sy + 0.5) / sub_n - 0.5
        gy = np.clip(np.floor((ys + oy * dy_step * sub_n) * GEBCO_H).astype(np.int32), 0, GEBCO_H - 1)
        for sx in range(sub_n):
            ox = (sx + 0.5) / sub_n - 0.5
            gx = np.floor(((xs + ox * dx_step * sub_n) % 1.0) * GEBCO_W).astype(np.int32) % GEBCO_W
            elev = sample_gebco_window(tile_cache, gx, gy)

            near_sea = coast_macro < 2.5
            is_ocean = near_sea & (
                ((elev <= 0) & ~(green_polder & (elev >= -15)))
                | ((elev <= 4) & (teal_shallow | navy_water | dark_lake))
            )
            land_accum += (~is_ocean).astype(np.float32)

    land_frac = land_accum / float(sub_n * sub_n)

    # Inland seas (Caspian, Aral, Dead Sea) and lakes (Great Lakes, Baikal, Victoria) have real
    # bathymetry slopes in GEBCO, so classify directly from NASA BMNG's 500m dark-water mask.
    inland_water = (coast_macro >= 2.5) & (dark_lake | navy_water)
    land_frac = np.where(inland_water, 0.0, land_frac)

    # Encode in [128..255]: 128 = 100% water, 255 = 100% land
    return np.clip(np.rint(128.0 + land_frac * 127.0), 128, 255).astype(np.uint8)


def needed_cog_ids_for_pack(entries: list[tuple[int, bytes]]) -> set[int]:
    cids: set[int] = set()
    for key, _ in entries:
        level = (key >> 16) & 0xf
        ty = (key >> 8) & 0xff
        tx = key & 0xff
        cols = BASE_COLS << level
        rows = BASE_ROWS << level
        lvl_w = cols * INNER
        lvl_h = rows * INNER
        u0 = ((tx * INNER - GUTTER - 2) % lvl_w) / lvl_w
        u1 = ((tx * INNER - GUTTER + TILE_SIZE + 2) % lvl_w) / lvl_w
        v0 = max(0.0, (ty * INNER - GUTTER - 2) / lvl_h)
        v1 = min(1.0, (ty * INNER - GUTTER + TILE_SIZE + 2) / lvl_h)
        cy0 = max(0, int(v0 * GEBCO_H) // COG_TILE)
        cy1 = min(COG_ROWS - 1, int(v1 * GEBCO_H) // COG_TILE)
        cx0 = int(u0 * GEBCO_W) // COG_TILE
        cx1 = int(u1 * GEBCO_W) // COG_TILE
        if cx0 <= cx1:
            cxs = range(cx0, cx1 + 1)
        else:
            cxs = list(range(cx0, COG_COLS)) + list(range(0, cx1 + 1))
        for cy in range(cy0, cy1 + 1):
            for cx in cxs:
                cids.add(cy * COG_COLS + cx)
    return cids


def process_pack(
    pack_id: int,
    tex_dir: Path,
    offsets: tuple[int, ...],
    counts: tuple[int, ...],
    coast_4k: np.ndarray,
    dl_workers: int,
    enc_workers: int,
) -> None:
    t0 = time.time()
    path = tex_dir / f"earth-tiles-{pack_id}.bin"
    raw = path.read_bytes()
    magic, ver, pid, cnt = struct.unpack_from("<IIII", raw, 0)
    assert magic == ECVT_MAGIC and ver == 1
    entries: list[tuple[int, bytes]] = []
    for i in range(cnt):
        key, off, length = struct.unpack_from("<III", raw, 16 + i * 12)
        entries.append((key, raw[off : off + length]))

    cids = sorted(needed_cog_ids_for_pack(entries))
    dl_bytes = sum(counts[c] for c in cids)
    print(f"[pack {pack_id}] {cnt} tiles -> fetching {len(cids)} GEBCO COG tiles ({dl_bytes / 1e6:.1f} MB) ...", flush=True)
    t_dl = time.time()
    tile_cache: dict[int, np.ndarray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=dl_workers) as ex:
        for tid, tile in ex.map(fetch_cog_tile, [(c, offsets[c], counts[c]) for c in cids]):
            tile_cache[tid] = tile
    print(f"[pack {pack_id}] fetched GEBCO in {time.time() - t_dl:.1f}s; encoding RGBA WebP ...", flush=True)

    def bake_one(item: tuple[int, bytes]) -> tuple[int, bytes]:
        key, blob = item
        level = (key >> 16) & 0xf
        ty = (key >> 8) & 0xff
        tx = key & 0xff
        rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)
        alpha = compute_tile_land_alpha(rgb, level, tx, ty, tile_cache, coast_4k)
        rgba = np.empty((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = alpha
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="WEBP", quality=84, alpha_quality=92, method=4, exact=True)
        return key, buf.getvalue()

    new_entries: list[tuple[int, bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=enc_workers) as ex:
        for res in ex.map(bake_one, entries):
            new_entries.append(res)

    tile_cache.clear()
    header_bytes = 16 + len(new_entries) * 12
    offset = header_bytes
    tmp = path.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<IIII", ECVT_MAGIC, 1, pack_id, len(new_entries)))
        for key, blob in new_entries:
            f.write(struct.pack("<III", key, offset, len(blob)))
            offset += len(blob)
        for _, blob in new_entries:
            f.write(blob)
    os.replace(tmp, path)
    print(
        f"[pack {pack_id}] DONE in {time.time() - t0:.1f}s -> {path.stat().st_size / (1024 * 1024):.2f} MiB",
        flush=True,
    )


def main() -> None:
    tex_dir = Path("public/textures")
    coast_4k = load_earth_coast(tex_dir / "earth-terrain.bin")
    offsets, counts = fetch_gebco_index()
    t_all = time.time()
    for pack_id in range(8):
        process_pack(pack_id, tex_dir, offsets, counts, coast_4k, dl_workers=20, enc_workers=min(16, os.cpu_count() or 8))
    total = sum((tex_dir / f"earth-tiles-{i}.bin").stat().st_size for i in range(8))
    print(f"ALL 8 EARTH PACKS BAKED WITH 463m GEBCO WATER MASK in {time.time() - t_all:.1f}s ({total / (1024 * 1024):.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
