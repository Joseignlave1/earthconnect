"""Build Moon & Mars paired albedo + relief-normal ECVT1 virtual tile packs.

Grid (per body):
  - Level 0: 24 x 12 tiles -> 12,192 x 6,096 logical px
  - Level 1: 48 x 24 tiles -> 24,384 x 12,192 logical px
    * Moon (r = 1,737.4 km): ~448 m/px (LROC WAC 400m color + LOLA LDEM_64 474m)
    * Mars (r = 3,396.2 km): ~875 m/px (USGS Viking 925m color + MOLA MEG064 925m)
  - Each tile is 512 x 512 WebP = 508 x 508 logical + 2 px seamless neighbor gutter.
  - 4 quadrant packs per body: {body}-tiles-0..3.bin (0=NW, 1=NE, 2=SW, 3=SE).
  - Key encoding: (kind << 20) | (level << 16) | (ty << 8) | tx
      kind 0 = sRGB display albedo
      kind 1 = tangent-space unit normal (R=east*0.5+0.5, G=north*0.5+0.5, B=up)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import os
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 512
GUTTER = 2
INNER = TILE_SIZE - 2 * GUTTER  # 508
BASE_COLS = 24
BASE_ROWS = 12
LEVELS = 2  # L0: 24x12, L1: 48x24
FULL_W = BASE_COLS * (1 << (LEVELS - 1)) * INNER  # 24384
FULL_H = BASE_ROWS * (1 << (LEVELS - 1)) * INNER  # 12192

ECVT_MAGIC = 0x54564345  # 'ECVT' little-endian
ECVT_VERSION = 1

BODY_PARAMS = {
    "moon": {"radius_m": 1_737_400.0, "relief": 2.2, "neutralize_viking_blue": False},
    "mars": {"radius_m": 3_396_190.0, "relief": 3.0, "neutralize_viking_blue": True},
}


def tile_key(kind: int, level: int, tx: int, ty: int) -> int:
    return (kind << 20) | (level << 16) | (ty << 8) | tx


def neutralize_viking_blue_inplace(arr: np.ndarray) -> None:
    """Match prepare-planet-terrain.py Mars Viking blue neutralization exactly."""
    rgb = arr.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    g_fixed = np.where(g > r, r + (g - r) * 0.25, g)
    b_fixed = np.where(b > g_fixed, g_fixed * 0.92 + (b - g_fixed) * 0.12, b)
    ice = np.clip((np.minimum(r, np.minimum(g, b)) - 130.0) / 50.0, 0.0, 1.0)
    rgb[:, :, 1] = g_fixed * (1.0 - ice) + g * ice
    rgb[:, :, 2] = b_fixed * (1.0 - ice) + b * ice
    np.clip(np.rint(rgb), 0, 255, out=rgb)
    arr[...] = rgb.astype(np.uint8)


def load_albedo_24k(path: Path, neutralize_blue: bool) -> np.ndarray:
    t0 = time.time()
    print(f"Loading albedo {path.name} -> {FULL_W}x{FULL_H} ...", flush=True)
    im = Image.open(path).convert("RGB")
    # Heal 1-pixel no-data black border columns (e.g. USGS Mars Viking col 0 at ±180°)
    # BEFORE resizing so Lanczos never bleeds black into the antimeridian seam.
    src_arr = np.asarray(im, dtype=np.uint8)
    if float(src_arr[:, 0].mean()) < 10.0 and float(src_arr[:, 1].mean()) > 20.0:
        src_arr = src_arr.copy()
        src_arr[:, 0] = ((src_arr[:, -1].astype(np.uint16) + src_arr[:, 1].astype(np.uint16) + 1) >> 1).astype(np.uint8)
        im.close()
        im = Image.fromarray(src_arr, mode="RGB")
    if im.size != (FULL_W, FULL_H):
        im = im.resize((FULL_W, FULL_H), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8).copy()
    im.close()
    if neutralize_blue:
        # Process in row bands to keep peak RAM modest
        for y0 in range(0, FULL_H, 1024):
            y1 = min(FULL_H, y0 + 1024)
            neutralize_viking_blue_inplace(arr[y0:y1])
    arr[0, :, :] = np.rint(arr[0].mean(axis=0)).astype(np.uint8)
    arr[-1, :, :] = np.rint(arr[-1].mean(axis=0)).astype(np.uint8)
    print(f"  albedo ready in {time.time() - t0:.1f}s", flush=True)
    return arr


def resample_dem_to_24k(raw_f32: np.ndarray) -> np.ndarray:
    """Bilinearly resample a 0°-centered float32 DEM (src_h, src_w) to (FULL_H, FULL_W)."""
    src_h, src_w = raw_f32.shape
    if (src_w, src_h) == (FULL_W, FULL_H):
        return raw_f32
    out = np.empty((FULL_H, FULL_W), dtype=np.float32)
    # Vectorized bilinear in row bands
    x_coord = (np.arange(FULL_W, dtype=np.float32) + 0.5) * (src_w / FULL_W) - 0.5
    x0 = np.floor(x_coord).astype(np.int32)
    fx = (x_coord - x0).astype(np.float32)
    x0 %= src_w
    x1 = (x0 + 1) % src_w

    y_coord = (np.arange(FULL_H, dtype=np.float32) + 0.5) * (src_h / FULL_H) - 0.5
    y0_all = np.clip(np.floor(y_coord).astype(np.int32), 0, src_h - 1)
    y1_all = np.clip(y0_all + 1, 0, src_h - 1)
    fy_all = np.clip(y_coord - y0_all, 0.0, 1.0).astype(np.float32)

    for y in range(FULL_H):
        r0 = raw_f32[y0_all[y]]
        r1 = raw_f32[y1_all[y]]
        v0 = r0[x0] * (1.0 - fx) + r0[x1] * fx
        v1 = r1[x0] * (1.0 - fx) + r1[x1] * fx
        fy = fy_all[y]
        out[y] = v0 * (1.0 - fy) + v1 * fy
    return out


def load_moon_dem_24k(path: Path) -> np.ndarray:
    t0 = time.time()
    print(f"Loading Moon LOLA LDEM_64 {path.name} -> {FULL_W}x{FULL_H} ...", flush=True)
    src_w, src_h = 23040, 11520
    mem = np.memmap(path, dtype="<i2", mode="r", shape=(src_h, src_w))
    raw = np.empty((src_h, src_w), dtype=np.float32)
    half = src_w // 2
    # Shift 180° horizontally so 0° longitude sits at center (matches prepare-planet-terrain.py)
    raw[:, :half] = mem[:, half:].astype(np.float32) * 0.5
    raw[:, half:] = mem[:, :half].astype(np.float32) * 0.5
    del mem
    dem = resample_dem_to_24k(raw)
    del raw
    print(f"  Moon DEM ready in {time.time() - t0:.1f}s", flush=True)
    return dem


def load_mars_dem_24k(src_dir: Path) -> np.ndarray:
    t0 = time.time()
    print(f"Loading Mars MOLA MEG064 quadrants -> {FULL_W}x{FULL_H} ...", flush=True)
    qw, qh = 11520, 5760
    raw = np.empty((qh * 2, qw * 2), dtype=np.float32)
    # 0°-centered layout: west half [-180..0] = 180..360E (*180gb), east half [0..+180] = 0..180E (*000gb)
    quads = [
        ("megt90n180gb.img", 0, qh, 0, qw),
        ("megt90n000gb.img", 0, qh, qw, qw * 2),
        ("megt00n180gb.img", qh, qh * 2, 0, qw),
        ("megt00n000gb.img", qh, qh * 2, qw, qw * 2),
    ]
    for name, y0, y1, x0, x1 in quads:
        mem = np.memmap(src_dir / name, dtype=">i2", mode="r", shape=(qh, qw))
        raw[y0:y1, x0:x1] = mem.astype(np.float32)
        del mem
    dem = resample_dem_to_24k(raw)
    del raw
    print(f"  Mars DEM ready in {time.time() - t0:.1f}s", flush=True)
    return dem


def smooth_polar_longitudes_inplace(grid: np.ndarray) -> None:
    """Isotropic longitudinal Gaussian smoothing near high latitudes to prevent orbital spokes."""
    t0 = time.time()
    h, w = grid.shape
    freqs = np.fft.rfftfreq(w, d=1.0)
    two_pi_f = 2.0 * np.pi * freqs
    for y in range(1, h - 1):
        lat = (0.5 - (y + 0.5) / h) * np.pi
        ratio = 1.0 / max(1e-4, float(np.cos(lat)))
        if ratio < 1.35:
            continue
        sigma = 0.5 * np.sqrt(ratio * ratio - 1.0)
        kernel = np.exp(-0.5 * (two_pi_f * sigma) ** 2)
        row_fft = np.fft.rfft(grid[y].astype(np.float64))
        grid[y] = np.fft.irfft(row_fft * kernel, n=w).astype(np.float32)
    grid[0, :] = float(grid[0].mean())
    grid[-1, :] = float(grid[-1].mean())
    print(f"  polar smoothing done in {time.time() - t0:.1f}s", flush=True)


def dem_to_tangent_normals(dem: np.ndarray, radius_m: float, relief: float) -> np.ndarray:
    """Compute (H, W, 3) uint8 tangent-space normals matching planetTerrain.ts hillNormal."""
    t0 = time.time()
    h, w = dem.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    d_angle = np.pi / h  # equals 2*pi/w since w == 2*h
    scale_n = relief / (2.0 * radius_m * d_angle)

    for y0 in range(0, h, 512):
        y1 = min(h, y0 + 512)
        ym = np.maximum(0, np.arange(y0, y1) - 1)
        yp = np.minimum(h - 1, np.arange(y0, y1) + 1)
        band = dem[y0:y1]
        # East derivative (periodic wrap in x)
        dh_e = np.roll(band, -1, axis=1) - np.roll(band, 1, axis=1)
        # North derivative (row index decreases northward)
        dh_n = dem[ym] - dem[yp]

        lat = (0.5 - (np.arange(y0, y1, dtype=np.float32) + 0.5) / h) * np.pi
        cos_lat = np.maximum(0.05, np.cos(lat))[:, None]

        # Match hillNormal = normalize(p - slope_e * E - slope_n * N)
        # => tangent normal = normalize(-slope_e, -slope_n, 1)
        tx = -(dh_e * (scale_n / cos_lat))
        ty = -(dh_n * scale_n)
        inv_len = 1.0 / np.sqrt(tx * tx + ty * ty + 1.0)
        tx *= inv_len
        ty *= inv_len
        tz = inv_len

        out[y0:y1, :, 0] = np.clip(np.rint((tx * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        out[y0:y1, :, 1] = np.clip(np.rint((ty * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        out[y0:y1, :, 2] = np.clip(np.rint(tz * 255.0), 0, 255).astype(np.uint8)

    print(f"  tangent normals ({w}x{h}) ready in {time.time() - t0:.1f}s", flush=True)
    return out


def downsample_2x_rgb(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    a = img.reshape(h // 2, 2, w // 2, 2, 3).astype(np.uint16)
    return ((a.sum(axis=(1, 3)) + 2) >> 2).astype(np.uint8)


def downsample_2x_dem(dem: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    return dem.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3), dtype=np.float32)


def extract_tile_with_gutter(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    h, w, _ = img.shape
    x0 = tx * INNER - GUTTER
    y0 = ty * INNER - GUTTER
    xs = (np.arange(x0, x0 + TILE_SIZE) % w)
    ys = np.clip(np.arange(y0, y0 + TILE_SIZE), 0, h - 1)
    return img[np.ix_(ys, xs)]


def encode_webp_bytes(tile_rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(tile_rgb, mode="RGB").save(buf, format="WEBP", quality=quality, method=5)
    return buf.getvalue()


def build_body_packs(
    body: str,
    albedo_l1: np.ndarray,
    dem_l1: np.ndarray,
    out_dir: Path,
    albedo_quality: int,
    normal_quality: int,
    workers: int,
) -> None:
    t_body = time.time()
    params = BODY_PARAMS[body]
    smooth_polar_longitudes_inplace(dem_l1)

    albedo_l0 = downsample_2x_rgb(albedo_l1)
    dem_l0 = downsample_2x_dem(dem_l1)

    normal_l1 = dem_to_tangent_normals(dem_l1, params["radius_m"], params["relief"])
    normal_l0 = dem_to_tangent_normals(dem_l0, params["radius_m"], params["relief"])
    del dem_l0, dem_l1

    layers = [
        (0, 0, albedo_l0, BASE_COLS, BASE_ROWS, albedo_quality),
        (0, 1, albedo_l1, BASE_COLS * 2, BASE_ROWS * 2, albedo_quality),
        (1, 0, normal_l0, BASE_COLS, BASE_ROWS, normal_quality),
        (1, 1, normal_l1, BASE_COLS * 2, BASE_ROWS * 2, normal_quality),
    ]

    # 4 quadrant packs: 0=NW, 1=NE, 2=SW, 3=SE
    pack_buckets: list[list[tuple[int, bytes]]] = [[] for _ in range(4)]

    def encode_job(args):
        kind, level, tx, ty, img, q, pack_id = args
        tile = extract_tile_with_gutter(img, tx, ty)
        return pack_id, tile_key(kind, level, tx, ty), encode_webp_bytes(tile, q)

    jobs = []
    for kind, level, img, cols, rows, q in layers:
        half_x = cols // 2
        half_y = rows // 2
        for ty in range(rows):
            py = 0 if ty < half_y else 1
            for tx in range(cols):
                px = 0 if tx < half_x else 1
                pack_id = py * 2 + px
                jobs.append((kind, level, tx, ty, img, q, pack_id))

    print(f"Encoding {len(jobs)} WebP tiles for {body} across {workers} threads ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for pack_id, key, data in ex.map(encode_job, jobs):
            pack_buckets[pack_id].append((key, data))

    total_bytes = 0
    for pack_id, items in enumerate(pack_buckets):
        items.sort(key=lambda pair: pair[0])
        count = len(items)
        header_bytes = 16 + count * 12
        offset = header_bytes
        index_parts = []
        payload_parts = []
        for key, data in items:
            index_parts.append(struct.pack("<III", key, offset, len(data)))
            payload_parts.append(data)
            offset += len(data)
        out_path = out_dir / f"{body}-tiles-{pack_id}.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack("<IIII", ECVT_MAGIC, ECVT_VERSION, pack_id, count))
            for part in index_parts:
                f.write(part)
            for part in payload_parts:
                f.write(part)
        size = out_path.stat().st_size
        total_bytes += size
        print(f"  wrote {out_path.name}: {count} entries, {size / (1024 * 1024):.2f} MiB", flush=True)

    print(
        f"Finished {body}: {sum(len(b) for b in pack_buckets)} entries, "
        f"{total_bytes / (1024 * 1024):.2f} MiB total in {time.time() - t_body:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Directory with downloaded Moon/Mars sources")
    parser.add_argument("--output", type=Path, default=Path("public/textures"))
    parser.add_argument("--bodies", nargs="+", choices=["moon", "mars"], default=["moon", "mars"])
    parser.add_argument("--albedo-quality", type=int, default=84)
    parser.add_argument("--normal-quality", type=int, default=86)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if "moon" in args.bodies:
        albedo = load_albedo_24k(args.sources / "lroc_color_poles.tif", False)
        dem = load_moon_dem_24k(args.sources / "ldem_64.img")
        build_body_packs("moon", albedo, dem, args.output, args.albedo_quality, args.normal_quality, args.workers)
        del albedo, dem

    if "mars" in args.bodies:
        albedo = load_albedo_24k(args.sources / "Mars_Viking_ClrMosaic_global_925m.tif", True)
        dem = load_mars_dem_24k(args.sources)
        build_body_packs("mars", albedo, dem, args.output, args.albedo_quality, args.normal_quality, args.workers)
        del albedo, dem


if __name__ == "__main__":
    main()
