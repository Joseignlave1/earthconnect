"""Build the 4K display DEMs (ECT2) and 8K albedo textures from NASA/USGS sources.

See public/textures/CREDITS.txt for dataset citations and source URLs.
"""

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ECT2_MAGIC = 0x45435432
ECT2_FLAG_DELTA_SHUFFLE = 1
ECT2_FLAG_POLAR_SMOOTHED = 2


def area_resample_dem(source_path: Path, src_w: int, src_h: int, dtype: str, scale: float, dst_w: int, dst_h: int) -> np.ndarray:
    """Area-average a raw signed-int16 cylindrical PDS DEM into (dst_h, dst_w) float32 metres, shifted 180°."""
    mem = np.memmap(source_path, dtype=dtype, mode="r", shape=(src_h, src_w))
    out = np.empty((dst_h, dst_w), dtype=np.float32)
    half_x = src_w // 2
    band_rows = 256
    # Precompute horizontal bin boundaries for fast exact area averaging.
    x_edges = np.linspace(0, src_w, dst_w + 1, dtype=np.int64)
    x_counts = (x_edges[1:] - x_edges[:-1]).astype(np.float32)

    for y0 in range(0, dst_h, band_rows):
        y1 = min(dst_h, y0 + band_rows)
        sy0 = int(np.floor(y0 * src_h / dst_h))
        sy1 = int(np.ceil(y1 * src_h / dst_h))
        # Read band and roll 180° horizontally so 0° longitude sits at the center.
        band = np.empty((sy1 - sy0, src_w), dtype=np.float32)
        raw = mem[sy0:sy1]
        if np.any(raw == -32768):
            raise ValueError(f"{source_path} contains missing (-32768) samples in rows {sy0}..{sy1}")
        band[:, :half_x] = raw[:, half_x:]
        band[:, half_x:] = raw[:, :half_x]
        band *= scale

        # Exact vertical area averaging into (y1 - y0, src_w)
        y_edges = np.linspace(y0 * src_h / dst_h, y1 * src_h / dst_h, y1 - y0 + 1)
        vert = np.empty((y1 - y0, src_w), dtype=np.float32)
        for i in range(y1 - y0):
            r0 = int(np.floor(y_edges[i])) - sy0
            r1 = max(r0 + 1, int(np.floor(y_edges[i + 1])) - sy0)
            vert[i] = band[r0:r1].mean(axis=0)

        # Exact horizontal area averaging using prefix sums
        csum = np.zeros((y1 - y0, src_w + 1), dtype=np.float64)
        np.cumsum(vert, axis=1, out=csum[:, 1:])
        out[y0:y1] = (csum[:, x_edges[1:]] - csum[:, x_edges[:-1]]) / x_counts

    return out


def smooth_polar_longitudes(grid: np.ndarray) -> np.ndarray:
    """Apply exact periodic longitudinal Gaussian smoothing so east-west pitch matches north-south pitch."""
    h, w = grid.shape
    freqs = np.fft.rfftfreq(w, d=1.0)  # cycles per pixel
    two_pi_f = 2.0 * np.pi * freqs
    result = grid.copy()
    for y in range(1, h - 1):
        lat = (0.5 - (y + 0.5) / h) * np.pi
        ratio = 1.0 / max(1e-4, float(np.cos(lat)))
        if ratio < 1.25:
            continue
        sigma = 0.5 * np.sqrt(ratio * ratio - 1.0)
        kernel = np.exp(-0.5 * (two_pi_f * sigma) ** 2)
        row_fft = np.fft.rfft(result[y].astype(np.float64))
        result[y] = np.fft.irfft(row_fft * kernel, n=w).astype(np.float32)
    # Close exact pole rows to their mean so all longitudes converge seamlessly.
    result[0, :] = float(result[0].mean())
    result[-1, :] = float(result[-1].mean())
    return result


def encode_ect2(elevations: np.ndarray, output: Path, zstd_level: int = 19) -> None:
    """Encode signed metre elevations as ECT2 (2D avg-delta + zigzag + byte-shuffle + Zstd)."""
    h, w = elevations.shape
    rounded = np.rint(elevations).astype(np.int32)
    if rounded.min() < -32768 or rounded.max() > 32767:
        raise ValueError(f"Elevation range {rounded.min()}..{rounded.max()} exceeds int16 metres")
    codes = (rounded + 32768).astype(np.uint16)

    # 2D predictor: pred(0,0)=32768, pred(0,x)=left, pred(y,0)=above, pred(y,x)=(left+above)>>1
    pred = np.empty_like(codes, dtype=np.int32)
    pred[0, 0] = 32768
    pred[0, 1:] = codes[0, :-1]
    pred[1:, 0] = codes[:-1, 0]
    pred[1:, 1:] = (codes[1:, :-1].astype(np.int32) + codes[:-1, 1:].astype(np.int32)) >> 1

    delta = (codes.astype(np.int32) - pred) & 0xFFFF
    signed = np.where(delta >= 32768, delta - 65536, delta).astype(np.int32)
    zigzag = ((signed << 1) ^ (signed >> 31)) & 0xFFFF
    hi = (zigzag >> 8).astype(np.uint8).tobytes()
    lo = (zigzag & 0xFF).astype(np.uint8).tobytes()
    payload = hi + lo

    proc = subprocess.run(
        ["zstd", f"-{zstd_level}", "-T0", "--content-size", f"--stream-size={len(payload)}", "-q", "-c"],
        input=payload,
        capture_output=True,
        check=True,
    )
    header = struct.pack(">I", ECT2_MAGIC) + struct.pack(
        "<III", w, h, ECT2_FLAG_DELTA_SHUFFLE | ECT2_FLAG_POLAR_SMOOTHED
    )
    output.write_bytes(header + proc.stdout)
    size_mb = (len(header) + len(proc.stdout)) / (1024 * 1024)
    print(f"{output}: {w} × {h} ECT2, {rounded.min()}..{rounded.max()} m, {size_mb:.2f} MiB")


def prepare_albedo(source: Path, output: Path, dst_w: int, dst_h: int, fmt: str, quality: int, neutralize_viking_blue: bool = False) -> None:
    """Resize a 0°-centered global albedo mosaic to (dst_w, dst_h) and close polar rows."""
    im = Image.open(source).convert("RGB")
    src_arr = np.asarray(im, dtype=np.uint8)
    if float(src_arr[:, 0].mean()) < 10.0 and float(src_arr[:, 1].mean()) > 20.0:
        src_arr = src_arr.copy()
        src_arr[:, 0] = ((src_arr[:, -1].astype(np.uint16) + src_arr[:, 1].astype(np.uint16) + 1) >> 1).astype(np.uint8)
        im.close()
        im = Image.fromarray(src_arr, mode="RGB")
    if im.size != (dst_w, dst_h):
        im = im.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8).copy()
    if neutralize_viking_blue:
        # USGS Viking MDIM 2.1 lacks a true blue band and gives dark basalt plains
        # (e.g. Syrtis Major) an artificial navy-blue tint (B > G > R). Gently clamp
        # excess blue/cyan while keeping neutral white polar caps and rusty plains intact.
        rgb = arr.astype(np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # Only attenuate blue/green channels where they exceed red (non-neutral cool cast).
        g_fixed = np.where(g > r, r + (g - r) * 0.25, g)
        b_fixed = np.where(b > g_fixed, g_fixed * 0.92 + (b - g_fixed) * 0.12, b)
        # Preserve bright neutral polar ice caps (high luminance where R, G, B are close).
        ice = np.clip((np.minimum(r, np.minimum(g, b)) - 130.0) / 50.0, 0.0, 1.0)
        rgb[:, :, 1] = g_fixed * (1.0 - ice) + g * ice
        rgb[:, :, 2] = b_fixed * (1.0 - ice) + b * ice
        arr = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    # Close exact pole rows to their average RGB so all longitudes agree at ±90°.
    arr[0, :, :] = np.rint(arr[0].mean(axis=0)).astype(np.uint8)
    arr[-1, :, :] = np.rint(arr[-1].mean(axis=0)).astype(np.uint8)
    out = Image.fromarray(arr, mode="RGB")
    if fmt == "webp":
        out.save(output, format="WEBP", quality=quality, method=6)
    else:
        out.save(output, format="JPEG", quality=quality, progressive=True, optimize=True)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"{output}: {dst_w} × {dst_h} {fmt.upper()} q{quality}, {size_mb:.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moon-dem", type=Path, help="LOLA LDEM_64 (23040x11520 <i2, 0.5 m)")
    parser.add_argument("--mars-dem", type=Path, help="MOLA MEG032 megt90n000fb.img (11520x5760 >i2, 1.0 m)")
    parser.add_argument("--moon-albedo", type=Path, help="LRO WAC lroc_color_poles_8k.tif")
    parser.add_argument("--mars-albedo", type=Path, help="USGS Mars_Viking_ClrMosaic_global_925m.tif")
    parser.add_argument("--earth-albedo", type=Path, help="NASA Blue Marble world.topo.200407.3x21600x10800.jpg")
    parser.add_argument("--dem-width", type=int, default=4096)
    parser.add_argument("--dem-height", type=int, default=2048)
    parser.add_argument("--albedo-width", type=int, default=8192)
    parser.add_argument("--albedo-height", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("public/textures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.moon_dem:
        dem = area_resample_dem(args.moon_dem, 23040, 11520, "<i2", 0.5, args.dem_width, args.dem_height)
        encode_ect2(smooth_polar_longitudes(dem), args.output / "moon-terrain.bin")
    if args.mars_dem:
        dem = area_resample_dem(args.mars_dem, 11520, 5760, ">i2", 1.0, args.dem_width, args.dem_height)
        encode_ect2(smooth_polar_longitudes(dem), args.output / "mars-terrain.bin")
    if args.moon_albedo:
        prepare_albedo(args.moon_albedo, args.output / "moon-albedo.webp", args.albedo_width, args.albedo_height, "webp", 86)
    if args.mars_albedo:
        prepare_albedo(args.mars_albedo, args.output / "mars-albedo.webp", args.albedo_width, args.albedo_height, "webp", 86, neutralize_viking_blue=True)
    if args.earth_albedo:
        prepare_albedo(args.earth_albedo, args.output / "earth-blue-marble.jpg", args.albedo_width, args.albedo_height, "jpeg", 86)


if __name__ == "__main__":
    main()
