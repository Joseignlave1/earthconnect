"""Crop a Mars HRSC/MOLA GeoTIFF to the Olympus Mons hero's local grid."""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def sunlight(heights, spacing):
    y, x = np.ogrid[:heights.shape[0], :heights.shape[1]]
    slope = np.zeros_like(heights, dtype=np.float32)
    # Match the northwest sun in main.ts. The angular ramp softens cast shadows.
    for distance in (1, 2, 3, 5, 7, 10, 14, 19, 25, 32, 41, 53, 68, 87, 110, 140, 175):
        dx, dz = round(-.445 * distance), round(-.893 * distance)
        sample = heights[np.clip(y + dz, 0, heights.shape[0] - 1),
                         np.clip(x + dx, 0, heights.shape[1] - 1)]
        ray_slope = (sample - heights) / (np.hypot(dx, dz) * spacing * 1000)
        np.maximum(slope, ray_slope, out=slope)
    visible = np.clip((.067 - slope + .015) / .03, 0, 1)
    return np.rint(visible**2 * (3 - 2 * visible) * 255).astype(np.uint8).tobytes()


def pack(source: str, output: Path):
    # This origin and scale match the HiRISE equirectangular crop in the hero.
    lon, lat = -137.70213346, 21.587221057388
    km_per_degree = np.pi * 3393.83 / 180
    east_scale = km_per_degree * np.cos(np.deg2rad(20))
    west, north, spacing = -240, -140, .4
    width, height = 1451, 1251
    # Shift by half a sample so the resampled raster's centers match the GPU grid.
    left = west - spacing / 2
    top = north - spacing / 2
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        with rasterio.open(source) as raster:
            window = from_bounds(
                lon + left / east_scale,
                lat - (top + height * spacing) / km_per_degree,
                lon + (left + width * spacing) / east_scale,
                lat - top / km_per_degree,
                raster.transform,
            )
            metres = raster.read(1, window=window, out_shape=(height, width),
                                 resampling=Resampling.bilinear)
    base = -4
    heights = metres.astype(np.int32) - base * 1000
    deltas = np.diff(heights, axis=1, prepend=0)
    zigzag = ((deltas << 1) ^ (deltas >> 31)).astype(np.uint16)
    packed = (zigzag & 255).astype(np.uint8).tobytes() + (zigzag >> 8).astype(np.uint8).tobytes()
    header = struct.pack('<IIffff', width, height, west, north, spacing, base)
    output.write_bytes(gzip.compress(header + packed + sunlight(heights, spacing), mtime=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='HRSC/MOLA blended 200 m DEM in east-longitude degrees')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'assets/olympus-region.bin')
    args = parser.parse_args()
    pack(args.source, args.output)
