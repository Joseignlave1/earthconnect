"""Bake the Olympus Mons lightmap with NumPy and Pillow from the original HiRISE IMG."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def bake(source: Path, output: Path, sun_azimuth: float, sun_elevation: float):
    dtm = np.memmap(source, dtype="<f4", mode="r", offset=25276, shape=(8239, 6319))
    stride = 3
    heights = dtm[540:4525:stride, 580:4741:stride].copy()
    if not np.isfinite(heights).all() or heights.min() <= 5000:
        raise ValueError("Expected the Layers in a Scarp on Olympus Mons HiRISE DTM.")
    rows, columns = heights.shape
    y, x = np.ogrid[:rows, :columns]
    metres = 1.0087288906343 * stride

    def shifted(dx, dz):
        return heights[np.clip(y + dz, 0, rows - 1), np.clip(x + dx, 0, columns - 1)]

    def horizon(azimuth, distances):
        highest = np.zeros_like(heights)
        for distance in distances:
            dx = round(math.cos(azimuth) * distance)
            dz = round(math.sin(azimuth) * distance)
            slope = (shifted(dx, dz) - heights) / (math.hypot(dx, dz) * metres)
            np.maximum(highest, slope, out=highest)
        return highest

    dz, dx = np.gradient(heights, metres)
    normal_length = np.sqrt(1 + dx * dx + dz * dz)
    east = -dx / normal_length
    south = -dz / normal_length

    # Average a small area light across the horizon to soften cast shadows.
    sun = np.zeros_like(heights)
    samples = (
        1,
        2,
        3,
        4,
        5,
        7,
        9,
        12,
        16,
        21,
        27,
        35,
        44,
        55,
        67,
        82,
        101,
        122,
        148,
        177,
        218,
        264,
        320,
        390,
        470,
    )
    for offset in (-0.051, -0.025, 0, 0.025, 0.051):
        slope = horizon(sun_azimuth + offset, samples)
        visible = np.clip((math.tan(sun_elevation) - slope + 0.032) / 0.071, 0, 1)
        sun += visible * visible * (3 - 2 * visible) * 0.2

    occlusion = np.zeros_like(heights)
    samples = (1, 2, 3, 4, 6, 8, 11, 16, 22, 31, 44, 63, 90, 122, 180, 250)
    for azimuth in np.linspace(0, 2 * math.pi, 10, endpoint=False):
        slope = horizon(azimuth, samples)
        occlusion += (1 - 1 / np.sqrt(1 + slope * slope)) * 0.1
    average = (shifted(4, 0) + shifted(-4, 0) + shifted(0, 4) + shifted(0, -4)) * 0.25
    concavity = np.clip((average - heights) / 6, 0, 1)
    sky = np.clip(1 - occlusion * 2.9 - concavity * 0.23, 0.10, 1)

    layers = np.stack([east * 0.5 + 0.5, south * 0.5 + 0.5, sun, sky], axis=-1)
    image = Image.fromarray((layers.clip(0, 1) * 255 + 0.5).astype("uint8"), "RGBA")
    image.thumbnail((1120, 1120), Image.Resampling.LANCZOS)
    encoding_steps = np.array([3, 3, 4, 4], dtype="uint16")
    packed = np.asarray(image, dtype="uint16")
    quantized = np.minimum(
        255, ((packed + encoding_steps // 2) // encoding_steps) * encoding_steps
    )
    Image.fromarray(quantized.astype("uint8")).save(output, lossless=True, method=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="DTEEC_001432_2015_001630_2015_U01.IMG"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "assets/olympus-lighting.webp",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=math.degrees(math.atan2(-0.893, -0.445)),
        help="sun azimuth in degrees east toward south",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=math.degrees(math.atan(0.067)),
        help="sun elevation in degrees above the terrain plane",
    )
    args = parser.parse_args()
    bake(
        args.source,
        args.output,
        math.radians(args.azimuth),
        math.radians(args.elevation),
    )
