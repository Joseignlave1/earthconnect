"""Build the display DEMs from the NASA PDS rasters listed in textures/CREDITS.txt."""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np

SOURCE_WIDTH = 5760
SOURCE_HEIGHT = 2880
OVERVIEW_FACTOR = 4


def prepare(source, output, byte_order, metres_per_unit):
    data = np.fromfile(source, dtype=f"{byte_order}i2")
    if data.size != SOURCE_HEIGHT * SOURCE_WIDTH or np.any(data == -32768):
        raise ValueError(f"{source} is incomplete or contains missing values")
    data = data.reshape(SOURCE_HEIGHT, SOURCE_WIDTH)
    height, width = SOURCE_HEIGHT // OVERVIEW_FACTOR, SOURCE_WIDTH // OVERVIEW_FACTOR
    # Labels specify pixel areas with geographic edges at 0..360 E / 90 N..90 S.
    # Shift 180 degrees so shared Earth UV coordinates use the same registration.
    data = np.roll(data, SOURCE_WIDTH // 2, axis=1)
    elevations = np.rint(
        data.reshape(height, OVERVIEW_FACTOR, width, OVERVIEW_FACTOR).mean(axis=(1, 3))
        * metres_per_unit
    ).astype(np.int32)
    if elevations.min() < -32768 or elevations.max() > 32767:
        raise ValueError(f"{source} elevation range exceeds the signed metre encoding")
    encoded = (elevations + 32768).astype(np.uint16)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = encoded >> 8
    rgba[:, :, 1] = encoded & 255
    rgba[:, :, 3] = 255
    packed = struct.pack("<4sII", b"ECT1", width, height) + rgba.tobytes()
    output.write_bytes(gzip.compress(packed, compresslevel=9, mtime=0))
    print(f"{output}: {width} × {height}, {elevations.min()}..{elevations.max()} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moon_lola_ldem_16", type=Path)
    parser.add_argument("mars_mola_megt90n000eb", type=Path)
    parser.add_argument("--output", type=Path, default=Path("public/textures"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    prepare(arguments.moon_lola_ldem_16, arguments.output / "moon-terrain.bin", "<", 0.5)
    prepare(arguments.mars_mola_megt90n000eb, arguments.output / "mars-terrain.bin", ">", 1.0)
