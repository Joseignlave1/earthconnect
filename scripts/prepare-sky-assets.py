#!/usr/bin/env python3
"""Build high-res 8K NASA Deep Star Map WebP from SVS #4851 16-bit EXR."""

from pathlib import Path
import Imath
import numpy as np
import OpenEXR
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_TEX = ROOT / "public" / "textures" / "nasa-deep-star-map.webp"
EXR_PATH = Path("/tmp/starmap_2020_8k.exr")


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def build_sky_webp():
    exr = OpenEXR.InputFile(str(EXR_PATH))
    dw = exr.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = np.stack(
        [
            np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
            for c in "RGB"
        ],
        axis=-1,
    )
    # Calibrate linear HDR EXR gently so Milky Way dust matches natural space luminance
    # while using the full 8-bit sRGB dynamic range cleanly with zero dark-floor grain.
    scaled = np.clip(rgb * 0.85, 0.0, 1.0)
    srgb = linear_to_srgb(scaled) * 255.0
    # Subtle TPDF dither only on visible nebula/sky pixels to prevent 8-bit Milky Way banding
    # while keeping pure-black deep space noise-free for tight WebP compression.
    rng = np.random.default_rng(4851)
    dither = (
        rng.random(srgb.shape, dtype=np.float32)
        - rng.random(srgb.shape, dtype=np.float32)
    ) * 0.50
    mask = (srgb > 2.5).astype(np.float32)
    u8 = np.clip(np.round(srgb + dither * mask), 0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")
    img.save(OUT_TEX, format="WEBP", quality=84, method=6)
    print(f"Wrote {OUT_TEX} ({w}x{h}, {OUT_TEX.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build_sky_webp()
