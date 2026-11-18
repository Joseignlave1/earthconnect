"""uv run --with rasterio --with numpy --with pillow --with scipy scripts/prepare-terrain.py [--reuse-elevation]."""
from pathlib import Path
import argparse
import gzip
import struct
import numpy as np
from PIL import Image
import rasterio
from scipy import ndimage

SOURCE = 'https://data.source.coop/ausantarctic/gebco/GEBCO_2026.tif'
ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / 'exports/earth-elevation.png'
binary = ROOT / 'public/textures/earth-terrain.bin'
parser = argparse.ArgumentParser()
parser.add_argument('--reuse-elevation', action='store_true', help='Keep the existing RG elevations and rebuild the coastline channel.')
args = parser.parse_args()
if args.reuse_elevation:
    previous_path = destination if destination.exists() else binary.with_name('earth-elevation.png')
    previous = np.asarray(Image.open(previous_path)).astype(np.int32)
    elevation = previous[:, :, 0] * 256 + previous[:, :, 1] - 32768
else:
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
        with rasterio.open(SOURCE) as src:
            elevation = src.read(1, out_shape=(2048, 4096), resampling=rasterio.enums.Resampling.average).astype(np.int32)
assert np.isfinite(elevation).all() and elevation.min() > -12000 and elevation.max() < 9000

# Only the connected ocean gets a blue boundary. Enclosed negative elevations
# (e.g. the Caspian basin) remain land for this layer. Join across the dateline.
labels, count = ndimage.label(elevation < 0)
parents = np.arange(count + 1)
def root(label):
    while parents[label] != label:
        parents[label] = parents[parents[label]]
        label = parents[label]
    return label
for west, east in zip(labels[:, 0], labels[:, -1]):
    if west and east:
        parents[root(east)] = root(west)
roots = np.array([root(label) for label in range(count + 1)])
joined = roots[labels]
sizes = np.bincount(joined.ravel())
sizes[0] = 0
ocean = joined == sizes.argmax()

# Signed distance in source pixels, rather than height/height derivatives.
# Padding makes distances periodic in longitude; values beyond 31.75 px saturate.
land = np.pad(~ocean, ((0, 0), (32, 32)), mode='wrap')
distance = ndimage.distance_transform_edt(land) - ndimage.distance_transform_edt(~land)
distance -= np.sign(distance) * .5
distance = distance[:, 32:-32]
code = (elevation + 32768).astype(np.uint16)
rgb = np.zeros((*code.shape, 3), dtype=np.uint8)
rgb[:, :, 0] = code >> 8
rgb[:, :, 1] = code & 255
rgb[:, :, 2] = np.rint(np.clip(distance, -31.75, 31.75) * 4 + 128).astype(np.uint8)
destination.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgb).save(destination, optimize=True)
decoded = np.asarray(Image.open(destination)).astype(np.int32)
assert np.array_equal(decoded[:, :, 0] * 256 + decoded[:, :, 1] - 32768, elevation)
# The runtime consumes bytes, never image/canvas readback: privacy browsers may
# deliberately perturb getImageData(), which corrupts the high elevation byte.
rgba = np.full((*code.shape, 4), 255, dtype=np.uint8)
rgba[:, :, :3] = rgb
payload = b'ECT1' + struct.pack('<II', code.shape[1], code.shape[0]) + rgba.tobytes()
# Avoid a .gz suffix: some static servers infer Content-Encoding and decompress
# it in fetch(), while a data URL leaves it compressed. Keep one decoding path.
binary.write_bytes(gzip.compress(payload, mtime=0))
assert gzip.decompress(binary.read_bytes()) == payload
# Regression landmarks: lowlands, deep ocean and an enclosed negative basin.
for longitude, latitude, is_land in [(-60, -10, True), (-100, 40, True), (-56, -33, True), (-30, 0, False), (51, 42, True)]:
    x = int((longitude + 180) / 360 * elevation.shape[1])
    y = int((90 - latitude) / 180 * elevation.shape[0])
    assert bool(decoded[y, x, 2] > 128) == is_land, (longitude, latitude)
print(f'{destination}: {destination.stat().st_size:,} bytes; {elevation.min()}..{elevation.max()} metres')
print(f'{binary}: {binary.stat().st_size:,} bytes; lossless numeric payload')
