"""One seamless MODIS-sinusoidal world grid, nominally 4 km, with native tiles.

278 cells exactly span one MODIS tile: 3999.822 m, avoiding overlaps or gaps at
native source boundaries. This is a new grid, not a relabeling of the AEQD pilot.
"""
import math
import re

import numpy as np
from rasterio.transform import from_origin

RADIUS = 6371007.181
SPAN = math.pi * RADIUS / 18
LEFT = -math.pi * RADIUS
TOP = math.pi * RADIUS / 2
SIZE = 278
CELL = SPAN / SIZE
FINE = CELL / 4
CRS = f'+proj=sinu +R={RADIUS} +nadgrids=@null +wktext +units=m +no_defs'
VERSION = 'modis-sinu-278-v1'


def tile_indices(key):
    match = re.fullmatch(r'h(\d{2})v(\d{2})', key)
    if not match:
        raise ValueError('Invalid native tile key')
    h, v = map(int, match.groups())
    if not (0 <= h < 36 and 0 <= v < 18):
        raise ValueError('Native tile outside global grid')
    return h, v


def transform(key, *, fine=False, halo=0):
    h, v = tile_indices(key)
    step = FINE if fine else CELL
    return from_origin(LEFT + h * SPAN - halo * step,
                       TOP - v * SPAN + halo * step, step, step)


def domain_mask(key, *, fine=False, halo=0):
    n = SIZE * (4 if fine else 1) + 2 * halo
    tr = transform(key, fine=fine, halo=halo)
    y = tr.f + (np.arange(n) + .5) * tr.e
    x = tr.c + (np.arange(n) + .5) * tr.a
    lat = y / RADIUS
    return ((np.abs(lat[:, None]) <= math.pi / 2)
            & (np.abs(x[None, :]) <= math.pi * RADIUS * np.maximum(0, np.cos(lat[:, None]))))


def tiles():
    return [f'h{h:02}v{v:02}' for v in range(18) for h in range(36)
            if domain_mask(f'h{h:02}v{v:02}').any()]


def definition():
    return dict(version=VERSION, crs_proj4=CRS, cell_m=CELL,
                nominal_cell_m=4000, native_tile_span_m=SPAN,
                global_shape=[18 * SIZE, 36 * SIZE], tile_shape=[SIZE, SIZE],
                origin=[LEFT, TOP], area_km2=CELL * CELL / 1e6,
                domain='Cell center inside the sinusoidal Earth; padding is NODATA.',
                scope='All Earth-domain native tiles; variable-specific source gaps remain missing.')
