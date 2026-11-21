# Olympus Mons scene assets

The stereo DTM and both matched RED orthophotos are from [Layers in a Scarp on Olympus Mons](https://www.uahirise.org/dtm/dtm.php?ID=PSP_001432_2015), courtesy of NASA/JPL-Caltech/UArizona. The red color, illumination and dust are artistic.

- `olympus-elevation.bin` uses `DTEEC_001432_2015_001630_2015_U01.IMG`, from zero-based columns 580–4740 and rows 540–4524 with a stride of four. Heights retain the Mars 2000 datum. The gzip payload starts with uint32 width and height, then float32 easting, southing, spacing, and base altitude in kilometres, all little-endian. Elevations above the base are quantized to eighths of a metre and row-delta encoded with zigzag values, stored as low and high byte planes.
- `olympus-ortho.webp` uses `PSP_001432_2015_RED_C_01_ORTHO.JP2`, cropped to the elevation grid and resized for the overview.
- `olympus-ortho-detail.webp` uses `PSP_001432_2015_RED_A_01_ORTHO.JP2` with a crop from `(3600, 2600)` to `(12200, 9400)`. Its source spacing is about 25 cm. The local WebP crop is resized and blended into the matching overview.
- `olympus-lighting.webp` is baked from the stereo elevations at three-pixel spacing. It packs east and south terrain normals, soft cast-shadow visibility averaged over five sun rays, and ambient exposure from ten horizon directions.
- `olympus-slab.webp` and `olympus-scree.webp` pack luma, tangent normals and roughness from Poly Haven's CC0 [Rocky Terrain](https://polyhaven.com/a/rocky_terrain) by Amal Kumar and [Rocky Gravel](https://polyhaven.com/a/rocky_gravel) by Dario Barresi and Dimitrios Savva. The shader uses these terrestrial scans for artistic surface detail.
- `olympus-region.bin` crops the [USGS HRSC/MOLA Blended DEM Global 200 m v2](https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m) to a local 580 × 500 km grid. Elevations use one-metre units in the same row-delta format, followed by a byte plane with baked sunlight visibility. Data credit NASA/GSFC, ESA/DLR/FU Berlin, and USGS Astrogeology. The HRSC contribution is [CC BY-SA 3.0 IGO](https://creativecommons.org/licenses/by-sa/3.0/igo/).
- `olympus-ctx.webp` comes from the [Murray Lab Global CTX Mosaic of Mars v1](https://murray-lab.caltech.edu/CTX/) by Jay Dickson and the Bruce Murray Laboratory for Planetary Visualization at Caltech, using NASA's Mars Reconnaissance Orbiter Context Camera imagery. It is resampled from the published Google Mars tile overlay to the local rectangle from −65 to 55 km east and −66 to 39 km south of the HiRISE origin.
- `earth-blue-marble.jpg` is a reduced copy of NASA's [Blue Marble: Next Generation](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography/) texture, credited to Reto Stöckli (NASA Goddard Space Flight Center). Earth's apparent size and placement were chosen for the composition.


To rebuild the baked lighting with NumPy and Pillow installed, run:

```sh
python src/hero/bake-lighting.py /path/to/DTEEC_001432_2015_001630_2015_U01.IMG
```

The regional elevation and its sunlight plane can be rebuilt with NumPy and Rasterio:

```sh
python src/hero/pack-regional-terrain.py /path/to/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
```

If changing the sunlight direction, update both bakes and the matching vector in `main.ts`.
