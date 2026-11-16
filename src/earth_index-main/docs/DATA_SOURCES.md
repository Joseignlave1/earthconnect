# Data sources and methodology

Reference for the challenge MVP, checked **2026-11-15**. The serving Earth snapshot is
`earth-ed5bc9904a5f` (revision
`ed5bc9904a5f7211e52e3ac5b21cd145fe2f69f6f38ddc0df98da08a69e90f0e`).
It combines products from different years and measurement methods; it is not a
simultaneous observation of Earth, the Moon and Mars.

## Earth: imported into the database

The database contains **25 environmental and quality fields**, on 460 MODIS
sinusoidal regions with 31,881,940 Earth-domain cell centers. Nominal cell spacing
is 4 × 4 km (exactly 3,999.822 m; about 16 km²/cell). Source resolution still
determines how much independent information a field contains.

| Source and citation | Imported product / period | Original support |
| --- | --- | --- |
| **EarthEnv topography / GMTED2010**, Amatulli et al. (2018). [Dataset](https://doi.org/10.1594/PANGAEA.867113), [methods paper](https://doi.org/10.1038/sdata.2018.40) | `elevation_1KMmd_GMTEDmd.tif`, `slope_1KMmn_GMTEDmd.tif`; static, derived from GMTED2010 | Nominal 1 km layers; mixed underlying observation dates |
| **NASA MODIS land cover**, Friedl & Sulla-Menashe (2022). [MCD12Q1 V061](https://doi.org/10.5067/MODIS/MCD12Q1.061) | `MCD12Q1.061`, **2024**; IGBP classification and QC | Annual, nominal 500 m |
| **NASA Terra/MODIS surface temperature**, Wan, Hook & Hulley (2021). [MOD11A2 V061](https://doi.org/10.5067/MODIS/MOD11A2.061) | `MOD11A2.061`, **2023**; day/night LST and QC, accessed through [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/modis-11A2-061) | 8-day clear-sky composites, nominal 1 km |
| **NASA POWER / MERRA-2**, NASA Langley Research Center. [Bulk-data documentation](https://power.larc.nasa.gov/docs/services/aws/), [meteorological source information](https://power.larc.nasa.gov/docs/faqs/data/) | POWER v10.0.0 public climatology Zarr, **2001–2020**; `T2M`, `PRECTOTCORR` | 0.5° latitude × 0.625° longitude; monthly/annual climatology |
| **ESA CCI Permafrost**, Westermann et al. (2025). [Northern Hemisphere extent v5.0](https://doi.org/10.5285/d235665772ec4b558e9a89ac85595e71) | **2023** `PFR`, `MODISLST_CRYOGRID-AREA4_PP`, `fv05.0` | Annual modeled permafrost fraction, 0.01° grid; northern product domain |
| **GLiM**, Hartmann & Moosdorf (2012). [Global Lithological Map Database v1.0, 0.5°](https://doi.org/10.1594/PANGAEA.788537), [methods paper](https://doi.org/10.1029/2012GC004370) | `glim_wgs84_0point5deg.txt.asc` from the [authors' archive](https://epic.awi.de/id/eprint/31092/1/hartmann-moosdorf_2012.zip) | Static 0.5° categorical map |

### Field lineage

Names below are the actual SQL columns in `measurements` / `cells`. Temperatures
are °C, elevations/relief/TPI are metres, slope is degrees, precipitation is
mm/year, and fractions use 0–1. Lithology is a categorical code.

| Source family | Stored fields |
| --- | --- |
| EarthEnv and derived terrain (7) | `elevation_median_m`, `slope_mean_deg`, `terrain_valid_fraction`, `relief_5km_m`, `relief_21km_m`, `tpi_5km_m`, `tpi_21km_m` |
| MODIS land cover (3) | `bare_fraction`, `permanent_snow_ice_fraction`, `landcover_valid_fraction` |
| MODIS surface temperature (9) | `surface_day_c`, `surface_night_c`, `surface_mean_c`, `surface_low_c`, `surface_high_c`, `surface_range_c`, `surface_day_valid_fraction`, `surface_night_valid_fraction`, `surface_paired_valid_fraction` |
| POWER climate (3) | `coldest_month_air_c`, `warmest_month_air_c`, `annual_precip_mm` |
| ESA permafrost (2) | `permafrost_fraction`, `permafrost_valid_fraction` |
| GLiM lithology (1) | `dominant_lithology_code` |

### How we process them

- **Terrain:** average the source elevation/slope onto an approximately 1 km
  intermediate grid, then aggregate to 4 km. Compute relief as neighborhood
  max-minus-min, and topographic position (TPI) as center elevation minus
  neighborhood mean, using 5 × 5 and 21 × 21 intermediate cells. Contexts need
  at least 90% valid neighbors; their final aggregation needs 50% valid subcells.
  The elevation field retains its historical `median` name, but our final
  aggregation is an average of source values. The source filename and TIFF
  description disagree about median versus maximum: this remains unresolved.
- **Land cover:** accept QC 0 or 2 and IGBP classes 1–17. Bare ground is class 16;
  permanent snow/ice is class 15. Fractions describe accepted classified support;
  a separate coverage field describes full-cell support. Permanent snow/ice is
  not seasonal snow persistence, and class 17 water is not bare land.
- **Surface temperature:** accept QC byte exactly zero and raw LST ≥7500;
  apply the 0.02 K scale and convert to °C. Average spatially before summarizing
  over time. A composite needs ≥50% supported cell area; an annual value needs
  ≥12 accepted composites and representation in all four calendar quarters.
  Means use accepted-area and composite-duration weights. `surface_mean_c`
  uses paired day/night support; low/high are the coldest nighttime / hottest
  daytime accepted composite means. Range is high-minus-low, not a measured
  daily diurnal amplitude. Coverage is accepted area-days divided by 365.
  The saved catalog has **44 of 46** expected 2023 composite start dates;
  January 1 and February 26 were absent. Missing observations stay missing.
- **Climate:** use monthly `T2M` minimum/maximum for coldest/warmest month;
  annual precipitation is the native annual mean daily rate ×365.25 days.
  Require complete monthly values and ≥99.9% valid spatial support. The coarse
  climate grid is resampled, so adjacent 4 km cells are not independent climate
  measurements. Earlier pilots used a different POWER baseline (2001–2025).
- **Permafrost:** convert `PFR` percent to fraction, reject values >100, and use
  the NetCDF CF coordinate vectors rather than its conflicting GeoTransform.
  Missing or outside-domain data remains unsupported. This is modeled ground
  state, not a measurement of water-ice concentration or an ice deposit.
- **Lithology:** sample the nearest source category; do not average codes.
  Source codes −9999 and 15 are missing. This is broad regional lithology,
  not 4 km mineral detection; the category dictionary is in API metadata.

Continuous fields are served as **DOUBLE**, retaining the precision of the
earlier quantized import: elevation/relief 1 m, slope 0.1°, TPI 0.25 m, air
temperature 0.01°C, surface temperature 0.02°C, precipitation 1 mm/year, and
fractions 0.001. Missing measurements are SQL `NULL`, never an invented zero.
The grid covers the planet, but individual variables have gaps, including
terrain limitations in Antarctica and substantial thermal QC exclusions.

## Moon and Mars: sampled on demand

`SamplePoint` queries public providers when a user selects a location. These
products are not imported into the Earth table. The compiled
[runtime source registry](../backend/src/planetary/sources.json) is the exact
inventory: **51 Moon and 28 Mars product entries**, including alternative
resolutions/footprints for the same quantity. An entry is not a promise that
the product returns a value everywhere.

| Body / family | Sources used | What they provide |
| --- | --- | --- |
| Moon terrain | [LOLA global DEM v06, 256 pixels/degree](https://trek.nasa.gov/moon/trekarcgis/rest/services/LRO_LOLA_DEM_Global_256ppd_v06/ImageServer) via NASA Trek | Reference-sphere elevation and inputs for derived terrain |
| Moon temperatures | LRO Diviner `diviner_tbol_hour12`, `hour00`, `min`, `max`, via [Trek](https://trek.nasa.gov/moon/trekarcgis2/rest/services/diviner_tbol_hour12/ImageServer) | Noon/midnight and extrema of bolometric surface-temperature composites; compilation 2009–2015, 0.5° service grid |
| Moon polar context | Trek illumination / Earth-visibility models, mapped shadow regions and surface-ice detections, northern WEH and ice-stability models | Product-specific polar context; modeled visibility/stability and mapped detections have different meanings |
| Moon composition / geology | Kaguya mineral products, Lunar Prospector and KPLO channels, unified lunar geology, via Trek | Mineral/spectral estimates, elemental/proxy channels and mapped geological attributes |
| Moon additional context | Trek Mini-RF radar, regional NAC/LOLA rock/roughness/slope products, Diviner Christiansen feature, gravity/magnetic/crustal models | Regional surface or broad geophysical context; retain each source's footprint and flags |
| Mars terrain | [Blended MOLA/HRSC DEM](https://trek.nasa.gov/mars/trekarcgis/rest/services/Mars_MOLA_blend200ppx_HRSC_DEM_clon0dd_200mpp_lzw/ImageServer) via Trek | Areoid-referenced elevation; nominal ~200 m service postings |
| Mars temperatures | ASU/JMARS `mars_yearly_avg_temperature_celsius`, `minimum`, `maximum`, through its [numeric-map service](https://ms-mars.mars.asu.edu/alb32?SERVICE=WMS&REQUEST=GetCapabilities) | Modeled annual surface-temperature summaries; 8 pixels/degree, not current weather |
| Mars surface materials | ASU/JMARS TES/THEMIS products and Odyssey hydrogen map; [TES product descriptions and references](https://tes.mars.asu.edu/products/) | Albedo, thermal inertia, surface dust index, water-equivalent hydrogen, TES mineral-model fractions and fit error |
| Mars mapped features | Trek global geology, OMEGA/CRISM hydrous-mineral detections, chlorides, dunes, valleys and glacial context | Exact-point intersections with the configured feature sublayer; attributes rather than a complete local inventory |

Trek numeric rasters use `getSamples`; feature layers use spatial queries.
ASU/JMARS returns numeric VICAR pixels through WMS, following the distinction
between [numeric and graphic maps](https://jmars.mars.asu.edu/numeric-maps).
We never infer scientific values from basemap colors. These requests use no
user-supplied NASA token. Responses include product IDs/URLs, units, spacing,
quality flags, status and retrieval time; retrieval time is not observation time.

Derived terrain uses body-specific distances: a local DEM stencil plus a bounded
21 × 21 window at 1 km sample spacing provides `slope_1km`, 5/21 km relief and
TPI. These are point-sampled estimates; Earth's values are aggregated cells.
Mars modeled temperature range is derived from valid max-minus-min values.
See the [sampler contract](../backend/README.md#moonmars-point-measurements)
and [source audit](../research/planetary-sources/README.md) for details.

Important interpretation rules:

- Moon elevation, Mars elevation and terrestrial elevation have different
  reference surfaces. Absolute heights are not automatically interchangeable.
- Earth MODIS temperatures and planetary model/composite temperatures describe
  different times and statistics. Editable targets, tolerances and weights
  express the aspects the user wants to compare.
- Some raw composition/proxy scales are unresolved. Respect
  `semantic_review_required` and `raw_source_scale`; successful numeric access
  does not establish a calibrated abundance. Hydrogen proxies and ice-stability
  models do not confirm an accessible ice deposit.
- Feature no-hit / `NO_DATA` does not establish absence. Outside coverage,
  source errors and timeouts have distinct statuses; retain them. A partial
  response can still contain usable measurements.
- The default builder compares available terrain and approximate thermal
  counterparts, with an editable bare-ground preference. Other returned
  planetary channels are contextual evidence, not automatically Earth-indexed
  equivalents. Scores express the query's preferences, not probabilities or
  certified landing/base suitability.

## Display imagery and attribution

Imagery provides visual context independently of the numeric datasets.
The current configuration and provider capabilities links are in
[`demo/src/maps.json`](../demo/src/maps.json).

| Map | Display product | Attribution shown |
| --- | --- | --- |
| Earth | NASA GIBS `BlueMarble_ShadedRelief_Bathymetry`, WMTS `500m` matrix | NASA GIBS |
| Moon | NASA Trek `LRO_WAC_Mosaic_Global_303ppd_v02` | NASA / USGS |
| Mars | NASA Trek `Mars_Viking_MDIM21_ClrMosaic_global_232m` | NASA / USGS |

The six packaged zoom-zero overview JPEGs come from those same display layers;
higher zooms request provider tiles. Display resolution does not determine
the resolution of the numeric value used for scoring.

## Credits and reuse

Credit the data authors as well as the hosting service; NASA Trek and Planetary
Computer host products from multiple teams and agencies.

- **EarthEnv:** Amatulli et al. (2018), dataset and methods paper linked above.
  The PANGAEA dataset record specifies **CC BY 3.0**.
- **MODIS:** cite Friedl & Sulla-Menashe (2022) for MCD12Q1 and Wan, Hook & Hulley
  (2021) for MOD11A2, using their product DOIs above. Follow NASA Earthdata's
  [data-use and citation guidance](https://doi.org/10.5067/DOC/ESCO/ESDS-RFC-055)
  and the [LP DAAC reuse guidance](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-data-use-and-citation).
  Credit Microsoft Planetary Computer for the MOD11A2 access path.
- **POWER:** acknowledge the NASA POWER Project at NASA Langley Research Center
  and identify the MERRA-2 climatology, version and baseline used here.
- **Permafrost:** cite Westermann et al. (2025) with the v5.0 dataset DOI, and
  acknowledge ESA CCI, Permafrost CCI and the data providers. Its
  [specific terms](https://artefacts.ceda.ac.uk/licences/specific_licences/esacci_permafrost_terms_and_conditions.pdf)
  permit use for any purpose subject to the stated acknowledgement/citation conditions.
- **GLiM:** cite Hartmann & Moosdorf (2012), dataset and paper above. The
  PANGAEA gridded dataset record specifies **CC BY 3.0**.
- **Planetary products and imagery:** retain the source-specific credits,
  product references and map attributions. The registry records source lineage,
  not a complete product-license audit; provider/product terms apply separately.
  Consult the [JMARS citation policy](https://jmars.mars.asu.edu/how-should-jmars-be-cited-published-works)
  and the metadata for each Trek product.

Our processing and derived results do not imply endorsement by the providers.

## Reproducing the snapshot

The [import methods](../db/WORLD_IMPORT.md) and
[database preparation guide](../db/README.md) describe acquisition, aggregation,
quantization and serving-file construction. Implementation entry points are
[`world_sources.py`](../scripts/world_sources.py),
[`world_features.py`](../scripts/world_features.py) and
[`prepare_duckdb.py`](../db/prepare_duckdb.py).

Keep the dataset backup with `research/world-4km/manifest.json`, `plan.json`,
`shared.json`, `METHODS.md`, `sources/*.json` and the frozen
`db/runtime/source-metadata.{json,pb}`. These local artifacts record original
URLs, granule IDs, byte counts, SHA-256 hashes, grid definitions, processing-code
hashes and per-layer precision/coverage. Large runtime/import artifacts are
ignored by Git. Shared raw sources persist; per-region raw assets were deleted
after successful checkpoints and must be reacquired by recorded identity to rebuild.

For planetary sampling, the checked-in registry and research audit preserve the
selected products and observed metadata; upstream services can change. For the
running Earth schema and per-column provenance, use the API `Metadata` RPC.
The older [candidate variable catalog](VARIABLES.md) is a research roadmap,
not the inventory of the current database.
