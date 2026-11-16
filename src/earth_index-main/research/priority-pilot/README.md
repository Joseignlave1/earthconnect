# Source-tested priority raster pilot

The priority variables have been downloaded/read, decoded, quantized and loaded
into Docker PostgreSQL/PostGIS. This is a working bounded prototype, not merely
a list of catalog hits. There are **16 aligned bands per area**, each with
100 × 100 projected 1 km pixels, for Atacama, Haughton Crater and Death Valley:
48 raster tiles and 480,000 stored pixel codes, including explicit NODATA.

The sites are centered on (-68.5, -23.5), (-89.82, 75.42), and (-116.82, 36.24)
respectively, using independent local AEQD grids. Coordinates are longitude,
latitude. The Atacama square is a northern Chile example, not a claim to cover
every scientifically studied Atacama analog location. Projected pixel area is
1 km²; ellipsoidal area differs slightly. This is not yet a global grid.

## What to prioritize

**P0: terrain, climate, bare ground and coverage.** These support immediately
useful environmental filters without having to infer everything from imagery.

**P1: permafrost, regional landform context and coarse lithology.** These add
context and help separate environments, with source-specific restrictions.
Permanent snow/ice class is included as a context layer; seasonal snow duration
is a separate quantity that remains unimplemented.

| Priority | SQL column | Unit / storage increment | Actual source and support |
| --- | --- | --- | --- |
| P0 | `elevation_median_m` | m / 1 | EarthEnv published nominal 1 km median elevation; area-average overlapping source medians; metadata conflict below |
| P0 | `slope_mean_deg` | degrees / 0.1 | EarthEnv mean native-derived slope, published nominal 1 km grid |
| P0 | `terrain_valid_fraction` | fraction / 0.001 | Accepted slope source area divided by target area |
| P0 | `coldest_month_air_c` | °C / 0.01 | NASA POWER 2001–2025 climatology: minimum of twelve monthly mean 2 m air temperatures |
| P0 | `warmest_month_air_c` | °C / 0.01 | Same source: maximum of monthly means |
| P0 | `annual_precip_mm` | mm/year / 0.1 | POWER corrected precipitation ANN mean daily rate × 365.24 days/year for this climatological period |
| P0 | `bare_fraction` | fraction / 0.001 | MODIS MCD12Q1.061, 2024, IGBP class 16 / accepted classified support; native ~463 m pixels |
| P0 | `landcover_valid_fraction` | fraction / 0.001 | Accepted MODIS classified area / target area |
| P1 | `permafrost_fraction` | fraction / 0.001 | ESA CCI v5, 2023 modeled PFR percent / 100, nominal 0.01° grid |
| P1 | `permafrost_valid_fraction` | fraction / 0.001 | Accepted CCI source area / target area |
| P1 | `relief_5km_m`, `relief_21km_m` | m / 1 | Max–min of derived 1 km elevation in 5×5 and 21×21 neighborhoods |
| P1 | `tpi_5km_m`, `tpi_21km_m` | m / 0.25 | Center elevation minus neighborhood mean at those scales |
| P1 | `dominant_lithology_code` | categorical integer | GLiM v1.0, 0.5° reduced-resolution global grid; containing source cell at target center |
| Context | `permanent_snow_ice_fraction` | fraction / 0.001 | MODIS IGBP class 15 / accepted classified support; not seasonal snow persistence |

All stored bands are standard PostGIS `16BUI`. Code 0 is NODATA; usable codes
decode as `code * scale + offset_value`. A measured physical zero has a nonzero
code. Numeric rounding error is verified against half the storage increment.
Units, source statistic and dates belong to the concrete raster layer, not just
the broader candidate variable. No custom codec or PostgreSQL extension beyond
ordinary PostGIS raster is used.

## Actual access and interpretation checks

* [EarthEnv/PANGAEA topography](https://doi.org/10.1594/PANGAEA.867113): anonymous
  remote raster windows worked for elevation and slope in all three areas.
  The published 1 km grid is 1/120° rather than constant 1 km spacing. Source
  elevation TIFF tags say “Maximum” despite the published median filename and
  catalog definition. We preserve that conflict in provenance and use the
  publication-defined statistic provisionally; resolve it before a scientific
  release. Separate precomputed roughness/TPI URLs returned HTTP 503 during
  checks, so they are not represented as imported source products. The supplied
  5/21 km descriptors are explicitly derived on the coarser grid. EarthEnv also
  documents high-latitude terrain artifacts; Haughton merits local DEM review.
* [NASA POWER climatology](https://power.larc.nasa.gov/docs/services/api/temporal/climatology/):
  six anonymous regional JSON requests worked. The actual MERRA-2 grid is
  **0.625° longitude × 0.5° latitude**. Values are aggregated by fractional
  spherical-area overlap of native cell supports. This creates aligned query
  pixels, not independently observed 1 km climate. Coldest/warmest monthly means
  are not weather extremes. Annual precipitation is a climatological estimate,
  not a measured calendar-year accumulation.
* [MODIS MCD12 user guide](https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf):
  authenticated Earthdata HDF4 downloads worked for all intersecting 2024
  granules. Only QC 0 (classified land) and 2 (classified water) are accepted.
  QC 1 is excluded even though unclassified land is labelled barren. Fractions
  use exact fractional support in the source sinusoidal projection, with
  classified coverage provided separately. Temporary original HDF files are
  deleted after reading the needed windows.
* [ESA CCI permafrost v5](https://doi.org/10.5285/d235665772ec4b558e9a89ac85595e71):
  the full 2023 file downloaded anonymously (12,448,396 bytes). Binary inspection
  verified PFR values 0–100 percent, fill 255, and latitude coordinates 25–85°N.
  The file's GeoTransform conflicts with its CF coordinate vectors; the latter
  control row order and placement. Atacama is outside the coordinate domain;
  Death Valley is inside it but the local PFR data are fill values. Both remain
  **NULL, never inferred zero**. Haughton has usable modeled frozen-ground data.
  PFR does not measure ice concentration or certify subsurface resources.
* [GLiM coarse lithology](https://doi.org/10.1594/PANGAEA.788537): the official
  38,670-byte ZIP downloaded anonymously. Its ASCII header, grid and class
  dictionary were checked. Category integers are copied from the containing
  native source cell and never averaged. Source class 15 (“no data”) is masked.
  Water and ice retain distinct context codes. This is regional geology,
  **not 1 km geology or local basalt/mineral detection**. Attribution: Hartmann
  and Moosdorf (2012), GLiM v1.0, CC BY 3.0.

The sources mix long-term terrain/climate with 2023 permafrost and 2024 land
cover. Their combination is an environmental screening baseline, not a
synchronized observation of a particular date. Source URLs, hashes, downloaded
granules, decoding rules, warnings, grids and quantization errors are in
[manifest.json](manifest.json); public API response evidence is in `sources/`.
No credentials or signed download URLs are retained.

## Does the pilot work?

The three profiles in [priorities.json](priorities.json) compare values at the
**same pixel**, requiring minimum coverage and rejecting missing required data:

| Profile | Atacama / 10,000 | Haughton / 10,000 | Death Valley / 10,000 |
| --- | ---: | ---: | ---: |
| Dry, bare, low mean slope | 9,782 | 0 | 5,153 |
| Hot, dry, bare, low mean slope | 0 | 0 | 5,153 |
| Cold, bare, low mean slope, modeled frozen ground | 0 | 7,865 | 0 |

All three reference-point cells pass their intended environmental profile.
The filters were developed while examining these examples; this is **not a blind
benchmark**, global ranking, or proof that the system can discover new planetary
analogs. The dry profile also selects Death Valley, as it reasonably should.
No site names or coordinates enter the predicates.

Full raster roundtrip checks compare all 480,000 pixel codes, including missing
pixels, to the derived files. The complete SQL match coordinate sets are checked
against independent NumPy masks; the integration checks also exercise the wide
view, geometry point lookups, read-only reader access and rejection of invalid
grid updates. See the [actual validation report](../pilots/database/priority-raster-validation.json)
and [load report](../pilots/database/priority-raster-load.json) for measured times,
storage scopes, center values and example matching coordinates.

The recorded reprocessing run took **59.1 seconds**, reusing cached POWER/CMR
responses and the CCI file while fetching terrain windows and MODIS granules.
Database ingestion took **0.61 seconds**, and integration checks took **1.89
seconds**. These are local pilot measurements, excluding initial dependency/image
setup, not estimates for processing the whole Earth. Each profile query took
roughly 0.26–0.42 seconds over the three regions in this run.

The 48 current raster values occupy **353,898 bytes as PostgreSQL raster
datums**, after ordinary PostgreSQL storage compression; uncompressed two-byte
pixel payload is 960,000 bytes. Relation/index totals also include an earlier
45-band research bundle: 360,448 bytes for layer metadata and indexes, and
868,352 bytes for raster tiles and indexes. The complete database was 53.2 MB,
including the H3 pilot and PostgreSQL/PostGIS catalogs. WAL and container images
are separate. Repeating the import reuses the same 48 layers.

These variables are sufficient for a useful first environmental demonstration.
They do **not** establish crater origin, mineral composition, detailed Mars Hill
basalt exposure, dunes, soil trafficability, or a contiguous flat landing area.
A low mean slope still permits small steep obstacles. Independent high-resolution
terrain and geological checks are the next validation step for shortlisted sites.

## Load and query

The derived tiles are retained in this repository. To import them, no Earthdata
login or original raster downloads are needed:

```sh
python3 scripts/setup_local_env.py
docker compose up -d --wait db
docker compose run --build --rm loader
docker compose run --rm --entrypoint python loader scripts/load_raster_pilot.py --report /reports/priority-raster-load.json
docker compose run --rm --entrypoint python loader scripts/test_raster_pilot.py --report /reports/priority-raster-validation.json
```

The first loader supplies the original variable catalog and H3 pilot. The raster
loader verifies file/code/processor hashes and imports in a single transaction.
Repeating an identical bundle reuses its layers. Changed processing produces a
new bundle; the local database may retain earlier research bundles. Always bind
the bundle SHA from the load report rather than querying across versions.

```sql
-- $1 = bundle_sha256 from priority-raster-load.json
SELECT site_key, x, y, elevation_median_m, slope_mean_deg,
       annual_precip_mm, warmest_month_air_c, bare_fraction
FROM research.priority_pilot_cells
WHERE bundle_sha256 = $1
  AND annual_precip_mm <= 250
  AND warmest_month_air_c >= 25
  AND bare_fraction >= 0.5
  AND slope_mean_deg <= 10
  AND landcover_valid_fraction >= 0.8
  AND terrain_valid_fraction >= 0.9;
```

`x` and `y` are 1-based pixel indices, not longitude/latitude. Layer metadata
contains the local SRID, origin and spacing to derive geometry. The reference
point lies at a grid corner and belongs to column/row (51,51) under floor
ownership. To decode a specific pixel without expanding the full tile:

```sql
SELECT feature_key, unit,
       ST_Value(rast, 1, $3::integer, $4::integer, true) * scale + offset_value AS value
FROM research.raster_tiles
WHERE bundle_sha256 = $1 AND site_key = $2;
```

`research.priority_pilot_cells` expands rasters on demand for bounded SQL
convenience. It does not provide a scalar index on every global pixel value.
Layer/extent selection and global query performance need further benchmarking.
Categorical code dictionaries are in `research.raster_layers.provenance`.
The Rust API may query these read-only views through `API_DATABASE_URL`; the
HTTP service itself is not implemented.

To regenerate the derived data with the already authorized Earthdata account:

```sh
uv run scripts/priority_pilot.py --credentials-file /private/path/to/earthdata-login
```

Processing runs on this computer. Terrain uses remote windows; climate uses
small JSON responses; MODIS requires temporary full granules; CCI currently
requires one small full NetCDF; GLiM is a small full ZIP. Source rasters are not
archived in the repository. Reprocessing changes the manifest and therefore the
bundle identity. Runtime depends on network access and upstream availability.

## Next priorities

1. Add actual seasonal snow persistence with cloud/valid-day coverage, rather
   than relabeling the available permanent snow/ice class.
2. Validate finer geological maps and selected high-resolution DEM windows for
   candidate locations, including the challenge's specific geological cases.
3. Freeze filters, then test independent regions and held-out known analog sites;
   evaluate false positives and connected usable area before a global import.
4. Scale the proven P0 subset first. Defer unavailable or scientifically weak
   variables instead of treating the full candidate catalog as guaranteed data.
