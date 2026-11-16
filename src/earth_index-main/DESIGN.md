# Earth Index database design

The serving database is **file-backed DuckDB 2.0**, with physical **DOUBLE** measurements on the nominal **4 × 4 km** global grid. Integer IDs and categorical codes stay integers. The Rust/Axum/Tonic API executes arbitrary read-only SQL with named parameters. Queries are never saved.

The global datasource is `earth-ed5bc9904a5f`, revision `ed5bc9904a5f7211e52e3ac5b21cd145fe2f69f6f38ddc0df98da08a69e90f0e`. It contains 460 native MODIS-sinusoidal regions, each 278 × 278 cells, at 3,999.822013548645 m spacing on a 6,371,007.181 m sphere. There are 35,550,640 rectangular addresses: 31,881,940 inside Earth and 3,668,700 padding addresses. Variable-specific source gaps remain NULL.

## Relations

| Relation | Representation |
| --- | --- |
| `measurements` | Physical column table: `region_id USMALLINT`, `pixel UINTEGER`, 24 nullable DOUBLE measurement/quality fields, `dominant_lithology_code SMALLINT` |
| `regions` | 460 rows with `region_id`, `site_key`, width/height, CRS/SRID, six DOUBLE affine coefficients, and WGS84 coverage geometry |
| `cells` | Convenience view joining the above, adding 1-based x/y and a lazily computed WGS84 cell footprint `geom`; padding geometry is NULL |
| `api_metadata` | Frozen protobuf metadata snapshot retaining units, definitions, per-region source provenance, grid transforms and loaded coverage WKB |
| `layout_identity` | Scientific source revision used to build this immutable snapshot |

`pixel` is zero-based row-major within a region: `x = pixel % width + 1`, `y = pixel // width + 1`. Region IDs are local to this revision; `site_key` identifies the native hXXvYY region. Do not rely on SQL result order.

The 25 measurement/quality fields are:

| Field | SQL type | Unit |
| --- | --- | --- |
| `annual_precip_mm` | DOUBLE | mm/year |
| `bare_fraction` | DOUBLE | 1 |
| `coldest_month_air_c` | DOUBLE | °C |
| `dominant_lithology_code` | SMALLINT | category_code |
| `elevation_median_m` | DOUBLE | m |
| `landcover_valid_fraction` | DOUBLE | 1 |
| `permafrost_fraction` | DOUBLE | 1 |
| `permafrost_valid_fraction` | DOUBLE | 1 |
| `permanent_snow_ice_fraction` | DOUBLE | 1 |
| `relief_21km_m` | DOUBLE | m |
| `relief_5km_m` | DOUBLE | m |
| `slope_mean_deg` | DOUBLE | degree |
| `surface_day_c` | DOUBLE | °C |
| `surface_day_valid_fraction` | DOUBLE | 1 |
| `surface_high_c` | DOUBLE | °C |
| `surface_low_c` | DOUBLE | °C |
| `surface_mean_c` | DOUBLE | °C |
| `surface_night_c` | DOUBLE | °C |
| `surface_night_valid_fraction` | DOUBLE | 1 |
| `surface_paired_valid_fraction` | DOUBLE | 1 |
| `surface_range_c` | DOUBLE | °C |
| `terrain_valid_fraction` | DOUBLE | 1 |
| `tpi_21km_m` | DOUBLE | m |
| `tpi_5km_m` | DOUBLE | m |
| `warmest_month_air_c` | DOUBLE | °C |

Units, methods, category dictionaries and missing-data caveats are returned by Metadata. The [scientific methods](db/WORLD_IMPORT.md) remain authoritative.

All continuous values, including quality fractions, are DOUBLE in the serving file. This first conversion decodes the existing quantized import and therefore retains its original rounding precision; it does not restore unrecorded source precision. Subsequent importers can supply native physical values without another serving quantization step. Previously imported PostgreSQL/PostGIS rasters and pilots remain intact as research/ingestion artifacts; they are not required by the running API.

## Scoring and wire format

The demo builds SQL from available target measurements, using generous optional hard filters and a weighted Gaussian similarity. Related terrain/thermal fields share family weights. Missing required measurements or insufficient coverage produce NULL; failed hard filters produce zero. Users can edit the SQL and named parameters directly.

The global Earth demo uses one WebGL2 score layer and no coverage clipping mask.
Lossless uint16 scores and their peak-preserving overviews are uploaded as integer
texture arrays on demand, with a 128 MiB LRU cache per active result. A small
native-grid lookup texture lets one screen draw address every region. The shader
maps WGS84 pixels directly into the shared geographic or spherical sinusoidal
grid; it does not rebuild projected patch images while panning. Threshold and
normalization changes update uniforms and reuse the same score textures. New
queries and clearing results release the old GPU cache. Highlights and selection
remain separate vector overlays.

Scoring arithmetic uses DOUBLE. SQL clamps each final score to [0,1] and quantizes it with `round_even(65534 * score, 0)::INTEGER`. Valid codes are **0..65534**; **65535** is reserved for missing evidence. Physical zero and missing evidence remain distinct.

`pack_u16_at(list(struct_pack(index := pixel, code := code)), width * height)` scatters codes into explicit addresses and returns a **little-endian BLOB, two bytes per cell**. This avoids sorting all cells. Duplicated/out-of-range addresses and invalid codes are errors; omitted addresses and NULL codes become 65535. `pack_u16(INTEGER[])` also packs an explicitly ordered list. Neither helper computes a score or hides query assumptions.

The SQL packers read native DuckDB vectors directly, including cached validity masks, and reuse scratch buffers. They avoid materializing nested Arrow arrays; the SQL signatures and byte format remain unchanged.

The final query returns one `(site_key, scores BLOB)` row per requested region. The frontend obtains shapes/transforms through Metadata and decodes BLOBs into Uint16Array buffers. A full rectangular world result is 71,101,280 score bytes before protobuf framing or transport compression. No DOUBLE measurement matrix is sent to the browser.

## Runtime and safety

The API shares one embedded engine, configured for eight worker threads and an 8 GB engine memory limit by default, with at most two active SQL requests. Each request owns a separate connection and read-only transaction. The database is mounted read-only; external file/network access and automatic extension installation/loading are disabled after the pinned Spatial extension is loaded, and configuration is locked. Only a single read-only query is accepted.

There is no server query deadline by default. Explicit caller deadlines and cancellation interrupt the request's connection. Response/request limits remain 128/64 MiB. Native gRPC and binary gRPC-Web share the existing recursive protobuf protocol; the legacy `pg_type` field now carries DuckDB types.

See [operations](db/README.md), [API](backend/README.md), and the [historical raster design](docs/POSTGIS_RASTER_DESIGN.md).
