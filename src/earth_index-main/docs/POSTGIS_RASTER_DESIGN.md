# Historical PostGIS raster design

Superseded for API serving by [the DuckDB design](../DESIGN.md). The scientific raster imports remain unchanged.

# Earth Index database design

Target: **4 × 4 km cells with 16-bit quantized measurement bands** for global
environmental screening. Each cell represents approximately 16 km². Scoring
expressions remain user-defined; decoded floating-point cell tables are not the
selected serving representation.

Implemented global dataset: 25 aligned U16 bands, 31,881,940 Earth-domain cells,
460 native MODIS-sinusoidal regions, published as `earth-ed5bc9904a5f`.
The exact equal-area spacing is 3,999.822 m (nominal 4 km), with 278 × 278 cells
per native region. Source gaps and non-Earth padding remain missing. See
[database operations](../db/README.md#global-4-km-dataset) and
[processing methods](../db/WORLD_IMPORT.md).

The bounded research pilots remain intact, including 16 aligned bands per
100 × 100 km area for three areas at 1 km spacing. See the
[priority pilot results](../research/priority-pilot/README.md). H3 observations are
retained separately. An [expanded stress pilot](../research/stress-pilot/README.md)
adds 24 diverse areas using the same layout and fixed environmental filters.
Global ingestion and byte-exact database verification are implemented. Full-world
query performance and scientific suitability calibration remain follow-up work.

## Storage decision

Use **standard tiled PostGIS rasters with 16-bit quantized integer bands** for
the selected 4 × 4 km Earth grid. PostgreSQL remains the authoritative
database and SQL engine; the API uses Rust/Axum. Use the ordinary PostGIS raster
extensions (`postgis` and `postgis_raster`) and PostgreSQL storage mechanisms. A custom raster codec or custom
native extension is not required.

The previous H3 cell/long-observation proposal is retained in
[the historical pilot design](H3_PILOT_DESIGN.md). Its scientific definitions
remain useful references, but the global raster store does not require a SQL row
and polygon for every pixel. Current database setup and imported observations
are documented in [database operations](../db/README.md).

## Grid and tiles

- A grid definition fixes the CRS, origin, pixel spacing, tile dimensions, axis
  order and footprint/missing-data policy. Changes create a new grid version.
- The implemented nominal **4 km** grid uses MODIS sinusoidal projection on a
  6,371,007.181 m sphere (custom SRID 930001). Exact spacing is 3,999.822013548645 m,
  aligning 278 cells to each native tile without seams. Its global origin is
  (-20,015,109.355797417, 10,007,554.677898709) m. A fixed longitude/latitude
  increment does not give constant-area cells. Local AEQD pilot grids remain
  separate grid versions.
- Pixel resolution and storage/network tile dimensions are separate choices.
  The existing pilot uses 100 × 100 pixels per tile, covering 100 × 100 km.
  Preserving that physical footprint at 4 km spacing would give 25 × 25 pixels;
  the implemented global region dimensions are 278 × 278 pixels.
- Migration `006` admits declared grid dimensions/affines and `008` adds clipped
  global coverage and safe cell geometry at poles and the map boundary. Applied
  migrations `002` and `003` remain unchanged.
- All compared layers share pixel alignment and tile coordinates. Record each
  source's native spatial support separately; resampling does not create finer
  independent measurements.
- Derive pixel position from the grid, tile coordinates and pixel row/column.
  Generate point or polygon geometry on demand.
- Edge tiles preserve the common alignment and use NODATA outside their valid
  footprint. Do not interpret padding as physical zero or a measurement.

## Minimal physical layout

Two relations are sufficient for raster observations, alongside the existing
variable definitions and source catalog:

| Relation | Stored information |
| --- | --- |
| `catalog.raster_layers` | Layer identity; variable/statistic and version; unit; grid definition/version; time/depth/other dimension selection; quantization scale and offset; source/processing provenance; validation/publication status |
| `data.raster_tiles` | Layer ID, tile column and row, and one single-band PostGIS `raster` of pixel type `16BUI` |

A layer is one variable/statistic on one grid for one fully specified dimensional
slice. Static elevation, daytime temperature for a particular interval, and a
quality/coverage fraction are separate layers. Metadata is shared by all tiles
in a layer, not repeated for every pixel. Different variables need no new SQL
columns or tables.

Use a primary key of `(layer_id, tile_row, tile_col)` and a layer foreign key.
Create a GiST index on `ST_ConvexHull(rast)` for spatial tile selection. Ingestion
must validate the expected grid, pixel type, band count, alignment and tile
coordinates; validate uniqueness and complete declared coverage before publishing
a layer. Missing tiles and existing tiles filled with NODATA have distinct
coverage meanings and must be described in the layer manifest.

Single-band tiles let queries load only the variables they need. Additional
tables, cached pixel rows and per-variable indexes require measured justification.
The [PostGIS loader](https://postgis.net/docs/using_raster_dataman.html) supports
tiled rasters, and [PostGIS pixel types](https://postgis.net/docs/RT_ST_BandPixelType.html)
include unsigned 16-bit integer bands.

## Quantization and missing data

Store numeric **integer codes**, not 16-bit floating-point values:

```text
physical_value = code * scale + offset
code = round((physical_value - offset) / scale)
```

Use codes `1..65535` for usable values. Reserve code `0` as the PostGIS band
NODATA value. A physical zero can still be represented by a nonzero code through
the chosen offset. Signed quantities are supported with an offset as well.

Scale must be finite and positive; offset must be finite. Choose and version
scale/offset per layer, consistently across every tile in that layer. Validate
finite source/derived values and reject code overflow or a usable value mapping
to NODATA. **Never silently clamp values to fit**: [PostGIS constructors](https://postgis.net/docs/RT_ST_AddBand.html) can clamp
out-of-range inputs, so validation must occur before constructing raster bands.

Round to the nearest representable code and record the maximum observed rounding
error; it must not exceed half the scale, allowing a documented floating-point
tolerance. Scale is a storage increment, not a measurement uncertainty or a claim
of scientific accuracy. Explicitly document threshold behavior near rounding
boundaries; queries are exact over decoded stored values.

Starting increments for the five real-data demo outputs:

| Output | Increment | Maximum rounding error |
| --- | --- | --- |
| Mean elevation | 1 m | 0.5 m |
| Mean slope | 0.1° | 0.05° |
| Fraction with native slope <5° | 0.001 | 0.0005 |
| Daytime land-surface temperature | 0.02 K | 0.01 K |
| Valid temperature footprint fraction | 0.001 | 0.0005 |

These are pilot choices, not a universal encoding for all 91 candidates. Before
loading another variable, check that its validated range and required increment
fit the usable code domain. A variable that cannot fit is deferred until an
explicit encoding decision is made, rather than silently losing precision.
Categorical codes use a versioned dictionary, not numeric scaling.

Quality/coverage quantities needed for filtering are separate aligned bands.
NODATA is not a score and must never compare as a valid numeric observation.
Retain source quality rules and missingness summaries in the layer manifest;
additional per-pixel missing-reason detail requires an explicitly counted band.

## SQL interface

Use standard PostGIS pixel/window functions with ordinary SQL views or SQL
functions to decode units. No native codec binding is needed.
[ST_Value](https://postgis.net/docs/RT_ST_Value.html) returns the stored integer
code as a SQL number, not an automatically scaled physical value. With NODATA
exclusion enabled, missing samples yield NULL; preserve that NULL when applying
the catalog's scale and offset.

For example, a point lookup using the implemented relations can use:

```sql
-- $1 is the layer ID; $2 is a point geometry already in the raster's CRS.
SELECT l.variable_key,
       ST_Value(t.rast, 1, $2, true) * l.scale + l.offset_value AS value,
       l.unit
FROM data.raster_tiles AS t
JOIN catalog.raster_layers AS l USING (layer_id)
WHERE t.layer_id = $1
  AND ST_ConvexHull(t.rast) && $2;
```

Future serving functions must enforce pixel-edge ownership and exclude
NULL/outside-tile results to prevent duplicate boundary measurements. The bounded
pilot has one tile per area/layer; point and pixel-corner lookups are checked
against the processed arrays. Transform longitude/latitude into the registered
raster CRS before lookup.

For multi-variable searches, filter candidate tiles by layer/time and spatial
extent, then compare decoded values at **the same pixel coordinates**. Expose
matching pixel IDs, values and optional geometry as ordinary query results.
Do not use independent tile averages or matching-area fractions as proof that
temperature and slope coincide. Connected area searches require an explicit
connectivity operation after constructing the common matching-pixel mask.

The serving goal is an **unsorted score raster**, rather than a global ranked
list. Evaluate the user's expression at each selected 4 km cell, then quantize
the resulting scores for compact delivery as compressed 16-bit arrays. The
shared protocol defines grid/order, byte order, score scale/offset and missing
codes; locations need not be repeated per cell. Request-visible chunks and LOD
remain delivery options. Coarser summaries are environmental screening values,
not proof that the same conditions coincide at finer resolution within a cell.

The spatial index accelerates tile selection; it does not index every pixel's
value. Start with standard raster operations on candidate tiles. Global searches
may scan many tiles, so do not promise indexed scalar-column performance or
instant arbitrary SQL over every pixel. Benchmark before adding conservative
tile min/max bounds or specialized indexes. Avoid materializing the full global
pixel grid as SQL rows merely to expose a convenient interface.

## Scientific ingestion and publication

Read original provider windows/subsets using bounded temporary storage. Retain
derived quantized rasters, the source references/hashes, methods and validation
results; a full local source-raster archive is not required. Quantization happens
**after** unit conversion, quality filtering and the declared aggregation.

Compute terrain derivatives at the required native support before aggregating
them. Preserve temperature window definitions and quality screening. Missing
vegetation measurements on another body are not automatically physical zero.
Units, time, coverage and body/source applicability remain explicit.

Load tiles in a transaction or into an unpublished layer. Validate provenance,
alignment, sample counts, missingness, every encoding error bound and a physical
value roundtrip before publication. Publish complete layers atomically and
version reprocessing/encoding changes instead of overwriting published outputs.
The Rust API uses a read-only database role. Further query limits and research
SQL execution controls can reuse the requirements in the historical design.

## Size and next validation

At roughly **31.875 million global 4 × 4 km cells**, one snapshot of 91 two-byte
scalar bands has about **5.80 GB of raw pixel payload**. The current 16-band
subset would have about **1.02 GB**, and a single 16-bit score layer about
**63.75 MB**. These are decimal units, using approximately 510 million km² for
the whole Earth including oceans and excluding grid padding. This is arithmetic, not a measured
PostGIS database size or a promise that all 91 candidates fit this encoding.
Tile rows, raster headers, spatial indexes, metadata, additional coverage/quality
bands and PostgreSQL operational storage are extra. Standard PostgreSQL storage
may compress tiles; no additional compression ratio is assumed in the baseline.
Extra dates, depths and statistics are additional layers.

The earlier approximately 57 GB estimate depended on extrapolating the
[k²-raster experiment](../research/k2-demo/README.md); that codec is not part of this
preferred design. The five quantized Atacama grids remain a separate codec
experiment.

Migrations `002` and `003` implement the bounded raster layout, provenance,
grid/type constraints and decoded SQL views. `research.priority_pilot_cells`
offers a wide, on-demand interface to aligned values; it does not materialize a
global cell table. Queries select an explicit `bundle_sha256`. The integration
checks verify every code, NODATA, point lookups and complete same-pixel profile
match sets against independent NumPy masks. See the
[pilot instructions](../research/priority-pilot/README.md) for measurements and
limitations. Global coverage and pole/map-edge geometry now have independent
checks; connected area searches across region boundaries and broad-query
performance remain subsequent work.

The [measurement-layout benchmark](../research/ranking-pilot/VALUE_LAYOUT.md)
tested decoded float64 cell rows using the existing 1 km pilot. The selected
4 km design retains 16-bit quantized measurements; that experiment does not
implement or validate the new grid. The separate 4 km expansion checks shared
fine/coarse behavior; the global dataset records its distinct native grid and
aggregation method explicitly.

## Implemented gRPC API

The [Rust/Axum + Tonic backend](../backend/README.md) executes stateless SQL with
named parameters over native gRPC and binary gRPC-Web. The Protobuf service
`earth_index.v1.EarthIndex` exposes `Query`, `Metadata`, `Health` and `SamplePoint`. Datasource
identity is a query request field. Each raster snapshot exposes cells/layers/tiles
SQL views. Metadata reports actual grids, coverage, units and provenance. SQL
controls filtering, scoring, ordering, limits and optional uint16 score packing.
Reflection enables the standard grpcui explorer. Serving views preserve the
quantized raster store; both the global 4 km dataset and original pilots are
available as separate datasource identities.
