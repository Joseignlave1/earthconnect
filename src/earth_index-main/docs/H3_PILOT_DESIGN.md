# Historical H3 pilot database design

Status: historical reference for the existing H3 pilot. The current target
storage design is [DESIGN.md](../DESIGN.md): tiled PostGIS rasters with 16-bit
quantized numeric values. This document preserves the earlier proposal and its
scientific/catalog requirements; its physical layout and implementation sequence
are not requirements for the new raster store. The implemented subset is
documented in [DATABASE.md](DATABASE.md).

## 1. Purpose and scope

Earth Index is a scientific geospatial feature store. Researchers must be able to
filter Earth regions by measurements and classifications, calculate their own
scores, and reproduce the resulting candidate rankings. Finding terrestrial
analogs for lunar and Martian environments is the initial use case.

**PostgreSQL is the authoritative database and SQL query engine. PostGIS provides
spatial types and operations.** This supersedes the DuckDB proposal in the
[original discussion](https://chatgpt.com/share/6af87ae4-707c-83e9-a894-8e84859a1289).
Parquet may be an ingestion/export format; it is not the live database.

The database owns the metadata catalog, geographic cells, derived observations,
published search snapshots, scientific functions, and query provenance. Source
rasters stay with their providers by default; the project stores their references,
not a local raster archive. Ingestion imports existing provider summaries where
their spatial and temporal support match the target feature. Otherwise, workers
read the necessary source windows, compute tile statistics, and load only those
statistics and their provenance into PostgreSQL. The later API, MCP server, and
map UI consume the same database interface. The API implementation language is
**Rust**; ingestion workers may retain the existing Python scientific tooling.

Catalog discovery reads metadata only. Computing a new tile mean, slope, or other
feature may require reading raster values even though those pixels are not retained.
Use remote windows/subsets where supported, or temporary downloads with a bounded,
evictable cache. Persistent raster storage is optional and must be explicitly
configured. Storage estimates must distinguish the database, peak temporary
working space, and the total size of upstream assets; upstream asset sizes are
not the project's required disk allocation. Result exports may use external storage.

The first implementation must support:

- Numeric, categorical, and tagged features without a migration per variable.
- Geographic, temporal, quality, and feature-value filters.
- Arbitrary SQL expressions for researcher-defined scoring and aggregation.
- Exact ranking over eligible published observations.
- Multiple datasets, source versions, processing versions, and resolutions.
- Explicit units, observation periods, missingness, uncertainty, and lineage.

Global raw-pixel storage, a custom query language, vector search, and automatic
scientific equivalence between planetary measurements are outside the MVP.

## 2. Baseline decisions

| Concern | Decision |
| --- | --- |
| Database | PostgreSQL 18; pin a supported patch release in deployment |
| Local deployment | Docker Compose with PostgreSQL/PostGIS and a persistent named volume |
| API language | Rust, using a separate database role for API reads |
| Required extension | PostGIS, with a release compatible with the deployment |
| Geographic grid | H3 for Earth, initially resolution 5 for regional screening |
| H3 integration | Generate/validate identifiers and boundaries in ingestion using a pinned H3 library; no required PostgreSQL H3 extension |
| Source arrays | Remain at provider URLs by default; bounded temporary reads/downloads for derived features, no mandatory local archive |
| Observation storage | Normalized, typed, long-form records |
| Common search path | Published wide projection backed by normalized observations |
| Metadata | Relational scientific fields plus preserved source JSONB |
| Search behavior | Hard filters followed by exact SQL scoring and ordering |
| Publication | Immutable releases; a release is one grid and analysis window |
| Identity | `bigint GENERATED ALWAYS AS IDENTITY` for internal entities; UUIDs for releases and query executions |
| Timestamps | `timestamptz`, serialized in UTC |

Resolution 5 is a starting choice, not a claim of 10 km square cells. H3 cells
have varying areas; its published average hexagon area at this resolution is
approximately 253 km². Store each cell's actual area and expose the grid
resolution. Finer grids are separate registered grids. See the
[H3 cell statistics](https://h3geo.org/docs/core-library/restable/).

Initial ingestion covers a bounded Earth region with terrain and surface
temperature. Global coverage is a later ingestion milestone, not a prerequisite
for implementing the database.

## 3. Schemas and ownership

| Schema | Responsibility |
| --- | --- |
| `catalog` | Dataset, source-variable, feature, and category definitions |
| `geo` | Grid definitions, cells, and versioned region boundaries |
| `ingest` | Processing runs, inputs, validation, and staging |
| `data` | Observations and immutable publication manifests |
| `science` | Versioned scientific SQL functions and aggregates |
| `research` | Supported researcher-facing views and search projections |
| `audit` | Saved query definitions, execution manifests, and result references |

All tables have explicit primary keys. Foreign keys use `RESTRICT` for scientific
records unless a lifecycle rule below explicitly allows deletion. Published
records cannot be updated or deleted by ingestion or query roles. Migrations use
a separate owner role.

In the table descriptions below, fields are required unless marked nullable.
Identifier fields reference the corresponding entity. `jsonb` defaults to an
empty object only where the object is genuinely optional; required scientific
metadata must not be satisfied by an empty placeholder.

## 4. Metadata catalog

### 4.1 Datasets and source assets

`catalog.datasets`

- `dataset_id bigint` PK; `dataset_key text` UNIQUE, stable machine identifier.
- `title text`, `provider text`, `description text`, `license_uri text` nullable.
- `created_at timestamptz`.

`catalog.dataset_versions`

- `dataset_version_id bigint` PK; `dataset_id bigint` FK.
- `version text`, `source_uri text`, `retrieved_at timestamptz`.
- `body text` initially constrained to `earth`.
- `coverage geometry(MultiPolygon,4326)` nullable; `temporal_extent tstzrange`
  nullable for static products.
- `source_metadata jsonb`, `metadata_sha256 text`.
- UNIQUE (`dataset_id`, `version`). A changed upstream artifact under the same
  provider version requires a distinct internal version, retaining the provider
  version in source metadata.

`catalog.assets`

- `asset_id bigint` PK; `dataset_version_id bigint` FK.
- `asset_key text`, `uri text`, `media_type text`, `byte_size bigint` nullable.
- `checksum_algorithm text`, `checksum text`; immutable content fingerprint.
- `source_metadata jsonb`; UNIQUE (`dataset_version_id`, `asset_key`).
- URIs contain no credentials. Ingestion must capture a content checksum before
  publication; a mutable URL or retrieval timestamp alone is insufficient.
  Record checksum provenance, including provider-supplied checksums when available.
  Catalog discovery may leave an unavailable checksum unknown; do not download a
  raster solely to populate its discovery metadata.

### 4.2 Array and source-variable metadata

`catalog.source_variables`

- `source_variable_id bigint` PK; `dataset_version_id bigint` FK.
- `variable_key text`, `data_type text`, `unit text` nullable for categories.
- `standard_name text` nullable, `dimensions jsonb`, `coordinates jsonb`.
- `crs_definition jsonb`, `nodata_definition jsonb`, `quality_definition jsonb`.
- `scale_factor double precision`, `add_offset double precision`.
- `spatial_support jsonb`, `temporal_support jsonb`, `source_metadata jsonb`.
- UNIQUE (`dataset_version_id`, `variable_key`).

`dimensions` is an ordered array of objects describing name, axis role, length,
coordinate reference, and values or bounds reference. Large coordinate vectors
remain in source assets. Record depth, spectral band, ensemble, and other axes,
not just latitude, longitude, and time. `spatial_support` describes pixel
footprint/resolution and point-versus-area support. `temporal_support` records
instantaneous, interval, composite, or climatological semantics and calendar.

Use the [STAC Datacube extension](https://github.com/stac-extensions/datacube) as
the interchange mapping for dimensions and variables, and
[CF conventions](https://cfconventions.org/) for scientific coordinate and array
semantics. Preserve original metadata alongside normalized fields. Record the
standard version used; do not claim complete STAC/CF compliance from a few fields.

### 4.3 Queryable feature definitions

`catalog.features`

- `feature_id bigint` PK; `feature_key text` UNIQUE; `title text`.
- Example keys: `terrain.slope.mean`, `terrain.elevation.stddev`,
  `temperature.surface.day.mean`, `geology.lithology`.

`catalog.feature_versions`

- `feature_version_id bigint` PK; `feature_id bigint` FK; `version integer`.
- `value_kind text`: `numeric`, `category`, or `boolean`.
- `canonical_unit text` nullable only for nonnumeric values; use `1` for ratios.
- `definition text`, `aggregation_method text`, `weighting_method text`.
- `spatial_policy jsonb`, `temporal_policy jsonb`, `missing_policy jsonb`.
- `uncertainty_definition jsonb`, `valid_domain jsonb`, `definition_sha256 text`.
- UNIQUE (`feature_id`, `version`). Published definitions are immutable.

`catalog.category_values`

- Composite PK (`feature_version_id`, `category_code text`).
- `label text`, `description text`, `source_mapping jsonb`.

Definitions distinguish slope in degrees from elevation variability in meters,
surface temperature from air temperature, absolute temperature from temperature
difference, and categorical class from inferred likelihood. A changed definition,
unit, aggregation, or missing-data rule creates a new feature version.

Metadata enables discovery and validation; ordinary PostgreSQL arithmetic does
not automatically enforce physical units. SQL authors must normalize compatible
quantities explicitly. Curated helpers and publication checks enforce their
declared unit contracts.

## 5. Spatial model

`geo.grids`

- `grid_id bigint` PK; `grid_key text` UNIQUE; `body text` = `earth`.
- `scheme text` = `h3`, `resolution smallint`, `library_version text`.
- `crs_definition jsonb`, `boundary_policy jsonb`, `area_method text`.
- Register a new grid identity if the boundary or area policy changes.

`geo.cells`

- Composite PK (`grid_id bigint`, `cell_id text`); FK to `geo.grids`.
- `geom geometry(MultiPolygon,4326)`, `centroid geography(Point,4326)`.
- `area_m2 double precision CHECK (area_m2 > 0)`.
- Cell IDs are canonical lowercase H3 strings, validated against the registered
  resolution during ingestion. They are not latitude/longitude coordinates.
- Boundaries must be nonempty and valid, with an explicit antimeridian split
  policy. Center coordinates use longitude, latitude order.

Use cell polygons for overlap filters and geodesic distances for meter-based
proximity. Never calculate square meters or meters from unprojected geometry
degrees. Store the area method; ingestion overlap weighting must use a consistent
documented area model. Geometry and geography have different operation and cost
characteristics; see the [PostGIS guidance](https://postgis.net/documentation/faq/geometry-or-geography/).

`geo.regions`

- `region_id bigint` PK; `region_key text`, `version text`, `name text`.
- `dataset_version_id bigint` FK; `geom geometry(MultiPolygon,4326)`.
- UNIQUE (`region_key`, `version`). Country/administrative boundaries have lineage.

Region queries must declare whether they mean centroid membership, any overlap,
or a minimum overlap fraction. A derived region-cell lookup may cache those
results, keyed by region version, grid, cell, and membership policy.

H3 parent relationships support navigation. They are not a replacement for
geometric intersection or valid overlap weights. Parent averages cannot safely
prune matching descendants. Hierarchical pruning is deferred until conservative
bounds and completeness conditions are implemented and tested.

Earth coordinates must not be reused for Moon/Mars geometry. Future planetary
support requires separate reference-system and body-aware spatial storage and
operations. The MVP can accept a target feature vector without storing planetary
geometry.

## 6. Processing and observation model

### 6.1 Processing lineage

`ingest.runs`

- `run_id bigint` PK; `idempotency_key text` UNIQUE.
- `status text`: `pending`, `running`, `validated`, or `failed`.
- `code_revision text`, `environment_digest text`, `parameters jsonb`.
- `created_at timestamptz`; `started_at timestamptz` and
  `finished_at timestamptz` nullable until their lifecycle events occur.
- `validation_report jsonb`, `error_summary text` nullable.

`ingest.run_inputs`

- Composite PK (`run_id`, `asset_id`, `source_variable_id`), all FKs.
- `selection jsonb`: explicit band/depth/time/other dimension selectors.
- Verify that the source variable and asset belong to the same dataset version.

`data.series`

- `series_id bigint` PK; `feature_version_id bigint`, `grid_id bigint`,
  `run_id bigint` FKs.
- `dimension_selection jsonb`, `series_sha256 text` UNIQUE.
- `status text`: `staging` or `sealed`; `sealed_at timestamptz` nullable.
- UNIQUE (`series_id`, `grid_id`) supports composite observation FKs.

A series is one feature definition, grid, processing run, and fully specified
slice of remaining dimensions. Separate day/night quantities or depth selections
must not collide in the same series. The canonical signature includes these
identities and selections. A sealed series and its input manifest are immutable.
Use separate runs when separate outputs require different input lineage.

### 6.2 Observation time

`data.time_windows`

- `window_id bigint` PK; `window_sha256 text` UNIQUE.
- `kind text`: `static`, `instant`, `interval`, or `climatology`.
- `time_start timestamptz` and `time_end timestamptz`, nullable by kind.
- `reference_period tstzrange` nullable; `calendar text`; `definition jsonb`.

An interval is `[time_start, time_end)` with `time_end > time_start`. An instant
has `time_start` and no end. Static values have neither timestamp. A climatology
has a bounded reference period and explicit season/month and aggregation
definition; it is not a measurement from an arbitrary synthetic year. Hashes are
computed from canonical validated metadata, not raw JSON serialization order.

The MVP supports the Gregorian calendar. Preserve other source calendars, but
reject conversion into queryable windows until a documented conversion or
calendar-aware representation exists. Acquisition time, observation period, and
processing time remain separate concepts.

### 6.3 Observations

`data.observations`

| Column | Meaning |
| --- | --- |
| `series_id bigint` | Processing and feature identity |
| `grid_id bigint`, `cell_id text` | Geographic support |
| `window_id bigint` | Observation/composite period |
| `value_numeric double precision` nullable | Canonical numeric measurement |
| `value_category text` nullable | Feature-version-scoped category code |
| `value_boolean boolean` nullable | Boolean measurement |
| `missing_reason text` nullable | Why there is no usable value |
| `valid_fraction double precision` nullable | Valid supported area-time fraction |
| `sample_count bigint` nullable | Accepted input sample count |
| `uncertainty double precision` nullable | Uncertainty in the feature's declared interpretation and unit |
| `quality_flags jsonb` | Decoded quality information, with source semantics retained |
| `support_metadata jsonb` | Per-observation deviations from series-level support |

Primary key: (`series_id`, `cell_id`, `window_id`). Foreign keys:
(`series_id`, `grid_id`) to series, (`grid_id`, `cell_id`) to cells, and
`window_id` to time windows. The grid redundancy permits database enforcement of
the observation's spatial identity.

Required invariants:

1. A usable row has exactly one non-NULL value field, matching the feature's
   `value_kind`; `missing_reason` is NULL. A missing row has all three values NULL
   and a non-NULL reason. Zero and `false` are valid values.
2. Numeric values and uncertainty are finite. Uncertainty is nonnegative;
   `sample_count >= 0`; `valid_fraction` is NULL or in `[0,1]`.
3. NULL quality metrics mean unknown, not zero. Define the denominator of
   `valid_fraction` in feature metadata; it is not necessarily a sample ratio.
4. Category codes belong to the observation's feature version. Valid-domain,
   support, and window checks follow that definition.
5. One natural observation key has one result. Reprocessing creates another
   series; it does not overwrite previously published values.

Single-row rules use CHECK constraints. Cross-table type/category/domain rules
use controlled ingestion procedures and constraint triggers, then publication
validation. A CHECK constraint must not query mutable catalog tables.

For a declared series coverage domain, ingestion writes explicit missing rows
where input exists but fails quality/coverage requirements. A completely absent
row means not produced or outside coverage; query views preserve this distinction.
Do not fabricate the full cell × time × variable Cartesian product.

`data.observation_tags` supports multiple classification labels on an observation:
PK (`series_id`, `cell_id`, `window_id`, `tag_code text`), FK to its observation.
Tag vocabulary and semantics are versioned in the feature definition. Tags used
as measurements have lineage; descriptive catalog labels are not site evidence.

## 7. Published releases and research views

`data.releases`

- `release_id uuid` PK; `release_key text` UNIQUE.
- `grid_id bigint`, `analysis_window_id bigint` FKs.
- `status text`: `draft`, `published`, or `withdrawn`.
- `manifest_sha256 text` and `published_at timestamptz`, nullable until published;
  `created_at timestamptz`, `description text`.

`data.release_members`

- PK (`release_id`, `feature_key text`).
- `series_id bigint`, `source_window_id bigint` FKs.
- `alignment_policy jsonb`; maps one published alias to one series/window.
- A member's series grid must equal the release grid. Alias definitions must
  identify the actual feature version, canonical unit, and quality policy.

A release is a fixed comparison snapshot. Monthly searches can select multiple
releases explicitly. Static terrain may be aligned to a temperature analysis
window through a recorded static-broadcast policy. A source composite or annual
mean cannot silently stand in for another observation period: a scientifically
different aggregation requires a derived series and its own lineage.

Publication validates every member, checks row uniqueness and coverage, builds
the search projection, and commits the published status in one transaction.
Readers never observe a partially built release. Replacing a source, fixing an
observation, or changing alignment creates a new release. Withdrawal hides a
release from default discovery while retaining its data and historical manifest.

Supported query surfaces:

- `research.datasets`, `research.features`: published metadata, versions,
  canonical units, available releases, coverage, and provenance.
- `research.cells`: registered Earth cells and supported spatial attributes.
- `research.observations`: release membership joined to long-form observations;
  exposes `release_id`, `feature_key`, typed values, quality, and lineage IDs.
- `research.tags`: published observation tags with release and feature identity.
- `research.earth_features`: stored wide projection for common search features;
  PK (`release_id`, `cell_id`), with the release's grid and analysis window.

The initial wide columns are `slope_mean_deg`, `elevation_stddev_m`,
`surface_temp_day_mean_k`, and `surface_temp_night_mean_k`, with corresponding
quality/coverage columns. Temperature coverage columns are
`surface_temp_day_valid_fraction` and `surface_temp_night_valid_fraction`.
Derived thermal amplitude requires a registered
definition and explicit temporal pairing. Albedo and geology can be added later.

Build wide rows over the union of member cells using left joins; a missing
feature remains NULL. The release manifest resolves each wide column to its
source series/window. Do not duplicate all lineage JSON on every wide row.
Adding a feature to the catalog or long-form store requires no schema migration;
promoting it to a typed wide column does. Unpromoted features remain queryable
through `research.observations`.

## 8. Query semantics and scientific functions

Researchers supply ordinary PostgreSQL SQL, including joins, CTEs, aggregates,
window functions, and custom arithmetic. No fixed analog-scoring formula is
embedded in storage. Separate mandatory filters from ranking expressions.

Example, with application-bound parameters and a release selected beforehand:

```sql
SELECT
    f.cell_id,
    ST_Y(c.centroid::geometry) AS latitude,
    ST_X(c.centroid::geometry) AS longitude,
    f.slope_mean_deg,
    f.surface_temp_day_mean_k,
    0.7 * science.normalized_distance_v1(f.slope_mean_deg, 3.0, 10.0)
      + 0.3 * science.normalized_distance_v1(
          f.surface_temp_day_mean_k, 278.15, 20.0
        ) AS score
FROM research.earth_features AS f
JOIN research.cells AS c
  ON c.grid_id = f.grid_id AND c.cell_id = f.cell_id
WHERE f.release_id = $1
  AND f.slope_mean_deg BETWEEN 0 AND 15
  AND f.surface_temp_day_mean_k IS NOT NULL
  AND f.surface_temp_day_valid_fraction >= 0.8
  AND ST_DWithin(
      c.centroid,
      ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography,
      $4
  )
ORDER BY score ASC, f.cell_id ASC
LIMIT $5;
```

`$2` is longitude, `$3` latitude, `$4` radius in meters. The query layer validates
parameter types and bounds. Stable tie-breaking is required for stored rankings.
Exact means exact over the selected release's eligible stored observations; it
does not imply perfect measurements or complete Earth coverage.

Initial function contracts:

| Function | Definition and edge behavior |
| --- | --- |
| `normalized_distance_v1(value, target, scale)` | `abs(value-target)/scale`; finite `scale > 0`, otherwise error; NULL inputs return NULL |
| `range_penalty_v1(value, lower, upper, scale)` | Zero inside inclusive bounds, normalized distance to nearest bound outside; reject reversed bounds or nonpositive scale |
| `circular_distance_v1(a, b, period)` | Shortest nonnegative distance modulo finite `period > 0`; NULL inputs return NULL |
| `weighted_mean_v1(value, weight)` aggregate | `sum(value*weight)/sum(weight)` over usable pairs; reject negative/nonfinite weights; no positive valid weight yields NULL |
| `kelvin_to_celsius_v1(value)` | Subtract 273.15 for absolute temperature; temperature differences use a distinct contract |

Validate finite inputs for all helpers. Unknown uncertainty is not perfect
confidence. No universal confidence score or planetary equivalence function is
introduced without a separately specified scientific definition. Missing features
are excluded by default in curated ranking examples. Explicit penalties or weight
renormalization must report available-feature coverage and be saved with SQL.

Implement helpers as reviewed SQL/PLpgSQL functions or aggregates with
`SECURITY INVOKER`. Mark volatility and parallel safety according to actual
behavior. Version function names when semantics change; keep historical versions
needed by retained queries. Metadata-reading functions are not `IMMUTABLE`.
See [PostgreSQL function definitions](https://www.postgresql.org/docs/current/sql-createfunction.html).

## 9. Indexing and physical layout

Initial indexes, in addition to primary/unique keys:

| Relation | Index | Purpose |
| --- | --- | --- |
| `geo.cells` | GiST (`geom`) | Polygon intersection |
| `geo.cells` | GiST (`centroid`) | Geodesic proximity |
| `geo.regions` | GiST (`geom`) | Region lookup |
| `data.observations` | B-tree (`series_id`, `window_id`, `value_numeric`, `cell_id`) WHERE value_numeric IS NOT NULL | Numeric range candidates |
| `data.observations` | B-tree (`series_id`, `window_id`, `value_category`, `cell_id`) WHERE value_category IS NOT NULL | Categorical candidates |
| `data.observation_tags` | B-tree (`tag_code`, `series_id`, `window_id`, `cell_id`) | Tag filters |
| `research.earth_features` | B-tree (`release_id`, `slope_mean_deg`, `cell_id`) | Common terrain constraint |
| `research.earth_features` | B-tree (`release_id`, `surface_temp_day_mean_k`, `cell_id`) | Common thermal constraint |

Index referencing FK columns when not already covered by a suitable leading
index. Spatial acceleration requires both a GiST index and an index-aware
predicate such as `ST_Intersects` or `ST_DWithin`; see the
[PostGIS spatial index guide](https://postgis.net/documentation/faq/spatial-indexes/).

Use scalar columns for measured values and common filters. JSONB holds metadata,
not the only representation of a measurement. Add GIN indexes only for measured
metadata/tag workloads; do not index every JSONB document or feature by default.

Start unpartitioned for the bounded pilot. Before global/time-series loading,
benchmark partitioning observations by series and/or time, and wide projections
by release. The chosen layout must match query predicates and retention needs.
PostgreSQL partitioned unique keys must include partition-key columns; a future
time partition therefore requires adapting observation identity/FKs rather than
assuming the current key works unchanged. See
[partitioning constraints](https://www.postgresql.org/docs/18/ddl-partitioning.html#DDL-PARTITIONING-DECLARATIVE-LIMITATIONS).

Bulk-load staging with `COPY`, publish validated batches, then `ANALYZE`. Use
`EXPLAIN (ANALYZE, BUFFERS)` on controlled benchmark queries to measure candidate
filtering, joins, sorting, and disk reads. Exact arbitrary scores may require a
scan and sort; no generic index accelerates every possible scoring function.
Vector indexes and conservative hierarchy bounds remain optional optimizations.

## 10. Ingestion and publication workflow

1. Register immutable dataset/asset/source-variable metadata and checksums.
2. Validate the target grid, feature version, dimensional selection, and windows.
3. Create or resume a run using a canonical idempotency key derived from input
   checksums, code/environment versions, parameters, and output definition.
4. Process source data outside PostgreSQL. Decode scale/offset and quality flags,
   handle nodata, convert units, and compute documented overlap/time weights.
   Resampling must not claim finer independent measurement support.
   Reuse suitable provider summaries; otherwise prefer windowed/subset reads.
   Bound temporary downloads and evict them after validated ingestion according
   to the cache policy; retaining a local source raster is not required.
5. `COPY` observations into run-scoped staging, then validate keys, finite values,
   geometry, temporal alignment, units, support, coverage, and lineage.
6. Insert validated observations under a staging series. Retries may reuse a
   matching staging run but must reject a conflicting payload for the same key.
   Seal only after completeness and validation checks pass.
7. Create a draft release from sealed series, build its wide projection, validate
   membership/alignment, and atomically publish it.
8. Record row counts, missing/quality distributions, checksums, and processing
   timings; update planner statistics and expose the release to discovery.

Failures leave no visible partial release. Staging cleanup is retryable and may
delete unpublished run data after a retention interval. Published observations,
manifests and definitions follow an explicit retention policy. Keep provider asset
references and fingerprints after temporary source files are evicted. Record that
recomputation depends on continued upstream availability unless optional archival
storage is configured; do not claim guaranteed source retention from a URL alone.

## 11. SQL execution boundary

The SQL interface exposes scientific data, not administrator access. Use separate
roles for migrations, ingestion, publication, query execution, and audit writing.
The query role has only SELECT on approved `research` surfaces and EXECUTE on
reviewed functions; it owns no scientific objects and has no role-switching,
schema-creation, temporary-table, extension, file, or server-program privileges.

Revoke default PUBLIC execution on project-defined functions and grant reviewed
signatures explicitly. Account for inherited/PUBLIC privileges when auditing the
role; a narrow GRANT alone does not remove existing access to built-in functions.
Apply the SQL allowlist as an additional boundary. Trusted views must not invoke
privileged or unreviewed functions.

User SQL runs through a PostgreSQL-aware parser and semantic allowlist. Permit a
single SELECT/approved EXPLAIN, including read-only CTEs. Reject modifying CTEs,
`SELECT INTO`, locks, multi-statements, arbitrary function/operator calls,
unapproved relations, configuration changes, and unreviewed table functions.
Resolve names against a fixed trusted search path and review allowed overloads,
casts, and operators, not just function spellings. Parameters are bound values.

Use a read-only transaction, statement/lock/idle-transaction timeouts, bounded
`work_mem` and temporary disk, concurrency limits, and result row/byte limits.
The service cancels timed-out workers and resets pooled sessions. Read-only
transactions alone are not a complete safety boundary; PostgreSQL describes them
as a high-level restriction in [SET TRANSACTION](https://www.postgresql.org/docs/current/sql-set-transaction.html).
Run public untrusted workloads on an isolated query deployment containing only
published scientific data, with network and infrastructure resource limits.

Audit/result ownership data lives outside that deployment or is inaccessible to
the scientific query role. An API authorization check is required for every
saved-query/result lookup. Adding private scientific datasets later requires an
explicit access model and tests before they enter researcher-visible surfaces.

## 12. Query provenance and result lifecycle

`audit.saved_queries`

- `saved_query_id uuid` PK; `owner_id text`, `name text`, `sql text`.
- `parameters jsonb`, `release_ids uuid[]`, `created_at timestamptz`.
- Saved definitions may be revised; executions copy the exact executed version.

`audit.query_executions`

- `query_id uuid` PK; `owner_id text`; `saved_query_id uuid` nullable.
- `sql text`, `parameters jsonb`, `release_manifest jsonb`.
- `dependency_manifest jsonb`: pinned region versions and any additional
  catalog/spatial dependencies outside release membership.
- `database_version text`, `postgis_version text`, `function_versions jsonb`.
- `execution_mode text` initially `exact`; `limits jsonb`.
- `status text`: `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
- `started_at timestamptz` nullable, `finished_at timestamptz` nullable.
- `row_count bigint` nullable, `result_schema jsonb`, `result_uri text` nullable,
  `result_checksum text` nullable, `expires_at timestamptz` nullable,
  `error_code text` nullable.

Resolve release IDs and manifests before execution and record the resolved
versions. Region filters select a specific immutable region version, not an
unversioned latest boundary. Results are materialized once, then paginated over that stored result;
pagination must not rerun SQL against a changing default release. Assign result
row ordinals for stable paging. Return query ID, schema/units where traceable,
row count, bounded preview, and release provenance to downstream consumers.

An arbitrary calculated expression has no inferred scientific unit unless the
caller or a reviewed helper supplies one. Mark such units unknown.

If execution or result limits are exceeded, fail or explicitly label truncation.
A partial result must never be reported as a completed exact ranking. Export
expiration does not delete the scientific execution manifest. Reproducibility
requires retained releases, function versions, and controlled deterministic SQL;
record or reject volatile/time-dependent constructs. Floating-point ties and
platform effects may prevent bit-for-bit reproducibility across environments.

## 13. Validation and operations

Use ordered SQL migrations with a recorded schema version. Pin database,
extension, ingestion-library, and scientific-function versions. Keep local/test
fixtures small and label synthetic values explicitly.

Required acceptance checks:

- Migrations work on an empty PostgreSQL/PostGIS database.
- Duplicate natural keys, wrong grid/feature types, invalid categories,
  nonfinite values, bad fractions, and malformed windows are rejected.
- Missing values, unknown quality, zero, and `false` remain distinguishable.
- Unit conversion and weighted aggregation match hand-calculated fixtures,
  including partial overlap, zero weight, missing inputs, and day/night pairing.
- Spatial fixtures cover region edges, the antimeridian, polar cells, and radius
  filters with known inside/outside outcomes.
- Retry ingestion does not duplicate rows or overwrite a sealed series.
- Failed publication exposes nothing; successful publication is atomic.
- Reprocessing produces a new release; old queries retain original lineage.
- Generic long-form SQL and the wide projection return equivalent measurements
  for the same release, with stable tie ordering.
- Query-role tests reject writes and prohibited SQL/function paths, enforce
  cancellation/result limits, and prevent cross-user result access.
- Benchmark fixtures cover spatial filtering, numeric/category/tag filtering,
  multi-feature joins, custom ranking, time comparisons, and metadata discovery.

Before production publication, configure encrypted connections, backups and
point-in-time recovery, then perform a restore test. Monitor ingestion lag,
query latency/cancellations, connections, storage/index size, autovacuum health,
and replica lag if used. Performance targets are set from representative pilot
benchmarks; there is no promised global query latency in this specification.

## 14. Implementation sequence and remaining choices

Start with a discovery census before scientific raster processing. The candidate
registry in `research/variables/candidates.json` records proposed source fields,
derived features, units, dimensions, applicability and scientific caveats. The
associated `source-checks.json` preserves NASA collection metadata and catalog
intersection checks for the three pilot areas. Candidate count includes spectral
bands, derived quantities and summary definitions; it is not a count of
independent physical measurements. Presence of source granules is not a
valid-pixel coverage assessment. Non-NASA sources can remain documented candidates
until their own spatial checks are completed.

Discovery has four gates: source/catalog availability; variable and QA mapping;
small-sample validity and scientific usefulness; then selection into an ingestion
cohort. Keep the catalog broad and allow a source to be deferred without removing
its definition. Each candidate must identify native support, measurement versus
model provenance, domain masks, missingness rules, and the proposed processing
operator. EVI is deferred; other biological features are optional Earth context.
No variable requires full-region or global processing merely to enter the catalog.

The first real extraction is recorded in `research/pilots/terrain/run.json`:
three 100 × 100 km regions, 959 intersecting H3 resolution-6 cells, and 12 terrain
summary fields per cell. This is a research pilot, not a published database
release or completion of the 91-candidate registry. `scripts/process_pilot_terrain.py`
uses remote native DEM windows and retains only CSV summaries, clipped cell
geometries and provenance; no source rasters are archived. Boundary-cell support
is explicitly the intersection with the pilot region. Product quality limitations,
static/mixed-epoch time semantics and derivative operators are recorded in its
run manifest.

Temperature extraction is recorded in `research/pilots/temperature-native/run.json`:
the same cells, 46 MOD11A2.061 composite windows covering 2024, and three output
statistics with separate valid-area fractions. Eight concurrent original-file
downloads completed the three regions in 103.6 seconds, with one temporary HDF4
file decoded at a time and no raster archive retained. Original headers, raw QA
zero, unit scaling, native-grid fractional overlap, paired support, source hashes
and explicit missingness are checked. These checks support a research artifact;
they do not constitute a reviewed global scientific release. The PostgreSQL pilot
import is implemented below; scientific publication remains subsequent work.

The frontend roadmap is exported in `catalog/variables.json`, with the complete
human-readable list in `docs/VARIABLES.md` and proposed TypeScript response types
in `catalog/types.ts`. Stable quantity IDs are distinct from statistic/operator
output IDs. Definitions expose proposed units, dimensional axes and availability;
all production-availability flags remain false until a database/API release is
loaded. Eight quantities have pilot statistics and aspect has component outputs;
the remaining candidates are planned or deferred. This contract is suitable for
building selectors, with categorical vocabularies and source-specific limits
added only after validation.

The initial numeric pilot subset is implemented in
`db/migrations/001_pilot_store.sql` and `scripts/load_pilots.py`. Docker Compose runs
PostgreSQL 18/PostGIS with a persistent volume. The loader verifies input hashes,
uses one transaction and a bundle identity for repeat imports, preserves source
manifests, seals series, and exposes explicitly named `research.pilot_*` views.
The imported domain contains 143,850 observation rows, including 13,639 explicit
missing values. All 91 candidate definitions are discoverable, while only the 15
realized numeric outputs have observation series. Full-cell geometry and clipped
region support are stored separately. A dedicated API role reads the research
views and has no write privileges. Measured storage is recorded in
`research/pilots/database/load.json`.

This migration is a bounded implementation of the design, not completion of the
full specification. Categorical/boolean observations, fully normalized provider
asset tables, production migration/ingestion role separation, immutable published
releases, wide release projections, arbitrary research SQL execution/auditing,
and the Rust HTTP API remain subsequent work. Until publication is implemented,
pilot views explicitly expose research artifacts and catalog production flags
remain false.

1. Database setup, roles, migrations, catalog, grid, and temporal constraints.
2. Processing lineage, typed observations, sealed series, and ingestion fixtures.
3. Immutable releases, research views, wide projection, and initial functions.
4. Terrain and temperature adapters for one bounded region and period.
5. Query execution controls, audit manifests, result storage, and benchmarks.
6. Expand coverage; add indexes/partitioning only from measured workload evidence.

Before real ingestion, select the pilot region, exact elevation/temperature
products and versions, analysis period, coverage thresholds, overlap/uncertainty
methods, and temporary-cache budget. Configure external storage only if result
exports or optional source archival require it. These are dataset configuration choices;
they do not postpone the core database implementation. Deployment sizing,
query/result limits, retention durations, and finer-grid promotion follow pilot
measurements.
