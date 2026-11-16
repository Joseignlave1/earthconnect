# Historical PostgreSQL ingestion and pilot operations

API serving now uses [DuckDB](README.md). The following documents the preserved scientific import store and legacy tools.

# PostgreSQL/PostGIS database

The database runs in Docker Compose with a digest-pinned PostgreSQL 18/PostGIS
image. It contains the global 25-band dataset and the original research pilots.
The application API uses Rust/Axum; the ingestion loader remains Python to work
with the scientific pipeline.

The selected global design uses **4 × 4 km cells and 16-bit quantized
measurements**. The global nominal 4 km grid is loaded; the exact spacing,
source limitations and resumable loader are described below. The original
1 km pilots use separate immutable bundle identities.

## Global 4 km dataset

Published datasource: **`earth-ed5bc9904a5f`**. SQL namespace:
`raster_ed5bc9904a5f7211e52e3ac5b21cd145`.

- 460 native regions, 25 feature bands, 11,500 single-band `16BUI` rasters.
- 31,881,940 Earth-domain cells at 3,999.822 m equal-area spacing (SRID 930001).
- Stored rectangular matrices include 3,668,700 padding cells outside Earth;
  their measurements and `cells.geom` are NULL. Source gaps also remain NULL.
- All 888,766,000 U16 codes matched the processed files byte-for-byte after import.
- 531,941,072 nonmissing stored feature values, including coverage/quality bands.
- About 284 MB of raster datums; about 409 MB for the entire database including
  the original bundles. Cluster WAL and container images are separate.

The global grid is a new dataset/version. Its POWER climatology is 2001–2020,
and annual precipitation uses 1 mm quantization to accommodate wet regions.
See [scientific methods](WORLD_IMPORT.md). Each layer exposes its actual scale,
offset, units, valid count, quantization error and source hashes.

The load commits one complete 25-band region at a time and validates the standard
PostGIS raster WKB without expanding values into floating-point SQL rows.
Restarting verifies/reuses committed regions; publication happens only after all
regions pass. The existing bundles' counts and hashes are checked unchanged.
The completed store is vacuum-frozen/analyzed. To reproduce or resume:

```sh
docker compose run --rm \
  -v "$PWD/research/world-4km:/workspace/research/world-4km" \
  --entrypoint python loader scripts/load_world.py --publish
```

Load/validation reports are in `research/world-4km/database.json`,
`database-file-validation.json`, `api-validation.json` and
`native-grpc-validation.json`. API validation compares six frozen targets across
three native regions with independent NumPy score buffers, and checks padding,
pole/map-edge geometry and all coverage polygons.

Use the datasource ID in `Query`; normal SQL and named parameters still apply:

```sql
SELECT site_key,x,y,elevation_median_m,surface_mean_c
FROM cells
WHERE site_key = :region AND x = :x AND y = :y;
```

For direct raster access without decoding all feature columns:

```sql
SELECT ST_Value(rast,1,:x,:y)*scale+offset_value AS value,unit
FROM tiles
WHERE site_key = :region AND feature_key = :feature;
```

`Metadata` supplies the 460 shared-grid definitions and precomputed geographic
coverage, clipped to Earth instead of transforming invalid native rectangles.
The demo retains a bounded pilot as its initial selection; global data remains
selectable. Full-world scoring optimization is separate from this data load.

The database contains both the original H3 observation pilot and the preferred
[16-bit quantized PostGIS raster pilot](../DESIGN.md). The
[priority pilot instructions](../research/priority-pilot/README.md) cover its
16 aligned bands across Atacama, Haughton and Death Valley. The
[historical specification](../docs/H3_PILOT_DESIGN.md) records the H3 layout.

The [27-area stress bundle](../research/stress-pilot/README.md) uses the same
raster tables and views. Load it with `scripts/load_raster_pilot.py --pilot-dir
research/stress-pilot` and an explicit report path. Bundle selection separates
the three-area baseline from the expanded geographic sample.

## Start and load

Requirements: a running Docker engine and Docker Compose. Rootless Docker works.

```sh
python3 scripts/setup_local_env.py
docker compose up -d --wait db
docker compose run --build --rm loader
docker compose run --rm --entrypoint python loader scripts/test_postgres.py
```

The setup helper generates private random passwords in `.env` with mode 0600 and
preserves an existing file. Never commit that file. To use a different host port,
adjust `POSTGRES_PORT` and the two host connection strings together.

`db` publishes only `127.0.0.1:5432`. The loader connects to `db:5432` on the
Compose network. PostgreSQL data persists in `earth-index_postgres_data`, mounted
at `/var/lib/postgresql`, as required by the
[PostgreSQL 18 PostGIS image](https://github.com/postgis/docker-postgis#%EF%B8%8F-pgdata-volume-path-change).
The host source directory is read-only inside the loader; only its `/reports`
mount is writable. The raster importer loads derived 16-bit tiles; original
provider raster archives are not stored in PostgreSQL.

```sh
docker compose ps
docker compose stop db
docker compose start db
docker compose down
```

Stopping containers or running `down` preserves the named volume. `down -v`
deletes the database volume; use it only when deliberately discarding local data.
The `.env` passwords initialize an empty database. Editing the file does not
rotate passwords in an existing volume.

## Contents and size

- 91 candidate quantity definitions; 15 realized numeric output definitions.
- 959 full H3 resolution-6 cells and their clipped supports in three pilot regions.
- 45 series: 12 terrain outputs × 3 regions, plus 3 temperature outputs × 3 regions.
- 47 distinct time windows: one static window and 46 MOD11A2 composites in 2024.
- 143,850 observation rows: 130,211 numeric values and 13,639 SQL NULL values with
  explicit missing reasons. Zero is preserved as a valid value.
- Source manifests, original source URLs and hashes, QA semantics, units,
  grid/operator versions, support and coverage fractions.

The source terrain and temperature artifacts occupy approximately 12.1 MB,
including 10.65 MB of CSVs. The initial database measured approximately 51 MB;
project tables including indexes and TOAST occupied 31 MB. Database size includes
PostgreSQL/PostGIS catalogs but excludes cluster WAL, other cluster databases,
Docker images, build caches and the source CSV files. These figures describe only
the three pilot regions. The [load report](../research/pilots/database/load.json)
records byte counts and table sizes; normal database operations can change them.
The local PostgreSQL volume measured about 195 MB including cluster-wide WAL,
system databases and other cluster files. This is separate from both the 51 MB
database measurement and the Docker image/build storage. A separate
[storage report](../research/pilots/database/storage.json) records these scopes.

## Import guarantees and scope

`scripts/load_pilots.py` verifies scientific code, inventory, geometry and CSV
hashes before writing. A transaction-scoped advisory lock serializes imports and
migrations. Migrations are ordered and checksum-checked. One transaction creates
the catalog, support geometry, provenance, series and observations. Failure
rolls back all project changes. A repeated bundle hash reuses the existing import
and checks its row counts instead of duplicating observations.

Numeric values must be finite, valid fractions must be in [0,1], missing rows
must have a reason and zero valid support, and usable values must have positive
support. Foreign keys enforce series/grid/region/cell identity. Static and
interval windows are distinct. Sealed series and observations reject mutation;
versioned metadata is append-only. The owner remains a development administrator,
not a production ingestion role.

Some raw HDF4 header strings contain U+0000, which PostgreSQL JSONB rejects.
Those strings are stored reversibly as a tagged object with `code_points` and
`earth_index_encoding`. Original manifests remain unchanged in the repository and
their SHA-256 hashes are stored. Do not treat encoded header strings as numeric
quality values. Scientific raster decoding is unchanged.

Integration checks compare every stored value, coverage fraction and unit against
the source CSVs, verify missingness and boundary/time semantics, reject invalid
values and mutation of sealed records, and exercise the API login permissions.
Fixture writes are rolled back. The first actual failed load also verified that
no observation table survived the transaction rollback.

The original H3 migration stores numeric observations only. Categorical/boolean
observations, normalized source-variable/asset relations, published releases,
wide release projections, arbitrary SQL execution controls and query audit
records were proposed in the historical pilot design. The current target
physical layout is instead specified in [DESIGN.md](../DESIGN.md). Candidate variables do not
imply globally available observations or a production scientific release.

## Rust API integration

Use `.env`'s `API_DATABASE_URL` for the Rust service. `DATABASE_URL` is the local
owner connection for administration/ingestion; if the Rust database driver
expects `DATABASE_URL`, supply the API URL as that service's environment value.
Do not pass the owner credentials to the application. See the
[backend guide](../backend/README.md) for the implemented gRPC/gRPC-Web API and grpcui explorer.

`db/migrations/` contains checksum-verified migrations. `db/migrate.py` applies
them and refreshes API datasource views without importing data. Compose runs it
before the API starts. After a new raster import, run `docker compose run --rm
migrate` to expose that bundle through the API. Existing 001–003 migration contents
and recorded checksums are preserved.

The `earth_index_api` login inherits SELECT access to research views, defaults to
read-only transactions and has a 10-second statement timeout. It has no access to
underlying project schemas or writes. These permissions protect the local pilot;
the API additionally uses explicit read-only transactions and always rolls them back.

Available views:

| View | Purpose |
| --- | --- |
| `research.variables` | The 91 candidate definitions plus two auxiliary coverage/permafrost definitions used by the raster pilot |
| `research.pilot_features` | Realized output IDs, units, definitions and series |
| `research.pilot_observations` | Sealed numeric pilot values, missingness, periods, fractions and provenance IDs |
| `research.cells` | Full H3 geometry, center and WGS84 ellipsoidal area |
| `research.cell_supports` | Region-clipped support geometry and area metadata |
| `research.raster_layers` | Bundle/layer identity, units, scale/offset, grid, source provenance and quality rules |
| `research.raster_tiles` | Raster tiles joined to decoding metadata |
| `research.raster_values` | Decoded nonmissing values with aligned pixel x/y coordinates |
| `research.priority_pilot_cells` | Wide on-demand columns for the 16 priority bands; filter by bundle and site |

For pilot ranking, use the [optimized SQL path](../research/ranking-pilot/OPTIMIZATION.md)
instead of expanding `research.priority_pilot_cells`. It reads only the required
aligned bands from the existing raster views, preserving the same per-pixel
scoring and missing-data rules. Retained parameterized queries are in
`research/pilots/database/fast-ranking-artifacts/`; replace the single psycopg
`%s` bundle placeholder with Rust/PostgreSQL `$1`. This path assumes the pilot's
aligned 100 × 100 single-tile grids. No migration or extra permission is needed.

Use parameterized SQL. For example, a Rust handler can bind a region key, output
ID and time range to the following query, then map the SQL names to the frontend
[observation contract](../catalog/types.ts):

```sql
SELECT variable_id, output_id, region_id_external, cell_id,
       value_numeric, unit, missing_reason, valid_area_fraction,
       time_kind, time_start, time_end_exclusive, run_id, bundle_sha256
FROM research.pilot_observations
WHERE region_id_external = $1
  AND output_id = $2
  AND time_start >= $3::timestamptz
  AND time_start < $4::timestamptz
ORDER BY time_start, cell_id;
```

For static terrain, use `time_kind = 'static'` instead of inventing timestamps.
The temperature difference is paired-pixel composite day minus night, not a
full diurnal amplitude. Kelvin differences must not have 273.15 subtracted.
Boundary cells summarize clipped geometry; map the support geometry from
`research.cell_supports` when presenting coverage. Do not label clipped coverage
as full-cell coverage.

For local administration without installing `psql` on the host:

```sh
docker compose exec db psql -U earth_index_owner -d earth_index
```

The container's local administrative socket is intentionally for development.
Production deployment still needs separate migration/ingestion identities,
connection encryption, backups, restore testing and release publication checks.

On this workstation, Docker was installed using the official
[rootless installation](https://docs.docker.com/engine/security/rootless/), with
the `rootless` context selected. Its user service can be controlled with
`systemctl --user start docker` or `systemctl --user stop docker`.
