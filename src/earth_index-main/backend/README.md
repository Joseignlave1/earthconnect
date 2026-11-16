# Rust gRPC API

Tonic serves native gRPC and binary gRPC-Web through Axum on port 8080, using an embedded, file-backed **DuckDB 2.0** database. The Svelte/TypeScript demo is served at `/demo`; reflection and grpcui provide the protobuf API explorer.

```sh
docker compose up --build -d api grpcui
```

Open [the demo](http://localhost:8080/demo) or [grpcui](http://localhost:8081). See [database preparation](../db/README.md) if the serving file is absent. No PostgreSQL service is needed for normal serving.

## Protocol

The [protobuf](proto/earth_index.proto) exposes Query, Metadata, Health and SamplePoint. Query accepts only `datasource_id`, a SQL string and a named parameter map. There are no saved queries or implicit score/filter rules. Native clients may negotiate gzip; browsers use binary gRPC-Web, with raw BLOB bytes rather than base64 on the wire.

`QueryResponse` retains positional columns/rows, including duplicate column names. The legacy `pg_type` protobuf field now describes DuckDB types. Integers within signed int64 remain exact; larger integers and DECIMAL values return strings. Lists/structs and JSON results return recursive values; BLOB and GEOMETRY results return bytes (geometry is WKB). SQL NULL remains a distinct null value.

Browser responses also negotiate HTTP `Content-Encoding: gzip` at level 1 using
`Accept-Encoding`. Browsers decompress the HTTP body before protobuf decoding;
gRPC-Web message frames remain unchanged. Clients that decline gzip still get
the uncompressed response. Native gRPC retains its protocol-level compression.

## SQL and bindings

Use `:name` placeholders and map keys without the colon. Repeated names share a binding; quoted strings, identifiers, comments and dollar-quoted strings are preserved. Missing/unused parameters, positional placeholders, multiple statements and transaction/session commands are rejected.

Scalars bind as BOOLEAN/BIGINT/DOUBLE/VARCHAR/BLOB. Homogeneous lists bind as native SQL lists, including nested and empty lists. Objects and heterogeneous JSON lists bind as JSON; binary values cannot be nested inside JSON. Bytes such as polygon WKB stay binary, e.g. `ST_GeomFromWKB(:boundary)`. DuckDB Spatial is available; use its SQL dialect rather than PostgreSQL-specific casts/functions.

```sql
SELECT site_key,x,y,surface_mean_c,slope_mean_deg
FROM cells
WHERE site_key=:region AND x=:x AND y=:y;
```

`measurements` is the fast physical column store (`region_id`, zero-based `pixel`, 25 fields). `regions` supplies shapes, grid transforms and coverage geometry. `cells` adds site_key, 1-based x/y and lazily computed WGS84 cell geometry. Metadata documents columns, units, all per-region provenance, coverage polygons and grid addressing.

## Compact map output

The demo does all scoring in DOUBLE and explicitly quantizes the final value in SQL:

```sql
round_even(65534 * greatest(0.0, least(1.0, score)), 0)::INTEGER
```

`0..65534` are valid scores and `65535` means missing evidence. `pack_u16(INTEGER[])` writes an ordered list as little-endian uint16 bytes. `pack_u16_at(STRUCT(index UINTEGER, code INTEGER)[], UINTEGER)` scatters explicitly addressed values into a fixed-length matrix without sorting millions of rows. Omitted addresses/NULL codes become 65535; invalid codes, duplicate/out-of-range addresses and lengths above five million are rejected. Neither helper calculates a score.

Both packers read DuckDB’s flattened native vectors and cached NULL masks directly, reusing a small scratch buffer per callback. They do not build intermediate Arrow lists/structs. The final packed BLOB is copied into DuckDB’s output storage.

The builder groups by integer region_id, joins the 460 region records after aggregation and returns `(site_key, scores BLOB)`. Shapes/transforms come from Metadata. A full rectangular world result has 71,101,280 packed score bytes before framing/compression. See [the builder](../demo/src/query.ts).

## Execution and limits

The shared engine defaults to eight threads and an 8 GB memory limit, with at most two active requests. Each request gets a separate connection and read-only transaction. The file is mounted read-only. External file/network access, extension autoloading and configuration changes are disabled after loading the pinned Spatial extension.

No query deadline is imposed by default (`STATEMENT_TIMEOUT_MS=0`). Explicit caller `grpc-timeout` deadlines include queue wait and interrupt database work. Abandoning/cancelling the RPC also interrupts its connection. SamplePoint and Metadata retain separate finite budgets. Limits default to 64 MiB requests and 128 MiB responses; oversized results return RESOURCE_EXHAUSTED. SQL errors return INVALID_ARGUMENT with `error-code=SQL_ERROR`, without PostgreSQL SQLSTATEs.

| Environment | Default |
| --- | --- |
| `DUCKDB_PATH` | `db/runtime/earth.duckdb` |
| `DUCKDB_SPATIAL_EXTENSION` | `db/runtime/spatial.duckdb_extension` (packaged path in Docker) |
| `DUCKDB_THREADS` | `8` |
| `DUCKDB_MEMORY_LIMIT` | `8GB` |
| `API_LISTEN` | `0.0.0.0:8080` |
| `STATEMENT_TIMEOUT_MS` | `0` |
| `MAX_RESPONSE_BYTES` | `134217728` |

Axum also exposes `/healthz`, `/docs` (grpcui redirect), `/proto/earth_index.proto` and `/proto/descriptor.bin`. Reflection v1/v1alpha, binary gRPC-Web CORS and native gzip remain supported.

## Development and verification

The pinned duckdb-rs binding downloads the matching official native library. `cargo run` and `cargo test` set the loader path; running the built binary directly requires libduckdb on the system library path. Docker packages it automatically.

```sh
cargo test --manifest-path backend/Cargo.toml
cargo test --manifest-path backend/Cargo.toml --test api --test grpc_web -- --ignored --test-threads=1
cargo clippy --manifest-path backend/Cargo.toml --all-targets -- -D warnings
```

Integration tests use the prepared snapshot; see [database operations](../db/README.md) for deployed fixture/global validation. [Example grpcui Query JSON](examples/query.json) uses the global datasource.

## Moon/Mars point measurements

`SamplePoint` accepts only a body, east-positive longitude [-180,180] and
latitude [-90,90]. Moon uses the IAU 2000 sphere; Mars uses planetocentric
latitude. It needs no NASA token and does not query or import planetary data
into the serving database. The RPC uses the same native gRPC and binary gRPC-Web service,
and appears automatically in grpcui/reflection.

```sh
grpcurl -plaintext -d @ localhost:8080 earth_index.v1.EarthIndex/SamplePoint < backend/examples/sample-point.json
cargo run --manifest-path backend/Cargo.toml --example sample_point -- mars 23.47 0.67
```

The [audited source registry](src/planetary/sources.json) enables 51 lunar and
28 Martian source products. It includes terrain elevation, lunar Diviner surface
temperatures, polar illumination and shadow, geology, mapped ice/mineral and
geomorphological features, mineral models, Mars albedo/thermal inertia/dust and
hydrogen proxies. Some products are regional; several provide alternative
measurements of the same concept. This is not 79 independent physical variables.

Each response has a measurement per configured source for the selected body,
plus a temperature range derived from available matched minimum/maximum maps
local slope/relief estimates from a complete 3×3 native DEM stencil, and
kilometre-scale terrain descriptors from a 21×21 grid at 1 km spacing.
Each measurement includes `variable_id`, `source_id`, `source_url`, label,
`value`, unit, native service pixel spacing/unit, quality flags, notes and status:

| Status | Meaning |
| --- | --- |
| AVAILABLE | Numeric sample or feature attribute records returned |
| NO_DATA | Missing/masked numeric value or no mapped feature at the exact point |
| OUTSIDE_COVERAGE | Outside the source's rectangular footprint |
| SOURCE_ERROR | HTTP/provider/decoding error or response too large |
| SOURCE_TIMEOUT | Source request or total sampling budget expired |

`value` uses the existing recursive Protobuf `Value`. Numeric samples are doubles
in source units; feature samples are lists of source attribute objects, limited
to 16 matches with an explicit truncation flag. No-match results are not evidence
of absence. Invalid/missing samples have no value; zero remains valid.

`partial` indicates source failures, timeouts or truncated feature matches.
Unknown units or scientific normalization are not guessed: those channels have
unit `raw` and `semantic_review_required` / `raw_source_scale` quality flags.
API access alone does not establish suitability for matching against Earth.
The [source audit](../research/planetary-sources/README.md) documents grid-spacing
mismatches, modeled versus observed temperatures and remaining calibration limits.
The rejected Mars temperature-range layer is never requested; a valid difference
is derived without a Celsius offset. Local Horn slope and max-minus-min relief
use planet-specific distances and retain the center elevation by sample location
ID. An incomplete/polar stencil leaves the local estimates missing. These are
not 4 km cell-mean slope or 5/21 km relief. A separate bounded window operation
now samples 441 heights at 1 km spacing, returning `slope_1km`, `relief_5km`,
`relief_21km`, `tpi_5km` and `tpi_21km`. The spherical tangent-plane grid works
at the poles and date line; relief/TPI require a valid center and >=90% window
support. Native point samples still differ from area-aggregated Earth cells,
identified by `point_samples_not_cell_averages`. Incomplete windows are flagged
and make the response partial. Trek rejects long GET URIs and POSTs, so the
window uses batches of at most 48 points, up to three active batches, sharing
the existing global eight-request semaphore and total 20-second deadline.
No full raster or additional credentials are needed.

The sampler has a 20-second total budget, 5-second upstream HTTP timeout, 6 active
source tasks per point and 8 HTTP requests globally. It cancels unfinished work
and returns completed values with timeout statuses. Caller `grpc-timeout` can
shorten the total budget. A deadline that expires while waiting for an identical
in-flight request uses gRPC DEADLINE_EXCEEDED. Invalid bodies/coordinates use
INVALID_ARGUMENT; a fully occupied point cache can return RESOURCE_EXHAUSTED.

An in-memory LRU-bounded cache holds up to 128 exact normalized points. Identical
in-flight requests share one computation; successful responses cache for six
hours and partial responses for 15 seconds. `cached` identifies reuse, and
`retrieved_at` preserves the source-retrieval time rather than claiming a current
observation. Coordinates are not rounded; equivalent antimeridian/signed-zero
and exact-pole coordinates are canonicalized. Cache contents disappear on restart.
Responses are capped at 256 KiB per upstream source; no full rasters or rendered
imagery are fetched. Only registry URLs are used; the client cannot supply a URL.
ASU responses sometimes omit a TLS close notification; the adapter stops at the
validated complete one-pixel VICAR boundary while retaining certificate validation.

Regenerate the registry from the reviewed research snapshots with
`python scripts/build_planetary_sampler_sources.py`. This requires no network.
Backend tests use local mock providers to cover numeric VICAR formats, CORS/gRPC-Web,
partial failures, time budgets, cancellation, cache coalescing, valid zero versus
NODATA, feature attributes and response-size limits. They do not depend on NASA
or ASU uptime.
