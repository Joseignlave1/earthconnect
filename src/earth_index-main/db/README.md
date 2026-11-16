# DuckDB serving database

`db/runtime/earth.duckdb` is the immutable, file-backed serving snapshot. The Rust API embeds DuckDB 2.0 and needs no PostgreSQL service. Continuous columns are physical DOUBLE; IDs and lithology codes are integers. The global snapshot is about 763 MB and includes all 25 fields, 460 grids, provenance and coverage metadata.

```sh
docker compose up --build -d api grpcui
```

Compose mounts the database file read-only and refuses to create an empty file if it is missing. Set `DUCKDB_PATH` to use another prepared snapshot. `DUCKDB_THREADS=8` and `DUCKDB_MEMORY_LIMIT=8GB` control the shared engine. The pinned Spatial extension is packaged in the API image. The current image targets Linux amd64.

## Rebuilding the serving snapshot

The current input is the validated physical-value benchmark file `research/world-4km/duckdb-double/world-double.duckdb`. Before replacing a PostgreSQL-backed API for the first time, export its global metadata:

```sh
cd demo
node --import tsx scripts/export-duckdb-metadata.ts
cd ..
```

This writes `db/runtime/source-metadata.pb` and `.json`. Retain both with the dataset backup. The exporter is a migration tool; do not re-export from the new API, whose advertised relations describe DuckDB. Subsequent rebuilds use these frozen files and require no running PostgreSQL service.

For a host build, fetch the pinned, checksum-verified Spatial extension, then:

```sh
sh db/fetch_spatial.sh
uv run db/prepare_duckdb.py
```

The preparer writes a staging file, verifies every field and NULL against the source, checkpoints, and atomically replaces the serving snapshot. It validates the scientific revision and coverage/grid counts. An interrupted build leaves the previous snapshot usable; restart the preparer to rebuild the staging file. Restart the API after replacing the file.

The current source has already been quantized during ingestion: converting it to DOUBLE preserves its import precision. No NASA data needs to be downloaded again for this conversion. Runtime files/native binaries are ignored by Git and excluded from Docker's build context.

## Validation

```sh
cargo test --manifest-path backend/Cargo.toml
cargo test --manifest-path backend/Cargo.toml --test api --test grpc_web -- --ignored --test-threads=1
cd demo
npm test
MATCHING_API=http://localhost:8080 node --import tsx --test tests/matching.integration.test.ts
node --import tsx scripts/verify-world-api.ts
```

The deployed verification checks six targets on three regions byte-for-byte and full-world score hashes against the original PostgreSQL reference. The original scientific import and PostgreSQL volume are preserved; start legacy services explicitly with `docker compose --profile legacy up -d db` when needed. See [legacy ingestion operations](POSTGIS.md) and [scientific methods](WORLD_IMPORT.md).
