# syntax=docker/dockerfile:1
FROM node:24-slim AS demo
WORKDIR /demo
COPY demo/package.json demo/package-lock.json ./
RUN npm ci
COPY demo/ ./
RUN npm run build

FROM rust:1.97.1-slim-trixie AS builder
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler ca-certificates curl && rm -rf /var/lib/apt/lists/*
# Pin the same DuckDB Spatial build as the data-file format and Rust binding.
COPY db/fetch_spatial.sh /fetch_spatial.sh
RUN sh /fetch_spatial.sh /spatial.duckdb_extension
WORKDIR /app
COPY backend/Cargo.toml backend/Cargo.lock ./
RUN mkdir src && printf 'fn main() {}\n' > src/main.rs && cargo build --locked --release && rm -rf src
COPY backend/build.rs ./
COPY backend/proto ./proto
COPY backend/src ./src
RUN touch src/main.rs && cargo build --locked --release && cp target/release/earth-index-api /earth-index-api && find target/release/build -path '*/out/libduckdb.so' -exec cp {} /libduckdb.so \;

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /libduckdb.so /usr/local/lib/libduckdb.so
COPY --from=builder /spatial.duckdb_extension /usr/local/lib/duckdb/spatial.duckdb_extension
RUN ldconfig
COPY --from=builder /earth-index-api /usr/local/bin/earth-index-api
COPY --from=demo /demo/dist /app/demo
ENV DEMO_DIR=/app/demo DUCKDB_SPATIAL_EXTENSION=/usr/local/lib/duckdb/spatial.duckdb_extension
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["earth-index-api"]
