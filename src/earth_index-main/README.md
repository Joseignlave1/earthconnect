# Earth Index

An Earth–Moon–Mars environmental analogue search demo with a **file-backed DuckDB database**, **Rust/Axum/Tonic API**, and **Svelte + TypeScript maps**.

The loaded global datasource `earth-ed5bc9904a5f` has 25 fields across 460 native sinusoidal regions: 31,881,940 Earth-domain cells at nominal **4 × 4 km** spacing. Continuous measurements use **DOUBLE**, IDs/categories use integers, and map scores are returned as packed **uint16** arrays. Source gaps remain missing. The serving snapshot is about 763 MB.

```sh
docker compose up --build -d api grpcui
```

Open [the demo](http://localhost:8080/demo) or [grpcui](http://localhost:8081). The demo starts with global coverage. Select a point on Moon or Mars to load available NASA measurements, inspect/edit the generated SQL and parameters, and score Earth. One cached GPU score layer, distant highlights and the score distribution help navigate candidates.

The API is a read-only SQL driver over native gRPC and binary gRPC-Web, with recursive named parameters, metadata, health, Moon/Mars sampling and reflection. Queries are not saved. The normal Compose stack needs no PostgreSQL service. Prepare the serving file first if it is not already present; it is excluded from Git and Docker build context.

- [Database specification](DESIGN.md)
- [Database preparation and operations](db/README.md)
- [API and protobuf contract](backend/README.md)
- [Frontend development](demo/README.md)
- [Fly.io deployment and dataset transfer](docs/DEPLOYMENT.md)
- [Scientific import methods and limitations](db/WORLD_IMPORT.md)
- [Data sources, citations and attribution](docs/DATA_SOURCES.md)
- [Preserved PostgreSQL ingestion and pilot tools](db/POSTGIS.md)
- [Historical PostGIS raster design](docs/POSTGIS_RASTER_DESIGN.md)

The serving file currently decodes the validated quantized import, retaining its original source precision. Switching its storage to DOUBLE does not recover precision discarded in that earlier import. The original rasters, research pilots, processing checkpoints and PostgreSQL volume remain intact.

## License

Original project source code is licensed under the [Apache License, Version 2.0](LICENSE.txt).
Third-party components and datasets retain their own licenses and attribution requirements;
see [Data sources, citations and attribution](docs/DATA_SOURCES.md).
