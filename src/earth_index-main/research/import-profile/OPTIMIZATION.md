# Reusable overlap weights and tile workers

`scripts/thermal_import.py` is the optimized thermal processor. It is separate from the frozen original processors so existing dataset fingerprints, files, and database revisions remain valid. `scripts/raster_overlap.py` provides the reusable aggregation primitive; `scripts/raster_source_cache.py` supplies deduplicated, hash-checked source downloads with pooled HTTPS connections.

Each CPU task owns one output tile and its complete annual series. It builds exact pixel-to-cell area weights once for each intersecting source grid, applies changing day/night/paired quality masks through compiled NumPy reductions, and combines source-tile pieces by accepted area before the original annual summaries. Arrays, QC rules, support thresholds, physical units and quantization conventions are preserved. Cartesian and spherical weights are obtained directly from exactextract; they are not a substitute resampling method.

## Run

The same command is available through the optional Compose `importer` service; it does not start or modify PostgreSQL:

```sh
docker compose run --build --rm importer \
  --sites atacama haughton deathvalley \
  --cell-m 4000 --side-km 100 --workers 8 --download-workers 8 \
  --output research/thermal-optimized-4km
```

The container writes project outputs only under the mounted `research/` directory. It defaults to container UID/GID 0, which maps to the host account with this machine's rootless Docker. On rootful Docker, set `IMPORT_UID`/`IMPORT_GID` to the desired host owner.

Generate a separate 4 km thermal pilot for selected existing footprints:

```sh
uv run scripts/thermal_import.py \
  --sites atacama haughton deathvalley \
  --cell-m 4000 --side-km 100 \
  --workers 8 --download-workers 8 \
  --output research/thermal-optimized-4km
```

The default 100 km footprint produces 25×25 four-kilometre cells. `--side-km 400` produces 100×100 four-kilometre cells and inventories the larger region, including intersecting native tile pieces. `--base-manifest` can select a different site manifest; `--all-sites` explicitly selects its entire inventory. No import is performed just by running tests or benchmarks.

For the current 1 km, 100 km pilot grid, the new processor can reuse matching base layers and emit the full 25-band manifest:

```sh
uv run scripts/thermal_import.py \
  --sites atacama --cell-m 1000 --side-km 100 \
  --workers 8 --download-workers 8 \
  --output research/thermal-optimized-1km
```

Four-kilometre output currently contains the nine thermal/support layers only. The other source families and existing database are not downscaled by this command. The legacy DB raster loader/schema still assumes the old grid; do not load new-grid output through that path until the separate grid migration is implemented. No database writes occur here.

`--verify` compares every optimized intermediate mean and accepted area with reference extraction. It is deliberately expensive and should be used on validation samples, not for maximum import throughput.

## Workers, memory and downloads

- NumPy reductions here are single-threaded; native BLAS/OpenMP pools are restricted to one thread per worker before imports. Eight processes can therefore process eight tiles on separate cores without Python's GIL serializing them.
- `--workers` controls CPU processes. `--download-workers` separately controls concurrent HTTP downloads. Download and CPU phases currently run sequentially; the two settings do not imply sixteen simultaneous CPU workers.
- Target geometry and an LRU cache of overlap plans remain local to a tile worker. Plans are keyed by native window shape/transform, support partial native-tile intersections, and have a 128 MiB cache budget per process. Oversized individual plans fail with guidance to reduce output tile size.
- Full native COG assets are deduplicated by unsigned URL across selected regions, downloaded once, and validated using SHA-256 when reused. Only the four required thermal bands are fetched. Temporary source files are shared for reading by workers.
- By default sources use the output's `.source-cache` directory and are deleted after successful completion. Interrupted runs retain them for resume. `--keep-source-cache` retains them after success. An explicit `--cache-dir` is a persistent shared cache and is always retained.
- Peak disk use can include all unique source assets selected by the run: this is not a streaming cache with a global disk quota. Select bounded groups of regions if necessary. Credentials/SAS signatures remain in memory and are not written into source metadata.
- Output files and checkpoints are written atomically. Re-running the same command reuses valid completed tiles; changed processor/dependency hashes, base manifest, grid or selected sites require a new output directory.

## Verification and measurement

```sh
uv run scripts/test_raster_overlap.py
uv run scripts/test_thermal_import.py
uv run scripts/test_thermal_pilot.py
uv run scripts/benchmark_overlap_import.py
```

Tests cover irregular/holey polygons, spherical areas, source boundaries, changing missingness, good QC zero, paired support, annual thresholds, U16 output equivalence, new-grid GeoTIFF transforms, corrupted cache recovery and checkpoint validation. The real-data benchmark additionally checks two seasonal dates each in Atacama, Amazon and Haughton, a complete available 44-composite year, and identical outputs across worker counts.

`optimized.json` records performance and scientific checks. CPU scaling uses sixteen distinct **overlapping** 400×400 km windows inside one native MODIS tile. It stresses independent tile processing and scheduling with cached real inputs; it does not represent sixteen independent geographic areas or sixteen source tile-years downloaded. Timings include geometry/weight construction, local decoding and process-pool startup. Reported average busy cores are summed task CPU seconds divided by wall seconds, excluding interpreter startup outside tasks.

The optimization is wired into the new thermal import command. The frozen terrain/land-cover/climate processors are unchanged; this is not yet the full multi-variable 4 km expansion/migration.

## Measured results, 2026-11-15

Same single-worker, cached-input 400×400 km region, 10,000 four-kilometre cells, 44 real 2023 composites:

| Processing | Elapsed |
| --- | ---: |
| Reference repeated spatial extraction | 48.03 s |
| Reusable overlap weights, including initial build | 2.74 s |

That is **17.6× faster** for this thermal CPU workload. Every code in all nine annual U16 layers is identical to the reference. Real seasonal intermediate means and accepted areas also match in the Atacama, Amazon and Haughton samples. This is a CPU speedup, not a 17.6× end-to-end import speedup: downloads now make up more of the remaining work.

Sixteen output windows × 44 composites, two trials in opposite worker-count orders:

| CPU workers | Mean elapsed | Average busy cores |
| ---: | ---: | ---: |
| 1 | 43.77 s | 0.98 |
| 4 | 13.85 s | 3.81 |
| 8 | 10.67 s | 6.86 |
| 16 | 10.00 s | 11.46 |

Eight is the default to leave useful headroom. Sixteen is supported and averaged ~6% less wall time here, while using considerably more CPU; individual trials did not consistently favor the same count. Worker count is not a guarantee of full utilization during startup, I/O or the final few tasks. Enough pending output tiles are needed to occupy all workers. The test uses overlapping real-data windows, so it does not establish sustained global throughput.

CLI smoke runs additionally generated the full 25-band Atacama 1 km bundle and a separate nine-band 4 km bundle. All 25 current-grid code hashes match the existing Atacama datasource. The 4 km files are 25×25 pixels with a 4,000 m pixel transform. Re-running the completed current-grid command reused its checkpoint without downloading or recomputing tiles. These are separate research outputs; the active PostgreSQL datasource was not modified.

Machine-readable repeated measurements are in `optimized.json`; `optimized-first-pass.json` preserves the preliminary single-trial run. The final report fingerprints the implemented processing dependencies. Source rasters are temporary benchmark inputs, not part of the dataset artifacts.

## Worldwide runtime projection

`scripts/estimate_world_import.py` inventories two global 2023 thermal dates and HEADs the four required assets, reusing previously measured sizes. Each date has 317 source tiles / 1,268 files. Payloads are 633.32 MB in January and 622.44 MB in July. Scaling their mean to 44 composites gives **27.63 GB / 55,792 files** for one available thermal year. This is a projection, not a full annual inventory. The complete 2024 MCD12Q1 CMR inventory has 315 granules totaling **1.52 GB**; all report sizes. Terrain globals add **1.20 GB**. Main source payload is therefore about **30.4 GB**, plus small/coarse ancillary inputs. `world-volume.json` preserves the unsigned source-size evidence.

Using 149 million km² as an approximate land-area assumption gives 9.31 million nominal 4 km cells. Scaling optimized eight-worker throughput by area alone gives ~10 minutes thermal CPU. Allow **15–45 minutes** for source-boundary pieces, coastal padding and geographic variation. The old terrain processor still needs its 1 km intermediate grid to preserve physical 5/21 km descriptors: scaling the measured ~1.64 single-core CPU seconds per 100 km patch gives ~6.8 core-hours before parallelism. Thermal speedup cannot be applied to those unchanged processors.

**Budget 4–8 hours runtime for a worldwide land snapshot with all current 25 bands, after the global grid and remaining importers are implemented.** Indicative phase allowances: downloads 1–2 hours, thermal CPU 15–45 minutes, other-variable preparation 1–3 hours, database loading/validation 15–45 minutes, with margin for inventory/retry/padding overhead. The measured download rate implies ~60 minutes for the main source payload; sustained global network/service throughput and database loading have not been tested. The lower-level ideal scales are not completion promises.

This means available evidence worldwide with source gaps marked NODATA. It does not supply ocean observations or fill missing polar terrain/thermal variables. Additional sources for those cases require a separate estimate. The current command supports manifest-defined regions; it is not yet an end-to-end global multi-variable runner. `world-eta.json` records assumptions, measured inputs and planning ranges; implementation time is excluded.
