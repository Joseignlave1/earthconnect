# Import measurements

The subsequent [overlap-weight optimization](OPTIMIZATION.md) implements source caching and tile workers and reports the new CPU measurements. The timings below describe the original aggregation path; transfer-volume estimates still apply.

Measured on this computer on 2026-11-15: Intel Core Ultra 9 285H, 16 available CPU threads/core IDs, approximately 31 GiB RAM. Existing scientific processors and database contents were not changed. These are source/import benchmarks, not an implemented expansion importer or an end-to-end deployment test.

## Download and CPU concurrency

Same 24 thermal COGs (15,172,409 bytes), two download trials in opposite worker-count orders. Each trial fetches the payload again; files and decoded values must remain identical. Provider/CDN caches cannot be flushed.

| Workers | Download seconds, average of two trials | Cached CPU seconds, 36 aggregation jobs |
| ---: | ---: | ---: |
| 1 | 10.80 | 37.86 |
| 4 | 4.42 | 13.49 |
| 8 | 3.28 | 10.50 |
| 16 | 4.15 | 11.73 |

Eight workers won both download trials and the CPU trial. Relative to one worker, eight achieved 3.30× download throughput and 3.61× processing throughput. Sixteen workers increased overhead/contention. This is not evidence of a universal provider concurrency limit. CPU timings include process startup and per-worker target geometry construction; larger persistent batches may behave differently. Downloading and processing were measured separately, not as overlapping pools.

The CPU work decodes native inputs, applies unchanged QC, performs exact fractional-area aggregation of day/night/paired temperature, and quantizes results. All output/support hashes match at every worker count. Each job has 10,000 target cells at 4 km, spanning 400×400 km, but the six concurrency samples include single source-tile intersections rather than complete mosaics of those regions.

Serial remote COG window reads of the same six four-band groups:

| Target region | Payload | HTTP requests | Elapsed | Process CPU |
| --- | ---: | ---: | ---: | ---: |
| 100×100 km, 1 km cells | 3.86 MB | 80 | 23.13 s | 1.77 s |
| 400×400 km, 4 km cells | 7.41 MB | 88 | 23.91 s | 1.95 s |

The 1 km test spent 21.54 seconds inside HTTP requests. Latency, not simply bytes, dominates this access pattern. Full COGs cost more bytes for a small window, but need only 24 GETs: 10.80 seconds even at one worker, or 3.28 seconds at eight. The comparison combines request consolidation with concurrency; these are not equally concurrent window/full-file implementations. A concurrent/prefetched range reader could also improve the window path. Recorded byte counts are HTTP payload, excluding headers/TLS. Some GDAL requests overlap, so summed HTTP durations must not be interpreted as an exact wall-time fraction.

## Full available thermal year at 4 km

One 400×400 km region centered within MODIS h11v11, with all target polygons verified inside the source tile:

- 44 available 2023 eight-day composites × four bands = 176 files, **107.45 MB**.
- Eight concurrent downloads: **12.71 s**, approximately 8.46 MB/s.
- Eight CPU workers: **11.07 s**; summed worker CPU 79.53 s.
- Annual support checks, summaries and nine U16 layers quantized in memory: **0.17 s**.
- Total sequential phases: **23.94 s**. No DB load is included.

This uses the frozen processor's QC byte exactly zero, area-overlap weighting, >=50% support per usable composite, >=12 composites, four-quarter support, duration weighting, day-high/night-low definitions, and reserved-zero U16 quantization. It does not change the scientific method. The mirror has 44 observed dates here, not a complete 46-date annual series. These remain clear-sky composite statistics, not all-weather daily extremes.

## Variables and actual inputs

The active datasource's 25 bands require six source families, not 25 independent raster downloads.

| Inputs | Outputs | Measured cost / batching opportunity |
| --- | --- | --- |
| MOD11A2 LST day/night + QC day/night | Six surface-temperature values + three support fractions | Above. Cache by tile/date/band; download each once, share across every intersecting target region. |
| EarthEnv/GMTED elevation + slope | Elevation, slope, 5/21 km relief and TPI, terrain support (seven bands) | Three 120×120 km halo windows: 6.10–8.33 s each, 7.46–14.08 MB, 14–15 requests; 1.57–1.70 s CPU. Aggregation ~0.84–0.88 s, contextual calculations ~0.05–0.08 s. These tests use the current 1 km intermediate terrain grid. |
| MCD12Q1 annual HDF, read only LC_Type1 + QC | Bare fraction, permanent ice/snow fraction, support | Authenticated Atacama file: 4.56 MB in 3.89 s; decode 0.095 s, two aggregates 0.581 s, geometry 0.339 s. This is one file/window, not a NASA concurrency test. Deduplicate annual granules before fetching. |
| NASA POWER T2M + PRECTOTCORR | Coldest/warmest monthly mean air temperature, annual precipitation | One regional request per parameter: 5.9 KB each, 0.92 / 0.67 s. Small payloads; request overhead dominates. Reuse native climate supports across target cells. |
| ESA CCI permafrost | Permafrost fraction + support | Entire northern-hemisphere 2023 source: 12.45 MB, 5.72 s. Download once. |
| GLiM coarse lithology | Dominant lithology | Entire ZIP: 38.7 KB, 1.47 s. Download once. |

Terrain TIFFs have **one full-longitude row per storage block**, not compact spatial tiles. Narrow windows still transfer complete compressed rows. Global elevation is 397.23 MB and slope 800.82 MB (HEAD sizes, not a full-download speed test). For corridors spanning most latitudes, fetch these two globals once (~1.20 GB) or maintain a shared row/range cache. Compute 5/21 km descriptors on the 1 km intermediate grid **before** 4 km aggregation, preserving physical neighborhood sizes. The source latitude domain is -56° to 84°; unsupported regions must remain missing.

Fresh land-cover source bytes/hash and decoded window match the existing import. Fresh Atacama terrain pixel hashes also match the existing source manifests. The 27-patch pilot previously made 6,200 thermal band reads of 5,808 unique assets; land cover made 35 downloads of 33 unique files (256.0 MB transferred vs 247.6 MB unique). Thus the scattered old pilot had modest duplication; the larger opportunity comes from contiguous regions and eliminating many latency-heavy requests.

## Expansion volume and conditional ETA

Using the earlier three land-clipped corridor geometries, intersected with native MODIS sinusoidal tiles. HEAD measurements cover all intersecting tiles on 2023-01-25 and 2023-07-12, four bands per tile/date. Scale the mean of those two seasonal totals to 44 dates:

| Corridor width | Land area | Nominal 4 km cells | Native source tiles | Estimated thermal payload |
| --- | ---: | ---: | ---: | ---: |
| 350 km | 12.34 million km² | 771,206 | 72 | 11.44 GB |
| 500 km | 17.51 million km² | 1,094,118 | 75 | 11.56 GB |
| 700 km | 24.14 million km² | 1,508,681 | 85 | 12.76 GB |

These are estimates, not all-year file inventories. No sample tiles were absent on either measured date. Geometry is simplified and intersections below 1 km² omitted; exact grid allocation, target buffers, seasonal compression, missing dates, additional representative blocks and source boundaries can change counts. Land area divided by 16 is nominal, not an allocated raster-cell count.

For the 350 km option, applying annual-test throughput gives **~23 minutes thermal download + ~14 minutes thermal processing** before padding, mosaicking, retries and other variables. Land cover adds roughly 0.54 GB if the previous 33-file mean (~7.5 MB) holds; it has not been inventoried for these corridors. Terrain adds ~1.20 GB if fetched globally. Working download volume is therefore roughly **13–15 GB** for the default terrain/land-cover/thermal families, before extra blocks or temporal revisions. The two globals and ancillary inputs can be reused across runs.

**Planning allowance: 1–3 hours for a refactored batched 12–17 million km² import**, conditional on similar sustained service/network throughput and bounded regional work. This is a throughput projection, not an end-to-end measured ETA; current scripts still hardcode 100×100 cells/1 km and cannot produce that expansion unchanged. The earlier 12–30 hour estimate scaled the old per-patch window workflow and was too pessimistic for this design. Query preparation, DB migration/validation, and importer implementation time are not benchmarked here.

## Proposed execution

1. Build one inventory keyed by product/version/year/native tile/date/band. Reuse matching cached assets after hash validation.
2. Fetch thermal COGs once with eight downloads in flight. For sparsely touched tiles, retain the option of range reads; whole-file batching is not universally smallest.
3. Process cached inputs in regional groups with eight workers, reuse projected geometry, and preserve missing-data/support semantics. Benchmark the combined download/CPU pipeline before assuming two simultaneous eight-worker pools are optimal.
4. Reuse terrain rows/globals; derive physical 5/21 km neighborhoods with halos; aggregate final 4 km cells. Read annual land-cover files once and fan out their two needed bands.
5. Use a bounded temporary source cache, bulk-load derived U16 results, validate coverage and values, then delete source files. No permanent raw-raster archive is required.

Here “batch” means an inventory-driven, deduplicated parallel downloader, rather than a single archive supplied by the provider. No server-side ZIP/extraction endpoint was benchmarked. Reducing output resolution reduces processing and final storage, but full native source-file downloads do not shrink by the same factor.

## Reproduce

```sh
uv run scripts/benchmark_import_pipeline.py
uv run scripts/benchmark_import_sources.py
uv run scripts/benchmark_import_sources.py --landcover-only --credentials-file /private/token-file
uv run scripts/benchmark_import_annual.py
uv run scripts/benchmark_import_volume.py --plan research/import-profile/corridor-plan.json
```

`sample-items.json` contains only unsigned public STAC metadata. SAS tokens are obtained in memory, never recorded. The optional token file contains an Earthdata bearer token, never committed. Raw thermal files use temporary `/tmp/earth-import-*-cache` directories. The benchmark processor's recorded hash predates changing its default sample path from `/tmp` to this reproducible sample; calculations are unchanged. `geometry_seconds` in CPU samples is the cached *initial* geometry build time, not additive per invocation. All times are wall seconds unless explicitly CPU; MB/GB are decimal.

Machine-readable evidence: `benchmark.json`, `annual.json`, `other-sources.json`, `volume.json`. The latter contains unsigned asset sizes from 680 HEAD requests. Existing database and scientific processors were left intact.
