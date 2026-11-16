# Read-only query experiments

No production settings, schemas or source data changed. Temporary settings reset
when benchmark sessions end. The immutable 4 km bundle and demo stay unchanged.

The current SQL view couples pixel-row cardinality to a zipped expansion of all
25 bands. This forces unused band expansion. The tested cellwise query instead
selects the runtime-required bands (11 columns for Valles, including two quality
fields), reads each patch once, scores at runtime and packs each patch separately.
It stores neither scores nor preferred target values, and keeps U16 source data.

## Measured settings

See `query-settings.json` for 3 rotated-order samples of five session-local
configurations against the existing query and the required-band per-patch query.
Current defaults: shared_buffers=128MB, effective_cache_size=4GB, work_mem=4MB,
jit=on, max_parallel_workers_per_gather=2, max_parallel_workers=8.

For the existing SQL, medians were 2142 ms default, 2018 ms with JIT off, 2088 ms
with 64MB work_mem, and 1916 ms with both. Raising work_mem removed per-patch
prototype temporary writes but showed no clear speed benefit in this sample.
The intentionally aggressive parallel-planning control launched 95 workers across
many small plan nodes and was slower (6281 ms); that is not merely a test of
raising max_parallel_workers_per_gather. Source raster functions are declared
parallel-safe; plan structure and work placement still matter.

A direct-array projection prototype was rejected after it timed out: carrying
large arrays through per-cell plans can duplicate work or create huge intermediate
tuples. This was read-only and left no persistent changes. The retained prototype
uses bounded per-patch unnesting rather than this rejected path.

## Chunk scheduling

`query-chunks.json` records a separate pooled-connection benchmark: 38 patches,
each 100x100 cells, assigned across 1/2/4/8 worker connections. Each transaction
is read-only; JIT and internal query parallelism are off; work_mem=4MB. Medians:
1049/535/316/203 ms. Timings include database result fetching and thread scheduling,
but exclude connection creation and gRPC. They are not directly interchangeable
with EXPLAIN instrumentation timings. Six targets were additionally checked at
eight workers, taking 203–235 ms each, with exact equality to all original API
score buffers (2.28M codes total). These are warm-cache pilot results, not global
or concurrent-user measurements.

Immutable bundle identity makes grid/layer metadata and native band caches safe
to reuse. A post-load VACUUM (FREEZE, ANALYZE) and appropriate cache sizing are
reasonable serving preparation, but have not been benchmarked here. work_mem is
per operation and worker; increasing it globally needs a concurrency budget.
Spatial partitioning/pruning helps bounded requests; a full-planet score still
visits every relevant chunk.

Chunk scheduling is exact here because scores are independent by cell and output
is grouped by patch. An arbitrary raw SQL query cannot generally be split and
concatenated: global aggregates, joins, windows, ORDER BY and LIMIT need their
original global semantics. The generic SQL driver must retain those semantics;
any fast path should apply only when chunk independence is explicit.

Reproduce:

```sh
uv run scripts/benchmark_query_settings.py
uv run scripts/benchmark_query_chunks.py
```
