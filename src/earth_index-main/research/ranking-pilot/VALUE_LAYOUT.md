# Measurement layout and per-cell scoring

Storing decoded measurement columns speeds up this bounded pilot, while keeping
the user's scoring expression flexible. This is an experiment with temporary
tables, not a replacement of the current raster database or CLI.

## Results

All paths evaluate the existing scoring expression over **270,000 cells**, with
the same eligibility, coverage and missing-evidence bounds. There is no score
ordering and no 10 × 10 km window aggregation. Aggregate counts and score sums
force evaluation of every result while excluding client transfer/serialization.

| Profile | Quantized rasters | Quantized rows, metadata join | Quantized rows, constant decoding | Decoded float64 rows |
| --- | ---: | ---: | ---: | ---: |
| Balanced hot/dry | 774.136 ms | 644.782 ms | 570.081 ms | 522.591 ms |
| Heat priority | 783.893 ms | 649.649 ms | 569.081 ms | 519.181 ms |
| Flatness priority | 783.560 ms | 649.143 ms | 572.220 ms | 521.163 ms |
| Cold ground | 898.705 ms | 698.757 ms | 601.291 ms | 552.008 ms |

These are server execution medians from five warm-cache `EXPLAIN ANALYZE` runs
per path/profile, with execution order rotated, on the same connection and
reader role. All tables are analyzed, there are no added indexes, and the normal
planner/JIT choices are retained. Plans and every sample are in the
[report](../pilots/database/value-layout-benchmark.json).

Decoded rows reduce execution time by **8.2–8.9%** against quantized rows with
constant decoding. They provide **1.48–1.63×** throughput relative to the raster
path. This separates two improvements: avoiding raster expansion and repeatedly
decoding integer measurements. It does not prove how much time each individual
arithmetic instruction costs; the physical row widths also differ.

The pilot uses one scale/offset pair per feature across all sites. The benchmark
checks this before testing the constant-decoding path. Its metadata-join control
shows the extra cost of obtaining those parameters per site; joining metadata
is not intrinsically necessary for every row when parameters are shared.

## What would be stored

Current stored codes decode to measurements with `value = code * scale + offset`.
Decoded cell rows instead persist the resulting temperature, rainfall, slope,
land-cover fractions and other measurements in native units. They do not contain
precomputed suitability curves or fixed final scores. The benchmark retains
float64 to preserve the existing decoded values; categories would remain codes.

PostgreSQL has signed two-byte `smallint`, rather than unsigned uint16. The
quantized-row control stores valid codes minus 32768 and uses NULL for missing
values. It reconstructs the original integer before scale/offset arithmetic.
All **2.7 million feature values**, including missing values, round-trip exactly
at all **270,000 cell positions**, with both metadata and constant decoding.
Candidate/unknown counts and lower/upper/evidence checksums also agree for every
profile, with floating-point sum tolerances recorded in the benchmark code.

This decodes the existing quantized data; it cannot restore precision discarded
during quantization. Float32 could save space but introduces additional rounding
and has not been tested here. Dequantization is also distinct from PostgreSQL's
physical page/TOAST decompression.

## Space and limits

The temporary heaps have ten measurement columns plus site and pixel coordinates:

* Quantized rows: **20,447,232 bytes**.
* Decoded float64 rows: **34,603,008 bytes**.
* Shared scale/offset metadata: **8,192 bytes**.

These are actual `pg_relation_size` heap sizes, including row/page overhead, not
just summed numeric field widths. They exclude indexes and do not predict the
size of a final global grid with different identifiers or columns. PostgreSQL
numeric widths are documented in its [type reference](https://www.postgresql.org/docs/18/datatype-numeric.html).

Queries materialize only the seven or nine fields needed by the profile. The
underlying row tables contain all ten tested features. The `MATERIALIZED` CTEs,
temporary spills, planner/JIT decisions and bounds calculations are retained
from the experiment; these measurements are not a lower bound for a future
single-score-only endpoint. They exclude binary packing, network transfer and
client rendering. No result is extrapolated to the whole Earth or to 4 km cells.
Session-local temporary tables also differ from persistent production tables,
including caching and parallel execution behavior.

The experiment creates only session-local tables and drops them at transaction
end. Source rasters, scoring specification, original evaluator, project schemas,
indexes and API permissions remain unchanged.

## Reproduce

```sh
docker compose run --rm --entrypoint python loader scripts/benchmark_value_layout.py --report /reports/value-layout-benchmark.json
```

See [benchmark_value_layout.py](../../scripts/benchmark_value_layout.py). Reports
record the source bundle, scoring-specification and benchmark-code hashes.
