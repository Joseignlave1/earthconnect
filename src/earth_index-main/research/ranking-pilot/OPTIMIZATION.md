# Faster ranking on the existing pilot

The optimized SQL evaluates all **270,000 pixels and 2,700 windows across 27
patches** in 1.10–1.29 seconds on this computer. It preserves the original
generous gate, continuous scores, evidence coverage, uncertainty bounds and
complete supported/optimistic ranking, including deterministic ties.

## Measured performance

| Preference | Original median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| Hot / dry / bare, balanced | 7.763 s | 1.101 s | 7.05× |
| Prioritize heat | 8.237 s | 1.155 s | 7.13× |
| Prioritize flatness | 8.178 s | 1.113 s | 7.35× |
| Cold / bare / frozen ground | 7.965 s | 1.285 s | 6.20× |

These are three paired samples per preference, with execution order alternated,
fresh connections, the read-only reader role and a 10-second statement timeout.
Timing includes connection, SQL execution and fetching all windows; container
startup, CLI formatting and Python ranking are outside it. These are local
warm-cache measurements, not production latency percentiles or a global-Earth
benchmark. The [full report](../pilots/database/fast-ranking-benchmark.json)
retains every sample, source/specification hashes and index fingerprints.

A separate four-client test completed every optimized query in 0.97–1.17 s.
This is one small concurrency test, rather than a sustained throughput estimate.

## What changed

The original wide view calls `ST_PixelAsPoints`, produces nearly four million
nonmissing band rows with geometries, then groups those rows back into 270,000
cells. An analyzed plan spent roughly 7.2 of 7.8 seconds obtaining that wide
source. It was the main bottleneck, before most scoring work.

[fast_ranking_query.py](../../scripts/fast_ranking_query.py) instead selects the
bands needed for the gate and current preference: seven for hot profiles, nine
for cold ground, out of sixteen stored bands. It uses `ST_DumpValues` and zipped
array expansion to combine aligned values directly into one row per pixel.
There is no point geometry construction or long-value regrouping. The original
scoring SQL is reused unchanged after substituting this source.

The 1–10,000 pixel ordinal preserves row-major coordinates. A missing band is
NULL-padded; even when every requested band is absent, the pixel positions
remain present. NODATA is converted to NULL before scale/offset decoding, so
physical zero retains its meaning. Per-pixel combinations still enforce
co-location before 10 × 10 km window averaging.

No index, extension, schema, quantization or source-data change was needed.
The raster store and wide convenience view remain available. Analyzed plans
are retained in [ranking-profiles](../pilots/database/ranking-profiles/); their
server execution times are diagnostic and distinct from the paired timings.

## Verification

The focused tests compare all 270,000 pixel coordinates and 2.7 million values
across the union of required bands with the verified TIFF arrays. Synthetic
checks cover physical zeros, NODATA, row boundaries and absent bands.

Every measured SQL result is independently compared with NumPy for all 2,700
windows, for each of four preferences: counts, evidence and score bounds agree
within 2.34 × 10⁻¹⁵. The full lower- and upper-bound window order and geographic
order match the original. All 15 existing project index definitions have the
same fingerprint before and after the benchmark. The frozen original evaluator,
specification and source bundle are checked rather than modified.

This improves execution only. Source resolution, coverage gaps and the
scientific/calibration limitations described in the [experiment](README.md)
still apply. The optimized reader assumes the existing aligned 100 × 100,
single-tile-per-layer pilot. A larger mosaic must join on tile coordinates and
validate alignment before using the same array technique.

## Run a query

```sh
docker compose run --rm --entrypoint python loader scripts/rank_pilot.py --profile heat_priority --geographic --limit 10
docker compose run --rm --entrypoint python loader scripts/rank_pilot.py --profile cold_ground --mode uncertain --geographic --limit 10
```

Profiles are `hot_balanced`, `heat_priority`, `flat_priority` and `cold_ground`.
The default `supported` mode orders positive lower-bound scores. `uncertain`
returns windows with incomplete evidence, ordered by their optimistic upper
bound; that queue is not an assertion of suitability. `--geographic` explicitly
keeps one best window per sampled patch; omit it to retain neighboring windows.
`--limit` controls returned results, after evaluating all windows. JSON includes
scores, evidence and pixel-window coordinates, with source/specification hashes.

The [retained SQL](../pilots/database/fast-ranking-artifacts/) can be used by the
future Rust API through the existing read-only role. Its single `%s` bundle
placeholder is psycopg syntax; use `$1` for a Rust PostgreSQL driver. The CLI
sorts the returned windows with the experiment's existing deterministic tie
rules. No Rust HTTP endpoint is implemented yet.

## Reproduce the measurements

```sh
docker compose run --rm --entrypoint python loader scripts/test_fast_ranking_query.py
docker compose run --rm --entrypoint python loader scripts/benchmark_fast_ranking.py --artifacts-dir /reports/fast-ranking-artifacts --report /reports/fast-ranking-benchmark.json
docker compose run --rm --entrypoint python loader scripts/profile_ranking_query.py --fast --report /reports/ranking-profiles/fast.json
```

The Compose workspace is read-only. `/reports` maps to
`research/pilots/database/` for generated reports and SQL artifacts.
