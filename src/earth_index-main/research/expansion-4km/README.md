# Timed 4 × 4 km expansion

A cold-cache experiment over 38 predeclared 400 × 400 km regions. Each region has
100 × 100 cells, with the same 25 U16 bands as the thermal pilot (including six
support bands). This is a purposive, diverse sample, not global or random coverage.
The original 1 km bundles and demo default remain separate.

Measured outcome: **20.0 minutes for cold import plus PostgreSQL load** versus
30.2 predicted; **22.3 minutes through validation and temporary-cache cleanup**.
The unique footprint is 6,065,781 km² (1.189% of Earth's surface, including water
and unsupported areas), about 22 times the original footprint. Raw transfers were
11.030 GB; the new compressed raster datums total 8.90 MB, excluding catalogue,
TOAST/index overhead and retained provenance/derived files.

All 9.5 million new codes and 6.75 million original active codes were verified.
The six unchanged target queries completed in 1.48–1.87 s each, returning 760 kB
of score values, and all 2.28 million output codes matched independent NumPy.
Shared-area per-region score correlations averaged 0.963–0.986 by target; 95th
percentile differences from the mean fine score were at most 0.0584. None of the
523 shared coarse scores >=0.75 lacked a fine subcell >=0.75. These checks cover
this sample only. Original dry/hot-dry/frozen-ground reference filters still
recover Atacama, Death Valley and Haughton respectively. Lunar absolute scores
remain weak, largely because of the thermal mismatch, and polar terrain gaps
remain missing.

Schema: `raster_8a7e289743ea9a7e0a2bf4106d7a0520`; staged datasource identity:
`earth-8a7e289743ea`.

`plan.json` was frozen before downloading rasters: 30.2 minutes predicted,
11.03 GB of estimated downloads. Catalog discovery, source transfers and eight
single-threaded CPU workers are timed in `execution.json` and `manifest.json`.
Planning, implementation and later diagnostic analysis are separate from import
time. Base-source transfers overlap the eight-worker thermal pool.

`database.json` records loading and exact PostGIS roundtrips. The schema is staged:
it has read-only SQL views but is omitted from Metadata, so the demo does not
switch automatically. Query its fully qualified `.cells` relation through the
existing SQL driver. `catalog.raster_bundle_publication` keeps that choice stable
across database startup/refresh. Publication is explicit via `--publish`.

## Scientific semantics

Terrain is first aggregated on a 1 km intermediate grid with a 10 km halo. The
original 5/21 km contextual neighborhoods are preserved before 4 × 4 reduction;
we do not turn those neighborhoods into 20/84 km contexts. Elevation and slope
use accepted-source area weights. Context descriptors average usable subcells,
requiring at least half of the sixteen subcells; each descriptor still requires
90% valid neighborhood support at 1 km.

Land-cover indicators are weighted by accepted classified source area. Permafrost
uses the CF coordinate vectors and explicit outside-domain missingness. Lithology
uses the containing native categorical cell at the target center, never an
average of category codes. Climate stays at its genuine coarse native support;
no high-resolution weather is invented. MODIS thermal QC, source overlap and
annual/seasonal support rules are the frozen reference rules.

These sources describe different periods: terrain/static context, 2024 annual
land cover, 2023 permafrost and clear-sky thermal summaries, and long-term climate.
Thermal lows/highs are extrema of observed 8-day composites, not daily extremes.
Earth–Moon/Mars thermal comparisons remain approximate. Antarctic terrain remains
missing because the terrain source stops at 56°S; its quality is zero.

## Reproduce

Use a new research output directory when declaring another experiment:

```sh
uv run scripts/plan_expansion.py --output research/another-expansion --target-minutes 30
docker compose run --rm --no-deps \
  --volume "${EARTH_INDEX_TOKEN_FILE}:/run/earthdata-token:ro" \
  --entrypoint python importer scripts/import_expansion.py \
  --output research/another-expansion --token-file /run/earthdata-token
docker compose run --rm --no-deps \
  --volume "${PWD}/research/another-expansion:/workspace/research/another-expansion" \
  --entrypoint python loader scripts/load_expansion.py --pilot-dir research/another-expansion
```

The token is an Earthdata bearer token stored in a private file outside Git. It
is never included in source manifests or download URLs. Raw source caches are
temporary; hashes and unsigned provenance persist. `execution.json` records their
final cleanup state. `--resume` requires unchanged processor/plan identity and
reuses completed checkpoints; successful completed experiments are immutable.

## Validation artifacts

- `source-precheck.json`: base-source checks for Atacama, Haughton and Antarctica.
- `validation/targets.json`: six frozen Moon/Mars samples, form criteria and SQL.
- `validation/*-matching.json`: baseline and expanded API query measurements.
- `validation/*-*.bin`: exact little-endian score matrices, 65535 means missing.
- `validation-report.json`: all-band ranges, masks, exact DB codes, SQL/NumPy
  equality, original bundle integrity, reference environmental filters, shared
  coverage comparison and subcell false-match diagnostics.
- `coverage.geojson`: densified geographic product footprints. The reported union
  counts overlap once, but includes water/missing evidence. Per-patch counts can
  overlap and are not unique global areas.
- `report.html`: measured timing, footprint map and unchanged target-score comparison.

The old/new resolution comparison uses their precisely aligned shared inner
96 × 96 km: new indices 38:62 correspond to original 1 km indices 2:98. Annual
thermal statistics do not commute with spatial averaging when QC/support changes,
so they are evaluated through resulting scores instead of assuming equality to
an average of old annual summaries. Categorical values are not averaged in the
shared-grid diagnostics. High coarse scores are compared with all sixteen fine
scores to expose smoothing effects.

These checks establish internal consistency and preservation of useful screening
behavior. They are not calibrated analogue ground truth or proof of connected,
flat, traversable terrain within a 16 km² cell.
