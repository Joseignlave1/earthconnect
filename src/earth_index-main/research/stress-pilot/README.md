# Frozen-filter environmental stress pilot

This expands the [three-area priority pilot](../priority-pilot/README.md) with
24 additional 100 × 100 km patches. Each has the same 16 bands and 100 × 100
projected 1 km pixels. The original three examples and source artifacts are
preserved. The expanded bundle has 27 areas, 432 raster tiles and 4,320,000
pixel codes, including NODATA. No global raster import is needed.

## Observed results

With unchanged thresholds, seven of the eight new plausible-match patches have
at least 100 eligible pixels under one or more profiles. All thirteen declared
negative-control patches have **zero eligible pixels**, even before applying
the 100-pixel patch-support convention. This is stronger evidence of broad
environmental separation than recovery of the original development examples.
It is not a global accuracy estimate: controls are purposively selected and
candidate expectations are not independently certified analog labels.

| Patch | Role | Dry/bare cells | Hot/dry cells | Cold/frozen cells |
| --- | --- | ---: | ---: | ---: |
| Atacama / northern Chile | reference | 9782 | 0 | 0 |
| Haughton Crater | reference | 0 | 0 | 7865 |
| Death Valley / Badwater vicinity | reference | 5153 | 5153 | 0 |
| Namib dune desert | plausible match | 9531 | 4667 | 0 |
| Central Sahara | plausible match | 10000 | 10000 | 0 |
| Wadi Rum / southern Jordan | plausible match | 8070 | 8070 | 0 |
| Salar de Uyuni | plausible match | 1785 | 0 | 0 |
| Central Australian desert | plausible match | 5229 | 5229 | 0 |
| Gobi desert | plausible match | 9988 | 2286 | 0 |
| Axel Heiberg Island | plausible match | 0 | 0 | 1608 |
| Central Svalbard | plausible match | 0 | 0 | 0 |
| Central Iceland | borderline | 0 | 0 | 0 |
| Amazon / Manaus vicinity | negative control | 0 | 0 | 0 |
| Congo Basin | negative control | 0 | 0 | 0 |
| Central Borneo | negative control | 0 | 0 | 0 |
| Finnish Lapland | negative control | 0 | 0 | 0 |
| Central Iowa | negative control | 0 | 0 | 0 |
| Central Uruguay | negative control | 0 | 0 | 0 |
| Serengeti region | negative control | 0 | 0 | 0 |
| Mekong Delta | negative control | 0 | 0 | 0 |
| Tokyo region | negative control | 0 | 0 | 0 |
| Central Alps | negative control | 0 | 0 | 0 |
| Everest / central Himalaya | negative control | 0 | 0 | 0 |
| Greenland interior ice sheet | negative control | 0 | 0 | 0 |
| Equatorial central Pacific | negative control | 0 | 0 | 0 |
| McMurdo Dry Valleys vicinity | coverage challenge | 0 | 0 | 0 |
| Almeria / southeast Spain | borderline | 84 | 84 | 0 |

Every patch has 10,000 projected pixels. Counts are eligibility under each
filter, not physical observations of zero or certainty of analog suitability.

Important findings:

* Svalbard has no matches. Every pixel fails the frozen coldest-month threshold
  of −20°C; the patch median is −14.77°C, and its median annual precipitation
  is 679.3 mm. A maritime Arctic candidate can therefore be excluded by this
  specific climate profile despite usable modeled permafrost. Do not infer that
  it cannot serve another analog purpose.
* Almeria has 84 eligible hot/dry pixels. Its largest four-neighbor component
  has 58 pixels. These results remain visible even though the predeclared
  100-pixel patch-support threshold is not met; small fractional coverage can
  still mean a substantial absolute candidate area. They require local review,
  not automatic dismissal as noise or certification as usable terrain.
* Axel Heiberg adds 1,608 cold/bare/frozen matches; its largest component has
  176 pixels. This provides a new geographic check beyond Haughton, while also
  showing why total passing area is not the same as one connected footprint.
* Antarctica has no cells with all required cold-profile measurements because
  terrain and northern permafrost are unavailable. This is a **coverage gap**,
  not evidence that the Dry Valleys environment fails the desired conditions.
* Iceland also has no matches. Bare volcanic land alone does not satisfy
  these particular arid or very cold environmental profiles.

Connected pixel areas are only coarse-grid descriptions; they do not establish
fine-scale flatness, obstacle clearance, trafficability, geology or crater origin.

The [full evaluation](../pilots/database/stress-evaluation.json) contains
per-predicate failures, missing-data counts, patch medians, connected-component summaries and
benchmark samples. The [SQL validation report](../pilots/database/stress-raster-validation.json)
verifies all 4,320,000 codes, all three complete match coordinate sets, 1,296
geometry point values and 27 wide-view reference cells. The
[reload report](../pilots/database/stress-raster-reload.json) confirms that an
identical import reuses all 432 layers.

## Test declaration

[sites.json](sites.json) records coordinates, roles, rationales and expectations
before processing. The sample deliberately spans different environments:

* Eight plausible matches: Namib, Sahara, Wadi Rum, Uyuni, central Australia,
  Gobi, Axel Heiberg Island and Svalbard.
* Thirteen negative controls: Amazon, Congo, Borneo, Finnish Lapland, Iowa,
  Uruguay, Serengeti, Mekong Delta, Tokyo, Alps, Himalaya, Greenland ice sheet
  and open Pacific Ocean.
* Two borderline areas: central Iceland and Almeria.
* One coverage challenge: McMurdo Dry Valleys vicinity, Antarctica.

The coordinates select broad research windows, not boundaries of named parks
or certified planetary analogs. Each square can mix several landscapes.
Environmental expectations are hypotheses, not pixel-level truth labels.
For example, [UNESCO's Namib description](https://whc.unesco.org/en/list/1430/)
includes dunes, fog and rocky features; passing a low-rainfall filter does not
mean absence of moisture or surface hazards. [Wadi Rum](https://whc.unesco.org/en/list/1377/)
has substantial cliffs and gorges; its valley floors may pass a mean-slope
filter while other terrain remains unsuitable.

[priorities.json](priorities.json) freezes the existing three filter definitions
and stores the SHA-256 of the original priority file. Neither thresholds nor
required coverage were retuned to the added patches. Original development
examples remain labelled `reference`, separate from the new areas. The processor
and evaluator reject changed filter/source identities.

A diagnostic “patch supports profile” means at least 100 of its 10,000 pixels
pass (1%, declared before processing). Reports also preserve exact match counts
and isolated matches. This threshold is not a scientific analog definition.

## Processing and database checks

The processor reuses unchanged scientific decoding/aggregation functions from
the original pilot. Four isolated processes avoid sharing GDAL/HDF4 library
state and keep source downloads bounded. Completed patches have hash-checked
checkpoints for resumable runs. Only derived quantized tiles and public evidence
are retained in the repository; original MODIS granules are temporary.

Two source-coverage adaptations are explicit:

* Polar NASA POWER requests include enough longitude span for the full projected
  square. Responses are aggregated using the same native-support calculation;
  their actual request URLs and response hashes are recorded.
* Antarctica is outside the EarthEnv terrain latitude domain and the northern
  permafrost product. Those measurements are NODATA with zero valid support,
  rather than invented observations. Failure of its frozen filter therefore
  cannot establish that the environment is unsuitable.

The inherited scientific limitations still apply: coarse climate and lithology,
the elevation TIFF metadata conflict, modeled permafrost and its coordinate
metadata conflict, classification uncertainty, mixed source dates, and a mean
slope that does not certify obstacle-free ground. See the original pilot for
source URLs, units and full interpretation rules.

Database integration checks compare every stored code, complete same-pixel SQL
match coordinate sets against NumPy, wide view center values, geometry lookups,
reader privileges, and rejected invalid grid updates. The evaluator reports
missingness separately from observed threshold failures, plus each predicate's
nonexclusive failure count. Four-neighbor component sizes describe only the
coarse matching mask, not connected traversable terrain at native resolution.

## Measured local processing and SQL cost

The 24 added patches processed in **125.7 seconds with four processes**, reusing
existing reference tiles and the downloaded CCI file. The expanded 432-layer
import took **5.43 seconds**. The comprehensive integration run took **24.25
seconds**, while an initial evaluation was also running. Benchmarks below were
repeated afterwards with no other project database tests/loaders running.

| Patches queried | Dry/bare median | Hot/dry median | Cold/frozen median |
| ---: | ---: | ---: | ---: |
| 3 | 0.67 s | 0.68 s | 0.72 s |
| 9 | 2.14 s | 2.15 s | 2.29 s |
| 18 | 3.73 s | 3.62 s | 4.23 s |
| 27 | 6.08 s | 5.48 s | 6.89 s |

All four simultaneous full-sample reader queries completed in 6.18–6.61
seconds, below the 10-second statement limit. Every successful benchmark count
was checked against independently evaluated arrays. These are small local
workload measurements, not production latency guarantees.

The current wide convenience view scans/expands candidate raster pixels. Its
cost grows substantially with the searched sample. This is usable for a bounded
demo but is **not evidence of fast global arbitrary queries**; global search
needs measured tile-level pruning or a more efficient raster query path.

The active expanded bundle has 2,514,518 bytes of raster datums after ordinary
PostgreSQL storage compression, versus 8,640,000 bytes of raw uint16 pixel
payload. Current raster relation/index totals include historical bundles:
1,474,560 bytes for metadata and 3,710,976 bytes for tiles. The complete database
was 57.1 MB including the H3 pilot and PostgreSQL/PostGIS catalogs; WAL and
container images are additional. Derived stress artifacts occupy about 12 MiB.
Do not extrapolate this heterogeneous pilot compression ratio to the Earth.

See [load measurements](../pilots/database/stress-raster-load.json) and the
[full benchmark samples](../pilots/database/stress-evaluation.json).

## Reproduce

Use the existing Docker database or initialize it as described in the original
pilot. Loading retained derived tiles does not require an Earthdata login:

```sh
docker compose run --build --rm loader
docker compose run --rm --entrypoint python loader scripts/load_raster_pilot.py --pilot-dir research/stress-pilot --report /reports/stress-raster-load.json
docker compose run --rm --entrypoint python loader scripts/test_raster_pilot.py --pilot-dir research/stress-pilot --report /reports/stress-raster-validation.json
docker compose run --rm --entrypoint python loader scripts/evaluate_stress_pilot.py --report /reports/stress-evaluation.json
```

For fresh processing using the authorized Earthdata account:

```sh
uv run scripts/stress_pilot.py --credentials-file /private/path/to/earthdata-login --workers 4
```

All versions share the ordinary PostGIS raster layout. Always bind the expanded
bundle SHA from the load report when querying `research.priority_pilot_cells`.
Use `site_key` to restrict the geographic sample. The original three-area
bundle can still be queried independently.

Benchmark queries use the existing wide SQL view, the reader role and a
10-second statement timeout. They count matches across 3, 9, 18 and 27 patches
with three sequential repetitions per profile, then run four clients together.
Timings include connection setup and local execution; the fixed warm-cache
order and small sample do not establish production latency percentiles.

This is a purposive environmental stress test. Its negative-control leakage
and plausible-site recovery are useful diagnostics, not estimates of global
precision/recall or certified planetary analog accuracy. Geology and relief are
imported context; the frozen filters do not evaluate their discriminating power.
