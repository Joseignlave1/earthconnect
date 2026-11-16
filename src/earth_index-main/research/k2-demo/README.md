# k²-raster pilot: 1 km² cells

This experiment tests the authors' actual C++ k²-raster implementation on the
existing Atacama 100 × 100 km pilot. It is independent of PostgreSQL; no database
schema or imported observations were changed.

**Conclusion:** retaining approximately 1 km² detail in compressed regular-grid
tiles is plausible. k²-raster preserves the grid and indexes its values; it does
not recover detail from H3 summaries, and its compression does not eliminate the
increase in sample count. It is a candidate to benchmark further, not a selected
production storage format.

## Inputs and resolution

The grid contains **10,000 cells**, each 1,000 × 1,000 projected metres in the
pilot's local WGS84 azimuthal equidistant CRS. Ground areas differ slightly from
the projected 1 km². This is a regular square grid, not H3 resolution 8.
Our existing Atacama H3 resolution 6 pilot contains 317 intersecting cells.

Previously retained CSVs contain coarse cell summaries. To obtain finer values,
the demo re-read the same public Copernicus DEM window and one original NASA
MOD11A2.061 granule for 2024-01-01–2024-01-08. The DEM decoded-window SHA-256 and
the NASA whole-file SHA-256 match the earlier pilot manifests. Original source
rasters were discarded; only the small **derived, quantized grids** and compressed
representations are retained here. `grid.json` records source hashes and methods.

Five aligned bands are indexed:

| Band | Stored increment | Maximum rounding error |
|---|---:|---:|
| Mean surface elevation | 1 m | 0.5 m |
| Mean native-pixel slope | 0.1° | 0.05° |
| Fraction of native terrain with slope <5° | 0.001 | 0.0005 |
| Mean daytime land-surface temperature | 0.02 K | 0.01 K |
| Valid temperature footprint fraction | 0.001 | 0.0005 |

The codec is lossless **after this explicit quantization**. Each band uses an
integer origin plus 16-bit codes, reserving zero for missing data. These are
rounding tolerances, not bounds on measurement or aggregation error. Threshold
comparisons near a rounding boundary may differ from the unrounded mean.

Slope is calculated on the native ~30 m DEM before aggregation. Averaging the
elevation first would erase terrain detail. Temperature comes from nominal
1 km MODIS pixels, approximately 926.6 m per side in its sinusoidal grid. It is
area-averaged onto the common local grid; that alignment creates no additional
thermal detail. Only QA byte exactly zero is accepted. Temperature is one 8-day
daytime surface-temperature composite; elevation is mixed-epoch DSM elevation.
Coverage fractions are separate from observed values and missingness.

## Measured storage

Payload sizes for all five bands, including k²-raster's serialized tree, indexes,
padding and per-band codec headers:

| Representation | Bytes | Versus packed uint16 |
|---|---:|---:|
| Raw int32, the upstream encoder input | 200,000 | 200% |
| Raw uint16, sufficient for these quantized bands | 100,000 | 100% |
| Standard k²-raster, k=2 | 61,048 | 61.0% |
| Heuristic k²-raster, k=2, one plain/vocabulary level | 62,843 | 62.8% |
| zlib level 9 on the uint16 bands | 44,636 | 44.6% |

All formats also require shared grid/scaling/provenance metadata, about 6.3 KiB
in `grid.json`, excluded from every payload comparison. `results.json` and the
visual are experiment reports, not necessary storage for the grid. The directory
intentionally retains competing encodings and baselines for reproducibility;
their sum is not the proposed storage cost.

Standard k²-raster achieves **1.64× compression against packed uint16**, or 3.28×
against int32. Ordinary compression is smaller here, but answering an arbitrary
value-range query from the zlib stream requires decompression. k²-raster navigates
its min/max index directly. The heuristic variant is slightly larger on these
small grids. This tests one configuration per variant, not a parameter search.

An H3 resolution 6 cell averages about 36 km² globally. Moving to 1 km² therefore
increases spatial sample count about 36-fold; a 1.64× codec gain does not cancel
that cost. A single global snapshot of 91 hypothetical two-byte scalar bands
would already be about 93 GB uncompressed, before masks, indexes, provenance,
extra statistics and time. Some actual variables need other types or dimensions.
Do not extrapolate this desert's measured compression to every variable or region.

## Queries and correctness

Both variants were serialized, reloaded, and checked against every input code:
**100,000 pixel comparisons**, **150 spatial/value-range queries**, and **2,600
window pixel comparisons** across both codecs. Coordinates from the range index
were also checked independently with NumPy. Tests include random rectangles,
empty result ranges and code zero. See `results.json` for counts and timings.

The example joint query requires all of these on the **same 1 km cell**:

- Daytime temperature ≥310 K.
- Temperature valid footprint fraction ≥0.95.
- At least 90% of native terrain support has slope <5°.

It returns **5,818 cells**. Using mean slope ≤5° instead returns 6,926 cells.
Band-specific compressed-index results are intersected by row/column, rather
than joining unrelated tile averages. This demo intersects coordinate sets;
it does not implement a fused multi-band tree traversal. The reported per-band
query timings exclude loading/I/O and do not time the complete joint query.

Largest four-neighbour connected component of qualifying cells: 3,902 cells,
with 84 components in total. This describes connected qualifying **footprints**;
it does not prove the underlying flat ground connects across their boundaries,
or that temperature and flatness coincide everywhere inside a 1 km footprint.
Sub-km site guarantees would still need finer aligned data and verification.

Queries take microseconds to fractions of a millisecond on this tiny warm grid.
Scans beat the index on several broad-result queries; min/max rejection is very
effective for impossible/selective ranges. This is not evidence of global query
latency. Benchmarks use native C++ for both scan and indexed coordinate retrieval,
median of seven batches of 200 repetitions, with allocations included.

## Reproduce

```sh
# Requires git, CMake, g++, Python and uv. No system install or database changes.
bash scripts/k2_demo_build.sh

# Reuse the retained derived codes; no network or NASA login needed:
uv run scripts/k2_demo.py
python scripts/test_k2_demo.py
python scripts/k2_demo_visual.py

# Optionally re-read the original sources using an Earthdata credential file:
uv run scripts/k2_demo.py --prepare --credentials-file /private/earthdata-file
```

Upstream is pinned to `fac218aab831f2611b5ae94f09f0ef12d1cd29ea` (v3.1).
It builds under `/tmp/earth-index-k2-upstream`. GCC 16 diagnoses existing member
name typos in SDSL and the raster's copy/move helpers; the build script explicitly
fixes those names. Codec/build/query algorithms remain unchanged. The upstream
library declares LGPL-2.1-or-later in its source; it is not vendored into this
project. A Rust API could use an appropriate native binding or a separately
implemented, verified Rust codec.

**Upstream defect found:** the uniform-root `get_values_window` branch starts
its column loop at `yend` instead of `yini`, producing an incomplete window for
constant rasters. Synthetic tests explicitly detect this defect for both variants
and both square/rectangular grids. The real-data bands are nonuniform and passed
all checks. The demo deliberately leaves that upstream algorithm unchanged;
production integration must fix this and expand testing. Passing the synthetic
suite includes successfully reproducing this known defect, not certifying the
upstream library for arbitrary inputs.

`demo.html` shows the verified cells with adjustable temperature/flatness filters
and the measured storage comparison. Its presentation filters operate on decoded
codes in the browser; the native compressed-index correctness/benchmark work is
performed by `k2_demo.py` and `k2_demo_query.cpp`.

## PostgreSQL implications

A possible layout is spatially indexed tile rows with aligned compressed band
payloads, plus tile-level bounds for initial filtering. Rust or a PostgreSQL
extension would then execute the fine-grid predicates and return matching cells.
PostgreSQL's GiST index on tile extents does **not** index values hidden inside a
`bytea`; storing a k²-raster blob alone does not provide fast arbitrary SQL over
its cells. A SQL function/extension and planner/query benchmarks would be required
to deliver the easy SQL interface discussed earlier. PostGIS raster and ordinary
compressed/tiled arrays remain worthwhile baselines for that next comparison.

## References

- [Authors' code and usage](https://gitlab.lbd.udc.es/fsilva/k2-raster).
- [Scalable and queryable compressed storage structure for raster data (2017)](https://doi.org/10.1016/j.is.2017.10.007).
- [Efficient processing of raster and vector data (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226943).
- [H3 cell counts and average areas](https://h3geo.org/docs/core-library/restable/).
