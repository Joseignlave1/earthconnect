# First real terrain extraction

This directory contains computed tile summaries, not source rasters. Run metadata
and methods are in [run.json](run.json); structural and numerical validation is in
[validation.json](validation.json). Scientific operator fixtures are run separately
with `uv run scripts/test_pilot_terrain.py`.

The complete three-region run took 96.6 seconds on this computer and recorded
1,612.2 MiB peak process RSS. This is a measured terrain-only pilot, not a runtime
forecast for the other candidate sources. No raster files or raster scratch space
were retained; remote window reads and processing arrays lived in memory. HTTP
transfer volume was not instrumented and must not be inferred from decoded sizes.

| Region | H3 resolution-6 cells | Numeric summary rows | Processing seconds |
| --- | ---: | ---: | ---: |
| Atacama | 317 | 3,804 | 25.5 |
| Central Iceland | 361 | 4,332 | 55.3 |
| Central Sahara | 281 | 3,372 | 15.7 |
| Total | 959 | 11,508 | 96.6 |

Each cell has elevation min/mean/max/population standard deviation, slope
mean/standard deviation/max, mean 3×3 height roughness, mean 3×3 topographic
position, mean 3×3 local relief, and mean downhill aspect eastness/northness.
Aspect components exclude slopes below 0.1 degrees; they are not ordinary
arithmetic means of aspect angles.

Only the portion of an H3 cell inside its pilot square is summarized. CSV
`target_area_m2` and `fraction_of_full_cell` preserve that support. Elevation is a
static mixed-epoch surface height relative to EGM2008, not a 2024 measurement.
Native elevation postings are used without reprojection or resampling, with
fractional polygon/pixel overlap and spherical area weighting. See the run's notes
for explicit operator definitions and approximation limits.

Source-asset seams lack cross-asset derivative neighborhoods and are excluded from
derivative support. `valid_area_fraction` reports the retained support, including
the nonflat condition for aspect. The current public asset metadata supplied no
auxiliary editing/filling/height-error/water masks; only nodata and finite checks
were applied. These are reviewable research outputs, not validated published
scientific measurements. Full upstream file checksums remain unknown; window
hashes and opaque provider ETags are recorded separately.

The CSV artifacts are ready for implementing PostgreSQL ingestion. No PostgreSQL
server has been installed or populated by this run. The other candidate variables
and NASA temperature extraction are not represented by these terrain results.
