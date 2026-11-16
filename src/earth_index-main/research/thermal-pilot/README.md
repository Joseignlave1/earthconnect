# MODIS ground-temperature revision

This adds six physical temperature summaries and three support layers to the
existing 27-patch, 1 km raster pilot. It creates a new immutable bundle; existing
H3 observations and previous raster revisions remain available.

Data: MOD11A2.061 Terra, 2023, from Microsoft Planetary Computer's public
cloud-optimized GeoTIFF mirror. The mirror's 2024 record is incomplete, so this
revision explicitly uses 2023. Catalogs and native decoded-window hashes are
retained; original rasters and public signed access URLs are not.

- `surface_day_c`, `surface_night_c`: valid-area × composite-duration weighted
  means of observed daytime/nighttime 8-day composites.
- `surface_mean_c`: weighted mean of paired daytime/nighttime native-pixel means,
  using common valid samples.
- `surface_low_c`, `surface_high_c`: coldest observed nighttime / hottest observed
  daytime composite, after per-window cell support screening.
- `surface_range_c`: `surface_high_c - surface_low_c`; a difference in Celsius
  degrees, not an absolute temperature and not a complete diurnal amplitude.
- `surface_day_valid_fraction`, `surface_night_valid_fraction`,
  `surface_paired_valid_fraction`: accepted clear-sky area × days / (cell area ×
  365). These combine temporal and spatial support; they are not the same as the
  terrain/land-cover coverage fractions.

All physical summaries require >=12 composites with >=50% accepted cell support
and observations in all four calendar quarters. Missing coverage stays NULL.
Native temperature values use scale 0.02 K and convert to Celsius once. Native
QC byte exactly zero is good; the QC GeoTIFF nodata tag must not mask it.
Composite QC summarizes majority daily flags, so it does not establish that
every daily constituent passed the same checks. Exact
fractional source-pixel overlap on the native sinusoidal projection merges tile
pieces before temporal aggregation.

These are clear-sky observed composite summaries, not annual all-weather
climatology or true daily temperature extremes. Default planetary comparisons
are explicitly approximate (different observation times, years, day lengths and
models), use editable 30°C tolerances and share one thermal weight family with
weight 0.5 by default. Surface-range remains accessible for manual queries;
including it automatically would repeat information already in low/high.

```sh
uv run scripts/test_thermal_pilot.py
uv run scripts/thermal_pilot.py --workers 6
uv run scripts/load_raster_pilot.py --pilot-dir research/thermal-pilot --report research/thermal-pilot/database.json
# Refresh API views after importing, using the existing Compose migrate service.
docker compose run --rm migrate
```

Per-site checkpoints permit a restart without reacquiring verified finished
patches. A checkpoint is reused only when the processor and tile hashes match.
Public source-window manifests hold stable unsigned asset links and decoded
pixel checksums. They do not establish independent provider checksum validation.
