# Native MOD11A2 temperature pilot

Completed research extraction for the 100 × 100 km Atacama, central Iceland and
Sahara regions. All 959 intersecting H3 resolution-6 cells use their intersection
with the pilot square, not their full cell geometry.

## Outputs

Each CSV has one row per cell and nominal 8-day composite window in 2024:

- `day_k`: area-weighted mean daytime surface temperature, in kelvin.
- `night_k`: area-weighted mean nighttime surface temperature, in kelvin.
- `paired_day_minus_night_k`: mean pixelwise composite difference, using only
  pixels valid in both fields, in kelvin. This is not a full diurnal amplitude.
- Each field has its own `_valid_area_fraction`, relative to the clipped cell.

There are 46 windows; the last window is six days, ending at 2025-01-01 exclusive.
These are composited land-surface retrievals, not air temperatures or instantaneous
daily observations. CSV empty fields mean no valid support after screening.

| Region | Cells | Cell-window rows | Missing feature values |
| --- | ---: | ---: | ---: |
| Atacama | 317 | 14,582 | 466 |
| Iceland | 361 | 16,606 | 13,102 |
| Sahara | 281 | 12,926 | 71 |
| Total | 959 | 44,114 | 13,639 |

The three output fields have 132,342 possible values, of which 118,703 are valid.
Missing values are retained; results are not interpolated across space or time.

## Method and limitations

`scripts/process_native_temperature.py` downloads 184 original MOD11A2.061 HDF4
files with eight workers, interleaving dates and regions. It decodes only the
intersecting native source window from each file and deletes the scratch file
immediately. HDF4 decoding is serial to avoid library races. Source files cannot
be window-read remotely through this driver, so full compressed payloads cross
the network even though only small windows are decoded.

Temperatures are decoded once from unsigned packed integers using the verified
scale factor 0.02, zero offset and kelvin units. Only physical source-valid
temperatures with raw QA byte exactly zero are used: good production/data flags,
emissivity error ≤0.01 and LST error ≤1 K. QA zero is a valid code. Composite QA
describes majority daily flags; it does not prove that every constituent daily
retrieval met these criteria.

Aggregation uses exact fractional polygon/pixel overlap on the native MODIS
sinusoidal grid (sphere radius 6,371,007.181 m). Its equal-area projection supplies
consistent spherical area weights. Disjoint source tiles are merged using valid
area weights; paired differences use common valid pixels. Geometries are densified
before projection. No temperature raster resampling is performed. Cloud and strict
QA screening can bias retained observations, especially in Iceland; inspect
coverage before comparing sites.

The optional AppEEARS Atacama export contained a date outside the requested year
and lacked two daytime QA files. No missing QA was inferred; the completed data
here came directly from original HDF4 fields. A comparison against the optional
AppEEARS export was not completed because its download redirect was outside the
consumer's configured allowlist.

## Provenance and checks

Per-region manifests record source granule IDs, canonical source URLs, retrieval
times, SHA-256 hashes computed from downloaded bytes, native headers and read
windows, plus output and cell-geometry hashes. The provider inventory supplied no
original-file checksum, so the recorded source hashes are not independent provider
checksum verification. Passwords, access tokens and signed redirect URLs are not
retained. Bearer authorization is stripped on signed storage redirects.

`run.json` records code/input hashes and measured resource use:

- Wall time: 103.65 seconds for all three regions.
- Peak process memory: 314.03 MiB.
- Original payload traffic: 884,221,143 bytes; this is not retained storage.
- Largest single scratch raster: 7,312,882 bytes.
- Retained source raster files: zero.
- Summaries and provenance: approximately 8.2 MB.

Reproduce with a private Earthdata login file outside the repository containing
`username:password`:

```sh
uv run scripts/process_native_temperature.py --credentials-file /private/path/to/login
uv run scripts/test_native_temperature.py
uv run scripts/test_nasa_temperature.py
python3 scripts/validate_nasa_temperature.py --native
```

Four native-HDF/security fixtures and eight shared scientific fixtures pass.
`validation.json` checks hashes, dates, cell counts, finite values, missingness,
support fractions, source metadata and absence of retained rasters. These are
research artifacts pending scientific review, database ingestion and publication.

Source: [NASA MOD11A2 Collection 6.1 file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD11A2).
