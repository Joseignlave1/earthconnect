# Resumable global 4 km acquisition

For source citations, credits and the current field inventory, see
[Data sources and methodology](../docs/DATA_SOURCES.md). This page describes the
acquisition job; the prepared data is now served through the
[DuckDB snapshot](README.md).

Run controls from the repository root:

```sh
scripts/world-import status
scripts/world-import logs
scripts/world-import pause
scripts/world-import resume
scripts/world-import stop
```

`pause` creates `research/world-4km/PAUSE`. In-flight source transfers and CPU jobs
finish; complete tiles checkpoint; new work waits. `resume` removes that marker
and starts the container if needed. `stop` asks Docker to stop with a three-minute
grace period. An immediate kill or reboot also preserves committed tiles. Resume
with the same Docker image: processor changes intentionally fail identity checks.
Initial shared-source preparation finishes its current stage before pausing.

The job runs detached in Docker. Eight independent CPU processes and two source
preparation threads (four transfer workers each) have bounded lookahead. The
container is limited to 10 logical CPUs and 12 GB RAM. A 20 GB free-space reserve
pauses new work. Five container retries recover startup failures; individual
regions retry six times. Check `status`/`logs` for a terminal failure.

Completed products are under `research/world-4km/tiles/<hXXvYY>/`; each region
has 25 single-band U16 TIFFs and a checkpoint published only after all files are
complete. Resume hashes every output, per-region source inventory and shared
inventory. Downloads are independently checksummed and atomically replaced;
partial files are never accepted. Per-region source rasters are deleted after a
successful checkpoint, leaving source URLs, sizes and SHA256 hashes in the source
inventory. Shared sources persist. Interrupting an active region may repeat that
region, but completed regions do not repeat. A final manifest is emitted only
after all global region checkpoints verify. This job prepares data; it does not
load or publish a replacement PostgreSQL datasource.

## Grid and support

The new seamless grid uses MODIS sinusoidal projection, sphere radius
6,371,007.181 m. 278 by 278 cells exactly cover each native 10-degree MODIS tile.
The exact spacing is **3,999.822013548645 m** (15.998576 km² per cell), nominally
4 km. This avoids seams, overlaps and a second thermal source tile at every grid
edge. The global rectangle is 10,008 by 5,004 cells; cells whose centers lie
outside the curved sinusoidal Earth domain are NODATA. All tiles with at least
one Earth-domain cell are processed, including ocean-only regions. Per-variable
coverage remains limited by the underlying products. The two pole points have
zero area and no independent cells.

This is a separate grid/version and processing method from the local AEQD pilot.
It is not an exact re-encoding of those pilot cells. Global area resampling uses
GDAL's average resampling, with exact coordinate transformation, rather than the
pilot's densified polygon/spherical-overlap implementation. Categorical lithology
uses nearest source-cell sampling. Coverage/QA layers remain separate from values.
U16 code zero means missing; physical zero has a nonzero code. The existing 25
feature names, units, scales and offsets come from the frozen thermal pilot,
except annual precipitation uses 1 mm steps (offset -1 mm), allowing up to
65,534 mm/year. The pilot's .1 mm encoding cannot represent very wet global cells.

## Variables and source semantics

- **Terrain (7):** elevation, slope, terrain valid fraction, 5/21 km relief and
  5/21 km TPI. EarthEnv/GMTED elevation/slope are first averaged to a native-grid
  approximately 1 km intermediate. Contexts use 5/21 pixel windows with a 10 pixel
  halo, at least 90% valid neighbors, then 4 by 4 reduction with at least 50%
  supported subcells for context fields. Elevation/slope use valid-area weights.
  The published elevation median filename conflicts with the TIFF's Maximum
  description; that original product ambiguity remains. No substitute data is
  invented south of the product's actual domain.
- **Land cover (3):** bare fraction, permanent snow/ice fraction and valid
  fraction, MCD12Q1.061 year 2024. Accept QC 0 or 2 and IGBP classes 1..17.
  Bare=16; permanent snow/ice=15; classified water=17 is not land. Fractions
  describe classified support and are accompanied by full-cell valid coverage.
- **Surface temperature (9):** day/night/paired annual mean, night composite low,
  day composite high, high-low range, and three valid-area-duration fractions.
  MOD11A2.061 Terra, all available 2023 composites from Planetary Computer.
  QC byte must equal zero; raw LST at least 7500; .02 K scale then Celsius.
  Paired temperature needs both day and night valid. Spatially average on the
  final grid before temporal summaries; require at least 50% supported cell
  area per accepted composite, at least 12 accepted composites and all four
  seasons. Weight means by accepted area and composite duration (last interval
  truncated to year end). Quality divides accepted area-days by 365, so absent
  composites remain absent evidence. These are clear-sky composite summaries,
  not instantaneous weather or daily extremes.
- **Air climate (3):** coldest/warmest monthly mean 2 m air temperature and
  annual precipitation. NASA POWER's public global MERRA-2 climatology Zarr,
  **2001–2020**. This is an explicit baseline difference from the pilot API's
  current 2001–2025 climatology. Read all 12 monthly values, requiring valid
  support; take monthly temperature min/max before spatial averaging. Annual
  precipitation is the native annual mean daily rate (13th coordinate) times
  the mean number of days in a climatological year. Temperature is already C;
  precipitation is mm/day. Native support remains .5 by .625 degrees, with
  longitude wrap. At least 99.9% valid spatial support is required. This bulk
  access avoids thousands of POWER regional API requests.
- **Permafrost (2):** ESA CCI v5 PFR year 2023 and valid fraction. CF coordinate
  vectors govern the georeference; flip ascending latitude, convert percent to
  fraction, values above 100 missing. Outside its northern product domain stays
  unsupported. Product GeoTransform conflicts with CF vectors; CF vectors win.
- **Lithology (1):** GLiM v1, .5-degree categorical source sampled at target
  centers; codes -9999 and 15 missing, no averaging of categories.

NASA bulk POWER documentation:
https://power.larc.nasa.gov/docs/services/aws/
https://power.larc.nasa.gov/docs/tutorials/service-data-request/aws/

Source metadata, object hashes, exact grid, algorithm hashes and each layer's
quantization error and valid count are persisted with the products. Passing
mechanical and quantization checks does not establish landing-site suitability.
Scientific comparison and database loading remain separate follow-up steps.

## First build and launch

```sh
docker compose --profile tools build importer
docker build -t earth-index-world-importer:latest -f db/world-importer.Dockerfile .
scripts/world-import start --token-file /absolute/path/to/existing/nasa-token
```

The token is mounted read-only as a Docker secret. Only its file path is retained
in the ignored, mode-0600 host configuration; token contents are never copied
into generated metadata, command arguments or the image.
