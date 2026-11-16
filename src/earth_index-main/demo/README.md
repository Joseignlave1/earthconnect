# Map demo

Svelte 5 + strict TypeScript, built with Vite. Svelte owns forms, tabs,
measurements, and result navigation; typed OpenLayers modules manage the maps.
Protobuf bindings are generated as TypeScript.

```sh
docker compose up --build -d api grpcui
```

Open <http://localhost:8080/demo>. Axum serves the built frontend from the API
container. For development, run `npm ci --prefix demo` and
`npm run dev --prefix demo`, then open <http://localhost:5173/demo/>. Vite proxies
the gRPC-Web service to port 8080.

## Flow

- Click Moon or Mars (or enter coordinates) to select a target. Enter selects
  the center of a focused offworld map. One binary gRPC-Web `SamplePoint` request
  retrieves numeric measurements through the Rust backend. Changing/clearing the
  target aborts the previous call and suppresses stale completions.
- The target measurements table shows values, units, source links and statuses.
  Hover a row for source grid spacing, scientific notes and quality flags.
  Missing/error/raw-scale values never become zero or calibrated measurements.
- The Form tab builds SQL from actual `cells` columns in `Metadata`. Select score
  variables, targets, tolerances and weights; optional min/max bounds reject cells
  without dropping their matrix positions. No enabled scoring criteria is an
  error, never a perfect match for every cell.
- By default, every available kilometre-scale terrain comparison is enabled:
  1 km Horn slope, 5/21 km relief and 5/21 km center-minus-mean terrain position.
  The backend samples 441 native DEM heights on a spherical tangent-plane grid
  at 1 km spacing. These are approximate counterparts to Earth's area-aggregated
  grid, not identical statistics. At least 90% of a relief/TPI window and its
  center must be valid. Missing target measurements are visibly omitted.
- Bare-ground target 1, tolerance 0.2, is an editable **analogue preference** for
  Earth IGBP barren class 16, not a remotely measured planetary land-cover class
  or independent land/water mask. It prevents flat forest/farmland from receiving
  high default analogue scores. Water with bare=0 scores poorly; unavailable
  ocean land-cover data stays missing. The form labels this preference separately.
- Scoring uses `exp(-0.5 * weighted_mean(squared_standardized_errors))`, a
  geometric mean of Gaussian similarities. A bad dimension cannot disappear
  behind several perfect similarities. Both relief scales share one weight
  family; both terrain-position scales share another, preventing correlated
  scales from accumulating votes. A family's weight is its largest member weight;
  member weights determine its mean error. Other variables are independent
  families. Categorical equality has zero loss; inequality has 25 (a 5-sigma
  mismatch), which approaches zero similarity. User weights remain editable.
- Default tolerances: slope `max(2 degrees, 35% of target)`; 5/21 km relief
  `max(50/150 m, 35% of target)`; 5/21 km terrain position `max(25/75 m, 50% of
absolute target)`. These global demo scales are editable and are not learned
  from known Earth sites or calibrated match probabilities. Related Earth
  coverage fractions must be >=0.9 when present in Metadata; otherwise the
  result is NULL, not a high score from thin evidence. NULL required values
  remain missing; min/max failures score zero.
- Absolute elevation fills but stays disabled because planetary datums differ.
  Relative terrain descriptors cancel the vertical datum. Planetary surface
  temperatures never substitute for Earth monthly air temperatures, and raw
  composition/geology fields have no fabricated crosswalk. Unsupported Earth
  variables remain available for manual criteria. The form reports measured
  versus preference evidence; scores describe only the selected dimensions.
- The Console tab exposes the exact SQL and named parameters. Editing either
  enters custom mode and disables the form builder. **Use form query** returns to
  the builder and replaces the custom text. Arbitrary SQL cannot be reverse-parsed
  into the form. Target changes do not rewrite custom SQL/params.
- The ground-temperature revision adds MODIS clear-sky daytime/nighttime means,
  cold/hot composite summaries and their range. Available planetary thermal
  targets auto-enable as **approximate** comparisons: Celsius conversion, 30°C
  editable tolerances, one shared thermal family and default weight 0.5. These
  preserve the accepted overpass/temporal mismatch instead of using air
  temperature as a substitute. See [thermal provenance](../research/thermal-pilot/README.md).
- **Run** executes `Query` against the selected datasource. **Cancel**, edits,
  datasource changes and target changes abort/invalidate old results. Errors are
  shown with no stale overlay left behind.
- Packed map results render on Earth. A clickable score histogram and native
  **Minimum score** slider filter the received matrix locally. Histogram heights
  use log counts so sparse high-score tails remain visible; counts include zero
  and exclude unknown. Zero scores never become picks, even at threshold zero.
  Defaults start at the lower of 75% or 90% of the best positive score,
  rounded down to the slider step. The displayed scores remain absolute.
- Four-neighbour connected qualifying cells form areas within each patch. The
  top 50 areas rank by peak score, then mean score, area size and stable address.
  Select/Go or Previous/Next jumps to an area and marks its bounds and peak.
  Threshold edits update visibility/counts without moving the map or querying
  the database again. **Best** reveals and jumps to the strongest area.
- **Best** and the result-patch selector zoom
  to a patch; **World**/Reset restores the global view. Click Earth to inspect a
  cell's patch, x/y and score. The overlay fades from transparent to purple,
  scaled to the best positive score across the entire result (never per patch).
  Zero, missing, and below-threshold cells are transparent. The legend shows the
  actual best percentage; scores, thresholds, and ranking remain absolute.
- Zoom-out selects max-pooled raster overviews: a small strong region survives
  downsampling instead of being averaged away or skipped. Overviews keep the
  original patch extent, handle odd grid sizes and preserve missing/zero codes.
  Zoom-in restores native cells; inspection always reads the native matrix.
- Up to 100 connected areas reaching 90% of the best score get tiny violet
  highlights at their actual peak cells when their footprints are too small to
  read. The core is about 2–3 px wide with a faint halo capped at 8 px across.
  Highlights fade out as an area grows from 2 to 4 screen pixels, leaving its
  raster shape visible. Selection uses a thin hollow ring. A weak overall result
  can still have a bright best spot; its absolute percentage remains visible.
  Filtering does not rescale colors or issue another query.
- Earth imagery stays in color without a coverage mask or outlines. Missing
  measurements remain transparent in the score overlay.

The default global datasource contains 460 native sinusoidal regions and
31,881,940 Earth-domain cells at nominal 4 × 4 km spacing, with 25 fields.
Default form scoring and filters are demo choices, not a validated landing-site
suitability model. Queries and edits are not saved.

## Compact score contract

The builder returns one row per patch with:

- `site_key`: string matching a Metadata grid.
- `scores`: BLOB containing exactly `width*height` little-endian uint16 values,
  in row-major order (x fastest). `pack_u16_at` uses explicit zero-based pixel addresses without sorting cells.

Values 0..65534 encode scores 0..1; 65535 is missing. The builder keeps every
cell, including filtered-out cells as zero, so matrix positions remain valid.
Custom SQL using this contract gets the same overlay; unknown/duplicate patches
and malformed/sparse byte arrays are rejected. Other SQL result shapes appear
as a table, with an explicit first-50-row display limit.

Dimensions, CRS WKT and affine transforms come from Metadata, with no hardcoded
patch list or grid size. The current renderer supports a regular grid in a single
spherical sinusoidal or geographic CRS. One WebGL2 layer projects the geographic
Earth view into the native score grid and samples uint16 texture arrays. Native
scores and max-pooled overviews are cached within a 128 MiB GPU budget;
threshold changes reuse textures. Native square cells can appear slanted after
projection onto the geographic map. The full global score payload is
71,101,280 bytes before compression; no
per-cell geometry is sent. Code 65535/NULL remains distinct from valid score zero.

## Imagery and builds

Moon/Mars use their native equirectangular NASA Trek grids; Earth uses NASA GIBS'
geographic grid. Details are in [src/maps.json](src/maps.json). The six local
level-zero JPEGs provide initial views; higher zoom levels fetch public imagery.
Numeric sampling goes through the backend, independently of basemap colors.

After protocol changes, regenerate the checked-in bindings with
`npm run generate --prefix demo` (requires protoc).

```sh
npm run check --prefix demo
npm test --prefix demo
npm run build --prefix demo
cargo test --manifest-path backend/Cargo.toml --test demo -- --ignored
```

Tests cover the backend RPC adapter, cancellation/stale targets, units and valid
zero, query generation/binding/missingness, packed byte order and shape checks,
WKB coverage and planetary grids. Component tests mount the Svelte app in jsdom
and exercise target autofill, custom SQL, cancellation, result filtering and
navigation, and cleanup on unmount. The Rust backend separately tests the native
DEM stencil, numeric services, SQL execution and gRPC transports.

## Adversarial checks

```sh
MATCHING_API=http://localhost:8080 npm test --prefix demo
npm run evaluate --prefix demo
```

The SQL fixtures execute the real builder output through gRPC-Web/DuckDB:
perfect terrain with non-bare cover, missing ocean data, low coverage, conflicting
terrain, duplicate relief scales, opposing characteristics in different cells,
NULL/zero distinction, and matrix-preserving hard filters. Unit tests cover
flagged units, absent targets, empty criteria, user override recovery and finite
input validation. Rust fixtures check physical distances at the poles/date line,
vertical translation invariance, 90% window support and batched sample IDs.

The live diagnostic samples six predeclared Moon/Mars points, compares the old
slope-only default against the current builder on all pilot patches and writes
`research/pilots/database/matching-evaluation.json`. It records missingness,
score distributions, per-patch scores and timings. It does not establish analogue
truth, tune thresholds per site, or exclude countries/patches by name.

For a self-contained result-navigation export using the current default datasource:

```sh
npm run navigation-preview --prefix demo
```

This writes `research/pilots/database/result-navigation.html` with the real score
matrix compressed inside the file, plus a small JSON run report. It bundles the
same Svelte histogram/threshold/shortlist component; area focus is shown
as a cell address in the standalone export. The `/demo` app also moves the map and
marks the selected area's bounds and peak.
