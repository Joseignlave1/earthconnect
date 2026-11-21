# earthconnect

Earth Connect compares Moon and Mars sites with conditions on Earth. Run
`npm install` and `npm run dev`, then open the shared sky workspace at
[localhost:5173](http://localhost:5173/). The same workspace opens at `/find`.

## Surface rendering lab

Open the standalone lab at [`/shader-test/`](http://localhost:5173/shader-test/).
This fullscreen WebGPU page
shows Earth analogue scores from a live Mars or Moon sample, a frozen snapshot,
or generated Perlin noise. It includes a plane/globe transition, orbit, zoom,
click inspection and score filters. `npm run test:shader` checks geometry,
picking and score data. The page is also included in `npm run build`.

Choose **Live planetary analogues** under **Score data**, select Mars or Moon,
enter latitude (−90 to 90) and longitude (−180 to 180) in degrees, and click
**Run analogue query**. The lab samples the point through Earth Index gRPC-Web
and applies the analogue measurement selection, tolerances, family weights
and coverage rules. **Effective criteria** shows the targets, tolerances
and weights used for the displayed map. Approximate temperature comparisons
and Earth-only preferences are labeled. Editing inputs leaves the old map
visible with its original query identified until a new result is ready.

The served lab loads about 183 MB of compressed Earth data into WebGPU at
startup, while displaying the selected snapshot. The status under **Score
data** shows when live queries are available. Queries use
`buildEarthQuery(criteria)` and `earth.dispatchQuery(query)` to score the
resident Earth grid on the renderer's WebGPU device. The served lab reads the
world data and rendering assets from a pinned CDN release. Use a WebGPU browser
on HTTPS or localhost. Live sampling contacts the Earth Index API, configurable
with `VITE_EARTH_INDEX_API_URL`.

Scores use a Magma color ramp, from dark violet through magenta and orange to
pale cream. HDR rendering is enabled by default: linear-light scene colors use
an RGBA16F target, with stronger score-dependent emission at the highest values.
Only radiance above 1 feeds a reduced-resolution bloom filter; an ACES fitted
curve and sRGB conversion produce the final image. Exposure and bloom are
adjustable, and HDR can be toggled for comparison. This is internal HDR with
tone-mapped output, so it works on ordinary displays. Surface, bloom,
presentation and cursor passes use WGSL and raw WebGPU,
without a rendering engine or WebGL fallback. Initialization is asynchronous;
device loss rebuilds GPU resources from the retained numeric data. Rendering
returns to idle after each requested update.

Rendering uses native-resolution **4× MSAA** at every display density. The
multisampled HDR color resolves once into a reusable scene texture. There is
no temporal reconstruction, jitter, motion-vector generation or history buffer.

Work is cached according to its inputs:

- Camera, viewport, morph, relief, terrain and score-data changes rebuild the
  surface and geographic annotation textures. With no-data limits enabled, they
  also refresh score coverage, as does the peak-preserving LOD toggle.
- Palette, score filters, opacity, contours, imagery and HDR changes rebuild
  surface shading while reusing geographic coordinates and coverage.
- Bloom extraction and blur run only after the scene changes, or when bloom is
  first enabled after a scene change. Bloom strength reuses that filtered image.
- Exposure, bloom strength and the no-data toggle update a cached tone-mapped
  display image. No-data limits are off by default. Enabling them builds the
  coverage mask if needed and reuses the current globe shading.
- Hover, selection and the space cursor only present that cached image with
  markers. A stationary pointer and idle view schedule no animation frames.

“Measure GPU passes” profiles orbit, hover, exposure and bloom-strength updates
at the current viewport with WebGPU timestamp queries. Each activity excludes
24 warmup frames and measures 60 samples. The orbit breakdown reports surface,
annotations, bloom, tone-map/boundary composition and presentation separately.
Cached stages report zero. CPU, compilation, queue waiting and display
presentation wait are excluded. Unsupported GPU timestamps are reported
explicitly rather than replaced with CPU timings.

Terrain lighting is baked into a geographic `r8unorm` texture with filtered
mipmaps. The sun direction is calculated locally from the navigation timestamp
using [USNO solar coordinates](https://aa.usno.navy.mil/faq/sun_approx) and
[Greenwich sidereal time](https://aa.usno.navy.mil/faq/GAST), then frozen until
reload. The panel shows the captured UTC time and subsolar point. No solar API
request, geolocation or runtime dependency is required, including offline.

Terrain gradients are computed once from the smoothed numeric elevation grid,
using physical east/north sampling distances along great circles and WGS84
metric radii. This avoids tiny longitude spacings amplifying quantization near
the poles. Water stays flat at sea level. Lighting combines the real sun with
50% ambient and a gentle 20% cartographic fill, preserving night-side relief;
the direct sun contributes up to 30%. Heatmap colors and emission remain unlit.

Orbit, zoom, morph, imagery, score/query changes and postprocessing reuse this
world-space light texture. Terrain exaggeration rebuilds illumination and its
mips while reusing the gradients; new terrain or device recovery rebuilds both.
The first/last rows of each lighting mip converge to one polar value. Projection,
coastlines and screen-width elevation contours still render at the current view.

Score tiles stay `r16uint` texture arrays with their exact native values and
max overviews. Numeric terrain stays `rg32float`; hardware anisotropic filtering
is used with `float32-filterable`, otherwise equivalent directional filtering
runs in the terrain shader without reducing height precision.

The surface lab defaults to **Live planetary analogues** (running Earth Score on
WebGPU from a Mars or Moon query) and also offers **Synthetic Perlin noise** in
the selector.

Score rendering separates dim context, bright matches, and filtering:

- **Context floor / opacity** draw a dim violet layer from raw scores, independent
  of highlight calibration. Context cannot produce HDR glow. Its defaults are
  a 0.10 floor and 40% opacity.
- **Highlight baseline / color focus** calibrate the bright upper tail. Defaults
  follow the exploratory settings: 0.31 baseline, top 22% focus, 76% highlight
  opacity. Bright candidates fade in continuously above the dim violet range.
- **Minimum score filter** hides both score layers below an inclusive threshold,
  leaving terrain and inspection available. It is off at zero. Changing it does
  not recalculate colors or emission. The UI reports the native cells passing
  the filter and disables “Go to best score” when none pass.

Auto contrast counts original equal-area 4 km cells in a worker histogram,
excluding missing values and preserving valid zeros. The default focus places
color anchors at the 78th, 97.8th, and 99.78th percentiles of cells strictly above
highlight baseline, plus the maximum. A minimum score span prevents repeated or
nearly identical quantiles from amplifying quantization noise. Colors describe
relative quality within that query's candidate population.

Calibration changes only with the dataset, highlight baseline, or focus control.
Context controls, the minimum filter, camera movement, globe/plane transitions,
and display LOD never change the scale. “Locked · compare queries” preserves
highlight colors, baseline, and glow calibration across dataset changes while
context and filtering remain adjustable. Auto mode recalibrates when unlocked;
Manual mode retains the upper numeric band and contrast controls. The legend
labels raw scores along the bright part of Magma; weak context has its own ramp.

Auto peak emission follows the upper color tail with a query-wide glow budget,
reducing whitening from broad plateaus and tied maxima. Exposure and bloom remain
manual. “Go to best score” centers and inspects the highest native cell in the
currently loaded dataset when it passes the minimum filter.

Faint solid red outlines mark valid-to-missing score boundaries, including polar
limits, the date-line gap and interior holes. Enable “No-data limits” to show
the outlines. The geographic annotation pass also caches a display-resolution
`r8unorm` validity mask while the toggle is on. The cached display composition
traces the mask at a fixed CSS-pixel width, with soft anti-aliased edges.
The line has no animated dot phase. Zero remains
valid; context, contrast and minimum-score filtering do not alter validity.
The line follows display-resolution score LOD, excludes the globe silhouette
and emits no HDR bloom. No global mask texture or backend request is needed.

All score sources use the production MODIS sinusoidal grid: nominal 4 km
cells (3999.822 m), 460 tiles of 278 × 278 uint16 values, 31,881,940 Earth
cells and 35,550,640 entries including missing padding (71.1 MB). Noise and
maximum overviews are generated in a cancellable Web Worker, then uploaded
into integer texture arrays. Display controls update style uniforms; contrast controls also recalibrate the small histogram, without regenerating score textures. Zoom far
enough to inspect individual cells; the inspector shows their native address.

NASA Blue Marble imagery provides the background; the heatmap opacity slider
blends the noise over it. Source and credit information is in
[the texture credits](public/textures/CREDITS.txt).

`npm run export:shader` creates `exports/earth-connect-surface-lab.html`, with
the image, terrain, styles, scripts and noise worker embedded.
The file opens from disk and supports synthetic noise offline.
Use the served `/shader-test/` page for live planetary queries and their Earth
dataset.

The former public demo URL on Fly now links to the full Earth Connect workspace
on Cloudflare Pages.

The default basemap is grey shaded relief on a WGS84 ellipsoid, with a blue
coastline and adjustable elevation contours. The blue line uses a signed distance
to the connected ocean boundary, keeping lowlands and enclosed negative basins
free of false coastline highlights. GEBCO 2026 supplies a
4096 × 2048 visual elevation overview (~10 km at the equator); the score grid
remains 4 km. Terrain exaggeration changes geometry and shading, while contour
spacing and inspected elevations remain in unexaggerated metres. Oceans stay
geometrically at sea level. Satellite imagery is an optional background.
Contours use mild, fixed source smoothing plus normal minification filtering.
Major contours (500 m by default) stay visible; half- and quarter-interval
families fade in together as the global camera scale gets closer. Local slope
controls line antialiasing rather than visibility, avoiding gaps in flat areas.
Ocean depth contours use subdued blue with their own major interval (1000 m
by default) and the same nested zoom transitions. Zero metres stays reserved
for the geographic coastline. Bathymetry is drawn on the flat ocean surface
beneath the score overlay.
Hillshade uses its own detail level and the coastline keeps sharper sampling.
The flat ocean does not receive seabed hillshade. Elevation filtering retains
float32 precision, using hardware filtering or shader filtering. Native 4 km
scores and raw height inspection are unaffected by this display filtering.
Globe coordinates are recovered from interpolated geodetic directions, with
wrapped longitude gradients and anisotropic filtering when available. Pole
vertices and the polar rows of each display level share averaged values to
prevent radial wedges. Picking follows the same coordinate mapping throughout
the plane-to-globe transition.
The overview and coarse global terrain mesh are for regional visualization,
not detailed local terrain analysis. `scripts/prepare-terrain.py` reproduces
the lossless packed elevation PNG and compressed numeric runtime data. Runtime
loading avoids canvas readback, which privacy browsers can perturb as part of
fingerprinting protection. Source credits accompany the textures.

## Standalone shared sky workspace

The full shared sky workspace with the WGSL editor is hosted at
[earth-connect.pages.dev](https://earth-connect.pages.dev/). Build and publish
it from the repository root with:

```sh
npm run build:pages
npx wrangler@4.129.0 pages deploy dist-pages --project-name earth-connect --branch main
```

The Pages build includes the startup HTML, workspace scripts, workers, and
cross-origin isolation headers. Both application builds use the asset CDN at
runtime and can build from a fresh checkout without the world data.
Planetary sampling uses the Earth Index API.

The workspace at `/` and `/find` loads its runtime files from the Earth Connect
asset CDN. The startup screen tracks the bytes still
needed for the current release. Verified files are saved in the browser's
IndexedDB and reused by content hash. On subsequent visits, the workspace
revalidates the release manifest with its ETag. If the CDN is unavailable,
the workspace can open using a previously saved complete release.

Add `?refreshCache=1` to the workspace URL or the CDN-backed HTML export to download
all release files again. Shader drafts and layout preferences stay saved.
Remove the query parameter to resume normal cache reuse.

`npm run export:workspace` creates `exports/earth-connect-workspace.html`.
Open it in a browser with WebGPU, or serve it over HTTPS. The HTML contains
the workspace UI and downloads its data from the CDN on first use. Use
`npm run export:workspace -- --offline` to create the fully embedded
`exports/earth-connect-workspace-offline.html` for opening without a
connection. The embedded export is about 285 MiB.

The offline exports, `npm run build:compute`, and the data-dependent test
commands fetch their required assets from the pinned CDN release on first use.
Run `npm run fetch:assets` to prepare all assets in advance, or pass
`-- world` or `-- render` to fetch only that group. The command verifies
file sizes and SHA-256 hashes for both cached and downloaded assets. If it
reports a local integrity error, move the named file aside and rerun it.
Once the local files are verified, these builds and tests can reuse them
without downloading them again.

The globe opens with a live WebGPU score map from the same Hello, World! shader
used in the editable workspace. The scene, score maps, and planetary coordinate
picker use the local data.
**Copy sampled WGSL values** contacts the Earth Index API when requested.
To use another compatible sampling server, set `VITE_EARTH_INDEX_API_URL`
when running either export command.

## WGSL editor

A new workspace opens [hello-world.wgsl](src/workspace/shaders/hello-world.wgsl)
and its preview. Rotate the globe to follow the greeting around Earth, or edit
the glyphs and watch the preview update. In an existing workspace, use
**More Actions** and **Reopen WGSL File…** to find the example.

Right-click a WGSL tab and choose **Open Preview to the Side** to open its
interactive map. Drag preview tabs or use the editor group controls to place
multiple maps side by side. Each preview follows its own shader, and valid
edits update its map automatically. Press Ctrl+Enter or Cmd+Enter in a source
editor or its preview to run that shader immediately. From a preview’s
**More Actions** menu, choose **Reveal Shader Source** to return to its WGSL.

The Apollo 11, Jezero, and Tycho tabs each contain a separate comparison. Choose **New WGSL Shader** from
**More Actions** to write your own independent comparison. **New WGSL Helper**
adds a file to the original score module so that `score.wgsl` can call its functions.
The **More Actions** menu also offers rename, reopen, and Vim keybindings.
Files and the Vim setting are saved in the browser.

Every Earth cell exposes `cell.latitude_deg` and `cell.longitude_deg` in decimal
degrees at its center, positive north and east. It also exposes `cell.grid_x`
and `cell.grid_y` as zero-based integer column and row indices in the full
10,008 × 5,004 sinusoidal raster. Columns increase rightward and rows increase
southward. Geographic coordinates are `NO_DATA` in raster padding.

WGSL can sample fixed coordinates with `moon_point(latitude, longitude)` or
`mars_point(latitude, longitude)`. Use degrees with east-positive longitude.
Coordinates must be statically resolvable without using values from an Earth
cell. Bind the sampled point to a `let` inside `score` or a helper.

```wgsl
fn score(cell: EarthCell) -> f32 {
  let lunar = moon_point(18.65, -133.8);
  let slope = lunar.terrain.slope_1km_deg;
  if (!has_data(slope) || !has_data(cell.slope_mean_deg)) {
    return NO_SCORE;
  }
  let difference = (cell.slope_mean_deg - slope) / 3.0;
  return exp(-0.5 * difference * difference);
}
```

Pick a point on Moon or Mars. Select the coordinates in the preview footer
to copy signed latitude and longitude in decimal degrees. Right-click the
point or its coordinates, or use the adjacent **Point actions** button, to
copy a WGSL expression, insert it into the preview’s shader, or copy sampled
fields as a WGSL constructor. Selecting a point leaves the current shader and
Earth scores unchanged.

The VS Code editor and WGSL analyzer load with the workspace. The analyzer runs in a
local WebAssembly worker and needs cross-origin isolation. Vite dev and preview
provide the headers, and the build includes a `_headers` file for static hosts
that use it. On other hosts, serve the app and worker from the same origin with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: credentialless`. Without isolation, the editor
provides local highlighting, completion, hover and WebGPU compiler diagnostics.

## Earth Score compute module

[Earth Score](src/earth-score/README.md) provides a standalone WebGPU API for
full-world analogue queries against the frozen Earth Index dataset. Build it
with `npm run build:compute`. The module accepts a caller-owned GPU device and
exposes addressable score buffers for downstream consumers.

The [Earth Index client files](vendor/earth-index/README.md) document the
upstream sampling contract, dataset provenance, and licenses.
