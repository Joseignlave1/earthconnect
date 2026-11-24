# earthconnect

Earth Connect compares Moon and Mars sites with conditions on Earth. Run
`npm install` and `npm run dev`, then open the shared sky workspace at
[localhost:5173](http://localhost:5173/). The same workspace opens at `/find`.

## WebGPU renderer & data

Rendering uses raw WebGPU and WGSL with native-resolution **4× MSAA**, an
internal RGBA16F HDR target with bloom and ACES tone mapping, and reverse-Z
depth across the shared solar system scene. Run `npm run test:shader` to check
geometry, picking, terrain decoding, and score data.

- **4 km score grid:** MODIS sinusoidal grid (460 tiles of 278 × 278 uint16
  values, 31,881,940 Earth cells) scored on WebGPU via `EarthScore`, stored in
  `r16uint` texture arrays with peak-preserving max overviews and automatic
  upper-tail Magma color calibration.
- **Planetary terrain & virtual tiles:** GEBCO 2026 Earth elevation + coast
  distance (`rg32float`), Moon/Mars 4K ECT2 DEMs (`rg8unorm`), 8K global albedo
  textures, and regional 500 m Earth / 24K Moon & Mars ECVT1 virtual tile packs
  streamed on demand into GPU LRU texture pools. Source credits are in
  [public/textures/CREDITS.txt](public/textures/CREDITS.txt).
- **Lighting & inspection:** Sun position is computed locally from the load
  timestamp; Shift-hover (or Shift-click to lock) opens an optical loupe over
  Earth, Moon, and Mars. Device loss automatically rebuilds GPU resources.

## Shared sky workspace

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
