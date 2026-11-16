# Browser performance baseline

These scripts instrument a separate, production-minified build under `/tmp`.
They do not change the application's renderer, the deployed image, the API, or
the database. Run against the existing Docker API on port 8080.

```sh
cd demo
node --import tsx scripts/profile-pipeline.ts --prepare-only
node --import tsx scripts/browser-profile.ts
```

In another terminal, install an optional browser driver outside this project and
point it at a local Chromium executable:

```sh
npm install --prefix /tmp/earth-browser-profile/driver playwright-core@1.58.2 --no-audit --no-fund
PLAYWRIGHT_MODULE=/tmp/earth-browser-profile/driver/node_modules/playwright-core/index.mjs \
CHROMIUM_PATH=/path/to/chrome \
node --import tsx scripts/run-browser-profile.ts
```

Repeat the driver with `--controls-only` to record `browser-controls.json`:
first hide the coverage clipping layer while retaining all scores, then clear
scores as well, repeating the same panning sequence three times per state.
This changes only the temporary browser's state and isolates
coverage clipping from score rendering/source refresh and the base map.
The current global demo has no coverage layer; this mode therefore compares the
single score layer against base imagery only. Historical mask measurements are
retained in `research/world-4km/pipeline-profile/browser-profile/`.

The default is a headed browser using ANGLE/OpenGL, so rendering uses the same
hardware driver as a normal desktop window. `PROFILE_HEADLESS=1` is available;
inspect the recorded renderer because software rendering changes interaction
results. `BROWSER_PROFILE_DIR` changes the temporary output directory for both
scripts. Close the build server after the driver finishes.

`browser.json` records the environment, raw RPC stages, instrumented scenarios,
and control runs against the uninstrumented deployed application. There is one
warmup and three measured query runs. All 460 score buffers are SHA-256 checked
against the independently validated world-query reference outside timed work.

The same 12-update pan and zoom sequences run with coverage only and with world
scores, plus detail panning and threshold changes. Overview interactions start
at four times the world-fit zoom so map constraints permit actual movement.
Each update waits for an animation opportunity. Delivered update rate and frame
gaps describe responsiveness; they are not GPU presentation/FPS measurements.

Timing categories include uint16 decoding, summary, histogram, overview
generation, connected areas, canvas drawing/PNG encoding, projection
registration, layer/source construction, coverage clipping, reprojection, and
OpenLayers rendering. `total_ms` includes instrumented child calls; `self_ms`
excludes them. Parent/child totals must not be added together. Async RPC elapsed
time is separate from synchronous call-stack timing.
The current renderer also records atlas construction and GPU initialization,
texture-cache access and rendering. Synthetic GPU tests cover integer values,
missing data, sparse regions, odd-grid LOD footprints, multiple texture banks,
threshold reuse, cache eviction and context restoration. World probes compare
GPU pixels to the CPU reference outside the timed query/interaction scenarios.
The high-frequency native `drawImage` API is deliberately not wrapped; timing
millions of individual calls would perturb raster reprojection measurements.

Long tasks identify main-thread stalls. Map render hooks measure synchronous
JavaScript/canvas calls, excluding the compositor and GPU completion. Heap
snapshots use Chromium's approximate `performance.memory` counters and are not
a retained-memory/leak analysis. Settling windows include image loading and
subsequent map frames; external basemap fetches are not controlled.

The query status timer stops before the reactive result controls and eventual
browser painting. The profiling driver's action timer includes the Svelte
update flush; neither should be mistaken for complete GPU presentation time.
No WASM speedup is assumed or measured by this baseline.
