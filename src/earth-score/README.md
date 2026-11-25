# Earth Score

Earth Score loads the frozen 4 km Earth Index measurements and scores all 35,550,640 stored addresses on a caller-owned WebGPU device. The module exposes GPU buffers suitable for a subsequent compute or render pass.

## Build and use

Use Node.js 24 or later. Run `npm ci` and `npm run build:compute` from the repository root. The standalone ES module, TypeScript declarations, decoder worker, and world data land in `exports/earth-score/`. Serve that directory over HTTPS or loopback HTTP and keep its relative paths intact. The regular application build uses its own output directory.

```ts
import { DEFAULT_SCORE_SHADER, loadEarthScore, SCORE_MISSING } from './earth-score/earth-score.mjs'

const earth = await loadEarthScore(device, {
  manifestUrl: new URL('./earth-score/world/manifest.json', import.meta.url),
  // signal: controller.signal,
})

try {
  await earth.dispatchShader(DEFAULT_SCORE_SHADER)
  const point = earth.layout.addressAt(-66.3, -24.2)
  if (point) {
    const { scores } = earth.chunks[point.chunk]
    // Bind scores as array<u32>. scores[point.row] is SCORE_MISSING or 0..65534.
    console.log(scores, point.row, SCORE_MISSING)
  }
} finally {
  earth.destroy()
}
```

Use the default WebGPU device limits for the bundled world. Loading allocates approximately 534 MB of GPU buffers plus transient decode, upload and driver memory. It uses four dedicated workers, a maximum of eight admitted fetch/decode jobs, eager uploads in completion order, and a queue fence every eight uploads. Each worker transfers its compact result. The caller's device coordinates the queue writes. This worker setup works without cross-origin isolation.

A dispatch submits synchronously and returns a promise for GPU completion. Submit a consumer pass on the same queue after that call and before dispatching another query if the consumer needs that result. Every dispatch overwrites the shared score buffers. Consume the results before destroying the instance. Destroy is idempotent and releases the module's buffers while the device remains available to its caller. Pass an `AbortSignal` to cancel a load and release partially allocated resources.

## Query with a Mars or Moon sample

Import `buildEarthQuery` from `src/earth-score/index.ts` in the app or from `earth-score.mjs` in the standalone build. Pass the current explorer `Criterion[]` after applying the coordinate sample and any user edits.

```ts
const sample = await samplePoint(body, latitude, longitude, signal)
const sampledCriteria = applySample(criteria, body, sample)
const query = buildEarthQuery(sampledCriteria)
await earth.dispatchShader(earth.queryShaderSource(query), signal)
// Use earth.chunks[i].scores and earth.layout for the map.
```

`buildEarthQuery` returns an immutable `EarthQuery`. Build a fresh query when the sample or settings change. Enabled criteria use their supplied target, tolerance, weight and group. Numeric terms use squared normalized distance. Categorical terms use a loss of 0 for a match and 25 for a mismatch. Members of a family form a weighted mean of their losses. Each family's largest member weight determines its share of the final loss. The shader computes `exp(-0.5 * min(loss, 1000))` and rounds to the nearest score code, with ties to even.

Either hard bound applies even when its criterion is disabled. Hard bounds are inclusive. Only active criteria contribute required values and any explicitly supplied coverage fields. Coverage must reach 0.9. Missing inputs or insufficient coverage produce 65,535 and take precedence over hard filter rejection, which produces 0. At least one enabled criterion with a finite target is required.

All 19 measurement fields in the bundled world are available, including lithology, climate, permafrost and the full thermal set. Coverage columns can be referenced through `qualityKey`. The builder validates active field names and numeric parameters. Dispatch checks that the loaded world has the fields referenced by the query. `dispatchQuery` uses the same score buffer layout and queue ordering as `dispatch`.

## Custom WGSL scores

Pass a single `fn score(cell: EarthCell) -> f32` to `earth.dispatchShader(source, signal)`.
The custom function runs once per indexed Earth cell using the decoded physical
measurements and coverage fractions in `SCORE_SHADER_FIELDS`. These values
are float32, including lithology's integer category codes. The editor's
`SCORE_SHADER_PRELUDE` describes the fields and defines the helpers.

```wgsl
fn score(cell: EarthCell) -> f32 {
  if (!has_data(cell.slope_mean_deg) || !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }
  return cell.bare_fraction * exp(-cell.slope_mean_deg / 8.0);
}
```

The data sentinel `NO_DATA` identifies missing measurements. Use `has_data`
before doing arithmetic with a required field, and return `NO_SCORE` for cells
that should have no score. Finite nonnegative returns are clamped to 0–1.
Negative or nonfinite returns yield the missing score code. Set any coverage
thresholds in the function.

The browser compiles the function as WGSL. The engine supplies the entry point
and GPU bindings. The module can define regular WGSL functions, constants,
loops, and arrays. Inputs are supplied through `EarthCell`.
`ScoreShaderError.diagnostics`
includes browser compiler errors and input-scope errors, with one-based input
coordinates when available. Dispatch writes into the same chunk buffers as
the baseline and criteria queries. Callers should serialize dispatch and
publication when copying those buffers into a renderer. An aborted dispatch
can leave scratch scores from batches that already ran.

`earth.queryShaderSource(query)` generates an editable module from a
`buildEarthQuery` plan. It uses the same coverage rules, inclusive encoded
bounds, categorical targets, and normalized family weights as
`dispatchQuery`. The formula works in source measurement codes to preserve
narrow numeric tolerances.

## Earth cell coordinates

| `EarthCell` field | WGSL type | Meaning                                                                                   |
| ----------------- | --------- | ----------------------------------------------------------------------------------------- |
| `latitude_deg`    | `f32`     | Latitude at the cell center in decimal degrees, positive north                            |
| `longitude_deg`   | `f32`     | Longitude at the cell center in decimal degrees, positive east                            |
| `grid_x`          | `u32`     | Zero-based column in the full 10,008-column MODIS sinusoidal raster, increasing rightward |
| `grid_y`          | `u32`     | Zero-based row in the full 5,004-row raster, increasing southward                         |

Grid indices retain their positions across missing tiles and dispatch chunks. In
padding outside the curved Earth footprint, both geographic fields are
`NO_DATA` and grid indices remain available. Use `has_data(cell.latitude_deg)`
or `has_data(cell.longitude_deg)` before using geographic coordinates. To get a
tile-local column or row, use the corresponding grid index modulo `278u`.

```wgsl
fn score(cell: EarthCell) -> f32 {
  if (!has_data(cell.latitude_deg)) {
    return NO_SCORE;
  }
  return 1.0 - smoothstep(15.0, 30.0, abs(cell.latitude_deg));
}
```

## Planetary point inputs

The workspace resolves `moon_point(latitude, longitude)` and
`mars_point(latitude, longitude)` before running the score. Arguments are
fixed coordinates in degrees, with east-positive longitude. The intrinsic
returns `MoonPoint` or `MarsPoint`. Each result has
`latitude_deg`, `longitude_deg`, `sampled`, `partial`, and common
`terrain.relief_5km_m`, `terrain.relief_21km_m`,
`terrain.tpi_5km_m`, `terrain.tpi_21km_m`, `terrain.slope_1km_deg`
and `terrain.elevation_m` fields. Each body uses its own elevation reference.

```wgsl
fn score(cell: EarthCell) -> f32 {
  let point = moon_point(-8.97, 15.50);
  let relief = point.terrain.relief_5km_m;
  if (!has_data(relief) || !has_data(cell.relief_5km_m)) {
    return NO_SCORE;
  }
  let difference = (cell.relief_5km_m - relief) / 300.0;
  return exp(-0.5 * difference * difference);
}
```

Moon temperatures such as `temperature.noon_k` are Diviner bolometric
surface temperatures in kelvin. Mars temperatures such as
`temperature.mean_c` are KRC modeled annual surface temperatures in degrees
Celsius. Both temperature ranges are differences with no Celsius offset.
The declared structs document the remaining fields and units, including
lunar polar visibility, composition and radar, and Martian albedo, thermal
inertia and TES modeled mineral fractions.

A point can be sampled successfully and still have missing fields. Use
`has_data` before arithmetic. Fields come from finite, available numerical
measurements with a supported unit. Structured features, text, and raw layers
with unresolved scales are outside the shader interface. A missing field
stays missing even when other sources returned values for the same point.

For a separate caller, the optional third argument to `dispatchShader`
carries `{ samples, calls }`. A call identifies its body, full original
UTF-16 source span (`offset` and `length`), and index into `samples`
(`slot`). Supply these locations from the WGSL analysis step.
`freezePlanetaryShaderPoint(sample)` returns a typed WGSL constructor
expression using the same canonical coordinates and measurements as the
read-only GPU input buffer.

## Layout and scoring

- `layout.sites` follows the frozen database's region order. Each site contributes exactly 278 × 278 row-major pixels, including padding. `addressOf(siteKey, pixel)` and `locate(address)` return the global address, chunk index and chunk-local row. A padding address has a null geographic center. `addressAt(longitude, latitude)` projects onto the same MODIS sinusoidal grid in degrees.
- Each `chunks[i].scores` has one little-endian uint32 per row and supports `STORAGE` and `COPY_SRC`. Scores 0–65,534 map linearly to 0–1. Code 65,535 marks missingness. Chunk order and `firstAddress` are fixed by the manifest.
- Each `chunks[i].packedInput` contains the original uint16 source codes in lossless 128-address blocks. `sourceCodeWGSL` provides random access when a caller binds the storage array as `packedInput`. The four u32 header words are magic, block size, rows and field count. The field-major descriptors store an absolute payload word offset and `base | width << 16 | missingMode << 21`. Source code 0 is missing. For valid codes, `lookup[fieldIndex * 65536 + code]` holds the exact float32 bits. Field order, names and encodings are available through `fields`.
- The baseline uses nine analogue features in five weighted families. It requires both terrain and land-cover coverage at or above 0.9 and valid codes for all nine features. Callers can override a feature's target, tolerance or weight. Tolerances and weights must be positive finite float32 values. The shader applies the original family normalization, squared-distance penalties and exponential score quantization.

## Assets and verification

The frozen dataset consists of 36 independent Zstd-9 frames (183,395,621 bytes total, largest under 12 MiB). The build retrieves missing frames from the pinned CDN release, verifies their size and SHA-256, and copies them byte-for-byte. Run `npm run fetch:assets -- world` to prepare the frames in advance. The runtime checks each frame's size, Zstd integrity and packed header. The verification fixtures additionally pin every frame, packed chunk and original code block with SHA-256. The bundled dataset credits travel with the world files in [CREDITS.md](../../compute-data/world/CREDITS.md).

```sh
npm run test:compute
npx playwright-core install chromium
npm run test:compute:gpu
```

The GPU test requires a hardware WebGPU adapter and runs the standalone production build over loopback. Set `EARTH_SCORE_CHROMIUM` to an installed Chromium executable or `EARTH_SCORE_CHROMIUM_ARGS` to a JSON array of launch flags when the default launcher needs adjustment for your GPU. The suite compares the baseline and a +0.25 slope-target query against independent NumPy references at all addresses. It checks a criteria query and an impossible hard filter against a double-precision reference in a real world chunk. It also checks exact baseline restoration, source-code reconstruction, resource cleanup and failed loads. [Reference provenance](../../tests/earth-score/README.md).

The shipped decoder binary uses reference Zstd 1.5.7. To rebuild it, install Rust with the `wasm32-unknown-unknown` target and Clang, then run `bash tools/earth-score-decoder/build.sh`. The included Cargo lockfile pins the dependencies. Normal npm builds use the checked-in binary. The decoder was built with Rust 1.97.1.
