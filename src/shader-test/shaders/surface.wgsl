override TILE_SIZE: u32 = 256u;
override BANK_SIZE: u32 = 256u;
override HIGHLIGHT_START: f32 = 0.22;
override HIGHLIGHT_FULL: f32 = 0.32;
override VIRTUAL_TILE_SIZE: f32 = 512.0;
override VIRTUAL_TILE_INNER: f32 = 508.0;
override VIRTUAL_TILE_GUTTER: f32 = 2.0;
override EARTH_TILE_BASE_COLS: f32 = 32.0;
override EARTH_TILE_BASE_ROWS: f32 = 16.0;

// Explicit 16-byte-aligned uniform layout shared with the CPU writer.
struct SurfaceUniforms {
  camera: mat4x4f,
  anchors: vec4f,
  cutoff: f32,
  contrast: f32,
  topRange: f32,
  emission: f32,
  contextFloor: f32,
  contextOpacity: f32,
  minimumScore: f32,
  highlightOpacity: f32,
  morph: f32,
  relief: f32,
  bestScore: f32,
  scoreMipCount: f32,
  landContourInterval: f32,
  depthContourInterval: f32,
  contourOpacity: f32,
  pixelRatio: f32,
  contourDetail: vec2f,
  _padDetail: f32,
  hdr: f32,
  autoPalette: f32,
  showGrid: f32,
  preservePeaks: f32,
  showImagery: f32,
  terrainReady: f32,
  nativeDepth: f32,
  earthReady: f32,
  earthTilesReady: f32,
  southPoleHeight: f32,
  northPoleHeight: f32,
  meshRelief: f32,
  _padPoles: f32,
  sunDirection: vec3f,
  _padSun: f32,
  eyePos: vec3f,
  _padEye: f32,
  camRight: vec3f,
  tanHalfFovX: f32,
  camUp: vec3f,
  tanHalfFovY: f32,
  camBack: vec3f,
  centerDepth: f32,
  centerNdc: vec2f,
  invViewportPx: vec2f,
}

@group(0) @binding(0) var<uniform> u: SurfaceUniforms;
@group(0) @binding(1) var lookup: texture_2d<u32>;
@group(0) @binding(2) var earth: texture_2d<f32>;
@group(0) @binding(3) var terrain: texture_2d<f32>;
@group(0) @binding(4) var geographicSampler: sampler;
@group(0) @binding(5) var terrainNormals: texture_2d<f32>;

const PI: f32 = 3.141592653589793;

fn elevation(uv: vec2f, demMip: f32) -> f32 {
  if (uv.y <= 0.) { return u.southPoleHeight; }
  if (uv.y >= 1.) { return u.northPoleHeight; }
  if (u.terrainReady < .5) { return 0.; }
  let maxLod = max(0., f32(textureNumLevels(terrain)) - 1.);
  return terrainLevel(vec2f(uv.x, 1. - clamp(uv.y, 0., 1.)), clamp(demMip, 0., maxLod)).r;
}

struct EarthPatchIn {
  @location(0) local: vec3f,
  @location(1) patchOrigin: vec2f,
  @location(2) patchSize: vec2f,
  @location(3) demMip: f32,
  @location(4) skirtDepth: f32,
}

struct SurfaceVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) geodetic: vec3f,
  @location(2) worldPos: vec3f,
}

fn computeSurfaceVertex(input: EarthPatchIn) -> SurfaceVertex {
  let uv = input.patchOrigin + input.local.xy * input.patchSize;
  let lon = (uv.x - .5) * 2. * PI;
  let lat = (uv.y - .5) * PI;
  let skirt = input.local.z * input.skirtDepth;
  let height = max(0., elevation(uv, input.demMip)) / 6378137. * u.meshRelief;
  let e2 = .0066943799901413165;
  let n = inverseSqrt(1. - e2 * sin(lat) * sin(lat));
  var normal = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  if (uv.y <= 0. || uv.y >= 1.) { normal = vec3f(0., sign(lat), 0.); }
  let baseSphere = vec3f(n * normal.x, n * (1. - e2) * normal.y, n * normal.z);
  let sphere = baseSphere + normal * (height - skirt);
  let plane = vec3f(equalEarth(lon, lat), height - skirt);
  let position = vec4f(mix(plane, sphere, u.morph), 1.);
  var clip = u.camera * position;
  // The standalone lab uses OpenGL depth. The solar scene uses native reverse-Z.
  if (u.nativeDepth < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SurfaceVertex(clip, uv, normal, position.xyz);
}

@vertex fn surfaceVertex(input: EarthPatchIn) -> SurfaceVertex {
  return computeSurfaceVertex(input);
}

struct AnnotationVertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(perspective, sample) uv: vec2f,
  @location(1) @interpolate(perspective, sample) geodetic: vec3f,
}

@vertex fn annotationVertex(input: EarthPatchIn) -> AnnotationVertex {
  let v = computeSurfaceVertex(input);
  return AnnotationVertex(v.position, v.uv, v.geodetic);
}

struct SmallSurfaceVertex {
  @builtin(position) position: vec4f,
}

@vertex fn smallSurfaceVertex(@builtin(vertex_index) vertex: u32) -> SmallSurfaceVertex {
  let depth = u.centerDepth;
  if (depth <= 1.001) { return SmallSurfaceVertex(vec4f(2., 2., 0., 1.)); }
  let clipCenter = u.camera * vec4f(0., 0., 0., 1.);
  if (clipCenter.w <= 0.) { return SmallSurfaceVertex(vec4f(2., 2., 0., 1.)); }
  let center = clipCenter.xy / clipCenter.w;
  let cx = -dot(u.eyePos, u.camRight);
  let cy = -dot(u.eyePos, u.camUp);
  let d2x = cx * cx + depth * depth;
  let d2y = cy * cy + depth * depth;
  let denom = max(depth * depth - 1., 1e-6);
  let extX = (sqrt(max(0., d2x - 1.)) + abs(cx) / depth) / (denom * u.tanHalfFovX);
  let extY = (sqrt(max(0., d2y - 1.)) + abs(cy) / depth) / (denom * u.tanHalfFovY);
  let pad = 3. * u.invViewportPx;
  let corner = center + unitQuadCorner(vertex) * (vec2f(extX, extY) + pad);
  return SmallSurfaceVertex(vec4f(corner, 0., 1.));
}

struct SmallSurfaceOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

fn wrappedGradient(delta: vec2f) -> vec2f {
  return vec2f(delta.x - floor(delta.x + .5), delta.y);
}

// 5-tap Catmull-Rom bicubic height filter for smooth zoomed-in ridges, valleys, and contours.
fn sampleHeightBicubic(sampleUv: vec2f) -> f32 {
  let size = vec2f(textureDimensions(terrain));
  let invSize = 1. / size;
  let pos = sampleUv * size - .5;
  let f = fract(pos);
  let tc1 = floor(pos) + .5;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = f2 - .5 * (f3 + f);
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.;
  let w3 = .5 * (f3 - f2);
  let w2 = 1. - w0 - w1 - w3;
  let w12 = w1 + w2;
  let tc0 = (tc1 - 1.) * invSize;
  let tc3 = (tc1 + 2.) * invSize;
  let tc12 = (tc1 + w2 / w12) * invSize;
  let s0 = terrainLevel(vec2f(tc12.x, tc0.y), 0.).r;
  let s1 = terrainLevel(vec2f(tc0.x, tc12.y), 0.).r;
  let s2 = terrainLevel(vec2f(tc12.x, tc12.y), 0.).r;
  let s3 = terrainLevel(vec2f(tc3.x, tc12.y), 0.).r;
  let s4 = terrainLevel(vec2f(tc12.x, tc3.y), 0.).r;
  let sum = s0 * (w12.x * w0.y) + s1 * (w0.x * w12.y) + s2 * (w12.x * w12.y)
    + s3 * (w3.x * w12.y) + s4 * (w12.x * w3.y);
  let total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return sum / total;
}

fn surfaceUvFrom(rawUv: vec2f, rawGeodetic: vec3f) -> vec2f {
  let geodetic = normalize(rawGeodetic);
  var longitude = 0.;
  if (length(geodetic.xz) > .00000001) { longitude = atan2(geodetic.x, geodetic.z); }
  let globeUv = vec2f(longitude / (2. * PI) + .5, atan2(geodetic.y, length(geodetic.xz)) / PI + .5);
  var uv = rawUv + wrappedGradient(globeUv - rawUv) * u.morph;
  uv.x = fract(uv.x);
  return uv;
}

fn surfaceUv(input: SurfaceVertex) -> vec2f {
  return surfaceUvFrom(input.uv, input.geodetic);
}

fn loadScore(address: vec2f, lod: i32) -> u32 {
  if (address.x < 0. || address.x >= 36. || address.y < 0. || address.y >= 18.) { return 65535u; }
  let cell = clamp(address, vec2f(0.), vec2f(35.99999, 17.99999));
  let slot = textureLoad(lookup, vec2i(floor(cell)), 0).r;
  if (slot == 65535u) { return 65535u; }
  let bank = slot / BANK_SIZE;
  let layer = i32(slot % BANK_SIZE);
  let size = TILE_SIZE >> u32(lod);
  let pixel = clamp(vec2i(fract(cell) * f32(size)), vec2i(0), vec2i(i32(size) - 1));
  return loadScoreBank(bank, pixel, layer, lod);
}

struct ScoreSample {
  score: f32,
  valid: f32,
  density: f32,
}

fn loadScoreBilinear(address: vec2f, lod: i32, cellStep: vec2f) -> ScoreSample {
  let sizeI = i32(TILE_SIZE >> u32(lod));
  let size = f32(sizeI);
  let pos = address * size - .5;
  let baseF = floor(pos);
  let base = vec2i(baseF);
  let rawF = pos - baseF;
  // At lod 0 when zoomed in (cell > 1 screen pixel), sharpen interpolation to a
  // 1-pixel anti-aliased edge so individual 4 km cells stay crisp and readable.
  let f = select(rawF, clamp((rawF - .5) / max(cellStep, vec2f(.04)) + .5, vec2f(0.), vec2f(1.)), lod == 0);
  let tile = vec2i(floor(baseF / size));
  let local = base - tile * sizeI;
  var codes = vec4u(65535u);
  if (local.x >= 0 && local.x < sizeI - 1 && local.y >= 0 && local.y < sizeI - 1) {
    // Fast path (>99% of pixels): one tile lookup + immediate early-out over oceans.
    if (tile.x < 0 || tile.x >= 36 || tile.y < 0 || tile.y >= 18) { return ScoreSample(0., 0., 0.); }
    let slot = textureLoad(lookup, tile, 0).r;
    if (slot == 65535u) { return ScoreSample(0., 0., 0.); }
    let bank = slot / BANK_SIZE;
    let layer = i32(slot % BANK_SIZE);
    codes = loadScoreQuadBank(bank, local, layer, lod);
  } else {
    let invSize = 1. / size;
    codes = vec4u(
      loadScore((baseF + vec2f(.5, .5)) * invSize, lod),
      loadScore((baseF + vec2f(1.5, .5)) * invSize, lod),
      loadScore((baseF + vec2f(.5, 1.5)) * invSize, lod),
      loadScore((baseF + vec2f(1.5, 1.5)) * invSize, lod),
    );
  }
  let wx0 = 1. - f.x;
  let wy0 = 1. - f.y;
  var wSum = 0.;
  var vSum = 0.;
  var sMax = 0.;
  if (codes.x != 65535u) { let w = wx0 * wy0; let s = f32(codes.x) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax, s); }
  if (codes.y != 65535u) { let w = f.x * wy0; let s = f32(codes.y) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax, s); }
  if (codes.z != 65535u) { let w = wx0 * f.y; let s = f32(codes.z) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax, s); }
  if (codes.w != 65535u) { let w = f.x * f.y; let s = f32(codes.w) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax, s); }
  if (wSum <= .001) { return ScoreSample(0., 0., 0.); }
  let avg = vSum / wSum;
  let density = clamp(avg / max(sMax, 1e-4), 0., 1.);
  return ScoreSample(avg, wSum, density);
}

struct ScorePointSample {
  score: f32,
  valid: f32,
}

struct ScoreProjection {
  address: vec2f,
  dx: vec2f,
  dy: vec2f,
}

fn projectScoreUv(uv: vec2f, uvDx: vec2f, uvDy: vec2f) -> ScoreProjection {
  let latitude = (uv.y - .5) * PI;
  let cosLat = cos(latitude);
  let sinTerm = (uv.x - .5) * PI * sin(latitude);
  let address = vec2f(18. + (uv.x - .5) * 36. * cosLat, (1. - uv.y) * 18.);
  let dx = vec2f(36. * (cosLat * uvDx.x - sinTerm * uvDx.y), -18. * uvDx.y);
  let dy = vec2f(36. * (cosLat * uvDy.x - sinTerm * uvDy.y), -18. * uvDy.y);
  return ScoreProjection(address, dx, dy);
}

fn sampleScoreTrilinear(address: vec2f, lodF: f32, cellStep: vec2f) -> ScoreSample {
  let maxLod = i32(u.scoreMipCount) - 1;
  let clamped = clamp(lodF, 0., f32(maxLod));
  let lo = i32(floor(clamped));
  let hi = min(lo + 1, maxLod);
  let t = clamped - f32(lo);
  let a = loadScoreBilinear(address, lo, cellStep);
  if (hi == lo || t <= .001) { return a; }
  let b = loadScoreBilinear(address, hi, cellStep);
  if (a.valid <= .001 && b.valid <= .001) { return ScoreSample(0., 0., 0.); }
  if (a.valid <= .001) { return ScoreSample(b.score, b.valid * t, b.density); }
  if (b.valid <= .001) { return ScoreSample(a.score, a.valid * (1. - t), a.density); }
  return ScoreSample(mix(a.score, b.score, t), mix(a.valid, b.valid, t), mix(a.density, b.density, t));
}

fn loadScorePointTrilinear(address: vec2f, lodF: f32) -> ScorePointSample {
  let maxLod = i32(u.scoreMipCount) - 1;
  let clamped = clamp(lodF, 0., f32(maxLod));
  let lo = i32(floor(clamped));
  let hi = min(lo + 1, maxLod);
  let t = clamped - f32(lo);
  let cLo = loadScore(address, lo);
  if (hi == lo || t <= .001) {
    if (cLo == 65535u) { return ScorePointSample(0., 0.); }
    return ScorePointSample(f32(cLo) / 65534., 1.);
  }
  let cHi = loadScore(address, hi);
  let vLo = select(0., 1., cLo != 65535u);
  let vHi = select(0., 1., cHi != 65535u);
  let valid = mix(vLo, vHi, t);
  if (valid <= .001) { return ScorePointSample(0., 0.); }
  let sLo = select(0., f32(cLo) / 65534., cLo != 65535u);
  let sHi = select(0., f32(cHi) / 65534., cHi != 65535u);
  let score = (sLo * vLo * (1. - t) + sHi * vHi * t) / valid;
  return ScorePointSample(score, valid);
}

fn scoreCode(uv: vec2f) -> u32 {
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  let proj = projectScoreUv(uv, uvDx, uvDy);
  if (u.preservePeaks <= .5) { return loadScore(proj.address, 0); }
  let majorLength = max(length(proj.dx), length(proj.dy));
  let singleLod = i32(clamp(floor(log2(max(1., majorLength * f32(TILE_SIZE)))), 0., u.scoreMipCount - 1.));
  return loadScore(proj.address, singleLod);
}

fn sampleScore(uv: vec2f, uvDx: vec2f, uvDy: vec2f) -> ScoreSample {
  let proj = projectScoreUv(uv, uvDx, uvDy);
  let address = proj.address;
  let addressDx = proj.dx;
  let addressDy = proj.dy;
  let tileSizeF = f32(TILE_SIZE);
  let cellStep = max(abs(addressDx), abs(addressDy)) * tileSizeF;
  if (u.preservePeaks <= .5) {
    let s = loadScoreBilinear(address, 0, cellStep);
    return ScoreSample(s.score, s.valid, 1.);
  }
  let dxLength = length(addressDx);
  let dyLength = length(addressDy);
  let majorLength = max(dxLength, dyLength);
  let singleLodF = clamp(log2(max(1., majorLength * tileSizeF)), 0., u.scoreMipCount - 1.);
  if (singleLodF <= .001) {
    let s = loadScoreBilinear(address, 0, cellStep);
    return ScoreSample(s.score, s.valid, 1.);
  }

  // Smoothly blend isotropic trilinear sampling with an up-to-7-tap anisotropic
  // footprint walk along grazing limbs. Average valid coverage across taps so
  // NO_SCORE boundaries (e.g. coastlines, glyphs) never smear outward at the limb.
  let xLonger = dxLength > dyLength;
  let major = select(addressDy, addressDx, xLonger);
  let crossStep = select(addressDx, addressDy, xLonger);
  let axis = major / max(majorLength, .000001);
  let span = majorLength + abs(dot(crossStep, axis));
  let width = abs(axis.x * crossStep.y - axis.y * crossStep.x);
  let anisoRatio = span / max(width, 1e-6);
  let anisoW = smoothstep(1.35, 2.4, anisoRatio) * smoothstep(.05, .45, singleLodF);
  let mipWeight = smoothstep(0., 2.2, singleLodF);
  if (anisoW <= .001) {
    let center = sampleScoreTrilinear(address, singleLodF, cellStep);
    let balanced = mix(1., center.density, mipWeight * .45);
    return ScoreSample(center.score, center.valid, balanced);
  }
  let maxTaps = 7.;
  let anisoLodF = clamp(log2(max(1., max(span / maxTaps, width) * tileSizeF)), 0., u.scoreMipCount - 1.);
  let taps = u32(clamp(2. * ceil((anisoRatio - 1.) * .5) + 1., 3., maxTaps));
  var sumScore = 0.;
  var sumValid = 0.;
  var bestScore = 0.;
  for (var tap = 0u; tap < taps; tap++) {
    let offset = ((f32(tap) + .5) / f32(taps) - .5) * span;
    let s = loadScorePointTrilinear(address + axis * offset, anisoLodF);
    if (s.valid > .001) {
      sumScore += s.score * s.valid;
      sumValid += s.valid;
      bestScore = max(bestScore, s.score);
    }
  }
  let anisoValid = sumValid / f32(taps);
  let anisoAvg = select(0., sumScore / max(sumValid, 1e-4), sumValid > .001);
  let anisoDensity = select(0., clamp(anisoAvg / max(bestScore, 1e-4), 0., 1.), sumValid > .001);
  if (anisoW >= .999) {
    if (anisoValid <= .001) { return ScoreSample(0., 0., 0.); }
    let balanced = mix(1., anisoDensity, mipWeight * .45);
    return ScoreSample(anisoAvg, anisoValid, balanced);
  }
  let center = sampleScoreTrilinear(address, singleLodF, cellStep);
  let finalValid = mix(center.valid, anisoValid, anisoW);
  if (finalValid <= .001) { return ScoreSample(0., 0., 0.); }
  let finalScore = (center.score * center.valid * (1. - anisoW) + anisoAvg * anisoValid * anisoW) / finalValid;
  let rawDensity = mix(center.density, anisoDensity, anisoW);
  let balanced = mix(1., rawDensity, mipWeight * .45);
  return ScoreSample(finalScore, finalValid, balanced);
}

fn packSplit16(x: f32) -> vec2f {
  let hi = round(x * 256.) / 256.;
  return vec2f(hi, x - hi);
}

@fragment fn annotationFragment(input: AnnotationVertex) -> @location(0) vec4f {
  let uv = surfaceUvFrom(input.uv, input.geodetic);
  return vec4f(packSplit16(uv.x), packSplit16(uv.y));
}

struct AnnotationOutput {
  @location(0) uv: vec4f,
  @location(1) coverage: vec2f,
}

@fragment fn coverageAnnotationFragment(input: AnnotationVertex) -> AnnotationOutput {
  let uv = surfaceUvFrom(input.uv, input.geodetic);
  return AnnotationOutput(vec4f(packSplit16(uv.x), packSplit16(uv.y)), vec2f(1., select(1., 0., scoreCode(uv) == 65535u)));
}

struct EarthImagery {
  color: vec3f,
  tileWeight: f32,
  landMask: f32,
  normalLocal: vec3f,
  normalWeight: f32,
}

fn calibrateS2(rawRgb: vec3f) -> vec3f {
  // Calibrate Sentinel-2 L3 BOA reflectance (baseline-lifted & brighter) onto Blue Marble's
  // linear exposure, black point, and white balance.
  return clamp(
    vec3f(.593, .507, .516) * pow(max(rawRgb - vec3f(.030, .030, .040), vec3f(0.)), vec3f(1.50, 1.45, 1.55)),
    vec3f(0.),
    vec3f(1.),
  );
}

fn earthSlotFade(idx: u32) -> f32 {
  return earthTileFades[idx >> 2u][idx & 3u];
}

struct EarthAlbedoSample {
  rgb: vec3f,
  land: f32,
  w: f32,
}

fn sampleEarthAlbedoSlot(
  slot: u32,
  level: u32,
  cell: vec2i,
  tileCoord: vec2f,
  imgUv: vec2f,
  imgDx: vec2f,
  imgDy: vec2f,
  coast: f32,
) -> EarthAlbedoSample {
  let innerScale = VIRTUAL_TILE_INNER / VIRTUAL_TILE_SIZE;
  let baseGrid = vec2f(EARTH_TILE_BASE_COLS, EARTH_TILE_BASE_ROWS);
  let maxShift = 5u;
  let grid = baseGrid * exp2(f32(level));
  let gradScale = grid * innerScale;
  let dx = imgDx * gradScale;
  let dy = imgDy * gradScale;
  let tileMip = log2(max(1., max(length(dx), length(dy)) * VIRTUAL_TILE_SIZE));
  if (level == 0u && tileMip >= 1.5) {
    return EarthAlbedoSample(vec3f(0.), 0., 0.);
  }
  // Integer-cell tile origin eliminates 1 px wrap hairlines at tile edges.
  let tileOrigin = vec2f(vec2u(cell) >> vec2u(maxShift - min(level, maxShift)));
  let localUv = clamp(tileCoord * grid - tileOrigin, vec2f(0.), vec2f(1.));
  let tileUv = localUv * innerScale + (VIRTUAL_TILE_GUTTER / VIRTUAL_TILE_SIZE);
  let high = textureSampleGrad(earthTilePool, earthTileSampler, tileUv, i32(slot), dx, dy);
  // Alpha encodes exact 30 m Copernicus land fraction in [0.5 .. 1.0] (128=water, 255=land).
  let highLand = clamp((high.a - .5) * 2., 0., 1.);
  var s2Cal = calibrateS2(high.rgb);
  // Low-frequency Blue Marble radiometric harmonization removes Sentinel-2 MGRS
  // tile-to-tile seasonal exposure steps across regional views while keeping 61 m detail crisp.
  let s2Mip = select(4., 3., level <= 2u);
  let bmLod = max(0., select(3., 2., level <= 2u) - f32(level));
  let s2LowSample = textureSampleLevel(earthTilePool, earthTileSampler, tileUv, i32(slot), s2Mip);
  let inlandSafe = smoothstep(.5, 2.2, coast) * smoothstep(.93, .995, s2LowSample.a);
  let harmonizeW = mix(.88, .20, smoothstep(1.5, 4.5, f32(level))) * inlandSafe;
  if (u.earthReady > .5 && harmonizeW > .001) {
    let s2LowCal = calibrateS2(s2LowSample.rgb);
    let bmLow = textureSampleLevel(earth, geographicSampler, imgUv, bmLod).rgb;
    let ratio = clamp(bmLow / max(s2LowCal, vec3f(.015)), vec3f(.50), vec3f(2.0));
    s2Cal = clamp(s2Cal * mix(vec3f(1.), ratio, harmonizeW), vec3f(0.), vec3f(1.));
  }
  let highRgb = mix(high.rgb, s2Cal, highLand);
  // L0 fades smoothly into 8K Blue Marble at orbital altitude; L1..L5 also fade gracefully if
  // their 5-mip chain runs out on rapid zoom-out or grazing views without an active ancestor.
  let w = select(1. - smoothstep(3.5, 4.5, tileMip), 1. - smoothstep(.75, 1.45, tileMip), level == 0u);
  return EarthAlbedoSample(highRgb, highLand, w);
}

fn sampleEarthNormalSlot(
  slot: u32,
  level: u32,
  cell: vec2i,
  tileCoord: vec2f,
  imgDx: vec2f,
  imgDy: vec2f,
) -> vec4f {
  let innerScale = VIRTUAL_TILE_INNER / VIRTUAL_TILE_SIZE;
  let baseGrid = vec2f(EARTH_TILE_BASE_COLS, EARTH_TILE_BASE_ROWS);
  let maxShift = 5u;
  let grid = baseGrid * exp2(f32(level));
  let gradScale = grid * innerScale;
  let dx = imgDx * gradScale;
  let dy = imgDy * gradScale;
  let tileMip = log2(max(1., max(length(dx), length(dy)) * VIRTUAL_TILE_SIZE));
  if (level == 0u && tileMip >= 1.5) {
    return vec4f(0., 0., 1., 0.);
  }
  let origin = vec2f(vec2u(cell) >> vec2u(maxShift - min(level, maxShift)));
  let localUv = clamp(tileCoord * grid - origin, vec2f(0.), vec2f(1.));
  let tileUv = localUv * innerScale + (VIRTUAL_TILE_GUTTER / VIRTUAL_TILE_SIZE);
  let ns = textureSampleGrad(earthNormalPool, earthTileSampler, tileUv, i32(slot), dx, dy);
  let n = normalize(vec3f(ns.xy * 2. - 1., max(.05, ns.z)));
  let w = select(1. - smoothstep(3.5, 4.5, tileMip), 1. - smoothstep(.75, 1.45, tileMip), level == 0u);
  return vec4f(n, w);
}

fn sampleEarthImagery(uv: vec2f, uvDx: vec2f, uvDy: vec2f, coast: f32) -> EarthImagery {
  let imgUv = vec2f(uv.x, 1. - uv.y);
  let imgDx = vec2f(uvDx.x, -uvDx.y);
  let imgDy = vec2f(uvDy.x, -uvDy.y);
  let coarseLand = smoothstep(-.25, .25, coast);
  var outColor = textureSampleGrad(earth, geographicSampler, imgUv, imgDx, imgDy).rgb;
  var outTileW = 0.;
  var outLand = coarseLand;
  var outNormal = vec3f(0., 0., 1.);
  var outNormalW = 0.;

  if (u.earthTilesReady > .001) {
    let tileCoord = vec2f(fract(imgUv.x), clamp(imgUv.y, 0., .999999));
    let fullDims = vec2i(textureDimensions(earthTilePage));
    let halfH = fullDims.y >> 1;
    let pageDims = vec2f(f32(fullDims.x), f32(halfH));
    let cell = vec2i(clamp(floor(tileCoord * pageDims), vec2f(0.), pageDims - 1.));
    let innerScale = VIRTUAL_TILE_INNER / VIRTUAL_TILE_SIZE;
    let baseGrid = vec2f(EARTH_TILE_BASE_COLS, EARTH_TILE_BASE_ROWS);

    // Top half of earthTilePage: Sentinel-2 RGB + 30 m Copernicus land/water alpha
    let entry = textureLoad(earthTilePage, cell, 0);
    if (entry.r < 255u) {
      var slot = entry.r;
      var level = entry.g;
      let grad0 = baseGrid * exp2(f32(level)) * innerScale;
      let mip0 = log2(max(1., max(length(imgDx * grad0), length(imgDy * grad0)) * VIRTUAL_TILE_SIZE));
      var usedFallback = false;
      if (mip0 > 1.75 && entry.b < 255u) {
        slot = entry.b;
        level = entry.a;
        usedFallback = true;
      }
      let fade = earthSlotFade(slot);
      let primary = sampleEarthAlbedoSlot(slot, level, cell, tileCoord, imgUv, imgDx, imgDy, coast);
      if (fade >= .999) {
        if (primary.w > .001) {
          let w = primary.w * u.earthTilesReady;
          outColor = mix(outColor, primary.rgb, w);
          outTileW = w;
          outLand = mix(coarseLand, primary.land, w);
        }
      } else if (!usedFallback && entry.b < 255u) {
        let parentFade = earthSlotFade(entry.b);
        let parent = sampleEarthAlbedoSlot(entry.b, entry.a, cell, tileCoord, imgUv, imgDx, imgDy, coast);
        let pW = parent.w * parentFade * u.earthTilesReady;
        let cW = primary.w * u.earthTilesReady;
        let parentColor = mix(outColor, parent.rgb, pW);
        let childColor = mix(outColor, primary.rgb, cW);
        let parentLand = mix(coarseLand, parent.land, pW);
        let childLand = mix(coarseLand, primary.land, cW);
        outColor = mix(parentColor, childColor, fade);
        outTileW = mix(pW, cW, fade);
        outLand = mix(parentLand, childLand, fade);
      } else if (primary.w * fade > .001) {
        let w = primary.w * fade * u.earthTilesReady;
        outColor = mix(outColor, primary.rgb, w);
        outTileW = w;
        outLand = mix(coarseLand, primary.land, w);
      }
    }

    // Bottom half of earthTilePage: 60 m Copernicus 10-bit DEM tangent-space normals
    let nEntry = textureLoad(earthTilePage, cell + vec2i(0, halfH), 0);
    if (nEntry.r < 255u) {
      var nSlot = nEntry.r;
      var nLevel = nEntry.g;
      let nGrad0 = baseGrid * exp2(f32(nLevel)) * innerScale;
      let nMip0 = log2(max(1., max(length(imgDx * nGrad0), length(imgDy * nGrad0)) * VIRTUAL_TILE_SIZE));
      var usedNFallback = false;
      if (nMip0 > 1.75 && nEntry.b < 255u) {
        nSlot = nEntry.b;
        nLevel = nEntry.a;
        usedNFallback = true;
      }
      let nFade = earthSlotFade(160u + nSlot);
      let primaryN = sampleEarthNormalSlot(nSlot, nLevel, cell, tileCoord, imgDx, imgDy);
      if (nFade >= .999) {
        outNormal = primaryN.xyz;
        outNormalW = primaryN.w * u.earthTilesReady;
      } else if (!usedNFallback && nEntry.b < 255u) {
        let parentNFade = earthSlotFade(160u + nEntry.b);
        let parentN = sampleEarthNormalSlot(nEntry.b, nEntry.a, cell, tileCoord, imgDx, imgDy);
        let pNW = parentN.w * parentNFade;
        let cNW = primaryN.w;
        let nwSum = pNW * (1. - nFade) + cNW * nFade;
        let nMixT = select(1., (cNW * nFade) / max(1e-5, nwSum), nwSum > 1e-4);
        outNormal = normalize(mix(parentN.xyz, primaryN.xyz, nMixT));
        outNormalW = mix(pNW, cNW, nFade) * u.earthTilesReady;
      } else {
        outNormal = primaryN.xyz;
        outNormalW = primaryN.w * nFade * u.earthTilesReady;
      }
    }
  }
  return EarthImagery(outColor, outTileW, outLand, outNormal, outNormalW);
}

fn oceanHash3(p: vec3f) -> f32 {
  var q = fract(p * .1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn oceanNoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * f * (f * (f * 6. - 15.) + 10.);
  let n000 = oceanHash3(i);
  let n100 = oceanHash3(i + vec3f(1., 0., 0.));
  let n010 = oceanHash3(i + vec3f(0., 1., 0.));
  let n110 = oceanHash3(i + vec3f(1., 1., 0.));
  let n001 = oceanHash3(i + vec3f(0., 0., 1.));
  let n101 = oceanHash3(i + vec3f(1., 0., 1.));
  let n011 = oceanHash3(i + vec3f(0., 1., 1.));
  let n111 = oceanHash3(i + vec3f(1., 1., 1.));
  let nx0 = mix(vec2f(n000, n001), vec2f(n100, n101), u.x);
  let nx1 = mix(vec2f(n010, n011), vec2f(n110, n111), u.x);
  let nxy = mix(nx0, nx1, u.y);
  return mix(nxy.x, nxy.y, u.z);
}

// Precomputed mix(0.66, 1.52, smoothstep(0.24, 0.78, 0.5)) — Naga does not support smoothstep() in const expressions.
const OCEAN_WIND_MEAN_ROUGHNESS: f32 = 1.06612203;

// Seam-free 3D mesoscale ocean wind field (~150-1000 km fronts & calm slicks) modulating
// Cox-Munk RMS wave slope, analytically band-limited using a proven Frobenius Jacobian bound
// (||J_p||_2 <= 1 + 0.22 * 5.2 * sqrt(3) * 1.875 * wWarp ≈ 1 + 3.72 * wWarp) so all
// sub-Nyquist noise evaluations—including the warp itself—are skipped when filtered out.
fn oceanWindRoughness(n: vec3f, nDx: vec3f, nDy: vec3f) -> f32 {
  let flow = vec3f(n.x, n.y * 1.45, n.z);
  let flowDx = vec3f(nDx.x, nDx.y * 1.45, nDx.z);
  let flowDy = vec3f(nDy.x, nDy.y * 1.45, nDy.z);
  let flowStep = max(length(flowDx), length(flowDy));
  // Even without warp stretching, once the lowest octave (7.5) exceeds Nyquist all octaves are 0.5.
  if (7.5 * flowStep >= .85) { return OCEAN_WIND_MEAN_ROUGHNESS; }
  let wWarp = 1. - smoothstep(.30, .85, 5.2 * flowStep);
  let pStep = flowStep * (1. + 3.72 * wWarp);
  let w0 = 1. - smoothstep(.30, .85, 7.5 * pStep);
  let w1 = 1. - smoothstep(.30, .85, 17.5 * pStep);
  let w2 = 1. - smoothstep(.30, .85, 38. * pStep);
  if (w0 <= .001 && w1 <= .001 && w2 <= .001) { return OCEAN_WIND_MEAN_ROUGHNESS; }
  var warp = vec3f(0.);
  if (wWarp > .001) {
    warp = (vec3f(
      oceanNoise3(flow * 5.2 + vec3f(3.1, 1.7, 5.3)),
      oceanNoise3(flow * 5.2 + vec3f(8.4, 2.9, 1.2)),
      oceanNoise3(flow * 5.2 + vec3f(1.5, 7.6, 4.8)),
    ) - vec3f(.5)) * wWarp;
  }
  let p = flow + warp * .22;
  var o0 = .5;
  if (w0 > .001) { o0 = mix(.5, oceanNoise3(p * 7.5), w0); }
  var o1 = .5;
  if (w1 > .001) { o1 = mix(.5, oceanNoise3(p * 17.5 + vec3f(4.2, 1.9, 6.7)), w1); }
  var o2 = .5;
  if (w2 > .001) { o2 = mix(.5, oceanNoise3(p * 38. + vec3f(9.1, 5.3, 2.4)), w2); }
  let wind = o0 * .52 + o1 * .32 + o2 * .16;
  return mix(.66, 1.52, smoothstep(.24, .78, wind));
}

fn earthRawHeightLod(nDx: vec3f, nDy: vec3f) -> f32 {
  let terrainSize = vec2f(textureDimensions(terrain));
  let texelAngle = PI / terrainSize.y;
  let pixelAngle = max(length(nDx), length(nDy));
  return log2(max(1e-4, pixelAngle / texelAngle));
}

fn evalEarthTerrain(uv: vec2f, uvDx: vec2f, uvDy: vec2f, rawHeightLod: f32) -> vec2f {
  let pair = terrainPair(uv, uvDx, uvDy);
  var height = pair.r;
  if (u.terrainReady > .5 && rawHeightLod < -1.) {
    let bicubicW = 1. - smoothstep(-2., -1., rawHeightLod);
    height = mix(height, sampleHeightBicubic(vec2f(uv.x, 1. - uv.y)), bicubicW);
  }
  return vec2f(height, pair.g);
}

fn evalEarthPalette(sample: ScoreSample) -> f32 {
  let score = sample.score;
  let base = u.cutoff;
  let best = u.bestScore;
  var strength = 0.;
  if (sample.valid > .001 && score > base && best > base) {
    strength = clamp((score - base) / max(.000001, best - base), 0., 1.);
  }
  let automatic = u.autoPalette > .5;
  return select(palettePosition(strength, u.topRange, u.contrast), autoPalette(score, base, u.anchors), automatic);
}

fn evalHeatColor(displayPalette: f32, palette: f32, localRelief: f32, contour: f32, pGrad: f32, showIso: bool) -> vec3f {
  // Day/night-independent local relief hillshade draped onto the heatmap so mountains,
  // ridges, and basins stay legible beneath highlighted scores.
  let reliefShade = mix(1., localRelief, .55 * (1. - .60 * pow(displayPalette, 3.)));
  var heat = magma(displayPalette) * reliefShade;
  // Let subtle elevation contours etch through zoomed-in hotspots.
  heat = mix(heat, heat * .80, contour * u.contourOpacity * .38);
  // Crisp warm-gold iso-contour ring around top-tier peaks when zoomed into coherent fields.
  let isoRing = (1. - smoothstep(.6, 1.8, abs(palette - .84) / pGrad)) * smoothstep(.24, .06, pGrad);
  return mix(heat, vec3f(1., .86, .52), isoRing * .32 * select(0., 1., showIso));
}

fn shadeEarth(
  uv: vec2f, N: vec3f, worldPos: vec3f, inputUv: vec2f, smallAngle: f32,
  uvDx: vec2f, uvDy: vec2f, nDx: vec3f, nDy: vec3f,
  sample: ScoreSample, palette: f32, pGradIn: f32,
  height: f32, coast: f32, hGrad: vec2f, coastGrad: f32,
  flatEdgeGrad: vec2f,
) -> vec3f {
  let score = sample.score;
  let base = u.cutoff;
  let best = u.bestScore;
  let visibleScore = sample.valid > .001 && score >= u.minimumScore;
  var strength = 0.;
  if (sample.valid > .001 && score > base && best > base) { strength = clamp((score - base) / max(.000001, best - base), 0., 1.); }
  let automatic = u.autoPalette > .5;
  let contextStrength = clamp((score - u.contextFloor) / max(.000001, 1. - u.contextFloor), 0., 1.);
  let contextCoverageRaw = select(0., smoothstep(0., .22, contextStrength) * u.contextOpacity * sample.valid, visibleScore);
  // Progressive opacity: translucent violet outer fringe lets basemap relief breathe,
  // while high-scoring crimson/orange/gold hotspots become bold and unmistakable.
  let onset = smoothstep(HIGHLIGHT_START - .02, HIGHLIGHT_FULL + .02, palette);
  let tier = smoothstep(HIGHLIGHT_START, .90, palette);
  let densityMod = mix(.64, 1., smoothstep(.15, .85, sample.density));
  let progressive = onset * mix(.44, 1.20, pow(tier, .85)) * densityMod;
  let coverageRaw = select(0., clamp(progressive * u.highlightOpacity * sample.valid, 0., .95), visibleScore && best > base);
  let satSample = sampleEarthImagery(uv, uvDx, uvDy, coast);
  // Both 8K Blue Marble and 500 m virtual tiles are rgba8unorm-srgb (linear mipmaps & samples).
  let satellite = toDisplay(satSample.color);
  let highResImagery = satSample.tileWeight;
  let exactLandMask = satSample.landMask;
  let waterMask = 1. - exactLandMask;
  // Physical solar & viewing vectors (used for per-fragment hillshading, water SSS, Fresnel sheen, glint, and atmosphere).
  let V = normalize(u.eyePos - worldPos);
  let L = normalize(u.sunDirection);
  let geoLon = (uv.x - .5) * 2. * PI;
  let geoLat = (uv.y - .5) * PI;
  let cosLon = cos(geoLon);
  let sinLon = sin(geoLon);
  let cosLat = cos(geoLat);
  let sinLat = sin(geoLat);
  let geoDir = vec3f(cosLat * sinLon, sinLat, cosLat * cosLon);
  let geoEast = vec3f(cosLon, 0., -sinLon);
  let geoNorth = vec3f(-sinLat * sinLon, cosLat, -sinLat * cosLon);
  // DPR-normalized minor singular value of dN/d(screen) gives the true angular scale of the
  // globe everywhere (pole- and limb-free), preventing max-pyramid inflation and bloom overreach
  // when Earth shrinks to a tiny distant disc (e.g., viewed from the Moon).
  let nxx = dot(nDx, nDx);
  let nyy = dot(nDy, nDy);
  let nxy = dot(nDx, nDy);
  let nDet = max(0., nxx * nyy - nxy * nxy);
  let nMax = max(1e-12, .5 * (nxx + nyy + length(vec2f(nxx - nyy, 2. * nxy))));
  let minorAngle = max(sqrt(nDet / nMax), smallAngle);
  let globeScaleLod = log2(max(1., minorAngle * (18. * f32(TILE_SIZE) / PI) * u.pixelRatio));
  let farEmberFade = 1. - smoothstep(3.5, 5.2, globeScaleLod) * u.morph;
  let farInflation = smoothstep(3.8, 6.6, globeScaleLod) * u.morph * select(0., 1., u.preservePeaks > .5);
  let displayPalette = palette * mix(1., .72, farInflation);
  let contextCoverage = contextCoverageRaw * mix(1., .40, farInflation);
  let coverage = coverageRaw * mix(1., .42, farInflation);
  // Hardware-filtered mipmapped tangent-space normal + multi-scale ridge-AO map (1 fetch/px).
  let pixelAngle = max(length(nDx), length(nDy));
  let sampleUv = vec2f(uv.x, 1. - uv.y);
  let imgDx = vec2f(uvDx.x, -uvDx.y);
  let imgDy = vec2f(uvDy.x, -uvDy.y);
  var hillNormal = geoDir;
  var valleyAo = 1.;
  if (u.terrainReady > .5) {
    let normalSample = textureSampleGrad(terrainNormals, geographicSampler, sampleUv, imgDx, imgDy);
    let reliefScale = u.relief / 20.;
    var localN = normalize(vec3f((normalSample.xy * 2. - 1.) * reliefScale, max(.02, normalSample.z)));
    if (satSample.normalWeight > .001 && exactLandMask > .01) {
      let hiScale = clamp(reliefScale, 0., 1.5);
      let hiLocalN = normalize(vec3f(satSample.normalLocal.xy * hiScale, max(.05, satSample.normalLocal.z)));
      localN = normalize(mix(localN, hiLocalN, satSample.normalWeight * exactLandMask));
    }
    hillNormal = normalize(geoEast * localN.x + geoNorth * localN.y + geoDir * localN.z);
    let ridge = normalSample.a * 2. - 1.;
    valleyAo = 1. + ridge * .13 * exactLandMask * clamp(reliefScale, 0., 1.25);
  } else if (satSample.normalWeight > .001 && exactLandMask > .01) {
    let hiScale = clamp(u.relief / 20., 0., 1.5);
    let hiLocalN = normalize(vec3f(satSample.normalLocal.xy * hiScale, max(.05, satSample.normalLocal.z)));
    let localN = normalize(mix(vec3f(0., 0., 1.), hiLocalN, satSample.normalWeight * exactLandMask));
    hillNormal = normalize(geoEast * localN.x + geoNorth * localN.y + geoDir * localN.z);
  }
  let oneMinusV = clamp(1. - dot(N, V), 1.e-4, 1.);
  let ndotl = dot(N, L);
  let daylight = smoothstep(-.035, .045, ndotl);
  let sunDay = smoothstep(-.10, .22, ndotl);
  // Distance-aware unified linear-space lighting pass for both hemispheres: keep the dramatic
  // midnight-blue night side from space/Hero while smoothly lifting night-side ambient when
  // zoomed in so dark-cartographic forests, rivers, and 60 m DEM relief stay legible past ACES's
  // dark toe while preserving a smooth day/night terminator.
  let zoomInspect = 1. - smoothstep(.0006, .0045, minorAngle * u.pixelRatio);
  let fillSide = normalize(cross(V, vec3f(0., 1., 0.)) + vec3f(1e-4, 0., 0.));
  let fillDir = normalize(geoDir + .42 * V + .26 * fillSide + .22 * vec3f(0., 1., 0.));
  let fill = max(0., dot(hillNormal, fillDir));
  let ambientRelief = (.56 + .48 * fill) * valleyAo;
  let sunDiffuse = max(0., dot(hillNormal, L)) * daylight;
  let nightAmbient = mix(vec3f(.24, .28, .36), vec3f(.58, .62, .70), zoomInspect);
  let sunLight = mix(vec3f(.82, .80, .75), vec3f(.50, .48, .44), zoomInspect);
  let globeLightLinear = nightAmbient * ambientRelief + sunLight * sunDiffuse;
  // Utilitarian flat map uses classic NW cartographic hillshade across all longitudes.
  let flatSun = normalize(geoDir * .70710678 - geoEast * .5 + geoNorth * .5);
  let flatDiffuse = max(0., dot(hillNormal, flatSun));
  let flatLightLinear = vec3f((.34 + .52 * flatDiffuse) * valleyAo);
  let lightLinear = mix(flatLightLinear, globeLightLinear, u.morph);
  let localRelief = clamp(mix((.62 + .54 * flatDiffuse) * valleyAo, ambientRelief, u.morph), .68, 1.32);

  var background = vec3f(.17, .185, .20);
  var waterSssLinear = vec3f(0.);
  if (u.showImagery > .5) {
    background = satellite * .78;
  } else {
    // Dark cartographic satellite basemap: brings out real biomes, mountain chains,
    // and optical bathymetry while softly compressing polar ice caps.
    var land = vec3f(.19, .205, .22);
    if (u.earthReady > .5) {
      let luma = dot(satellite, vec3f(.299, .587, .114));
      let iceCompress = 1. / (1. + max(0., luma - .42) * 1.65);
      let muted = mix(vec3f(luma), satellite, .60) * iceCompress;
      land = vec3f(.10, .115, .13) + muted * .52;
    }
    land += vec3f(.055, .060, .066) * pow(clamp(height / 5500., 0., 1.), .75);

    // Log-compressed bathymetry ramp across 0 .. 11 km so every depth tier reads clearly
    // as deeper—shelves, slopes, ridges, abyssal plains, deep basins, and hadal trenches—
    // without crushing 4-6 km open ocean to pitch black:
    //   0 .. -200m (shelves/gulfs): muted coastal slate-teal tinted by local satellite hue
    //   -200 .. -2000m (slopes/banks): marine cerulean-slate
    //   -2000 .. -5500m (ridges/abyssal plains): deep oceanic navy
    //   -5500 .. -11000m (deep basins/hadal trenches): dark midnight indigo
    let depth = select(max(0., -height), 240., height >= 0. && waterMask > .05);
    let dNorm = clamp(log2(1. + depth / 350.) / 5.0, 0., 1.);
    let shelfZone = exp(-depth / 260.);
    let slopeZone = exp(-depth / 1800.);
    let shelfCol = vec3f(.108, .242, .298);
    let slopeCol = vec3f(.072, .176, .268);
    let abyssCol = vec3f(.048, .106, .198);
    let hadalCol = vec3f(.022, .046, .098);
    var ocean = mix(
      mix(shelfCol, slopeCol, clamp(dNorm / .35, 0., 1.)),
      mix(abyssCol, hadalCol, clamp((dNorm - .72) / .28, 0., 1.)),
      clamp((dNorm - .35) / .37, 0., 1.),
    );
    // Tint shallow water with muted local satellite chroma so reefs, bays, and warm desert coasts blend naturally.
    if (u.earthReady > .5) {
      let satLuma = dot(satellite, vec3f(.299, .587, .114));
      let satMuted = mix(vec3f(satLuma), satellite, .58);
      let satWater = vec3f(.028, .044, .064) + satMuted * vec3f(.34, .42, .46);
      let hasLocalWaterColor = smoothstep(.04, .24, max(satellite.g, satellite.b));
      ocean = mix(ocean, satWater, mix(.06, .52, shelfZone * hasLocalWaterColor));
      ocean *= (.94 + .16 * vec3f(satMuted.r * .5 + satMuted.g * .5, satMuted.g * .65 + satMuted.b * .35, satMuted.b));
    }
    // Two-layer water optics: look into the depth volume at nadir; cool sky Fresnel sheen at grazing angles.
    // Underwater bathymetry hillshade is carried once in linear light by lightLinear.
    let skySheen = vec3f(.09, .18, .32) * pow(oneMinusV, 3.2) * .24 * u.morph;
    ocean += skySheen * (1. - .45 * slopeZone);
    background = mix(ocean, land, exactLandMask);

    // Subtle sunlit shallow-water subsurface scattering (soft sea-glass lift over shallow shelves & reefs).
    let sssDepth = exp(-depth / 200.) * (1. - .38 * oneMinusV);
    let sssSun = mix(.58, smoothstep(-.02, .28, ndotl), u.morph);
    let sssTint = mix(vec3f(.025, .145, .185), clamp(satellite * vec3f(.32, .56, .58), vec3f(.02, .10, .14), vec3f(.12, .28, .30)), .45);
    waterSssLinear = sssTint * sssDepth * sssSun * .11 * waterMask * (1. - coverage);
  }
  let contour = terrainContoursSplit(height, hGrad, u.landContourInterval, u.depthContourInterval, u.contourDetail, u.pixelRatio);
  let contourColor = select(vec3f(.40, .43, .47), vec3f(.24, .44, .58), height < 0.);
  background = mix(background, contourColor, contour * u.contourOpacity * select(.55, 1., u.showImagery < .5 && u.earthReady < .5));
  let coastPixels = abs(coast) / max(.0001, coastGrad);
  let shore = 1. - smoothstep(.35, 1.15, coastPixels);
  // Completely fade out the coarse 4K shoreline vector once 500 m alpha-masked shorelines take over.
  let shoreStrength = .36 * (1. - highResImagery) * (1. - .75 * coverage);
  background = mix(background, vec3f(.20, .54, .76), select(0., shore * shoreStrength, u.terrainReady > .5));
  // Physical Cox-Munk / GGX ocean sun glint with V·H Schlick Fresnel (F0 = 0.02) and
  // two-way Rayleigh/aerosol airmass reddening near the terminator (fades out on flat map).
  let hv = L + V;
  let H = hv * inverseSqrt(max(1.e-6, dot(hv, hv)));
  let ndoth = clamp(dot(N, H), 0., 1.);
  let vdoth = clamp(dot(V, H), 0., 1.);
  let nlSpec = max(ndotl, 0.);
  let nvSpec = max(dot(N, V), .05);
  let specDay = smoothstep(-.02, .07, ndotl);
  let glintMask = waterMask * specDay * u.morph * (1. - coverage);
  // Rotation-invariant geometric specular AA (largest eigenvalue of normal covariance);
  // adaptive angular gate widens with specAaSq while still skipping negligible far tails.
  let specAaSq = min(.35 * nMax, .18);
  let minNdoth = mix(.80, .55, smoothstep(.002, .05, specAaSq));
  var specLinear = vec3f(0.);
  if (glintMask > .001 && ndoth > minNdoth) {
    let fv = 1. - vdoth;
    let fv2 = fv * fv;
    let glintFresnel = .02 + .98 * (fv2 * fv2 * fv);
    let roughScale = oceanWindRoughness(N, nDx, nDy);
    let a1 = .052 * roughScale;
    let a1Sq = clamp(a1 * a1 + specAaSq, 1e-5, 1.);
    let d1Den = ndoth * ndoth * (a1Sq - 1.) + 1.;
    let d1 = a1Sq / (PI * d1Den * d1Den);
    let vis1 = .5 * nlSpec / max(nlSpec * sqrt(nvSpec * nvSpec * (1. - a1Sq) + a1Sq) + nvSpec * sqrt(nlSpec * nlSpec * (1. - a1Sq) + a1Sq), 1e-4);
    let a2 = .125 * roughScale;
    let a2Sq = clamp(a2 * a2 + specAaSq, 1e-5, 1.);
    let d2Den = ndoth * ndoth * (a2Sq - 1.) + 1.;
    let d2 = a2Sq / (PI * d2Den * d2Den);
    let vis2 = .5 * nlSpec / max(nlSpec * sqrt(nvSpec * nvSpec * (1. - a2Sq) + a2Sq) + nvSpec * sqrt(nlSpec * nlSpec * (1. - a2Sq) + a2Sq), 1e-4);
    let airmass = 1. / (nlSpec + .08) + 1. / (nvSpec + .08);
    let airTrans = exp(-max(0., airmass - 2.) * vec3f(.15, .24, .40));
    let specBrdf = (d1 * vis1 * .72 + d2 * vis2 * .28) * glintFresnel * smoothstep(minNdoth, minNdoth + .07, ndoth);
    let specRaw = vec3f(1., .97, .92) * airTrans * (specBrdf * .38) * glintMask;
    specLinear = specRaw / (vec3f(1.) + specRaw * .10);
  }
  let fresnel = pow(oneMinusV, 4.2);
  let horizon = pow(oneMinusV, 8.0);
  // Subtle warm sunset scatter along the grazing terminator limb, boosted by Mie forward scatter when backlit.
  let backScatter = pow(max(0., dot(-V, L)), 5.0);
  let tw = (ndotl + .01) / .13;
  let twilight = exp(-(tw * tw)) * pow(oneMinusV, 4.2) * (1. + 2.4 * backScatter);
  let atmoLinear = (
    vec3f(.20, .52, 1.0) * (fresnel * .28 + horizon * .52) * sunDay +
    vec3f(1., .42, .12) * twilight * .13 +
    vec3f(.05, .12, .24) * horizon * .22 * (1. - sunDay)
  ) * u.morph * (1. - .55 * coverage);
  let baseLinear = toLinear(background);
  let litLinear = baseLinear * lightLinear;
  let pGrad = max(1e-4, pGradIn);
  var result: vec3f;
  if (u.hdr > .5) {
    var surfaceLinear = litLinear + waterSssLinear + specLinear;
    if (contextCoverage > .001) {
      let contextHeat = magma(.08 + .11 * sqrt(contextStrength));
      surfaceLinear = mix(surfaceLinear, toLinear(contextHeat), contextCoverage);
    }
    if (coverage > .001) {
      let heat = evalHeatColor(displayPalette, palette, localRelief, contour, pGrad, visibleScore && best > base);
      let heatLinear = toLinear(clamp(heat, vec3f(0.), vec3f(1.))) * mix(.68, 1.16, farEmberFade);
      let peakGlow = pow(select(strength, palette, automatic), 5.5) * densityMod * farEmberFade;
      let ember = mix(heatLinear, vec3f(1., .58, .14), .28) * (u.emission * 1.15 * peakGlow);
      surfaceLinear = mix(surfaceLinear, heatLinear + ember, coverage);
    }
    result = surfaceLinear + atmoLinear;
  } else {
    var litDisplay = pow(max(litLinear, vec3f(0.)), vec3f(1. / 2.2));
    if (contextCoverage > .001) {
      let contextHeat = magma(.08 + .11 * sqrt(contextStrength));
      litDisplay = mix(litDisplay, contextHeat, contextCoverage);
    }
    if (coverage > .001) {
      let heat = evalHeatColor(displayPalette, palette, localRelief, contour, pGrad, visibleScore && best > base);
      litDisplay = mix(litDisplay, heat, coverage);
    }
    result = litDisplay + pow(max(waterSssLinear + specLinear + atmoLinear, vec3f(0.)), vec3f(1. / 2.2));
  }
  if (u.showGrid > .5) {
    let lines = uv * vec2f(24., 12.);
    // Use longitude-aligned angular footprint away from the pole, blending to full pixelAngle
    // right inside the polar cap so unresolved meridians converge smoothly to mean duty.
    let lonAngular = mix(max(abs(dot(nDx, geoEast)), abs(dot(nDy, geoEast))), pixelAngle, smoothstep(.08, .02, cosLat));
    let polarStepU = 24. * lonAngular / max(2. * PI * cosLat, 1e-5);
    let stepU = max(length(vec2f(uvDx.x, uvDy.x)) * 24., polarStepU * u.morph);
    let stepV = length(vec2f(uvDx.y, uvDy.y)) * 12.;
    let gridWidth = 1.1 * u.pixelRatio;
    let gridAa = mix(1., .7, smoothstep(1., 1.8, u.pixelRatio));
    let line = max(
      filteredPulse(lines.x, stepU, gridWidth, gridAa, .45, 0.),
      filteredPulse(lines.y, stepV, gridWidth, gridAa, .45, 0.),
    );
    let gridColor = vec3f(.61, .72, .77);
    result = mix(result, select(gridColor, toLinear(gridColor), u.hdr > .5), line * .17);
  }
  // Subtle ~1 CSS px neatline framing the Equal Earth flat map against deep space.
  let flatEdgeWeight = 1. - smoothstep(.02, .22, u.morph);
  if (flatEdgeWeight > .001) {
    let edgeDistX = min(inputUv.x, 1. - inputUv.x) / max(1e-6, flatEdgeGrad.x);
    let edgeDistY = min(inputUv.y, 1. - inputUv.y) / max(1e-6, flatEdgeGrad.y);
    let edgeCssPx = min(edgeDistX, edgeDistY) / max(1., u.pixelRatio);
    let neatline = (1. - smoothstep(.65, 1.65, edgeCssPx)) * flatEdgeWeight;
    let borderColor = vec3f(.38, .48, .60);
    result = mix(result, select(borderColor, toLinear(borderColor), u.hdr > .5), neatline * .44);
  }
  return result;
}

@fragment fn surfaceFragment(input: SurfaceVertex) -> @location(0) vec4f {
  let uv = surfaceUv(input);
  let N = normalize(input.geodetic);
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  let nDx = dpdx(N);
  let nDy = dpdy(N);
  let rawHeightLod = earthRawHeightLod(nDx, nDy);
  let tc = evalEarthTerrain(uv, uvDx, uvDy, rawHeightLod);
  let hGrad = vec2f(dpdx(tc.x), dpdy(tc.x));
  let coastGrad = fwidth(tc.y);
  let sample = sampleScore(uv, uvDx, uvDy);
  let palette = evalEarthPalette(sample);
  let pGrad = fwidth(palette);
  let flatEdgeGrad = vec2f(
    length(vec2f(dpdx(input.uv.x), dpdy(input.uv.x))),
    length(vec2f(dpdx(input.uv.y), dpdy(input.uv.y))),
  );
  return vec4f(shadeEarth(
    uv, N, input.worldPos, input.uv, 0., uvDx, uvDy, nDx, nDy,
    sample, palette, pGrad, tc.x, tc.y, hGrad, coastGrad, flatEdgeGrad,
  ), 1.);
}

struct EarthRayHit {
  worldPos: vec3f,
  N: vec3f,
  uv: vec2f,
  hit: bool,
}

fn evalEarthRay(destPos: vec2f) -> EarthRayHit {
  let ndc = vec2f(destPos.x * u.invViewportPx.x - 1., 1. - destPos.y * u.invViewportPx.y);
  let tanX = u.tanHalfFovX;
  let tanY = u.tanHalfFovY;
  let depth = u.centerDepth;
  let dNdc = ndc - u.centerNdc;
  let evalPos = u.camRight * (dNdc.x * depth * tanX) + u.camUp * (dNdc.y * depth * tanY);
  let dirLocal = normalize(u.camRight * (ndc.x * tanX) + u.camUp * (ndc.y * tanY) - u.camBack);
  let invAxis = vec3f(1., 1.003364089821, 1.);
  let posE = evalPos * invAxis;
  let dirE = normalize(dirLocal * invAxis);
  let q = posE - dirE * dot(posE, dirE);
  let disc = 1. - dot(q, q);
  let hit = disc >= 0.;
  let pHitE = select(normalize(q), q - dirE * sqrt(max(0., disc)), hit);
  let worldPos = vec3f(pHitE.x, pHitE.y / 1.003364089821, pHitE.z);
  let N = normalize(pHitE * invAxis);
  var lonHit = 0.;
  if (length(N.xz) > 1e-8) { lonHit = atan2(N.x, N.z); }
  let uv = vec2f(fract(lonHit / (2. * PI) + .5), atan2(N.y, length(N.xz)) / PI + .5);
  return EarthRayHit(worldPos, N, uv, hit);
}

@fragment fn smallSurfaceFragment(
  input: SmallSurfaceVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallSurfaceOut {
  // Evaluate analytical per-sample ellipsoid raycasts and neighbor rays directly from
  // framebuffer coordinates for crisp sub-pixel silhouettes and quad-independent gradients,
  // with explicit sample_mask per WebGPU spec.
  let destPos = input.position.xy;
  let h0 = evalEarthRay(destPos);
  if (!h0.hit) { discard; }
  let hXp = evalEarthRay(destPos + vec2f(.5, 0.));
  let hXm = evalEarthRay(destPos - vec2f(.5, 0.));
  let hYp = evalEarthRay(destPos + vec2f(0., .5));
  let hYm = evalEarthRay(destPos - vec2f(0., .5));
  var uvDx = wrappedGradient(hXp.uv - hXm.uv);
  var nDx = hXp.N - hXm.N;
  if (hXp.hit != hXm.hit) {
    uvDx = select(wrappedGradient(h0.uv - hXm.uv), wrappedGradient(hXp.uv - h0.uv), hXp.hit) * 2.;
    nDx = select(h0.N - hXm.N, hXp.N - h0.N, hXp.hit) * 2.;
  }
  var uvDy = wrappedGradient(hYp.uv - hYm.uv);
  var nDy = hYp.N - hYm.N;
  if (hYp.hit != hYm.hit) {
    uvDy = select(wrappedGradient(h0.uv - hYm.uv), wrappedGradient(hYp.uv - h0.uv), hYp.hit) * 2.;
    nDy = select(h0.N - hYm.N, hYp.N - h0.N, hYp.hit) * 2.;
  }
  let rawHeightLod = earthRawHeightLod(nDx, nDy);
  let tc0 = evalEarthTerrain(h0.uv, uvDx, uvDy, rawHeightLod);
  var tcDx: vec2f;
  if (hXp.hit != hXm.hit) {
    let tcSide = evalEarthTerrain(select(hXm.uv, hXp.uv, hXp.hit), uvDx, uvDy, rawHeightLod);
    tcDx = select(tc0 - tcSide, tcSide - tc0, hXp.hit) * 2.;
  } else {
    tcDx = evalEarthTerrain(hXp.uv, uvDx, uvDy, rawHeightLod) - evalEarthTerrain(hXm.uv, uvDx, uvDy, rawHeightLod);
  }
  var tcDy: vec2f;
  if (hYp.hit != hYm.hit) {
    let tcSide = evalEarthTerrain(select(hYm.uv, hYp.uv, hYp.hit), uvDx, uvDy, rawHeightLod);
    tcDy = select(tc0 - tcSide, tcSide - tc0, hYp.hit) * 2.;
  } else {
    tcDy = evalEarthTerrain(hYp.uv, uvDx, uvDy, rawHeightLod) - evalEarthTerrain(hYm.uv, uvDx, uvDy, rawHeightLod);
  }
  let hGrad = vec2f(tcDx.x, tcDy.x);
  let coastGrad = abs(tcDx.y) + abs(tcDy.y);
  let sample0 = sampleScore(h0.uv, uvDx, uvDy);
  let palette0 = evalEarthPalette(sample0);
  var pGrad = 0.;
  if (u.highlightOpacity > .001 && u.bestScore > u.cutoff) {
    var pDx: f32;
    if (hXp.hit != hXm.hit) {
      let pSide = evalEarthPalette(sampleScore(select(hXm.uv, hXp.uv, hXp.hit), uvDx, uvDy));
      pDx = select(palette0 - pSide, pSide - palette0, hXp.hit) * 2.;
    } else {
      pDx = evalEarthPalette(sampleScore(hXp.uv, uvDx, uvDy)) - evalEarthPalette(sampleScore(hXm.uv, uvDx, uvDy));
    }
    var pDy: f32;
    if (hYp.hit != hYm.hit) {
      let pSide = evalEarthPalette(sampleScore(select(hYm.uv, hYp.uv, hYp.hit), uvDx, uvDy));
      pDy = select(palette0 - pSide, pSide - palette0, hYp.hit) * 2.;
    } else {
      pDy = evalEarthPalette(sampleScore(hYp.uv, uvDx, uvDy)) - evalEarthPalette(sampleScore(hYm.uv, uvDx, uvDy));
    }
    pGrad = abs(pDx) + abs(pDy);
  }
  let flatEdgeGrad = vec2f(length(vec2f(uvDx.x, uvDy.x)), length(vec2f(uvDx.y, uvDy.y)));
  let smallAngle = max(u.centerDepth * u.tanHalfFovY * u.invViewportPx.y, 1e-4);
  let result = shadeEarth(
    h0.uv, h0.N, h0.worldPos, h0.uv, smallAngle, uvDx, uvDy, nDx, nDy,
    sample0, palette0, pGrad, tc0.x, tc0.y, hGrad, coastGrad, flatEdgeGrad,
  );
  var clip = u.camera * vec4f(h0.worldPos, 1.);
  if (u.nativeDepth < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SmallSurfaceOut(vec4f(result, 1.), clip.z / clip.w, 1u << sampleIdx);
}

struct SmallAnnotationOut {
  @location(0) uv: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn smallAnnotationFragment(
  input: SmallSurfaceVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallAnnotationOut {
  let h0 = evalEarthRay(input.position.xy);
  if (!h0.hit) { discard; }
  var clip = u.camera * vec4f(h0.worldPos, 1.);
  if (u.nativeDepth < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SmallAnnotationOut(vec4f(packSplit16(h0.uv.x), packSplit16(h0.uv.y)), clip.z / clip.w, 1u << sampleIdx);
}

struct SmallCoverageAnnotationOut {
  @location(0) uv: vec4f,
  @location(1) coverage: vec2f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn smallCoverageAnnotationFragment(
  input: SmallSurfaceVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallCoverageAnnotationOut {
  let h0 = evalEarthRay(input.position.xy);
  if (!h0.hit) { discard; }
  var clip = u.camera * vec4f(h0.worldPos, 1.);
  if (u.nativeDepth < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SmallCoverageAnnotationOut(
    vec4f(packSplit16(h0.uv.x), packSplit16(h0.uv.y)),
    vec2f(1., select(1., 0., scoreCode(h0.uv) == 65535u)),
    clip.z / clip.w,
    1u << sampleIdx,
  );
}
