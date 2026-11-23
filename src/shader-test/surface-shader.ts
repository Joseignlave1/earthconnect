import { colorShader, magmaShader } from './color.ts'
import { contourShader } from './contours.ts'
import { equalEarthShader } from './geometry.ts'
import { autoStyleShader, HIGHLIGHT_FULL, HIGHLIGHT_START, scoreStyleShader } from './score-style.ts'
import { SIZE } from './world-grid.ts'

// Every field after the matrix is a vec4, shared with the packed CPU uniform
// buffer. Keeping the layout explicit avoids per-frame bind-group allocation.
const surfaceUniforms = `
struct SurfaceUniforms {
  camera: mat4x4f,
  anchors: vec4f,
  style: vec4f, // baseline, contrast, top band, emission
  context: vec4f, // floor, opacity, minimum filter, highlight opacity
  view: vec4f, // morph, relief, best score, mip count
  contour: vec4f, // land interval, depth interval, opacity, pixel ratio
  detail: vec4f, // minor/fine contour opacity, reserved, HDR
  flags: vec4f, // automatic palette, grid, preserve peaks, imagery
  status: vec4f, // terrain ready, native depth, earthReady, earthTilesReady
  poles: vec4f,
  sun: vec4f, // Earth-fixed unit sun direction, reserved
  eye: vec4f, // Earth-fixed camera eye in Earth-radius units, reserved
}
@group(0) @binding(0) var<uniform> u: SurfaceUniforms;
`

export function surfaceShader(banks: number, bankSize: number, filterableTerrain: boolean) {
  return `
diagnostic(off, derivative_uniformity);
${surfaceUniforms}
@group(0) @binding(1) var lookup: texture_2d<u32>;
@group(0) @binding(2) var earth: texture_2d<f32>;
@group(0) @binding(3) var terrain: texture_2d<f32>;
@group(0) @binding(4) var geographicSampler: sampler;
@group(0) @binding(5) var illumination: texture_2d<f32>;
${Array.from({ length: banks },(_,i) => `@group(0) @binding(${6 + i}) var scores${i}: texture_2d_array<u32>;`).join('\n')}
@group(0) @binding(${6 + banks}) var earthTilePage: texture_2d<u32>;
@group(0) @binding(${7 + banks}) var earthTilePool: texture_2d_array<f32>;
@group(0) @binding(${8 + banks}) var earthTileSampler: sampler;
const PI: f32 = 3.141592653589793;
${colorShader}
${magmaShader}
${scoreStyleShader}
${autoStyleShader}

${filterableTerrain ? `
fn terrainLevel(uv: vec2f,lod: f32) -> vec2f {
  return textureSampleLevel(terrain,geographicSampler,uv,lod).rg;
}
fn terrainGradient(uv: vec2f,a: vec2f,b: vec2f) -> vec2f {
  return textureSampleGrad(terrain,geographicSampler,uv,a,b).rg;
}` : `
// Full precision terrain still works without float32-filterable. Only this
// sampling helper changes; heights and coast distances are never quantized.
fn terrainTexel(pixel: vec2i,lod: i32) -> vec2f {
  let size = vec2i(textureDimensions(terrain,lod));
  let p = vec2i(((pixel.x % size.x) + size.x) % size.x,clamp(pixel.y,0,size.y - 1));
  return textureLoad(terrain,p,lod).rg;
}
fn terrainBilinear(uv: vec2f,lod: i32) -> vec2f {
  let p = uv * vec2f(textureDimensions(terrain,lod)) - .5;
  let base = vec2i(floor(p));
  let f = fract(p);
  return mix(mix(terrainTexel(base,lod),terrainTexel(base + vec2i(1,0),lod),f.x),
    mix(terrainTexel(base + vec2i(0,1),lod),terrainTexel(base + vec2i(1,1),lod),f.x),f.y);
}
fn terrainLevel(uv: vec2f,lod: f32) -> vec2f {
  let level = clamp(lod,0.,f32(textureNumLevels(terrain) - 1u));
  let low = i32(floor(level));
  let high = min(low + 1,i32(textureNumLevels(terrain)) - 1);
  return mix(terrainBilinear(uv,low),terrainBilinear(uv,high),fract(level));
}
fn terrainGradient(uv: vec2f,a: vec2f,b: vec2f) -> vec2f {
  let size = vec2f(textureDimensions(terrain));
  let major = length(a * size);
  let minor = length(b * size);
  let taps = u32(clamp(ceil(major / max(1.,minor)),1.,16.));
  let lod = log2(max(1.,max(minor,major / f32(taps))));
  var value = vec2f(0.);
  for (var i = 0u; i < taps; i++) {
    value += terrainLevel(uv + a * ((f32(i) + .5) / f32(taps) - .5),lod);
  }
  return value / f32(taps);
}`}
fn elevation(uv: vec2f) -> f32 {
  if (uv.y == 0.) { return u.poles.x; }
  if (uv.y == 1.) { return u.poles.y; }
  return terrainLevel(vec2f(uv.x,1. - clamp(uv.y,0.,1.)),0.).r;
}
${equalEarthShader}
struct SurfaceVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) geodetic: vec3f,
  @location(2) worldPos: vec3f,
}
@vertex fn surfaceVertex(@location(0) uv: vec2f) -> SurfaceVertex {
  let lon = (uv.x - .5) * 2. * PI;
  let lat = (uv.y - .5) * PI;
  let height = max(0.,elevation(uv)) / 6378137. * u.view.y;
  let e2 = .0066943799901413165;
  let n = inverseSqrt(1. - e2 * sin(lat) * sin(lat));
  var normal = vec3f(cos(lat) * sin(lon),sin(lat),cos(lat) * cos(lon));
  if (uv.y == 0. || uv.y == 1.) { normal = vec3f(0.,sign(lat),0.); }
  let sphere = vec3f(n * normal.x,n * (1. - e2) * normal.y,n * normal.z) + normal * height;
  let plane = vec3f(equalEarth(lon,lat),height);
  let position = vec4f(mix(plane,sphere,u.view.x),1.);
  var clip = u.camera * position;
  // The standalone lab uses OpenGL depth. The solar scene uses native reverse-Z.
  if (u.status.y < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SurfaceVertex(clip,uv,normal,position.xyz);
}
fn wrappedGradient(delta: vec2f) -> vec2f {
  return vec2f(delta.x - floor(delta.x + .5),delta.y);
}
fn terrainPair(uv: vec2f,dx: vec2f,dy: vec2f) -> vec2f {
  let size = vec2f(textureDimensions(terrain));
  let a = vec2f(dx.x,-dx.y) * size;
  let b = vec2f(dy.x,-dy.y) * size;
  let xx = a.x * a.x + b.x * b.x;
  let yy = a.y * a.y + b.y * b.y;
  let xy = a.x * a.y + b.x * b.y;
  let spread = length(vec2f(xx - yy,2. * xy));
  var angle = 0.;
  if (spread > .000001) { angle = .5 * atan2(2. * xy,xx - yy); }
  let axis = vec2f(cos(angle),sin(angle));
  let major = sqrt(max(.000001,.5 * (xx + yy + spread)));
  let minor = abs(a.x * b.y - a.y * b.x) / major;
  let invSize = 1. / size;
  let sampleUv = vec2f(uv.x,1. - uv.y);
  let perp = vec2f(-axis.y,axis.x);
  let minHeight = 2.828427125; // exp2(1.5)
  // At globe and medium zoom (minor >= exp2(1.5)), height and coast share the exact same
  // footprint, cutting rg32float terrain fetches in half.
  let sharp = terrainGradient(sampleUv,axis * max(major,1.) * invSize,perp * max(minor,1.) * invSize);
  if (minor >= minHeight) { return sharp; }
  let soft = terrainGradient(sampleUv,axis * max(major,minHeight) * invSize,perp * minHeight * invSize);
  return vec2f(soft.r,sharp.g);
}
${contourShader}
fn surfaceUv(input: SurfaceVertex) -> vec2f {
  let geodetic = normalize(input.geodetic);
  var longitude = 0.;
  if (length(geodetic.xz) > .00000001) { longitude = atan2(geodetic.x,geodetic.z); }
  let globeUv = vec2f(longitude / (2. * PI) + .5,atan2(geodetic.y,length(geodetic.xz)) / PI + .5);
  var uv = input.uv + wrappedGradient(globeUv - input.uv) * u.view.x;
  uv.x = fract(uv.x);
  return uv;
}
fn loadScore(address: vec2f,lod: i32) -> u32 {
  if (address.x < 0. || address.x >= 36. || address.y < 0. || address.y >= 18.) { return 65535u; }
  let cell = clamp(address,vec2f(0.),vec2f(35.99999,17.99999));
  let slot = textureLoad(lookup,vec2i(floor(cell)),0).r;
  var code = 65535u;
  if (slot != 65535u) {
    let bank = slot / ${bankSize}u;
    let layer = i32(slot % ${bankSize}u);
    let size = ${SIZE} >> u32(lod);
    let pixel = clamp(vec2i(fract(cell) * f32(size)),vec2i(0),vec2i(i32(size) - 1));
    ${Array.from({ length: banks },(_,i) => `${i ? 'else ' : ''}if (bank == ${i}u) {
      code = textureLoad(scores${i},pixel,layer,lod).r;
    }`).join('\n')}
  }
  return code;
}
struct ScoreSample {
  score: f32,
  valid: f32,
  density: f32,
}
fn loadScoreBilinear(address: vec2f,lod: i32,cellStep: vec2f) -> ScoreSample {
  let sizeI = i32(${SIZE} >> u32(lod));
  let size = f32(sizeI);
  let pos = address * size - .5;
  let baseF = floor(pos);
  let base = vec2i(baseF);
  let rawF = pos - baseF;
  // At lod 0 when zoomed in (cell > 1 screen pixel), sharpen interpolation to a
  // 1-pixel anti-aliased edge so individual 4 km cells stay crisp and readable.
  let f = select(rawF,clamp((rawF - .5) / max(cellStep,vec2f(.04)) + .5,vec2f(0.),vec2f(1.)),lod == 0);
  let tile = vec2i(floor(baseF / size));
  let local = base - tile * sizeI;
  var c00 = 65535u;
  var c10 = 65535u;
  var c01 = 65535u;
  var c11 = 65535u;
  if (local.x >= 0 && local.x < sizeI - 1 && local.y >= 0 && local.y < sizeI - 1) {
    // Fast path (>99% of pixels): one tile lookup + immediate early-out over oceans.
    if (tile.x < 0 || tile.x >= 36 || tile.y < 0 || tile.y >= 18) { return ScoreSample(0.,0.,0.); }
    let slot = textureLoad(lookup,tile,0).r;
    if (slot == 65535u) { return ScoreSample(0.,0.,0.); }
    let bank = slot / ${bankSize}u;
    let layer = i32(slot % ${bankSize}u);
    ${Array.from({ length: banks },(_,i) => `${i ? 'else ' : ''}if (bank == ${i}u) {
      c00 = textureLoad(scores${i},local,layer,lod).r;
      c10 = textureLoad(scores${i},local + vec2i(1,0),layer,lod).r;
      c01 = textureLoad(scores${i},local + vec2i(0,1),layer,lod).r;
      c11 = textureLoad(scores${i},local + vec2i(1,1),layer,lod).r;
    }`).join('\n')}
  } else {
    let invSize = 1. / size;
    c00 = loadScore((baseF + vec2f(.5,.5)) * invSize,lod);
    c10 = loadScore((baseF + vec2f(1.5,.5)) * invSize,lod);
    c01 = loadScore((baseF + vec2f(.5,1.5)) * invSize,lod);
    c11 = loadScore((baseF + vec2f(1.5,1.5)) * invSize,lod);
  }
  let wx0 = 1. - f.x;
  let wy0 = 1. - f.y;
  var wSum = 0.;
  var vSum = 0.;
  var sMax = 0.;
  if (c00 != 65535u) { let w = wx0 * wy0; let s = f32(c00) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax,s); }
  if (c10 != 65535u) { let w = f.x * wy0; let s = f32(c10) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax,s); }
  if (c01 != 65535u) { let w = wx0 * f.y; let s = f32(c01) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax,s); }
  if (c11 != 65535u) { let w = f.x * f.y; let s = f32(c11) / 65534.; wSum += w; vSum += w * s; sMax = max(sMax,s); }
  if (wSum <= .001) { return ScoreSample(0.,0.,0.); }
  let avg = vSum / wSum;
  let density = clamp(avg / max(sMax,1e-4),0.,1.);
  return ScoreSample(avg,wSum,density);
}
fn sampleScoreTrilinear(address: vec2f,lodF: f32,cellStep: vec2f) -> ScoreSample {
  let maxLod = i32(u.view.w) - 1;
  let clamped = clamp(lodF,0.,f32(maxLod));
  let lo = i32(floor(clamped));
  let hi = min(lo + 1,maxLod);
  let t = clamped - f32(lo);
  let a = loadScoreBilinear(address,lo,cellStep);
  if (hi == lo || t <= .001) { return a; }
  let b = loadScoreBilinear(address,hi,cellStep);
  if (a.valid <= .001 && b.valid <= .001) { return ScoreSample(0.,0.,0.); }
  if (a.valid <= .001) { return ScoreSample(b.score,b.valid * t,b.density); }
  if (b.valid <= .001) { return ScoreSample(a.score,a.valid * (1. - t),a.density); }
  return ScoreSample(mix(a.score,b.score,t),mix(a.valid,b.valid,t),mix(a.density,b.density,t));
}
fn loadScorePointTrilinear(address: vec2f,lodF: f32) -> vec2f {
  let maxLod = i32(u.view.w) - 1;
  let clamped = clamp(lodF,0.,f32(maxLod));
  let lo = i32(floor(clamped));
  let hi = min(lo + 1,maxLod);
  let t = clamped - f32(lo);
  let cLo = loadScore(address,lo);
  if (hi == lo || t <= .001) {
    return select(vec2f(0.,0.),vec2f(f32(cLo) / 65534.,1.),cLo != 65535u);
  }
  let cHi = loadScore(address,hi);
  let vLo = select(0.,1.,cLo != 65535u);
  let vHi = select(0.,1.,cHi != 65535u);
  let valid = mix(vLo,vHi,t);
  if (valid <= .001) { return vec2f(0.,0.); }
  let sLo = select(0.,f32(cLo) / 65534.,cLo != 65535u);
  let sHi = select(0.,f32(cHi) / 65534.,cHi != 65535u);
  let score = (sLo * vLo * (1. - t) + sHi * vHi * t) / valid;
  return vec2f(score,valid);
}
fn scoreCode(uv: vec2f) -> u32 {
  let latitude = (uv.y - .5) * PI;
  let address = vec2f(18. + (uv.x - .5) * 36. * cos(latitude),(1. - uv.y) * 18.);
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  if (u.flags.z <= .5) { return loadScore(address,0); }
  let addressDx = vec2f(36. * (cos(latitude) * uvDx.x - (uv.x - .5) * PI * sin(latitude) * uvDx.y),-18. * uvDx.y);
  let addressDy = vec2f(36. * (cos(latitude) * uvDy.x - (uv.x - .5) * PI * sin(latitude) * uvDy.y),-18. * uvDy.y);
  let majorLength = max(length(addressDx),length(addressDy));
  let singleLod = i32(clamp(floor(log2(max(1.,majorLength * ${SIZE}.))),0.,u.view.w - 1.));
  return loadScore(address,singleLod);
}
fn sampleScore(uv: vec2f,uvDx: vec2f,uvDy: vec2f) -> ScoreSample {
  let latitude = (uv.y - .5) * PI;
  let address = vec2f(18. + (uv.x - .5) * 36. * cos(latitude),(1. - uv.y) * 18.);
  let addressDx = vec2f(36. * (cos(latitude) * uvDx.x - (uv.x - .5) * PI * sin(latitude) * uvDx.y),-18. * uvDx.y);
  let addressDy = vec2f(36. * (cos(latitude) * uvDy.x - (uv.x - .5) * PI * sin(latitude) * uvDy.y),-18. * uvDy.y);
  let cellStep = max(abs(addressDx),abs(addressDy)) * ${SIZE}.;
  if (u.flags.z <= .5) {
    let s = loadScoreBilinear(address,0,cellStep);
    return ScoreSample(s.score,s.valid,1.);
  }
  let dxLength = length(addressDx);
  let dyLength = length(addressDy);
  let majorLength = max(dxLength,dyLength);
  let singleLodF = clamp(log2(max(1.,majorLength * ${SIZE}.)),0.,u.view.w - 1.);
  if (singleLodF <= .001) {
    let s = loadScoreBilinear(address,0,cellStep);
    return ScoreSample(s.score,s.valid,1.);
  }

  // Smoothly blend isotropic trilinear sampling with an up-to-7-tap anisotropic
  // footprint walk along grazing limbs. Average valid coverage across taps so
  // NO_SCORE boundaries (e.g. coastlines, glyphs) never smear outward at the limb.
  let xLonger = dxLength > dyLength;
  let major = select(addressDy,addressDx,xLonger);
  let crossStep = select(addressDx,addressDy,xLonger);
  let axis = major / max(majorLength,.000001);
  let span = majorLength + abs(dot(crossStep,axis));
  let width = abs(axis.x * crossStep.y - axis.y * crossStep.x);
  let anisoRatio = span / max(width,1e-6);
  let anisoW = smoothstep(1.35,2.4,anisoRatio) * smoothstep(.05,.45,singleLodF);
  let mipWeight = smoothstep(0.,2.2,singleLodF);
  let center = sampleScoreTrilinear(address,singleLodF,cellStep);
  if (anisoW <= .001) {
    let balanced = mix(1.,center.density,mipWeight * .45);
    return ScoreSample(center.score,center.valid,balanced);
  }
  let maxTaps = 7.;
  let anisoLodF = clamp(log2(max(1.,max(span / maxTaps,width) * ${SIZE}.)),0.,u.view.w - 1.);
  let taps = u32(clamp(2. * ceil((anisoRatio - 1.) * .5) + 1.,3.,maxTaps));
  var sumScore = 0.;
  var sumValid = 0.;
  var bestScore = 0.;
  for (var tap = 0u; tap < taps; tap++) {
    let offset = ((f32(tap) + .5) / f32(taps) - .5) * span;
    let s = loadScorePointTrilinear(address + axis * offset,anisoLodF);
    if (s.y > .001) {
      sumScore += s.x * s.y;
      sumValid += s.y;
      bestScore = max(bestScore,s.x);
    }
  }
  let anisoValid = sumValid / f32(taps);
  let anisoAvg = select(0.,sumScore / max(sumValid,1e-4),sumValid > .001);
  let anisoDensity = select(0.,clamp(anisoAvg / max(bestScore,1e-4),0.,1.),sumValid > .001);
  let finalValid = mix(center.valid,anisoValid,anisoW);
  if (finalValid <= .001) { return ScoreSample(0.,0.,0.); }
  let finalScore = (center.score * center.valid * (1. - anisoW) + anisoAvg * anisoValid * anisoW) / finalValid;
  let rawDensity = mix(center.density,anisoDensity,anisoW);
  let balanced = mix(1.,rawDensity,mipWeight * .45);
  return ScoreSample(finalScore,finalValid,balanced);
}
@fragment fn annotationFragment(input: SurfaceVertex) -> @location(0) vec2f {
  return surfaceUv(input);
}
struct AnnotationOutput { @location(0) uv: vec2f, @location(1) coverage: f32, }
@fragment fn coverageAnnotationFragment(input: SurfaceVertex) -> AnnotationOutput {
  let uv = surfaceUv(input);
  return AnnotationOutput(uv,select(1.,.5,scoreCode(uv) == 65535u));
}
struct EarthImagery { color: vec3f, tileWeight: f32, landMask: f32, }
fn sampleEarthImagery(uv: vec2f,uvDx: vec2f,uvDy: vec2f,coast: f32) -> EarthImagery {
  let imgUv = vec2f(uv.x,1. - uv.y);
  let imgDx = vec2f(uvDx.x,-uvDx.y);
  let imgDy = vec2f(uvDy.x,-uvDy.y);
  let coarseLand = smoothstep(-.25,.25,coast);
  if (u.status.w > .5) {
    let tileCoord = vec2f(fract(imgUv.x),clamp(imgUv.y,0.,.999999));
    let cell = vec2i(clamp(floor(tileCoord * vec2f(160.,80.)),vec2f(0.),vec2f(159.,79.)));
    let entry = textureLoad(earthTilePage,cell,0);
    if (entry.r < 255u) {
      let innerScale = 508. / 512.;
      var slot = entry.r;
      var grid = vec2f(40.,20.) * exp2(f32(entry.g));
      var gradScale = grid * innerScale;
      var dx = imgDx * gradScale;
      var dy = imgDy * gradScale;
      var tileMip = log2(max(1.,max(length(dx),length(dy)) * 512.));
      if (tileMip > 1.75 && entry.b < 255u) {
        slot = entry.b;
        grid = vec2f(40.,20.) * exp2(f32(entry.a));
        gradScale = grid * innerScale;
        dx = imgDx * gradScale;
        dy = imgDy * gradScale;
        tileMip = log2(max(1.,max(length(dx),length(dy)) * 512.));
      }
      if (tileMip < 2.8) {
        let localUv = fract(tileCoord * grid);
        let tileUv = localUv * innerScale + (2. / 512.);
        let high = textureSampleGrad(earthTilePool,earthTileSampler,tileUv,i32(slot),dx,dy);
        // Alpha encodes exact ~463m GEBCO + lake land fraction in [0.5 .. 1.0] (128=water, 255=land).
        let highLand = clamp((high.a - .5) * 2.,0.,1.);
        if (tileMip <= 2.0) { return EarthImagery(high.rgb,1.,highLand); }
        let base = textureSampleGrad(earth,geographicSampler,imgUv,imgDx,imgDy).rgb;
        let w = 1. - smoothstep(2.0,2.8,tileMip);
        return EarthImagery(mix(base,high.rgb,w),w,mix(coarseLand,highLand,w));
      }
    }
  }
  return EarthImagery(textureSampleGrad(earth,geographicSampler,imgUv,imgDx,imgDy).rgb,0.,coarseLand);
}
fn oceanHash3(p: vec3f) -> f32 {
  var q = fract(p * .1031);
  q += dot(q,q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
fn oceanNoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * f * (f * (f * 6. - 15.) + 10.);
  let n000 = oceanHash3(i);
  let n100 = oceanHash3(i + vec3f(1.,0.,0.));
  let n010 = oceanHash3(i + vec3f(0.,1.,0.));
  let n110 = oceanHash3(i + vec3f(1.,1.,0.));
  let n001 = oceanHash3(i + vec3f(0.,0.,1.));
  let n101 = oceanHash3(i + vec3f(1.,0.,1.));
  let n011 = oceanHash3(i + vec3f(0.,1.,1.));
  let n111 = oceanHash3(i + vec3f(1.,1.,1.));
  let nx0 = mix(vec2f(n000,n001),vec2f(n100,n101),u.x);
  let nx1 = mix(vec2f(n010,n011),vec2f(n110,n111),u.x);
  let nxy = mix(nx0,nx1,u.y);
  return mix(nxy.x,nxy.y,u.z);
}
// Seam-free 3D mesoscale ocean wind field (~150-1000 km fronts & calm slicks) modulating
// Cox-Munk RMS wave slope: calm patches sharpen/brighten the core and darken the skirt,
// while windy belts widen into a softer silver-gold sheen.
fn oceanWindRoughness(n: vec3f) -> f32 {
  let flow = vec3f(n.x,n.y * 1.45,n.z);
  let warp = vec3f(
    oceanNoise3(flow * 5.2 + vec3f(3.1,1.7,5.3)),
    oceanNoise3(flow * 5.2 + vec3f(8.4,2.9,1.2)),
    oceanNoise3(flow * 5.2 + vec3f(1.5,7.6,4.8)),
  ) - vec3f(.5);
  let p = flow + warp * .22;
  let wind = oceanNoise3(p * 7.5) * .52 + oceanNoise3(p * 17.5 + vec3f(4.2,1.9,6.7)) * .32 + oceanNoise3(p * 38. + vec3f(9.1,5.3,2.4)) * .16;
  return mix(.66,1.52,smoothstep(.24,.78,wind));
}
@fragment fn surfaceFragment(input: SurfaceVertex) -> @location(0) vec4f {
  let uv = surfaceUv(input);
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  let sample = sampleScore(uv,uvDx,uvDy);
  let score = sample.score;
  let base = u.style.x;
  let best = u.view.z;
  let visibleScore = sample.valid > .001 && score >= u.context.z;
  var strength = 0.;
  if (sample.valid > .001 && score > base && best > base) { strength = clamp((score - base) / max(.000001,best - base),0.,1.); }
  let automatic = u.flags.x > .5;
  let palette = select(palettePosition(strength,u.style.z,u.style.y),autoPalette(score,base,u.anchors),automatic);
  let contextStrength = clamp((score - u.context.x) / max(.000001,1. - u.context.x),0.,1.);
  let contextCoverageRaw = select(0.,smoothstep(0.,.22,contextStrength) * u.context.y * sample.valid,visibleScore);
  let contextHeat = magma(.08 + .11 * sqrt(contextStrength));
  // Progressive opacity: translucent violet outer fringe lets basemap relief breathe,
  // while high-scoring crimson/orange/gold hotspots become bold and unmistakable.
  let onset = smoothstep(${HIGHLIGHT_START} - .02,${HIGHLIGHT_FULL} + .02,palette);
  let tier = smoothstep(${HIGHLIGHT_START},.90,palette);
  let densityMod = mix(.64,1.,smoothstep(.15,.85,sample.density));
  let progressive = onset * mix(.44,1.20,pow(tier,.85)) * densityMod;
  let coverageRaw = select(0.,clamp(progressive * u.context.w * sample.valid,0.,.95),visibleScore && best > base);
  let terrainSample = terrainPair(uv,uvDx,uvDy);
  let height = terrainSample.r;
  let coast = terrainSample.g;
  let satSample = sampleEarthImagery(uv,uvDx,uvDy,coast);
  let satellite = satSample.color;
  let highResImagery = satSample.tileWeight;
  let exactLandMask = satSample.landMask;
  let waterMask = 1. - exactLandMask;
  let light = textureSampleGrad(illumination,geographicSampler,vec2f(uv.x,1. - uv.y),vec2f(uvDx.x,-uvDx.y),vec2f(uvDy.x,-uvDy.y)).r;
  // Physical solar & viewing vectors (used for water volume SSS, Fresnel sky sheen, ocean glint, and atmosphere).
  let N = normalize(input.geodetic);
  let V = normalize(u.eye.xyz - input.worldPos);
  let L = normalize(u.sun.xyz);
  // DPR-normalized minor singular value of dN/d(screen) gives the true angular scale of the
  // globe everywhere (pole- and limb-free), preventing max-pyramid inflation and bloom overreach
  // when Earth shrinks to a tiny distant disc (e.g., viewed from the Moon).
  let nDx = dpdx(N);
  let nDy = dpdy(N);
  let nxx = dot(nDx,nDx);
  let nyy = dot(nDy,nDy);
  let nxy = dot(nDx,nDy);
  let nDet = max(0.,nxx * nyy - nxy * nxy);
  let nMax = max(1e-12,.5 * (nxx + nyy + length(vec2f(nxx - nyy,2. * nxy))));
  let globeScaleLod = log2(max(1.,sqrt(nDet / nMax) * (18. * ${SIZE}. / PI) * u.contour.w));
  let farEmberFade = 1. - smoothstep(3.5,5.2,globeScaleLod) * u.view.x;
  let farInflation = smoothstep(3.8,6.6,globeScaleLod) * u.view.x * select(0.,1.,u.flags.z > .5);
  let displayPalette = palette * mix(1.,.72,farInflation);
  let contextCoverage = contextCoverageRaw * mix(1.,.40,farInflation);
  let coverage = coverageRaw * mix(1.,.42,farInflation);
  let oneMinusV = clamp(1. - dot(N,V),1.e-4,1.);
  let ndotl = dot(N,L);
  let sunDay = smoothstep(-.10,.22,ndotl);
  let macroLight = .493 + .46 * smoothstep(-.035,.045,ndotl) * max(0.,ndotl);
  let localRelief = clamp(light / max(.25,macroLight),.72,1.28);
  // Utilitarian flat map uses even cartographic hillshade across all longitudes instead of a dark night half.
  let flatLight = clamp(.82 * localRelief,.56,1.02);
  let surfaceLight = mix(flatLight,light,u.view.x);

  var background = vec3f(.17,.185,.20);
  var waterSssLinear = vec3f(0.);
  if (u.flags.w > .5) {
    background = satellite * .78;
  } else {
    // Dark cartographic satellite basemap: brings out real biomes, mountain chains,
    // and optical Beer-Lambert bathymetry while softly compressing polar ice caps.
    var land = vec3f(.19,.205,.22);
    if (u.status.z > .5) {
      let luma = dot(satellite,vec3f(.299,.587,.114));
      let iceCompress = 1. / (1. + max(0.,luma - .42) * 1.65);
      let muted = mix(vec3f(luma),satellite,.60) * iceCompress;
      land = vec3f(.10,.115,.13) + muted * .52;
    }
    land += vec3f(.05,.055,.06) * clamp(height / 5500.,0.,1.);

    // Per-channel Beer-Lambert optical water absorption, tuned to the dark-cartographic land
    // luminance so shallow shelves stay restrained & harmonious while depth stays unmistakable:
    //  0 .. -200m (shelves/gulfs): muted coastal slate-teal tinted by muted local satellite hue
    //  -200 .. -2000m (slopes/banks): muted marine cerulean-slate
    //  -2000 .. -4500m (ridges/basins): deep oceanic navy
    //  -4500 .. -10000m (hadal trenches): dark midnight indigo
    let depth = select(max(0.,-height),240.,height >= 0. && waterMask > .05);
    let absorb = exp(-depth / vec3f(145.,850.,3600.));
    let trench = exp(-max(0.,depth - 3200.) / 3600.);
    let shelfZone = exp(-depth / 260.);
    let slopeZone = exp(-depth / 1800.);
    let deepWater = vec3f(.024,.042,.076) * (.58 + .42 * trench) + vec3f(.014,.052,.110) * absorb.b * .88;
    let midSlope = vec3f(.028,.098,.155) * absorb.g * .74;
    let shallowTeal = vec3f(.042,.128,.152) * (absorb.r * .45 + absorb.g * .55) * .56;
    var ocean = deepWater + midSlope * (1. - .35 * shelfZone) + shallowTeal * shelfZone;
    // Tint shallow water with muted local satellite chroma so reefs, bays, and warm desert coasts blend naturally.
    if (u.status.z > .5) {
      let satLuma = dot(satellite,vec3f(.299,.587,.114));
      let satMuted = mix(vec3f(satLuma),satellite,.58);
      let satWater = vec3f(.028,.044,.064) + satMuted * vec3f(.34,.42,.46);
      let hasLocalWaterColor = smoothstep(.04,.24,max(satellite.g,satellite.b));
      ocean = mix(ocean,satWater,mix(.16,.54,shelfZone * hasLocalWaterColor));
      ocean *= (.90 + .22 * vec3f(satMuted.r * .5 + satMuted.g * .5,satMuted.g * .65 + satMuted.b * .35,satMuted.b));
    }
    // Sculpted underwater bathymetry hillshade: continental slopes, mid-ocean ridges, and trenches read in 3D.
    let seafloorRelief = clamp(light / max(.25,macroLight),.55,1.48);
    ocean *= mix(1.,seafloorRelief,.74 * (1. - .30 * shelfZone));
    // Two-layer water optics: look into the depth volume at nadir; cool sky Fresnel sheen at grazing angles.
    let skySheen = vec3f(.09,.18,.32) * pow(oneMinusV,3.2) * .28 * (.30 + .70 * sunDay) * u.view.x;
    ocean += skySheen * (1. - .45 * slopeZone);
    background = mix(ocean,land,exactLandMask);

    // Subtle sunlit shallow-water subsurface scattering (soft sea-glass lift over shallow shelves & reefs).
    let sssDepth = exp(-depth / 200.) * (1. - .38 * oneMinusV);
    let sssSun = mix(.58,smoothstep(-.02,.28,ndotl),u.view.x);
    let sssTint = mix(vec3f(.025,.145,.185),clamp(satellite * vec3f(.32,.56,.58),vec3f(.02,.10,.14),vec3f(.12,.28,.30)),.45);
    waterSssLinear = sssTint * sssDepth * sssSun * .11 * waterMask * (1. - coverage);
  }
  let interval = select(u.contour.x,u.contour.y,height < 0.);
  let contour = terrainContours(height,vec2f(dpdx(height),dpdy(height)),interval,u.detail.xy,u.contour.w,1.);
  let contourColor = select(vec3f(.40,.43,.47),vec3f(.24,.44,.58),height < 0.);
  background = mix(background,contourColor,contour * u.contour.z * select(.55,1.,u.flags.w < .5 && u.status.z < .5));
  let coastPixels = abs(coast) / max(.0001,fwidth(coast));
  let shore = 1. - smoothstep(.35,1.15,coastPixels);
  // Completely fade out the coarse 4K shoreline vector once 500 m alpha-masked shorelines take over.
  let shoreStrength = .36 * (1. - highResImagery) * (1. - .75 * coverage);
  background = mix(background,vec3f(.20,.54,.76),select(0.,shore * shoreStrength,u.status.x > .5));
  // Physical Cox-Munk / GGX ocean sun glint with V·H Schlick Fresnel (F0 = 0.02) and
  // two-way Rayleigh/aerosol airmass reddening near the terminator (fades out on flat map).
  let hv = L + V;
  let H = hv * inverseSqrt(max(1.e-6,dot(hv,hv)));
  let ndoth = clamp(dot(N,H),0.,1.);
  let vdoth = clamp(dot(V,H),0.,1.);
  let nlSpec = max(ndotl,0.);
  let nvSpec = max(dot(N,V),.05);
  let specDay = smoothstep(-.02,.07,ndotl);
  let glintMask = waterMask * specDay * u.view.x * (1. - coverage);
  var specLinear = vec3f(0.);
  if (glintMask > .001 && ndoth > .80) {
    let fv = 1. - vdoth;
    let fv2 = fv * fv;
    let glintFresnel = .02 + .98 * (fv2 * fv2 * fv);
    let roughScale = oceanWindRoughness(N);
    let a1 = .052 * roughScale;
    let a1Sq = a1 * a1;
    let d1Den = ndoth * ndoth * (a1Sq - 1.) + 1.;
    let d1 = a1Sq / (PI * d1Den * d1Den);
    let vis1 = .5 * nlSpec / max(nlSpec * sqrt(nvSpec * nvSpec * (1. - a1Sq) + a1Sq) + nvSpec * sqrt(nlSpec * nlSpec * (1. - a1Sq) + a1Sq),1e-4);
    let a2 = .125 * roughScale;
    let a2Sq = a2 * a2;
    let d2Den = ndoth * ndoth * (a2Sq - 1.) + 1.;
    let d2 = a2Sq / (PI * d2Den * d2Den);
    let vis2 = .5 * nlSpec / max(nlSpec * sqrt(nvSpec * nvSpec * (1. - a2Sq) + a2Sq) + nvSpec * sqrt(nlSpec * nlSpec * (1. - a2Sq) + a2Sq),1e-4);
    let airmass = 1. / (nlSpec + .08) + 1. / (nvSpec + .08);
    let airTrans = exp(-max(0.,airmass - 2.) * vec3f(.15,.24,.40));
    let specBrdf = (d1 * vis1 * .72 + d2 * vis2 * .28) * glintFresnel * smoothstep(.80,.87,ndoth);
    let specRaw = vec3f(1.,.97,.92) * airTrans * (specBrdf * .38) * glintMask;
    specLinear = specRaw / (vec3f(1.) + specRaw * .10);
  }
  let fresnel = pow(oneMinusV,4.2);
  let horizon = pow(oneMinusV,8.0);
  // Subtle warm sunset scatter along the grazing terminator limb, boosted by Mie forward scatter when backlit.
  let backScatter = pow(max(0.,dot(-V,L)),5.0);
  let twilight = exp(-pow((ndotl + .01) / .13,2.)) * pow(oneMinusV,4.2) * (1. + 2.4 * backScatter);
  let atmoLinear = (
    vec3f(.20,.52,1.0) * (fresnel * .28 + horizon * .52) * sunDay +
    vec3f(1.,.42,.12) * twilight * .13 +
    vec3f(.05,.12,.24) * horizon * .22 * (1. - sunDay)
  ) * u.view.x * (1. - .55 * coverage);
  // Cool moonlit cartographic fill on the night hemisphere so continents, islands, coastlines,
  // and ocean bathymetry stay clearly readable through the ACES shadow toe while remaining
  // strictly darker and cooler than the sunlit day side.
  let baseLinear = toLinear(background);
  let nightOcean = waterMask * (baseLinear * vec3f(.14,.22,.32) + vec3f(.0024,.0054,.0118) * (.72 + .28 * localRelief));
  let nightLand = exactLandMask * (baseLinear * vec3f(.10,.13,.19) + vec3f(.0038,.0048,.0068) * localRelief);
  let nightContour = select(vec3f(.0045,.0055,.0075),vec3f(.0030,.0080,.0150),height < 0.) * contour * u.contour.z;
  let nightShore = vec3f(.0055,.016,.028) * select(0.,shore * shoreStrength,u.status.x > .5);
  let nightFillLinear = (nightOcean + nightLand + nightContour + nightShore) * (1. - coverage) * (1. - sunDay) * u.view.x;
  // Day/night-independent local relief hillshade draped onto the heatmap so mountains,
  // ridges, and basins stay legible beneath highlighted scores.
  let reliefShade = mix(1.,localRelief,.55 * (1. - .60 * pow(displayPalette,3.)));
  var heat = magma(displayPalette) * reliefShade;
  // Let subtle elevation contours etch through zoomed-in hotspots.
  heat = mix(heat,heat * .80,contour * u.contour.z * .38);
  // Crisp warm-gold iso-contour ring around top-tier peaks when zoomed into coherent fields.
  let pGrad = max(1e-4,fwidth(palette));
  let isoRing = (1. - smoothstep(.6,1.8,abs(palette - .84) / pGrad)) * smoothstep(.24,.06,pGrad);
  heat = mix(heat,vec3f(1.,.86,.52),isoRing * .32 * select(0.,1.,visibleScore && best > base));
  var result = mix(mix(background * surfaceLight,contextHeat,contextCoverage),heat,coverage);
  if (u.detail.w > .5) {
    let context = mix(toLinear(background * surfaceLight) + waterSssLinear + specLinear + nightFillLinear,toLinear(contextHeat),contextCoverage);
    let heatLinear = toLinear(clamp(heat,vec3f(0.),vec3f(1.))) * mix(.68,1.16,farEmberFade);
    let peakGlow = pow(select(strength,palette,automatic),5.5) * densityMod * farEmberFade;
    let ember = mix(heatLinear,vec3f(1.,.58,.14),.28) * (u.style.w * 1.15 * peakGlow);
    result = mix(context,heatLinear + ember,coverage) + atmoLinear;
  } else {
    result = result + pow(max(waterSssLinear + specLinear + nightFillLinear + atmoLinear,vec3f(0.)),vec3f(1. / 2.2));
  }
  if (u.flags.y > .5) {
    let lines = uv * vec2f(24.,12.);
    let distanceToLine = abs(fract(lines + .5) - .5) / max((abs(uvDx) + abs(uvDy)) * vec2f(24.,12.),vec2f(.0001));
    let line = 1. - smoothstep(.4,1.1,min(distanceToLine.x,distanceToLine.y));
    let gridColor = vec3f(.61,.72,.77);
    result = mix(result,select(gridColor,toLinear(gridColor),u.detail.w > .5),line * .17);
  }
  // Subtle ~1 CSS px neatline framing the Equal Earth flat map against deep space.
  let flatEdgeWeight = 1. - smoothstep(.02,.22,u.view.x);
  if (flatEdgeWeight > .001) {
    let edgeDistX = min(input.uv.x,1. - input.uv.x) / max(1e-6,length(vec2f(dpdx(input.uv.x),dpdy(input.uv.x))));
    let edgeDistY = min(input.uv.y,1. - input.uv.y) / max(1e-6,length(vec2f(dpdx(input.uv.y),dpdy(input.uv.y))));
    let edgeCssPx = min(edgeDistX,edgeDistY) / max(1.,u.contour.w);
    let neatline = (1. - smoothstep(.65,1.65,edgeCssPx)) * flatEdgeWeight;
    let borderColor = vec3f(.38,.48,.60);
    result = mix(result,select(borderColor,toLinear(borderColor),u.detail.w > .5),neatline * .44);
  }
  return vec4f(result,1.);
}`
}
