import { colorShader, magmaShader } from './color.ts'
import { contourShader } from './contours.ts'
import { autoStyleShader, HIGHLIGHT_FULL, HIGHLIGHT_START, scoreStyleShader } from './score-style.ts'
import { SIZE } from './world-grid.ts'

// Every field after the matrix is a vec4, shared with the packed CPU uniform
// buffer. Keeping the layout explicit avoids per-frame bind-group allocation.
export const surfaceUniforms = `
struct SurfaceUniforms {
  camera: mat4x4f,
  anchors: vec4f,
  style: vec4f, // baseline, contrast, top band, emission
  context: vec4f, // floor, opacity, minimum filter, highlight opacity
  view: vec4f, // morph, relief, best score, mip count
  contour: vec4f, // land interval, depth interval, opacity, pixel ratio
  detail: vec4f, // minor/fine contour opacity, reserved, HDR
  flags: vec4f, // automatic palette, grid, preserve peaks, imagery
  status: vec4f, // terrain ready, native depth, reserved
  poles: vec4f,
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
struct SurfaceVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) geodetic: vec3f,
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
  let plane = vec3f(lon,sin(lat),height);
  let position = vec4f(mix(plane,sphere,u.view.x),1.);
  var clip = u.camera * position;
  // The standalone lab uses OpenGL depth. The solar scene uses native reverse-Z.
  if (u.status.y < .5) { clip.z = (clip.z + clip.w) * .5; }
  return SurfaceVertex(clip,uv,normal);
}
fn wrappedGradient(delta: vec2f) -> vec2f {
  return vec2f(delta.x - floor(delta.x + .5),delta.y);
}
fn terrainAt(uv: vec2f,dx: vec2f,dy: vec2f,minimumLod: f32) -> vec2f {
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
  let minimum = exp2(minimumLod);
  return terrainGradient(vec2f(uv.x,1. - uv.y),axis * max(major,minimum) / size,
    vec2f(-axis.y,axis.x) * max(minor,minimum) / size);
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
  let cell = clamp(address,vec2f(0.),vec2f(35.99999,17.99999));
  let slot = textureLoad(lookup,vec2i(floor(cell)),0).r;
  var code = 65535u;
  if (slot != 65535u) {
    let bank = slot / ${bankSize}u;
    let layer = i32(slot % ${bankSize}u);
    let size = ${SIZE} >> u32(lod);
    let pixel = vec2i(fract(cell) * f32(size));
    ${Array.from({ length: banks },(_,i) => `${i ? 'else ' : ''}if (bank == ${i}u) {
      code = textureLoad(scores${i},pixel,layer,lod).r;
    }`).join('\n')}
  }
  return code;
}
fn scoreCode(uv: vec2f) -> u32 {
  let latitude = (uv.y - .5) * PI;
  let address = vec2f(18. + (uv.x - .5) * 36. * cos(latitude),(1. - uv.y) * 18.);
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  if (u.flags.z <= .5) { return loadScore(address,0); }
  let addressDx = vec2f(36. * (cos(latitude) * uvDx.x - (uv.x - .5) * PI * sin(latitude) * uvDx.y),-18. * uvDx.y);
  let addressDy = vec2f(36. * (cos(latitude) * uvDy.x - (uv.x - .5) * PI * sin(latitude) * uvDy.y),-18. * uvDy.y);
  let dxLength = length(addressDx);
  let dyLength = length(addressDy);
  let majorLength = max(dxLength,dyLength);
  let singleLod = i32(clamp(floor(log2(max(1.,majorLength * ${SIZE}.))),0.,u.view.w - 1.));
  if (singleLod == 0) { return loadScore(address,singleLod); }

  // Longitude shear and grazing angles stretch the pixel footprint. Sample
  // along its long axis so the max mip does not widen peaks across it.
  let xLonger = dxLength > dyLength;
  let major = select(addressDy,addressDx,xLonger);
  let crossStep = select(addressDx,addressDy,xLonger);
  let axis = major / max(majorLength,.000001);
  let span = majorLength + abs(dot(crossStep,axis));
  let width = abs(axis.x * crossStep.y - axis.y * crossStep.x);
  let maxTaps = 7.;
  let footprint = max(span / maxTaps,width) * ${SIZE}.;
  let lod = i32(clamp(floor(log2(max(1.,footprint))),0.,u.view.w - 1.));
  if (lod >= singleLod) { return loadScore(address,singleLod); }
  // An odd tap count includes the score at the center of the pixel.
  let taps = u32(min(2. * ceil((span * ${SIZE}. / exp2(f32(lod)) - 1.) * .5) + 1.,maxTaps));
  var best = 65535u;
  for (var tap = 0u; tap < taps; tap++) {
    let offset = ((f32(tap) + .5) / f32(taps) - .5) * span;
    let score = loadScore(address + axis * offset,lod);
    if (score != 65535u) { best = max(select(best,0u,best == 65535u),score); }
  }
  return best;
}
@fragment fn annotationFragment(input: SurfaceVertex) -> @location(0) vec2f {
  return surfaceUv(input);
}
struct AnnotationOutput { @location(0) uv: vec2f, @location(1) coverage: f32, }
@fragment fn coverageAnnotationFragment(input: SurfaceVertex) -> AnnotationOutput {
  let uv = surfaceUv(input);
  return AnnotationOutput(uv,select(1.,.5,scoreCode(uv) == 65535u));
}
@fragment fn surfaceFragment(input: SurfaceVertex) -> @location(0) vec4f {
  let uv = surfaceUv(input);
  let uvDx = wrappedGradient(dpdx(uv));
  let uvDy = wrappedGradient(dpdy(uv));
  let latitude = (uv.y - .5) * PI;
  let code = scoreCode(uv);
  let score = f32(code) / 65534.;
  let base = u.style.x;
  let best = u.view.z;
  let visibleScore = code != 65535u && score >= u.context.z;
  var strength = 0.;
  if (code != 65535u && score > base && best > base) { strength = clamp((score - base) / max(.000001,best - base),0.,1.); }
  let automatic = u.flags.x > .5;
  let palette = select(palettePosition(strength,u.style.z,u.style.y),autoPalette(score,base,u.anchors),automatic);
  let contextStrength = clamp((score - u.context.x) / max(.000001,1. - u.context.x),0.,1.);
  let contextCoverage = select(0.,smoothstep(0.,.10,contextStrength) * u.context.y,visibleScore);
  let contextHeat = magma(.10 + .12 * sqrt(contextStrength));
  let coverage = select(0.,smoothstep(${HIGHLIGHT_START},${HIGHLIGHT_FULL},palette) * u.context.w,visibleScore && best > base);
  var background = vec3f(.16,.17,.18);
  if (u.flags.w > .5) { background = textureSampleGrad(earth,geographicSampler,vec2f(uv.x,1. - uv.y),vec2f(uvDx.x,-uvDx.y),vec2f(uvDy.x,-uvDy.y)).rgb * .78; }
  let height = terrainAt(uv,uvDx,uvDy,1.5).r;
  let coast = terrainAt(uv,uvDx,uvDy,0.).g;
  if (u.flags.w < .5) { background = select(background,vec3f(.055,.065,.075),coast < 0.); }
  let interval = select(u.contour.x,u.contour.y,height < 0.);
  let contour = terrainContours(height,vec2f(dpdx(height),dpdy(height)),interval,u.detail.xy);
  let contourColor = select(vec3f(.37,.40,.43),vec3f(.20,.30,.38),height < 0.);
  background = mix(background,contourColor,contour * u.contour.z);
  let coastPixels = abs(coast) / max(.0001,fwidth(coast));
  let shore = 1. - smoothstep(.35,1.1,coastPixels);
  background = mix(background,vec3f(.08,.43,.68),select(0.,shore * .85,u.status.x > .5));
  let heat = magma(palette);
  let light = textureSampleGrad(illumination,geographicSampler,vec2f(uv.x,1. - uv.y),vec2f(uvDx.x,-uvDx.y),vec2f(uvDy.x,-uvDy.y)).r;
  var result = mix(mix(background * light,contextHeat,contextCoverage),heat,coverage);
  if (u.detail.w > .5) {
    let context = mix(toLinear(background * light),toLinear(contextHeat),contextCoverage);
    result = mix(context,toLinear(heat) * (1. + u.style.w * pow(select(strength,palette,automatic),12.)),coverage);
  }
  if (u.flags.y > .5) {
    let lines = uv * vec2f(24.,12.);
    let distanceToLine = abs(fract(lines + .5) - .5) / max((abs(uvDx) + abs(uvDy)) * vec2f(24.,12.),vec2f(.0001));
    let line = 1. - smoothstep(.4,1.1,min(distanceToLine.x,distanceToLine.y));
    let gridColor = vec3f(.61,.72,.77);
    result = mix(result,select(gridColor,toLinear(gridColor),u.detail.w > .5),line * .17);
  }
  return vec4f(result,1.);
}`
}
