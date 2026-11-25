override PLANET_TILE_PAIRS: u32 = 48u;

struct View {
  camera: mat4x4f,
  sun: vec4f,        // body-fixed sun direction, HDR flag
  terrain: vec4f,    // reference radius metres, vertical exaggeration, major contours, DPR
  low: vec4f,        // sRGB low albedo, minor contour detail
  high: vec4f,       // sRGB high albedo, fine contour detail
  tiles: vec4f,      // x: virtual tile pool active (0/1), y: reserved, z: small-planet radial pad, w: relief fade
  eye: vec4f,        // body-local camera position (km), w: body kind (0 = Moon, 1 = Mars)
  camRight: vec4f,   // body-local camera right, w: tan(FOV/2) * aspect
  camUp: vec4f,      // body-local camera up, w: tan(FOV/2)
  camBack: vec4f,    // body-local camera back, w: view depth of body center (km)
  centerInfo: vec4f, // centerNdcX, centerNdcY, 2 / outWidth, 2 / outHeight
}

@group(0) @binding(0) var<uniform> u: View;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var albedoSampler: sampler;
@group(0) @binding(5) var tilePage: texture_2d<u32>;
@group(0) @binding(6) var tilePool: texture_2d_array<f32>;
@group(0) @binding(7) var tileSampler: sampler;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec3f,
}

fn computePlanetVertex(position: vec3f) -> Vertex {
  let smallPad = u.tiles.z;
  let reliefFade = u.tiles.w;
  let rSurf = u.terrain.x * .001;
  let refPos = normalize(position) * rSurf;
  let basePos = mix(refPos, position, reliefFade);
  let pos = select(basePos, refPos * (1. + smallPad), smallPad > 0.);
  return Vertex(u.camera * vec4f(pos, 1.), pos);
}

@vertex fn vertexMain(@location(0) position: vec3f) -> Vertex {
  return computePlanetVertex(position);
}

struct SmallVertex {
  @builtin(position) position: vec4f,
}

@vertex fn smallVertex(@builtin(vertex_index) vertex: u32) -> SmallVertex {
  let corners = array<vec2f, 6>(vec2f(-1., -1.), vec2f(1., -1.), vec2f(-1., 1.), vec2f(-1., 1.), vec2f(1., -1.), vec2f(1., 1.));
  let rSurf = u.terrain.x * .001;
  let depth = u.camBack.w;
  if (depth <= rSurf * 1.001) { return SmallVertex(vec4f(2., 2., 0., 1.)); }
  let clipCenter = u.camera * vec4f(0., 0., 0., 1.);
  if (clipCenter.w <= 0.) { return SmallVertex(vec4f(2., 2., 0., 1.)); }
  let center = clipCenter.xy / clipCenter.w;
  let cx = -dot(u.eye.xyz, u.camRight.xyz);
  let cy = -dot(u.eye.xyz, u.camUp.xyz);
  let r2 = rSurf * rSurf;
  let d2x = cx * cx + depth * depth;
  let d2y = cy * cy + depth * depth;
  let denom = max(depth * depth - r2, 1e-6);
  let extX = (rSurf * sqrt(max(0., d2x - r2)) + abs(cx) * r2 / depth) / (denom * u.camRight.w);
  let extY = (rSurf * sqrt(max(0., d2y - r2)) + abs(cy) * r2 / depth) / (denom * u.camUp.w);
  let pad = vec2f(3. * u.centerInfo.z, 3. * u.centerInfo.w);
  let corner = center + corners[vertex] * (vec2f(extX, extY) + pad);
  return SmallVertex(vec4f(corner, 0., 1.));
}

struct SmallOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

fn dirToUv(dir: vec3f) -> vec2f {
  let pi = 3.141592653589793;
  let lon = atan2(dir.x, dir.z);
  let lat = atan2(dir.y, length(dir.xz));
  return vec2f(lon / (2. * pi) + .5, .5 - lat / pi);
}

fn wrappedGrad(g: vec2f) -> vec2f {
  return vec2f(g.x - round(g.x), g.y);
}

fn decode(sample: vec4f) -> f32 {
  return dot(sample.rg, vec2f(65280., 255.)) - 32768.;
}

// 5-tap Catmull-Rom bicubic height filter for smooth zoomed-in crater rims & contours.
fn sampleHeightBicubic(uv: vec2f) -> f32 {
  let size = vec2f(textureDimensions(heights));
  let invSize = 1. / size;
  let pos = uv * size - .5;
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
  let s0 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc0.y), 0.));
  let s1 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc0.x, tc12.y), 0.));
  let s2 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc12.y), 0.));
  let s3 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc3.x, tc12.y), 0.));
  let s4 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc3.y), 0.));
  let sum = s0 * (w12.x * w0.y) + s1 * (w0.x * w12.y) + s2 * (w12.x * w12.y)
    + s3 * (w3.x * w12.y) + s4 * (w12.x * w3.y);
  let total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return sum / total;
}

fn samplePlanetHeight(uv: vec2f, rawLod: f32, maxLod: f32) -> f32 {
  var h = decode(textureSampleLevel(heights, gridSampler, uv, clamp(rawLod, 0., maxLod)));
  if (rawLod < -1.) {
    h = mix(h, sampleHeightBicubic(uv), 1. - smoothstep(-2., -1., rawLod));
  }
  return h;
}

fn shadePlanet(
  p: vec3f, localPos: vec3f, smallAngle: f32,
  pDx: vec3f, pDy: vec3f, uv: vec2f, uvDx: vec2f, uvDy: vec2f,
  texelAngle: f32, pixelAngle: f32, lod: f32, height: f32, hGrad: vec2f,
) -> vec3f {
  let pi = 3.141592653589793;
  // Minor singular value of dp/d(screen) gives the un-foreshortened angular scale everywhere.
  let pxx = dot(pDx, pDx);
  let pyy = dot(pDy, pDy);
  let pxy = dot(pDx, pDy);
  let pDet = max(0., pxx * pyy - pxy * pxy);
  let pMax = max(1e-12, .5 * (pxx + pyy + length(vec2f(pxx - pyy, 2. * pxy))));
  let minorAngle = max(sqrt(pDet / pMax), smallAngle);
  var tileAlbedo = vec3f(0.);
  var tileWeight = 0.;
  var tileNormal = p;
  var tileNormalWeight = 0.;
  if (u.tiles.x > .5) {
    let tileCoord = vec2f(fract(uv.x), clamp(uv.y, 0., .999999));
    let cell = vec2i(clamp(floor(tileCoord * vec2f(48., 24.)), vec2f(0.), vec2f(47., 23.)));
    let entry = textureLoad(tilePage, cell, 0);
    if (entry.r < 255u) {
      let innerScale = 508. / 512.;
      var slot = entry.r;
      var grid = vec2f(24., 12.) * exp2(f32(entry.g));
      var gradScale = grid * innerScale;
      var dx = uvDx * gradScale;
      var dy = uvDy * gradScale;
      var tileMip = log2(max(1., max(length(dx), length(dy)) * 512.));
      if (tileMip > 1.75 && entry.b < 255u) {
        slot = entry.b;
        grid = vec2f(24., 12.) * exp2(f32(entry.a));
        gradScale = grid * innerScale;
        dx = uvDx * gradScale;
        dy = uvDy * gradScale;
        tileMip = log2(max(1., max(length(dx), length(dy)) * 512.));
      }
      if (tileMip < 2.8) {
        let localUv = fract(tileCoord * grid);
        let tileUv = localUv * innerScale + (2. / 512.);
        tileWeight = 1. - smoothstep(2.0, 2.8, tileMip);
        tileAlbedo = textureSampleGrad(tilePool, tileSampler, tileUv, i32(slot), dx, dy).rgb;
        let polarSafe = smoothstep(.04, .14, length(p.xz));
        if (polarSafe > .001) {
          let nSample = textureSampleGrad(tilePool, tileSampler, tileUv, i32(slot + PLANET_TILE_PAIRS), dx, dy).rgb;
          let tn = normalize(vec3f(nSample.x * 2. - 1., nSample.y * 2. - 1., max(.05, nSample.z)));
          let east = normalize(vec3f(p.z, 0., -p.x));
          let north = cross(p, east);
          tileNormal = normalize(east * tn.x + north * tn.y + p * tn.z);
          tileNormalWeight = tileWeight * polarSafe;
        }
      }
    }
  }
  var hillNormal = tileNormal;
  if (tileNormalWeight < .999) {
    // 4-tap bilinear tangent central differences (linearly interpolates per-texel slopes)
    // instead of 6x5 bicubic taps; falls back to 3D direction steps near the poles.
    let lon = (uv.x - .5) * 2. * pi;
    let lat = (.5 - uv.y) * pi;
    let cosLon = cos(lon);
    let sinLon = sin(lon);
    let cosLat = cos(lat);
    let sinLat = sin(lat);
    let east = vec3f(cosLon, 0., -sinLon);
    let north = vec3f(-sinLat * sinLon, cosLat, -sinLat * cosLon);
    let s = min(.25, texelAngle * exp2(lod));
    var hE: f32;
    var hW: f32;
    var hN: f32;
    var hS: f32;
    if (abs(lat) < 1.48) {
      let du = s / (2. * pi * max(cosLat, .05));
      let dv = s / pi;
      hE = decode(textureSampleLevel(heights, gridSampler, uv + vec2f(du, 0.), lod));
      hW = decode(textureSampleLevel(heights, gridSampler, uv - vec2f(du, 0.), lod));
      hN = decode(textureSampleLevel(heights, gridSampler, uv - vec2f(0., dv), lod));
      hS = decode(textureSampleLevel(heights, gridSampler, uv + vec2f(0., dv), lod));
    } else {
      hE = decode(textureSampleLevel(heights, gridSampler, dirToUv(p + east * s), lod));
      hW = decode(textureSampleLevel(heights, gridSampler, dirToUv(p - east * s), lod));
      hN = decode(textureSampleLevel(heights, gridSampler, dirToUv(p + north * s), lod));
      hS = decode(textureSampleLevel(heights, gridSampler, dirToUv(p - north * s), lod));
    }
    let scale = u.terrain.y / (2. * s * u.terrain.x);
    let baseNormal = normalize(p - (east * (hE - hW) + north * (hN - hS)) * scale);
    hillNormal = normalize(mix(baseNormal, tileNormal, tileNormalWeight));
  }
  let sun = normalize(u.sun.xyz);
  let viewDir = normalize(u.eye.xyz - localPos);
  let pDotL = dot(p, sun);
  let daylight = smoothstep(-.035, .045, pDotL);
  let ndotl = max(0., dot(hillNormal, sun));
  let ndotv = max(.04, dot(hillNormal, viewDir));
  let isMars = u.eye.w > .5;
  var diffuse = ndotl;
  if (!isMars) {
    // Lommel-Seeliger lunar regolith BRDF: flat dusty opposition brightness + punchy crater relief near the terminator.
    let ls = (2. * ndotl) / max(.06, ndotl + ndotv);
    diffuse = mix(ndotl, ls, .72);
  } else {
    // Oren-Nayar rough desert dust BRDF (sigma = 0.72 rad) for authentic Martian matte-regolith falloff.
    let s2 = .5184;
    let A = 1. - .5 * s2 / (s2 + .33);
    let B = .45 * s2 / (s2 + .09);
    let lProj = sun - hillNormal * ndotl;
    let vProj = viewDir - hillNormal * ndotv;
    let lpLen = length(lProj);
    let vpLen = length(vProj);
    let cosPhi = select(0., clamp(dot(lProj, vProj) / max(1e-5, lpLen * vpLen), -1., 1.), lpLen > 1e-4 && vpLen > 1e-4);
    let sinA = sqrt(max(0., 1. - min(ndotl, ndotv) * min(ndotl, ndotv)));
    let tanB = sqrt(max(0., 1. - max(ndotl, ndotv) * max(ndotl, ndotv))) / max(.04, max(ndotl, ndotv));
    diffuse = ndotl * (A + B * max(0., cosPhi) * sinA * tanB);
  }
  // Distance-aware unified linear lighting: keep dramatic crescent phases when tiny in the
  // background (e.g. Moon seen from Earth), while lifting linear ambient smoothly by the
  // focused orbital view (~200 CSS px radius) so night-side craters & dark basalt stay readable
  // and strictly darker than any sunlit pixel across the terminator.
  let inspect = 1. - smoothstep(.0040, .022, minorAngle * u.terrain.w);
  let fillSide = normalize(cross(viewDir, vec3f(0., 1., 0.)) + vec3f(1e-4, 0., 0.));
  let fillDir = normalize(p + .45 * viewDir + .22 * fillSide + .18 * vec3f(0., 1., 0.));
  let fill = max(0., dot(hillNormal, fillDir));
  let ambientLinear = select(
    mix(.022, .12, inspect) + mix(.012, .065, inspect) * fill,
    mix(.026, .13, inspect) + mix(.014, .070, inspect) * fill,
    isMars,
  );
  let sunLinear = select(
    mix(1.05, .94, inspect),
    mix(.98, .88, inspect),
    isMars,
  ) * daylight * diffuse;
  let ambientTint = select(vec3f(.86, .93, 1.06), vec3f(1., .95, .91), isMars);
  let contours = terrainContours(
    height,
    hGrad,
    u.terrain.z,
    vec2f(u.low.w, u.high.w),
    u.terrain.w,
    0.,
  );
  let tone = clamp((height + 6000.) / 16500., 0., 1.);
  var surfaceLinear = toLinear(mix(u.low.rgb, u.high.rgb, tone));
  let albedoSize = vec2f(textureDimensions(albedo));
  let albedoLod = clamp(log2(max(1., pixelAngle / (pi / albedoSize.y))), 0., f32(textureNumLevels(albedo) - 1u));
  let albedoSample = textureSampleLevel(albedo, albedoSampler, uv, albedoLod);
  let hasAlbedo = albedoSample.a > .5 || tileWeight > .001;
  if (albedoSample.a > .5) {
    surfaceLinear = albedoSample.rgb;
  }
  if (tileWeight > .001) {
    // 24K tilePool stores sRGB albedo alongside linear normal layers, so linearize albedo to match 8K sRGB hardware decode.
    surfaceLinear = mix(surfaceLinear, toLinear(tileAlbedo), tileWeight);
  }
  let lineColor = mix(surfaceLinear * .38, surfaceLinear + vec3f(.055, .058, .062), .60);
  surfaceLinear = mix(surfaceLinear, lineColor, select(.45, .32, hasAlbedo) * contours);
  let litLinear = surfaceLinear * (ambientTint * ambientLinear + vec3f(sunLinear));
  // Thin dusty Martian atmospheric limb scatter (warm peach day limb + faint blue twilight).
  var marsAtmo = vec3f(0.);
  if (isMars) {
    let oneMinusV = clamp(1. - dot(p, viewDir), 1e-4, 1.);
    let sunDay = smoothstep(-.12, .22, pDotL);
    let fresnel = pow(oneMinusV, 4.0);
    let horizon = pow(oneMinusV, 7.5);
    let tw = (pDotL + .01) / .11;
    let twilight = exp(-(tw * tw)) * pow(oneMinusV, 4.2);
    marsAtmo = vec3f(.88, .54, .34) * (fresnel * .15 + horizon * .28) * sunDay
      + vec3f(.38, .56, .78) * twilight * .07
      + vec3f(.10, .06, .04) * horizon * .08 * (1. - sunDay);
  }
  let totalLinear = litLinear + marsAtmo;
  return select(toDisplay(totalLinear), totalLinear, u.sun.w > .5);
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  let pi = 3.141592653589793;
  let p = normalize(input.local);
  let uv = dirToUv(p);
  let pDx = dpdx(p);
  let pDy = dpdy(p);
  let uvDx = wrappedGrad(dpdx(uv));
  let uvDy = wrappedGrad(dpdy(uv));
  let size = vec2f(textureDimensions(heights));
  let texelAngle = pi / size.y;
  let pixelAngle = max(length(pDx), length(pDy));
  let maxLod = f32(textureNumLevels(heights) - 1u);
  let rawLod = log2(max(1e-4, pixelAngle / texelAngle));
  let lod = clamp(rawLod, 0., maxLod);
  let height = samplePlanetHeight(uv, rawLod, maxLod);
  let hGrad = vec2f(dpdx(height), dpdy(height));
  return vec4f(shadePlanet(p, input.local, 0., pDx, pDy, uv, uvDx, uvDy, texelAngle, pixelAngle, lod, height, hGrad), 1.);
}

struct PlanetRayHit {
  hitPos: vec3f,
  p: vec3f,
  uv: vec2f,
  hit: bool,
}

fn evalPlanetRay(destPos: vec2f, rSurf: f32) -> PlanetRayHit {
  let ndc = vec2f(destPos.x * u.centerInfo.z - 1., 1. - destPos.y * u.centerInfo.w);
  let tanX = u.camRight.w;
  let tanY = u.camUp.w;
  let depth = u.camBack.w;
  let dNdc = ndc - u.centerInfo.xy;
  let evalLocal = u.camRight.xyz * (dNdc.x * depth * tanX) + u.camUp.xyz * (dNdc.y * depth * tanY);
  let dir = normalize(u.camRight.xyz * (ndc.x * tanX) + u.camUp.xyz * (ndc.y * tanY) - u.camBack.xyz);
  let q = evalLocal - dir * dot(evalLocal, dir);
  let disc = rSurf * rSurf - dot(q, q);
  let hit = disc >= 0.;
  let hitPos = select(normalize(q) * rSurf, q - dir * sqrt(max(0., disc)), hit);
  let p = normalize(hitPos);
  return PlanetRayHit(hitPos, p, dirToUv(p), hit);
}

@fragment fn smallFragment(
  input: SmallVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallOut {
  let pi = 3.141592653589793;
  let rSurf = u.terrain.x * .001;
  // Evaluate analytical per-sample sphere raycasts and neighbor rays directly from framebuffer
  // coordinates for crisp sub-pixel silhouettes and quad-independent gradients, with explicit
  // sample_mask guaranteeing WebGPU per-sample execution.
  let destPos = input.position.xy;
  let h0 = evalPlanetRay(destPos, rSurf);
  if (!h0.hit) { discard; }
  let hXp = evalPlanetRay(destPos + vec2f(.5, 0.), rSurf);
  let hXm = evalPlanetRay(destPos - vec2f(.5, 0.), rSurf);
  let hYp = evalPlanetRay(destPos + vec2f(0., .5), rSurf);
  let hYm = evalPlanetRay(destPos - vec2f(0., .5), rSurf);
  var pDx = hXp.p - hXm.p;
  var uvDx = wrappedGrad(hXp.uv - hXm.uv);
  if (hXp.hit != hXm.hit) {
    pDx = select(h0.p - hXm.p, hXp.p - h0.p, hXp.hit) * 2.;
    uvDx = select(wrappedGrad(h0.uv - hXm.uv), wrappedGrad(hXp.uv - h0.uv), hXp.hit) * 2.;
  }
  var pDy = hYp.p - hYm.p;
  var uvDy = wrappedGrad(hYp.uv - hYm.uv);
  if (hYp.hit != hYm.hit) {
    pDy = select(h0.p - hYm.p, hYp.p - h0.p, hYp.hit) * 2.;
    uvDy = select(wrappedGrad(h0.uv - hYm.uv), wrappedGrad(hYp.uv - h0.uv), hYp.hit) * 2.;
  }
  let size = vec2f(textureDimensions(heights));
  let texelAngle = pi / size.y;
  let pixelAngle = max(length(pDx), length(pDy));
  let maxLod = f32(textureNumLevels(heights) - 1u);
  let rawLod = log2(max(1e-4, pixelAngle / texelAngle));
  let lod = clamp(rawLod, 0., maxLod);
  let height0 = samplePlanetHeight(h0.uv, rawLod, maxLod);
  var hDx: f32;
  if (hXp.hit != hXm.hit) {
    let hSide = samplePlanetHeight(select(hXm.uv, hXp.uv, hXp.hit), rawLod, maxLod);
    hDx = select(height0 - hSide, hSide - height0, hXp.hit) * 2.;
  } else {
    hDx = samplePlanetHeight(hXp.uv, rawLod, maxLod) - samplePlanetHeight(hXm.uv, rawLod, maxLod);
  }
  var hDy: f32;
  if (hYp.hit != hYm.hit) {
    let hSide = samplePlanetHeight(select(hYm.uv, hYp.uv, hYp.hit), rawLod, maxLod);
    hDy = select(height0 - hSide, hSide - height0, hYp.hit) * 2.;
  } else {
    hDy = samplePlanetHeight(hYp.uv, rawLod, maxLod) - samplePlanetHeight(hYm.uv, rawLod, maxLod);
  }
  let hGrad = vec2f(hDx, hDy);
  let smallAngle = max(u.camBack.w * u.camUp.w * u.centerInfo.w, 1e-4) / rSurf;
  let color = shadePlanet(h0.p, h0.hitPos, smallAngle, pDx, pDy, h0.uv, uvDx, uvDy, texelAngle, pixelAngle, lod, height0, hGrad);
  let clip = u.camera * vec4f(h0.hitPos, 1.);
  return SmallOut(vec4f(color, 1.), clip.z / clip.w, 1u << sampleIdx);
}

fn packSplit16Small(x: f32) -> vec2f {
  let hi = round(x * 256.) / 256.;
  return vec2f(hi, x - hi);
}

struct SmallAnnotationOut {
  @location(0) geo: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

fn evalSmallAnnotation(input: SmallVertex, sampleIdx: u32) -> SmallAnnotationOut {
  let rSurf = u.terrain.x * .001;
  let h0 = evalPlanetRay(input.position.xy, rSurf);
  if (!h0.hit) { discard; }
  let bodyOffset = select(2., 4., u.eye.w > .5);
  let clip = u.camera * vec4f(h0.hitPos, 1.);
  return SmallAnnotationOut(
    vec4f(packSplit16Small(h0.uv.x + bodyOffset), packSplit16Small(1. - h0.uv.y)),
    clip.z / clip.w,
    1u << sampleIdx,
  );
}

@fragment fn smallAnnotationFragment(
  input: SmallVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallAnnotationOut {
  return evalSmallAnnotation(input, sampleIdx);
}

struct SmallAnnotationCoverageOut {
  @location(0) geo: vec4f,
  @location(1) coverage: vec2f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn smallAnnotationCoverageFragment(
  input: SmallVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallAnnotationCoverageOut {
  let out = evalSmallAnnotation(input, sampleIdx);
  return SmallAnnotationCoverageOut(out.geo, vec2f(0., 0.), out.depth, out.mask);
}
