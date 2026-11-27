override PLANET_TILE_PAIRS: u32 = 48u;
override VIRTUAL_TILE_SIZE: f32 = 512.0;
override VIRTUAL_TILE_INNER: f32 = 508.0;
override VIRTUAL_TILE_GUTTER: f32 = 2.0;
override PLANET_TILE_BASE_COLS: f32 = 24.0;
override PLANET_TILE_BASE_ROWS: f32 = 12.0;

struct View {
  camera: mat4x4f,
  sunDirection: vec3f,
  hdr: f32,
  radiusMeters: f32,
  relief: f32,
  majorContourInterval: f32,
  pixelRatio: f32,
  lowAlbedo: vec3f,
  minorContourDetail: f32,
  highAlbedo: vec3f,
  fineContourDetail: f32,
  tilesReady: f32,
  _padTiles0: f32,
  _padTiles1: f32,
  reliefFade: f32,
  eyeKm: vec3f,
  bodyKind: f32, // 0 = Moon, 1 = Mars
  camRight: vec3f,
  tanHalfFovX: f32,
  camUp: vec3f,
  tanHalfFovY: f32,
  camBack: vec3f,
  centerDepthKm: f32,
  centerNdc: vec2f,
  invViewportPx: vec2f,
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

@vertex fn vertexMain(input: CubePatchIn) -> Vertex {
  let pos = displacedCubePatchPos(input, heights, gridSampler, u.radiusMeters * .001, u.relief, u.reliefFade);
  return Vertex(u.camera * vec4f(pos, 1.), pos);
}

struct SmallVertex {
  @builtin(position) position: vec4f,
}

@vertex fn smallVertex(@builtin(vertex_index) vertex: u32) -> SmallVertex {
  let rSurf = u.radiusMeters * .001;
  let depth = u.centerDepthKm;
  if (depth <= rSurf * 1.001) { return SmallVertex(vec4f(2., 2., 0., 1.)); }
  let clipCenter = u.camera * vec4f(0., 0., 0., 1.);
  if (clipCenter.w <= 0.) { return SmallVertex(vec4f(2., 2., 0., 1.)); }
  let center = clipCenter.xy / clipCenter.w;
  let cx = -dot(u.eyeKm, u.camRight);
  let cy = -dot(u.eyeKm, u.camUp);
  let r2 = rSurf * rSurf;
  let d2x = cx * cx + depth * depth;
  let d2y = cy * cy + depth * depth;
  let denom = max(depth * depth - r2, 1e-6);
  let extX = (rSurf * sqrt(max(0., d2x - r2)) + abs(cx) * r2 / depth) / (denom * u.tanHalfFovX);
  let extY = (rSurf * sqrt(max(0., d2y - r2)) + abs(cy) * r2 / depth) / (denom * u.tanHalfFovY);
  let pad = 3. * u.invViewportPx;
  let corner = center + unitQuadCorner(vertex) * (vec2f(extX, extY) + pad);
  return SmallVertex(vec4f(corner, 0., 1.));
}

struct SmallOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

fn wrappedGrad(g: vec2f) -> vec2f {
  return vec2f(g.x - round(g.x), g.y);
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
  let s0 = decodeHeight(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc0.y), 0.));
  let s1 = decodeHeight(textureSampleLevel(heights, gridSampler, vec2f(tc0.x, tc12.y), 0.));
  let s2 = decodeHeight(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc12.y), 0.));
  let s3 = decodeHeight(textureSampleLevel(heights, gridSampler, vec2f(tc3.x, tc12.y), 0.));
  let s4 = decodeHeight(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc3.y), 0.));
  let sum = s0 * (w12.x * w0.y) + s1 * (w0.x * w12.y) + s2 * (w12.x * w12.y)
    + s3 * (w3.x * w12.y) + s4 * (w12.x * w3.y);
  let total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return sum / total;
}

fn samplePlanetHeight(uv: vec2f, rawLod: f32, maxLod: f32) -> f32 {
  var h = decodeHeight(textureSampleLevel(heights, gridSampler, uv, clamp(rawLod, 0., maxLod)));
  if (rawLod < -1.) {
    h = mix(h, sampleHeightBicubic(uv), 1. - smoothstep(-2., -1., rawLod));
  }
  return h;
}

struct PlanetLod {
  texelAngle: f32,
  pixelAngle: f32,
  maxLod: f32,
  rawLod: f32,
  lod: f32,
}

fn computePlanetLod(pDx: vec3f, pDy: vec3f) -> PlanetLod {
  let size = vec2f(textureDimensions(heights));
  let texelAngle = PLANET_PI / size.y;
  let pixelAngle = max(length(pDx), length(pDy));
  let maxLod = f32(textureNumLevels(heights) - 1u);
  let rawLod = log2(max(1e-4, pixelAngle / texelAngle));
  let lod = clamp(rawLod, 0., maxLod);
  return PlanetLod(texelAngle, pixelAngle, maxLod, rawLod, lod);
}

struct PlanetShadeInput {
  p: vec3f,
  localPos: vec3f,
  smallAngle: f32,
  pDx: vec3f,
  pDy: vec3f,
  uv: vec2f,
  uvDx: vec2f,
  uvDy: vec2f,
  lod: PlanetLod,
  height: f32,
  hGrad: vec2f,
}

fn shadePlanet(s: PlanetShadeInput) -> vec3f {
  let p = s.p;
  let uv = s.uv;
  // Minor singular value of dp/d(screen) gives the un-foreshortened angular scale everywhere.
  let pxx = dot(s.pDx, s.pDx);
  let pyy = dot(s.pDy, s.pDy);
  let pxy = dot(s.pDx, s.pDy);
  let pDet = max(0., pxx * pyy - pxy * pxy);
  let pMax = max(1e-12, .5 * (pxx + pyy + length(vec2f(pxx - pyy, 2. * pxy))));
  let minorAngle = max(sqrt(pDet / pMax), s.smallAngle);
  var tileAlbedo = vec3f(0.);
  var tileWeight = 0.;
  var tileNormal = p;
  var tileNormalWeight = 0.;
  if (u.tilesReady > .5) {
    let tileCoord = vec2f(fract(uv.x), clamp(uv.y, 0., .999999));
    let pageDims = vec2f(textureDimensions(tilePage));
    let cell = vec2i(clamp(floor(tileCoord * pageDims), vec2f(0.), pageDims - 1.));
    let entry = textureLoad(tilePage, cell, 0);
    if (entry.r < 255u) {
      let innerScale = VIRTUAL_TILE_INNER / VIRTUAL_TILE_SIZE;
      let baseGrid = vec2f(PLANET_TILE_BASE_COLS, PLANET_TILE_BASE_ROWS);
      var slot = entry.r;
      var grid = baseGrid * exp2(f32(entry.g));
      var gradScale = grid * innerScale;
      var dx = s.uvDx * gradScale;
      var dy = s.uvDy * gradScale;
      var tileMip = log2(max(1., max(length(dx), length(dy)) * VIRTUAL_TILE_SIZE));
      if (tileMip > 1.75 && entry.b < 255u) {
        slot = entry.b;
        grid = baseGrid * exp2(f32(entry.a));
        gradScale = grid * innerScale;
        dx = s.uvDx * gradScale;
        dy = s.uvDy * gradScale;
        tileMip = log2(max(1., max(length(dx), length(dy)) * VIRTUAL_TILE_SIZE));
      }
      if (tileMip < 2.8) {
        let localUv = fract(tileCoord * grid);
        let tileUv = localUv * innerScale + (VIRTUAL_TILE_GUTTER / VIRTUAL_TILE_SIZE);
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
    let lon = (uv.x - .5) * 2. * PLANET_PI;
    let lat = (.5 - uv.y) * PLANET_PI;
    let cosLon = cos(lon);
    let sinLon = sin(lon);
    let cosLat = cos(lat);
    let sinLat = sin(lat);
    let east = vec3f(cosLon, 0., -sinLon);
    let north = vec3f(-sinLat * sinLon, cosLat, -sinLat * cosLon);
    let step = min(.25, s.lod.texelAngle * exp2(s.lod.lod));
    var hE: f32;
    var hW: f32;
    var hN: f32;
    var hS: f32;
    if (abs(lat) < 1.48) {
      let du = step / (2. * PLANET_PI * max(cosLat, .05));
      let dv = step / PLANET_PI;
      hE = decodeHeight(textureSampleLevel(heights, gridSampler, uv + vec2f(du, 0.), s.lod.lod));
      hW = decodeHeight(textureSampleLevel(heights, gridSampler, uv - vec2f(du, 0.), s.lod.lod));
      hN = decodeHeight(textureSampleLevel(heights, gridSampler, uv - vec2f(0., dv), s.lod.lod));
      hS = decodeHeight(textureSampleLevel(heights, gridSampler, uv + vec2f(0., dv), s.lod.lod));
    } else {
      hE = decodeHeight(textureSampleLevel(heights, gridSampler, dirToUv(p + east * step), s.lod.lod));
      hW = decodeHeight(textureSampleLevel(heights, gridSampler, dirToUv(p - east * step), s.lod.lod));
      hN = decodeHeight(textureSampleLevel(heights, gridSampler, dirToUv(p + north * step), s.lod.lod));
      hS = decodeHeight(textureSampleLevel(heights, gridSampler, dirToUv(p - north * step), s.lod.lod));
    }
    let scale = u.relief / (2. * step * u.radiusMeters);
    let baseNormal = normalize(p - (east * (hE - hW) + north * (hN - hS)) * scale);
    hillNormal = normalize(mix(baseNormal, tileNormal, tileNormalWeight));
  }
  let sun = normalize(u.sunDirection);
  let viewDir = normalize(u.eyeKm - s.localPos);
  let pDotL = dot(p, sun);
  let daylight = smoothstep(-.035, .045, pDotL);
  let ndotl = max(0., dot(hillNormal, sun));
  let ndotv = max(.04, dot(hillNormal, viewDir));
  let isMars = u.bodyKind > .5;
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
  let inspect = 1. - smoothstep(.0040, .022, minorAngle * u.pixelRatio);
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
    s.height,
    s.hGrad,
    u.majorContourInterval,
    vec2f(u.minorContourDetail, u.fineContourDetail),
    u.pixelRatio,
    0.,
  );
  let tone = clamp((s.height + 6000.) / 16500., 0., 1.);
  var surfaceLinear = toLinear(mix(u.lowAlbedo, u.highAlbedo, tone));
  let albedoSize = vec2f(textureDimensions(albedo));
  let albedoLod = clamp(log2(max(1., s.lod.pixelAngle / (PLANET_PI / albedoSize.y))), 0., f32(textureNumLevels(albedo) - 1u));
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
  return select(toDisplay(totalLinear), totalLinear, u.hdr > .5);
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  let p = normalize(input.local);
  let uv = dirToUv(p);
  let pDx = dpdx(p);
  let pDy = dpdy(p);
  let uvDx = wrappedGrad(dpdx(uv));
  let uvDy = wrappedGrad(dpdy(uv));
  let lod = computePlanetLod(pDx, pDy);
  let height = samplePlanetHeight(uv, lod.rawLod, lod.maxLod);
  let hGrad = vec2f(dpdx(height), dpdy(height));
  return vec4f(shadePlanet(PlanetShadeInput(p, input.local, 0., pDx, pDy, uv, uvDx, uvDy, lod, height, hGrad)), 1.);
}

struct PlanetRayHit {
  hitPos: vec3f,
  p: vec3f,
  uv: vec2f,
  hit: bool,
}

fn evalPlanetRay(destPos: vec2f, rSurf: f32) -> PlanetRayHit {
  let ndc = vec2f(destPos.x * u.invViewportPx.x - 1., 1. - destPos.y * u.invViewportPx.y);
  let tanX = u.tanHalfFovX;
  let tanY = u.tanHalfFovY;
  let depth = u.centerDepthKm;
  let dNdc = ndc - u.centerNdc;
  let evalLocal = u.camRight * (dNdc.x * depth * tanX) + u.camUp * (dNdc.y * depth * tanY);
  let dir = normalize(u.camRight * (ndc.x * tanX) + u.camUp * (ndc.y * tanY) - u.camBack);
  let q = evalLocal - dir * dot(evalLocal, dir);
  let disc = rSurf * rSurf - dot(q, q);
  let hit = disc >= 0.;
  let hitPos = select(normalize(q) * rSurf, q - dir * sqrt(max(0., disc)), hit);
  let p = normalize(hitPos);
  return PlanetRayHit(hitPos, p, dirToUv(p), hit);
}

struct SmallAxisGrad {
  pD: vec3f,
  uvD: vec2f,
}

fn computeSmallAxisGrad(h0: PlanetRayHit, hPos: PlanetRayHit, hNeg: PlanetRayHit) -> SmallAxisGrad {
  if (hPos.hit != hNeg.hit) {
    let pD = select(h0.p - hNeg.p, hPos.p - h0.p, hPos.hit) * 2.;
    let uvD = select(wrappedGrad(h0.uv - hNeg.uv), wrappedGrad(hPos.uv - h0.uv), hPos.hit) * 2.;
    return SmallAxisGrad(pD, uvD);
  }
  return SmallAxisGrad(hPos.p - hNeg.p, wrappedGrad(hPos.uv - hNeg.uv));
}

fn computeSmallHeightDiff(
  height0: f32,
  hPos: PlanetRayHit,
  hNeg: PlanetRayHit,
  rawLod: f32,
  maxLod: f32,
) -> f32 {
  if (hPos.hit != hNeg.hit) {
    let hSide = samplePlanetHeight(select(hNeg.uv, hPos.uv, hPos.hit), rawLod, maxLod);
    return select(height0 - hSide, hSide - height0, hPos.hit) * 2.;
  }
  return samplePlanetHeight(hPos.uv, rawLod, maxLod) - samplePlanetHeight(hNeg.uv, rawLod, maxLod);
}

@fragment fn smallFragment(
  input: SmallVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> SmallOut {
  let rSurf = u.radiusMeters * .001;
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
  let gx = computeSmallAxisGrad(h0, hXp, hXm);
  let gy = computeSmallAxisGrad(h0, hYp, hYm);
  let lod = computePlanetLod(gx.pD, gy.pD);
  let height0 = samplePlanetHeight(h0.uv, lod.rawLod, lod.maxLod);
  let hGrad = vec2f(
    computeSmallHeightDiff(height0, hXp, hXm, lod.rawLod, lod.maxLod),
    computeSmallHeightDiff(height0, hYp, hYm, lod.rawLod, lod.maxLod),
  );
  let smallAngle = max(u.centerDepthKm * u.tanHalfFovY * u.invViewportPx.y, 1e-4) / rSurf;
  let color = shadePlanet(PlanetShadeInput(
    h0.p, h0.hitPos, smallAngle, gx.pD, gy.pD, h0.uv, gx.uvD, gy.uvD, lod, height0, hGrad,
  ));
  let clip = u.camera * vec4f(h0.hitPos, 1.);
  return SmallOut(vec4f(color, 1.), clip.z / clip.w, 1u << sampleIdx);
}

struct SmallAnnotationOut {
  @location(0) geo: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

fn evalSmallAnnotation(input: SmallVertex, sampleIdx: u32) -> SmallAnnotationOut {
  let rSurf = u.radiusMeters * .001;
  let h0 = evalPlanetRay(input.position.xy, rSurf);
  if (!h0.hit) { discard; }
  let bodyOffset = select(2., 4., u.bodyKind > .5);
  let clip = u.camera * vec4f(h0.hitPos, 1.);
  return SmallAnnotationOut(
    vec4f(packSplit16(h0.uv.x + bodyOffset), packSplit16(1. - h0.uv.y)),
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
