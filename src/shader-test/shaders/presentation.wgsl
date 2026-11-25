@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var geography: texture_multisampled_2d<f32>;
@group(0) @binding(4) var sceneMsaa: texture_multisampled_2d<f32>;
@group(0) @binding(5) var bloomTex: texture_2d<f32>;

fn unpackGeo(g: vec4f) -> vec2f {
  return vec2f(g.x + g.y, g.z + g.w);
}

fn sampleLuma(uv: vec2f) -> f32 {
  let s = textureSampleLevel(source, linearSampler, clamp(uv, vec2f(.001), vec2f(.999)), 0.);
  let rgb = s.rgb / max(s.a, .0001);
  return dot(rgb, vec3f(.2126, .7152, .0722));
}

fn localLuma(screenUv: vec2f) -> f32 {
  let step = 6. * p.effects.z / max(p.viewport.zw, vec2f(1.));
  var sum = sampleLuma(screenUv);
  sum += sampleLuma(screenUv + vec2f(step.x, 0.));
  sum += sampleLuma(screenUv - vec2f(step.x, 0.));
  sum += sampleLuma(screenUv + vec2f(0., step.y));
  sum += sampleLuma(screenUv - vec2f(0., step.y));
  sum += sampleLuma(screenUv + vec2f(step.x, step.y) * .707);
  sum += sampleLuma(screenUv + vec2f(-step.x, step.y) * .707);
  sum += sampleLuma(screenUv + vec2f(step.x, -step.y) * .707);
  sum += sampleLuma(screenUv - vec2f(step.x, step.y) * .707);
  return sum / 9.;
}

fn sampleNearReticle(coordinates: vec2f, hasSel: bool, hasHover: bool, hoverBody: i32) -> bool {
  if (coordinates.y == -1.) { return false; }
  let bodyId = i32(floor(coordinates.x * .5));
  let bodyMorph = select(1., p.state.x, bodyId == 0);
  let uv = vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y);
  if (hasSel) {
    let sel = select(select(p.selEarth, p.selMoon, bodyId == 1), p.selMars, bodyId == 2);
    if (sel.z > .005 && length(worldTangent(uv, sel.xy, bodyMorph, sel.w)) <= 16.) { return true; }
  }
  if (hasHover && bodyId == hoverBody) {
    if (length(worldTangent(uv, p.hover.xy, bodyMorph, p.hover.w)) <= 16.) { return true; }
  }
  return false;
}

fn evalSampleReticle(
  s: vec4f, bg: f32, coordinates: vec2f, hasSel: bool, hasHover: bool, hoverBody: i32, aa: f32, hdr: bool,
) -> vec4f {
  if (coordinates.y == -1. || s.a <= .0001) { return s; }
  // Body IDs are packed in the integer part of x: Earth [0,1], Moon [2,3], Mars [4,5].
  let bodyId = i32(floor(coordinates.x * .5));
  let bodyMorph = select(1., p.state.x, bodyId == 0);
  let sel = select(select(p.selEarth, p.selMoon, bodyId == 1), p.selMars, bodyId == 2);
  let useSel = hasSel && sel.z > .005;
  let useHover = hasHover && bodyId == hoverBody;
  if (!useSel && !useHover) { return s; }
  let uv = vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y);
  var c = s.rgb / s.a;
  if (useSel) {
    let tSel = worldTangent(uv, sel.xy, bodyMorph, sel.w);
    c = applyPrecisionReticle(c, bg, tSel, length(tSel), aa, sel.z, hdr);
  }
  if (useHover) {
    let tHov = worldTangent(uv, p.hover.xy, bodyMorph, p.hover.w);
    c = applyPrecisionReticle(c, bg, tHov, length(tHov), aa, 1., hdr);
  }
  return vec4f(c * s.a, s.a);
}

fn loadLinearSample(samplePx: vec2i, k: i32, coords: vec2f, bakedSel: bool, hdr: bool) -> vec4f {
  var s = textureLoad(sceneMsaa, samplePx, k);
  if (bakedSel) {
    s = evalWorldMarkerSample(s, coords, p.selEarth, p.selMoon, p.selMars, p.state.x, p.effects.x, p.effects.z, hdr);
  }
  return s;
}

fn finishLinear(s: vec4f, uv: vec2f) -> vec3f {
  var c = s.rgb / max(s.a, .0001);
  if (p.hoverScreen.w > .5) {
    c = (c + textureSampleLevel(bloomTex, linearSampler, uv, 0.).rgb * p.effects.y) * p.effects.x;
    c = toDisplay(acesToneMap(c));
  }
  return c;
}

@fragment fn presentFragment(input: ScreenVertex) -> @location(0) vec4f {
  let scene = textureSampleLevel(source, linearSampler, input.uv, 0.);
  let bakedSel = p.state.y > .5;
  let isMotion = p.state.y > 1.5;
  let hasSel = !bakedSel && (p.selEarth.z > .005 || p.selMoon.z > .005 || p.selMars.z > .005);
  let hasHover = p.hover.z > .5;
  if (!hasSel && !hasHover) { return scene; }
  // Hover reticles have a hard 16 CSS px tangent-space cutoff; skip all 4x MSAA geography loads
  // outside a generous 36 CSS px screen-space radius around the cursor when no unbaked selection exists.
  if (!hasSel && p.hoverScreen.z > .5 && length(input.position.xy - p.hoverScreen.xy) > 36. * max(p.effects.z, 1.)) {
    return scene;
  }
  let pixel = vec2i(input.position.xy);
  let geoSize = vec2i(textureDimensions(geography));
  let outSize = vec2i(round(p.viewport.zw));
  let sameRes = geoSize.x == outSize.x && geoSize.y == outSize.y;
  let samplePx = select(
    clamp(vec2i(floor(input.uv * vec2f(geoSize))), vec2i(0), geoSize - vec2i(1)),
    clamp(pixel, vec2i(0), geoSize - vec2i(1)),
    sameRes,
  );
  let g0 = unpackGeo(textureLoad(geography, samplePx, 0));
  let g1 = unpackGeo(textureLoad(geography, samplePx, 1));
  let g2 = unpackGeo(textureLoad(geography, samplePx, 2));
  let g3 = unpackGeo(textureLoad(geography, samplePx, 3));
  if (g0.y == -1. && g1.y == -1. && g2.y == -1. && g3.y == -1.) { return scene; }
  let hoverBody = i32(round(p.viewport.y));
  if (!sampleNearReticle(g0, hasSel, hasHover, hoverBody)
    && !sampleNearReticle(g1, hasSel, hasHover, hoverBody)
    && !sampleNearReticle(g2, hasSel, hasHover, hoverBody)
    && !sampleNearReticle(g3, hasSel, hasHover, hoverBody)) { return scene; }
  let aa = clamp(.65 / max(p.effects.z, 1.), .15, .75);
  let displayOnlyHover = p.viewport.x > .5;
  if (displayOnlyHover) {
    let base = scene.rgb / max(scene.a, .0001);
    let bg = localLuma(input.uv);
    let sBase = vec4f(base, 1.);
    let h0 = evalSampleReticle(sBase, bg, g0, hasSel, hasHover, hoverBody, aa, false);
    let h1 = evalSampleReticle(sBase, bg, g1, hasSel, hasHover, hoverBody, aa, false);
    let h2 = evalSampleReticle(sBase, bg, g2, hasSel, hasHover, hoverBody, aa, false);
    let h3 = evalSampleReticle(sBase, bg, g3, hasSel, hasHover, hoverBody, aa, false);
    if (all(h0 == sBase) && all(h1 == sBase) && all(h2 == sBase) && all(h3 == sBase)) { return scene; }
    let outCol = (h0.rgb + h1.rgb + h2.rgb + h3.rgb) * .25;
    return vec4f(outCol * scene.a, scene.a);
  }
  let hdr = p.hoverScreen.w > .5;
  let l0 = loadLinearSample(samplePx, 0, g0, bakedSel, hdr);
  let l1 = loadLinearSample(samplePx, 1, g1, bakedSel, hdr);
  let l2 = loadLinearSample(samplePx, 2, g2, bakedSel, hdr);
  let l3 = loadLinearSample(samplePx, 3, g3, bakedSel, hdr);
  let plainAvg = (l0 + l1 + l2 + l3) * .25;
  if (plainAvg.a <= .0001) { return scene; }
  let bg = localLuma(input.uv);
  let h0 = evalSampleReticle(l0, bg, g0, hasSel, hasHover, hoverBody, aa, hdr);
  let h1 = evalSampleReticle(l1, bg, g1, hasSel, hasHover, hoverBody, aa, hdr);
  let h2 = evalSampleReticle(l2, bg, g2, hasSel, hasHover, hoverBody, aa, hdr);
  let h3 = evalSampleReticle(l3, bg, g3, hasSel, hasHover, hoverBody, aa, hdr);
  if (all(h0 == l0) && all(h1 == l1) && all(h2 == l2) && all(h3 == l3)) { return scene; }
  let hoverAvg = (h0 + h1 + h2 + h3) * .25;
  let hoverCol = finishLinear(hoverAvg, input.uv);
  let direct = !isMotion && p.effects.w < .5 && p.state.z < .5 && sameRes;
  if (direct) { return vec4f(hoverCol * scene.a, scene.a); }
  let base = scene.rgb / max(scene.a, .0001);
  let plainCol = finishLinear(plainAvg, input.uv);
  let outCol = clamp(base + (hoverCol - plainCol), vec3f(0.), vec3f(1.));
  return vec4f(outCol * scene.a, scene.a);
}
