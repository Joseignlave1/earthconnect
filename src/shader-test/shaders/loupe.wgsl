struct LoupeUniforms {
  centerPx: vec2f,
  radiusCss: f32,
  pixelRatio: f32,
  viewportPx: vec2f,
  exposure: f32,
  hdr: f32,
  aimOffsetCss: vec2f,
  locked: f32,
  _padState: f32,
  selectedOffsetCss: vec2f,
  hasSelected: f32,
  _padSelected: f32,
}

@group(0) @binding(0) var<uniform> loupe: LoupeUniforms;
@group(0) @binding(1) var loupeTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;

struct LoupeVertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
}

@vertex fn loupeVertex(@builtin(vertex_index) index: u32) -> LoupeVertex {
  let extent = loupe.radiusCss + 12.;
  let local = unitQuadCorner(index) * extent;
  let screen = (loupe.centerPx + local * loupe.pixelRatio) / loupe.viewportPx;
  return LoupeVertex(vec4f(screen * vec2f(2., -2.) + vec2f(-1., 1.), 0., 1.), local);
}

fn toneMapLoupe(uv: vec2f) -> vec3f {
  let s = textureSampleLevel(loupeTex, linearSampler, clamp(uv, vec2f(.002), vec2f(.998)), 0.);
  var c = s.rgb / max(s.a, .0001);
  if (loupe.hdr > .5) {
    c = toDisplay(acesToneMap(c * loupe.exposure));
  }
  return c;
}

@fragment fn loupeFragment(input: LoupeVertex) -> @location(0) vec4f {
  let p = input.local;
  let r = loupe.radiusCss;
  let d = length(p);
  let aa = max(.35, .6 / max(loupe.pixelRatio, 1.));
  let inside = 1. - smoothstep(r - aa, r + aa, d);
  let uv = p / (2. * r) + .5;
  var lens = toneMapLoupe(uv);
  // Subtle optical glass vignette near the bezel edge.
  let edgeVig = smoothstep(r - 7., r, d);
  lens = mix(lens, lens * .82, edgeVig * .65);

  let innerMask = 1. - smoothstep(r - 1.5 - aa, r - 1.5 + aa, d);

  // Selected Earth marker projected into loupe space (so clicked spots stay visible inside the lens).
  if (loupe.hasSelected > .5) {
    let sd = length(p - loupe.selectedOffsetCss);
    let selRingDist = abs(sd - 6.8);
    let selCase = max(1. - smoothstep(1.75 - aa, 1.75 + aa, selRingDist), 1. - smoothstep(2.4 - aa, 2.4 + aa, sd)) * innerMask;
    let selCore = max(1. - smoothstep(.8 - aa, .8 + aa, selRingDist), 1. - smoothstep(1.45 - aa, 1.45 + aa, sd)) * innerMask;
    lens = mix(lens, vec3f(.02, .028, .045), selCase * .88);
    lens = mix(lens, vec3f(.97, .98, 1.), selCore * .96);
  }

  let locked = loupe.locked > .5;
  let aim = select(vec2f(0.), loupe.aimOffsetCss, locked);
  let ap = p - aim;
  let ad = length(ap);
  let aimUv = aim / (2. * r) + .5;
  let aimCol = toneMapLoupe(aimUv);
  let aimLuma = dot(aimCol, vec3f(.2126, .7152, .0722));
  let aax = abs(ap.x);
  let aay = abs(ap.y);
  let armBand = smoothstep(3.4 - aa, 3.4 + aa, ad) * (1. - smoothstep(10. - aa, 10. + aa, ad));
  let armCore = max(1. - smoothstep(.52 - aa, .52 + aa, aax), 1. - smoothstep(.52 - aa, .52 + aa, aay)) * armBand;
  let armCase = max(1. - smoothstep(1.35 - aa, 1.35 + aa, aax), 1. - smoothstep(1.35 - aa, 1.35 + aa, aay)) * armBand;
  let dotCore = 1. - smoothstep(1.15 - aa, 1.15 + aa, ad);
  let dotCase = 1. - smoothstep(2.05 - aa, 2.05 + aa, ad);
  let crossCase = max(armCase, dotCase) * innerMask;
  let crossCore = max(armCore, dotCore) * innerMask;
  let caseAlpha = mix(.62, .92, smoothstep(.1, .65, aimLuma));
  lens = mix(lens, vec3f(.02, .028, .045), crossCase * caseAlpha);
  lens = mix(lens, vec3f(.97, .98, 1.), crossCore * .96);

  // Precision dual-stroke bezel + 4 cardinal ticks + soft floating shadow.
  let ax = abs(p.x);
  let ay = abs(p.y);
  let shadow = exp(-max(0., d - r) * .36) * .48 * (1. - inside);
  let bezelDist = abs(d - r);
  let bezelCase = 1. - smoothstep(2.1 - aa, 2.1 + aa, bezelDist);
  let bezelCore = 1. - smoothstep(.85 - aa, .85 + aa, bezelDist);
  let tickBand = smoothstep(r + .4 - aa, r + .4 + aa, d) * (1. - smoothstep(r + 4.4 - aa, r + 4.4 + aa, d));
  let tickCase = max(1. - smoothstep(1.4 - aa, 1.4 + aa, ax), 1. - smoothstep(1.4 - aa, 1.4 + aa, ay)) * tickBand;
  let tickCore = max(1. - smoothstep(.58 - aa, .58 + aa, ax), 1. - smoothstep(.58 - aa, .58 + aa, ay)) * tickBand;
  let ringCase = max(bezelCase, tickCase);
  let ringCore = max(bezelCore, tickCore);
  let isTopTick = p.y < 0. && ax < ay;
  let bezelTint = select(vec3f(.95, .97, 1.), vec3f(.68, .90, 1.), locked);
  let coreColor = mix(bezelTint, vec3f(.45, .84, 1.), select(0., 1., isTopTick) * tickCore * .85);

  let alpha = max(inside, max(shadow, max(ringCase * .88, ringCore)));
  if (alpha < .002) { discard; }
  var outCol = mix(vec3f(.015, .02, .035), lens, inside);
  outCol = mix(outCol, vec3f(.02, .028, .045), ringCase * .88);
  outCol = mix(outCol, coreColor, ringCore * .96);
  return vec4f(outCol, alpha);
}
