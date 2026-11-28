struct BridgeViewport {
  sizePx: vec2f,
  tanFov: vec2f,
}

struct BridgeTiming {
  near: f32,
  timeSeconds: f32,
  reducedMotion: f32,
  opacity: f32,
}

struct BridgeUniform {
  viewport: BridgeViewport,
  timing: BridgeTiming,
  marsColor: vec3f,
  arcHalfWidthPx: f32,
  earthColor: vec3f,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> u: BridgeUniform;

struct ArcInstance {
  @location(0) startPos: vec3f,
  @location(1) startT: f32,
  @location(2) endPos: vec3f,
  @location(3) endT: f32,
}

struct ArcVertex {
  @builtin(position) position: vec4f,
  @location(0) t: f32,
  @location(1) crossSide: f32,
  @location(2) halfWidthPx: f32,
}

fn projectView(p: vec3f) -> vec4f {
  let w = max(-p.z, u.timing.near * 1.001);
  return vec4f(p.x / u.viewport.tanFov.x, p.y / u.viewport.tanFov.y, u.timing.near, w);
}

@vertex fn arcVertex(
  @builtin(vertex_index) vIdx: u32,
  seg: ArcInstance,
) -> ArcVertex {
  let corners = array<vec2f, 6>(
    vec2f(0., -1.),
    vec2f(1., -1.),
    vec2f(0., 1.),
    vec2f(0., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  let corner = corners[vIdx];
  let clipA = projectView(seg.startPos);
  let clipB = projectView(seg.endPos);
  let ndcA = clipA.xy / clipA.w;
  let ndcB = clipB.xy / clipB.w;
  let deltaPx = (ndcB - ndcA) * u.viewport.sizePx * .5;
  let lenPx = max(length(deltaPx), 1e-4);
  let perpPx = vec2f(-deltaPx.y, deltaPx.x) / lenPx;
  let isEnd = corner.x > .5;
  var clip = select(clipA, clipB, isEnd);
  let t = select(seg.startT, seg.endT, isEnd);
  let halfW = u.arcHalfWidthPx;
  let offsetNdc = perpPx * (corner.y * halfW * 2.) / u.viewport.sizePx;
  clip.x += offsetNdc.x * clip.w;
  clip.y += offsetNdc.y * clip.w;
  return ArcVertex(clip, t, corner.y, halfW);
}

fn cometPulse(t: f32, head: f32, leadSigma: f32, tailSigma: f32) -> f32 {
  let d = t - head;
  let sigma = select(tailSigma, leadSigma, d > 0.);
  return exp(-(d * d) / (sigma * sigma));
}

@fragment fn arcFragment(input: ArcVertex) -> @location(0) vec4f {
  let t = input.t;
  let distPx = abs(input.crossSide) * input.halfWidthPx;
  let endFade = smoothstep(0., .028, t) * smoothstep(1., .972, t);

  // Whisper-thin astronomical hairline core pre-filtered to avoid 1x DPR diagonal staircasing.
  let spanFocus = mix(.86, 1.06, pow(abs(t * 2. - 1.), 1.5));
  let rawCore = .58 * spanFocus;
  let coreSigma = sqrt(rawCore * rawCore + .62 * .62);
  let core = exp(-pow(distPx / coreSigma, 2.)) * pow(rawCore / coreSigma, .65);
  let halo = exp(-pow(distPx / (2.5 * spanFocus), 2.)) * .17;

  // Smooth C1 traveling pulses gliding from Mars (t=0) to Earth (t=1) with no wrap discontinuity.
  let time = u.timing.timeSeconds;
  let staticMode = u.timing.reducedMotion > .5;
  let phaseA = select(fract(time / 8.8) * 1.44 - .22, .44, staticMode);
  let phaseB = select(fract(time / 8.8 + .5) * 1.44 - .22, .82, staticMode);
  let pulseA = cometPulse(t, phaseA, .026, .13);
  let pulseB = cometPulse(t, phaseB, .024, .10) * .55;
  let pulse = (pulseA + pulseB) * smoothstep(0., .05, t) * smoothstep(1., .95, t);

  // Warm Martian amber-ivory -> cool celestial silver -> Earth Magma gold-coral.
  let midTint = vec3f(.96, .89, .84);
  let baseTint = select(
    mix(u.marsColor, midTint, smoothstep(0., .5, t)),
    mix(midTint, u.earthColor, smoothstep(.5, 1., t)),
    t > .5,
  );
  let pulseTint = mix(vec3f(1.55, 1.18, .85), vec3f(1.9, 1.04, .64), t);
  let energy = (core * .46 + halo) * baseTint + (core * 1.25 + halo * 1.65) * pulse * pulseTint;
  let intensity = energy * endFade * u.timing.opacity;
  return vec4f(intensity, 0.);
}

struct AnchorInstance {
  @location(0) point: vec3f,
  @location(1) kind: f32,
  @location(2) radiusPx: f32,
}

struct AnchorVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) kind: f32,
  @location(2) radiusPx: f32,
}

@vertex fn anchorVertex(
  @builtin(vertex_index) vIdx: u32,
  anchor: AnchorInstance,
) -> AnchorVertex {
  let corners = array<vec2f, 6>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(-1., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  let corner = corners[vIdx];
  var clip = projectView(anchor.point);
  let offsetNdc = corner * (anchor.radiusPx * 2.) / u.viewport.sizePx;
  clip.x += offsetNdc.x * clip.w;
  clip.y += offsetNdc.y * clip.w;
  return AnchorVertex(clip, corner * anchor.radiusPx, anchor.kind, anchor.radiusPx);
}

@fragment fn anchorFragment(input: AnchorVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  if (r > input.radiusPx) { discard; }
  let tint = select(u.marsColor, u.earthColor, input.kind > .5);
  let coreD = r / 1.35;
  let core = exp(-(coreD * coreD)) * 1.35;
  let glowD = r / 4.2;
  let glow = exp(-(glowD * glowD)) * .24;
  let ringR = select(5.8, 4.8, input.kind > .5);
  let ringD = (r - ringR) / .78;
  let ring = exp(-(ringD * ringD)) * .46;
  let breath = select(.90 + .10 * sin(u.timing.timeSeconds * 1.2 + input.kind * 2.1), 1., u.timing.reducedMotion > .5);
  let color = tint * (core + glow + ring) * breath * u.timing.opacity;
  return vec4f(color, 0.);
}
