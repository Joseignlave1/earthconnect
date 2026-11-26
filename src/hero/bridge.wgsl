struct BridgeUniform {
  viewport: vec4f, // width, height, tanX, tanY
  timing: vec4f,   // near, timeSeconds, reducedMotion (0/1), opacity
  marsColor: vec4f,
  earthColor: vec4f,
}

@group(0) @binding(0) var<uniform> u: BridgeUniform;

struct ArcVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f, // x: t along arc [0,1], y: signed cross-width [-1,1]
  @location(1) halfWidthPx: f32,
}

fn projectView(p: vec3f) -> vec4f {
  let w = max(-p.z, u.timing.x * 1.001);
  return vec4f(p.x / u.viewport.z, p.y / u.viewport.w, u.timing.x, w);
}

@vertex fn arcVertex(
  @location(0) a: vec4f,      // xyz: camera-space start, w: tA
  @location(1) b: vec4f,      // xyz: camera-space end, w: tB
  @location(2) corner: vec4f, // x: end (0/1), y: side (-1/+1), z: halfWidthPx, w: unused
) -> ArcVertex {
  let clipA = projectView(a.xyz);
  let clipB = projectView(b.xyz);
  let ndcA = clipA.xy / clipA.w;
  let ndcB = clipB.xy / clipB.w;
  let deltaPx = (ndcB - ndcA) * u.viewport.xy * .5;
  let lenPx = max(length(deltaPx), 1e-4);
  let perpPx = vec2f(-deltaPx.y, deltaPx.x) / lenPx;
  let isEnd = corner.x > .5;
  var clip = select(clipA, clipB, isEnd);
  let t = select(a.w, b.w, isEnd);
  let offsetNdc = perpPx * (corner.y * corner.z * 2.) / u.viewport.xy;
  clip.x += offsetNdc.x * clip.w;
  clip.y += offsetNdc.y * clip.w;
  return ArcVertex(clip, vec2f(t, corner.y), corner.z);
}

fn cometPulse(t: f32, head: f32, leadSigma: f32, tailSigma: f32) -> f32 {
  let d = t - head;
  let sigma = select(tailSigma, leadSigma, d > 0.);
  return exp(-(d * d) / (sigma * sigma));
}

@fragment fn arcFragment(input: ArcVertex) -> @location(0) vec4f {
  let t = input.uv.x;
  let distPx = abs(input.uv.y) * input.halfWidthPx;
  let endFade = smoothstep(0., .028, t) * smoothstep(1., .972, t);

  // Whisper-thin astronomical hairline core + subtle optical aura.
  let spanFocus = mix(.86, 1.06, pow(abs(t * 2. - 1.), 1.5));
  let core = exp(-pow(distPx / (.58 * spanFocus), 2.));
  let halo = exp(-pow(distPx / (2.5 * spanFocus), 2.)) * .17;

  // Smooth C1 traveling pulses gliding from Mars (t=0) to Earth (t=1) with no wrap discontinuity.
  let time = u.timing.y;
  let staticMode = u.timing.z > .5;
  let phaseA = select(fract(time / 8.8) * 1.44 - .22, .44, staticMode);
  let phaseB = select(fract(time / 8.8 + .5) * 1.44 - .22, .82, staticMode);
  let pulseA = cometPulse(t, phaseA, .026, .13);
  let pulseB = cometPulse(t, phaseB, .024, .10) * .55;
  let pulse = (pulseA + pulseB) * smoothstep(0., .05, t) * smoothstep(1., .95, t);

  // Warm Martian amber-ivory -> cool celestial silver -> Earth Magma gold-coral.
  let midTint = vec3f(.96, .89, .84);
  let baseTint = select(
    mix(u.marsColor.rgb, midTint, smoothstep(0., .5, t)),
    mix(midTint, u.earthColor.rgb, smoothstep(.5, 1., t)),
    t > .5,
  );
  let pulseTint = mix(vec3f(1.55, 1.18, .85), vec3f(1.9, 1.04, .64), t);
  let energy = (core * .46 + halo) * baseTint + (core * 1.25 + halo * 1.65) * pulse * pulseTint;
  let intensity = energy * endFade * u.timing.w;
  return vec4f(intensity, 0.);
}

struct AnchorVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) kind: f32,
  @location(2) radiusPx: f32,
}

@vertex fn anchorVertex(
  @location(0) point: vec4f, // xyz: camera-space position, w: kind (0 = Mars, 1 = Earth)
  @location(1) quad: vec4f,  // xy: corner (-1/+1), z: radiusPx, w: unused
) -> AnchorVertex {
  var clip = projectView(point.xyz);
  let offsetNdc = quad.xy * (quad.z * 2.) / u.viewport.xy;
  clip.x += offsetNdc.x * clip.w;
  clip.y += offsetNdc.y * clip.w;
  return AnchorVertex(clip, quad.xy * quad.z, point.w, quad.z);
}

@fragment fn anchorFragment(input: AnchorVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  if (r > input.radiusPx) { discard; }
  let tint = select(u.marsColor.rgb, u.earthColor.rgb, input.kind > .5);
  let core = exp(-pow(r / 1.35, 2.)) * 1.35;
  let glow = exp(-pow(r / 4.2, 2.)) * .24;
  let ringR = select(5.8, 4.8, input.kind > .5);
  let ring = exp(-pow((r - ringR) / .58, 2.)) * .52;
  let breath = select(.90 + .10 * sin(u.timing.y * 1.2 + input.kind * 2.1), 1., u.timing.z > .5);
  let color = tint * (core + glow + ring) * breath * u.timing.w;
  return vec4f(color, 0.);
}
