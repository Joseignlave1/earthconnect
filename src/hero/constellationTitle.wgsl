// Shared uniforms for both compute and render passes.
struct TitleViewport {
  sizePx: vec2f,
  dpr: f32,
  opacity: f32,
}

struct StageRect {
  originPx: vec2f,
  sizePx: vec2f,
}

struct ViewBoxMeta {
  size: vec2f,
  fieldPad: f32,
  _padView: f32,
}

struct TitleTiming {
  timeSeconds: f32,
  dt: f32,
  outlineShift: vec2f,
}

struct TitleCounts {
  particleCount: f32,
  reducedMotion: f32, // 0/1
  _pad0: f32,
  _pad1: f32,
}

struct SlashInput {
  start: vec2f,
  end: vec2f,
  velocity: vec2f,
  isActive: f32,      // 0/1
  brushRadius: f32,
}

struct CtaInput {
  sinkPos: vec2f,     // black hole center in viewBox coords
  horizonViewR: f32,
  bottomViewY: f32,
  hover: f32,
  pressed: f32,
  _pad0: f32,
  _pad1: f32,
}

struct TitleUniform {
  viewport: TitleViewport,
  stage: StageRect,
  view: ViewBoxMeta,
  timing: TitleTiming,
  counts: TitleCounts,
  slash: SlashInput,
  cta: CtaInput,
}

struct StarGpu {
  pos: vec2f,   // viewBox coords
  mag: f32,     // 1/2/3
  phase: f32,
  color: vec3f, // linear spectral tint
  spiked: f32,  // 0/1
}

struct EdgeGpu {
  start: vec2f,     // trimmed viewBox start
  end: vec2f,       // trimmed viewBox end
  baseOpacity: f32,
  baseWidthPx: f32,
  kind: f32,        // 0 = letter, 1 = sightline
  _pad: f32,
}

struct DustHomeGpu {
  pos: vec2f,   // home viewBox coords
  size: f32,    // viewBox px
  seed: f32,    // 0..1
  color: vec3f, // sculpted rest linear RGB
  kind: f32,    // 0 = outline, 1 = inner bevel (or CTA tidal stream)
}

struct TitleDustState {
  pos: vec2f,       // live viewBox coords
  z: f32,           // liveZ (-0.35..+0.35, +toward viewer)
  settleFlash: f32, // 0..1
  vel: vec2f,       // viewBox / s
  vz: f32,          // units / s
  maxDisp: f32,     // peak displacement since last settle
}

struct CtaDustState {
  pos: vec2f,        // live viewBox coords
  settleFlash: f32,  // 0..1 (core letter flash)
  releaseDelay: f32, // seconds until stream grain detaches
  fallProgress: f32, // 0..1 radial infall progress for stream grains
  _pad: f32,
}

// ---------- Helpers (hash21 & valueNoise provided by noise.wgsl) ----------

fn fbm(p: vec2f) -> f32 {
  var v = 0.;
  var amp = .54;
  var q = p;
  for (var i = 0; i < 3; i++) {
    v += amp * valueNoise(q);
    q = mat2x2f(.8, -.6, .6, .8) * q * 2.04 + vec2f(17.3, 9.2);
    amp *= .5;
  }
  return v;
}

fn curl2d(p: vec2f) -> vec2f {
  let e = .35;
  let n1 = fbm(p + vec2f(0., e));
  let n2 = fbm(p - vec2f(0., e));
  let n3 = fbm(p + vec2f(e, 0.));
  let n4 = fbm(p - vec2f(e, 0.));
  return vec2f(n1 - n2, -(n3 - n4)) / (2. * e);
}

fn closestOnSeg(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let ab = b - a;
  let len2 = dot(ab, ab);
  if (len2 <= 1e-5) {
    return a;
  }
  let t = clamp(dot(p - a, ab) / len2, 0., 1.);
  return a + ab * t;
}

fn fieldSizeOf(uVal: TitleUniform) -> vec2f {
  return uVal.view.size + vec2f(uVal.view.fieldPad * 2.);
}

fn viewToFieldUv(uVal: TitleUniform, viewPos: vec2f) -> vec2f {
  return (viewPos + vec2f(uVal.view.fieldPad)) / fieldSizeOf(uVal);
}

fn quadCorner(vIdx: u32) -> vec2f {
  let corners = array<vec2f, 6>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(-1., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  return corners[vIdx];
}

// ============================================================================
// COMPUTE PASS A: Shared 2D Fluid Velocity & Slash Wake Field (172 x 80)
// Stores (vx, vy, wakeEnergy, 0) in rgba16float ping-pong.
// ============================================================================

@group(0) @binding(0) var<uniform> uVel: TitleUniform;
@group(0) @binding(1) var velSrc: texture_2d<f32>;
@group(0) @binding(2) var velSampler: sampler;
@group(0) @binding(3) var velDst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn velCompute(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(velDst);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + vec2f(.5)) / vec2f(dims);
  let fSize = fieldSizeOf(uVel);
  let viewPos = uv * fSize - vec2f(uVel.view.fieldPad);
  let dt = uVel.timing.dt;
  let reduced = uVel.counts.reducedMotion > .5;

  if (reduced) {
    textureStore(velDst, vec2i(gid.xy), vec4f(0.));
    return;
  }

  // Semi-Lagrangian self-advection + gentle diffusion so fluid swirls carry themselves forward.
  let cur = textureSampleLevel(velSrc, velSampler, uv, 0.);
  let backUv = clamp(uv - (cur.xy * dt) / fSize, vec2f(0.), vec2f(1.));
  let texel = vec2f(1.) / vec2f(dims);
  let adv = textureSampleLevel(velSrc, velSampler, backUv, 0.);
  let lap = (
    textureSampleLevel(velSrc, velSampler, clamp(backUv + vec2f(texel.x, 0.), vec2f(0.), vec2f(1.)), 0.) +
    textureSampleLevel(velSrc, velSampler, clamp(backUv - vec2f(texel.x, 0.), vec2f(0.), vec2f(1.)), 0.) +
    textureSampleLevel(velSrc, velSampler, clamp(backUv + vec2f(0., texel.y), vec2f(0.), vec2f(1.)), 0.) +
    textureSampleLevel(velSrc, velSampler, clamp(backUv - vec2f(0., texel.y), vec2f(0.), vec2f(1.)), 0.)
  ) * .25;

  var vel = mix(adv.xy, lap.xy, min(.35, dt * 6.));
  vel *= exp(-dt * 2.4);

  // Wake channel lingers (~1.25s) and diffuses slightly so revealed constellations stay open cleanly.
  var wake = mix(adv.z, lap.z, min(.22, dt * 3.5));
  wake = max(0., wake - dt * (.48 + wake * .42));

  // Inject unbroken continuous slash along slash.start -> slash.end.
  if (uVel.slash.isActive > .5) {
    let q = closestOnSeg(viewPos, uVel.slash.start, uVel.slash.end);
    let diff = viewPos - q;
    let d = length(diff);
    let r = uVel.slash.brushRadius;
    if (d < r * 1.45) {
      let falloff = exp(-pow(d / (r * .62), 2.));
      let outDir = select(vec2f(0., -1.), diff / d, d > .5);
      let slashVel = uVel.slash.velocity;
      let speed = length(slashVel);
      let tAnim = uVel.timing.timeSeconds;
      let curlLo = curl2d(viewPos * .032 + vec2f(tAnim * .35, -tAnim * .26));
      let curlHi = curl2d(viewPos * .078 + vec2f(-tAnim * .42, tAnim * .31));
      let curl = curlLo * .65 + curlHi * .45;
      let partStrength = 58. + min(110., speed * .22);
      let injectVel = outDir * (partStrength * falloff) + slashVel * (.56 * falloff) + curl * ((145. + min(140., speed * .28)) * falloff);
      vel += injectVel * min(1., dt * 15.);
      let vLen = length(vel);
      if (vLen > 460.) {
        vel = vel * (460. / vLen);
      }
      let wakeShape = smoothstep(r * 1.05, r * .18, d);
      wake = max(wake, wakeShape);
    }
  }

  textureStore(velDst, vec2i(gid.xy), vec4f(vel, wake, 0.));
}

// ============================================================================
// COMPUTE PASS B: Title & CTA GPU Stardust Simulation
// ============================================================================

@group(0) @binding(0) var<uniform> uDust: TitleUniform;
@group(0) @binding(1) var<storage, read> dustHome: array<DustHomeGpu>;
@group(0) @binding(2) var<storage, read_write> dustState: array<TitleDustState>;
@group(0) @binding(3) var dustVelTex: texture_2d<f32>;
@group(0) @binding(4) var dustSampler: sampler;
@group(0) @binding(5) var<storage, read_write> ctaDustState: array<CtaDustState>;

@compute @workgroup_size(128)
fn dustCompute(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let count = u32(uDust.counts.particleCount);
  if (idx >= count) {
    return;
  }
  let dh = dustHome[idx];
  let homePos = dh.pos + uDust.timing.outlineShift;
  let seed = dh.seed;
  let dt = uDust.timing.dt;

  if (uDust.counts.reducedMotion > .5) {
    dustState[idx] = TitleDustState(homePos, 0., 0., vec2f(0.), 0., 0.);
    return;
  }

  let st = dustState[idx];
  var pos = st.pos;
  var z = st.z;
  var settleFlash = max(0., st.settleFlash * exp(-dt * 5.4));
  var vel = st.vel;
  var vz = st.vz;
  var maxDisp = st.maxDisp;

  let uv = clamp(viewToFieldUv(uDust, pos), vec2f(0.), vec2f(1.));
  let field = textureSampleLevel(dustVelTex, dustSampler, uv, 0.);
  let fluidVel = field.xy;
  let localWake = field.z;

  let seed2 = fract(seed * 17.317 + .29);
  let seed3 = fract(seed * 53.719 + .61);
  let seed4 = fract(seed * 97.413 + .13);
  let scatterAngle = (seed2 - .5) * 2.15;
  let cs = cos(scatterAngle);
  let sn = sin(scatterAngle);

  // Soft Gaussian slash impulse with wide per-particle speed and angular fan.
  if (uDust.slash.isActive > .5) {
    let q = closestOnSeg(pos, uDust.slash.start, uDust.slash.end);
    let diff = pos - q;
    let d = length(diff);
    let rEff = uDust.slash.brushRadius * (.64 + .74 * seed2);
    if (d < rEff * 1.4) {
      let falloff = exp(-pow(d / (rEff * .58), 2.));
      let angle = seed * 6.28318;
      let fallbackDir = vec2f(cos(angle), sin(angle));
      let rawOut = select(fallbackDir, diff / d, d > .8);
      let fannedDir = vec2f(rawOut.x * cs - rawOut.y * sn, rawOut.x * sn + rawOut.y * cs);
      let slashVel = uDust.slash.velocity;
      let speed = length(slashVel);
      let grainPower = .4 + 1.35 * seed3;
      let push = (72. + min(195., speed * .36)) * grainPower;
      let impScale = falloff * min(1., dt * 13.);
      vel += (fannedDir * push + slashVel * (.26 + .34 * seed2)) * impScale;
      let zBias = (seed4 - .48) * 1.4;
      let zKick = zBias * (.55 + min(.9, speed * .0015));
      vz += zKick * impScale;
    }
  }

  let prevDisp = length(pos - homePos);
  let coupling = .55 + .85 * seed2;
  vel += (fluidVel * coupling - vel) * min(.36, dt * (1.8 + localWake * 2.4));

  // Skip curl/fBm noise evaluations completely for settled particles at rest.
  let time = uDust.timing.timeSeconds;
  let calm = 1. - smoothstep(.08, .72, localWake);
  let grainSpring = .68 + .72 * seed;
  let springK = mix(1.4, 21.5, calm * calm) * grainSpring;
  let damping = mix(2.2, 8.2, calm);
  let toHome = homePos - pos;
  let swirlWeight = smoothstep(1.5, 16., prevDisp) * (1. - calm * .8);
  var swirl = vec2f(0.);
  if (swirlWeight > .0001) {
    swirl = curl2d(pos * .048 + vec2f(time * .36 + seed * 3.1, -time * .28 + seed2 * 2.7)) * (48. + 42. * seed3);
  }
  let acc = toHome * springK + swirl * swirlWeight - vel * damping;

  let zSwirlWeight = smoothstep(2., 18., prevDisp) * (1. - calm * .8);
  var zSwirl = 0.;
  if (zSwirlWeight > .0001) {
    zSwirl = (fbm(pos * .05 + vec2f(time * .3, seed4 * 7.1)) - .5) * .55 * zSwirlWeight;
  }
  let accZ = -z * (springK * .95) + zSwirl - vz * (damping * .9);

  vel += acc * dt;
  pos += vel * dt;
  vz += accZ * dt;
  z = clamp(z + vz * dt, -.35, .35);

  let disp = length(pos - homePos);
  maxDisp = max(maxDisp, disp);

  if (maxDisp > 11. && disp < 3.4 && prevDisp >= 3.0 && localWake < .38) {
    settleFlash = max(settleFlash, clamp((maxDisp - 10.) / 26., .25, 1.));
    maxDisp = 0.;
  }

  if (localWake < .01 && disp < .18 && length(vel) < 1.2 && abs(z) < .015 && abs(vz) < .08) {
    pos = homePos;
    vel = vec2f(0.);
    z = 0.;
    vz = 0.;
    maxDisp = 0.;
  }

  dustState[idx] = TitleDustState(pos, z, settleFlash, vel, vz, maxDisp);
}

@compute @workgroup_size(128)
fn ctaDustCompute(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let count = u32(uDust.counts.particleCount);
  if (idx >= count) {
    return;
  }
  let dh = dustHome[idx];
  let homePos = dh.pos + uDust.timing.outlineShift;
  let seed = dh.seed;
  let dt = uDust.timing.dt;

  if (uDust.counts.reducedMotion > .5) {
    ctaDustState[idx] = CtaDustState(homePos, 0., 0., 0., 0.);
    return;
  }

  let st = ctaDustState[idx];
  let settleFlash = max(0., st.settleFlash * exp(-dt * 5.4));
  let seed2 = fract(seed * 17.317 + .29);
  let seed3 = fract(seed * 53.719 + .61);
  let seed4 = fract(seed * 97.413 + .13);

  let hover = uDust.cta.hover;
  let pressed = uDust.cta.pressed;
  let sinkPos = uDust.cta.sinkPos;
  let horizonViewR = uDust.cta.horizonViewR;
  let bottomViewY = uDust.cta.bottomViewY;
  let isStream = dh.kind > .5;
  let time = uDust.timing.timeSeconds;

  let toSink = sinkPos - homePos;
  let distCenter = length(toSink);
  let sinkDir = toSink / max(distCenter, 1.);

  if (!isStream) {
    let tension = (1. - hover) * (.32 + .40 * seed2) * (.65 + .35 * sin(time * 2.4 + seed * 19.));
    let restedPos = homePos + sinkDir * tension;
    let lockedPos = mix(restedPos, homePos, smoothstep(.05, .7, hover));
    let flash = max(settleFlash, max(hover * .30, pressed * .95));
    ctaDustState[idx] = CtaDustState(lockedPos, flash, 0., 0., 0.);
    return;
  }

  // Stream grains: zero-momentum, irreversible radial infall straight toward the black hole center.
  var age = st.fallProgress;
  var releaseDelay = st.releaseDelay;
  let lifeSpeed = .24 + .18 * seed2;
  let anchored = hover > .08;

  if (age > .0001) {
    age += dt * lifeSpeed;
    if (age >= 1.0) {
      if (anchored) {
        age = 0.;
        releaseDelay = seed * 2.6;
      } else {
        age = fract(age);
      }
    }
  } else {
    if (anchored) {
      releaseDelay = max(releaseDelay, seed * 2.6);
    } else if (releaseDelay > 0.) {
      releaseDelay = max(0., releaseDelay - dt);
    } else {
      age = dt * lifeSpeed;
    }
  }

  let perp = vec2f(-sinkDir.y, sinkDir.x);
  let distHorizon = max(36., distCenter - horizonViewR * .94);
  let distBottom = select(distHorizon, max(44., (bottomViewY + 26. - homePos.y) / max(sinkDir.y, .12)), sinkDir.y > .12);
  let travelDist = min(distHorizon, distBottom) * (.94 + .18 * seed3);

  let fallFrac = pow(age, 1.55);
  let spreadEnv = smoothstep(0., .32, age);
  let lateralJitter = ((seed4 - .5) * 9. + sin(time * 1.6 + seed * 31. + age * 5.5) * 1.8) * spreadEnv;
  let radialPos = homePos + sinkDir * (fallFrac * travelDist) + perp * lateralJitter;

  ctaDustState[idx] = CtaDustState(radialPos, 0., releaseDelay, age, 0.);
}

// ============================================================================
// RENDER PASSES (4x MSAA rgba16float HDR)
// ============================================================================

@group(0) @binding(0) var<uniform> u: TitleUniform;
@group(0) @binding(1) var<storage, read> stars: array<StarGpu>;
@group(0) @binding(2) var velSamplerR: sampler;
@group(0) @binding(3) var<storage, read> edges: array<EdgeGpu>;
@group(0) @binding(4) var<storage, read> dustHomeR: array<DustHomeGpu>;
@group(0) @binding(5) var<storage, read> dustStateR: array<TitleDustState>;
@group(0) @binding(6) var velTexR: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> ctaDustStateR: array<CtaDustState>;

fn sampleTitleParting(viewPos: vec2f) -> f32 {
  let uv = clamp(viewToFieldUv(u, viewPos), vec2f(0.), vec2f(1.));
  return textureSampleLevel(velTexR, velSamplerR, uv, 0.).z;
}

fn sampleCtaParting(viewPos: vec2f) -> f32 {
  let nx = clamp(viewPos.x / max(u.view.size.x, 1.), 0., 1.);
  return clamp(u.cta.hover * 1.45 - nx * .45, 0., 1.);
}

// ---------- Pass 1: GPU Stardust Particles ----------

struct DustVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) color: vec3f,
  @location(2) energy: f32,
  @location(3) corePx: f32,
  @location(4) maxR: f32,
}

@vertex fn dustVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> DustVertex {
  let dh = dustHomeR[iIdx];
  let ds = dustStateR[iIdx];
  let homePos = dh.pos + u.timing.outlineShift;
  let pos = ds.pos;
  let z = ds.z;
  let settleFlash = ds.settleFlash;
  let disp = length(pos - homePos);
  let loose = smoothstep(1.2, 16., disp);
  let pxPerView = u.stage.sizePx.x / u.view.size.x;

  let nearShift = max(0., z);
  let farShift = max(0., -z);
  let baseCore = dh.size * pxPerView;
  let scatterSize = mix(1.0, .70 + .36 * dh.seed, loose);
  let depthSize = 1. + z * .22;
  let corePx = max(.58, baseCore * scatterSize * depthSize);
  let maxR = corePx * 2.45;

  let centerPx = u.stage.originPx + (pos / u.view.size) * u.stage.sizePx;
  let c = quadCorner(vIdx);
  let posPx = centerPx + c * maxR;
  let ndc = vec2f((posPx.x / u.viewport.sizePx.x) * 2. - 1., 1. - (posPx.y / u.viewport.sizePx.y) * 2.);

  let twinkle = 1. - loose * .34 + loose * .34 * sin(u.timing.timeSeconds * 5.2 + dh.seed * 29.);
  let isBevel = dh.kind > .5;
  let warmLift = vec3f(1.26, .98, .66);
  let coolFar = vec3f(.78, .88, 1.12);
  let condenseGlow = vec3f(1.42, 1.28, 1.02);
  var rgb = mix(dh.color, warmLift, loose * (.32 + nearShift * .18));
  rgb = mix(rgb, coolFar, farShift * .35);
  rgb += condenseGlow * (settleFlash * .36);

  let baseAlpha = select(.56, .20, isBevel);
  let depthFade = 1. - farShift * .24 - nearShift * .12;
  let energy = baseAlpha * mix(1., .66, loose * .65) * depthFade * twinkle + settleFlash * .28;

  return DustVertex(vec4f(ndc, 0., 1.), c * maxR, rgb, energy, corePx, maxR);
}

@vertex fn ctaDustVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> DustVertex {
  let dh = dustHomeR[iIdx];
  let ds = ctaDustStateR[iIdx];
  let homePos = dh.pos + u.timing.outlineShift;
  let pos = ds.pos;
  let settleFlash = ds.settleFlash;
  let disp = length(pos - homePos);
  let loose = smoothstep(1.2, 16., disp);
  let pxPerView = u.stage.sizePx.x / u.view.size.x;

  let baseCore = dh.size * pxPerView;
  let scatterSize = mix(1.0, .70 + .36 * dh.seed, loose);
  let corePx = max(.58, baseCore * scatterSize);
  let maxR = corePx * 2.45;

  let centerPx = u.stage.originPx + (pos / u.view.size) * u.stage.sizePx;
  let c = quadCorner(vIdx);
  let posPx = centerPx + c * maxR;
  let ndc = vec2f((posPx.x / u.viewport.sizePx.x) * 2. - 1., 1. - (posPx.y / u.viewport.sizePx.y) * 2.);

  let isStream = dh.kind > .5;
  let ctaPressed = u.cta.pressed;
  let ctaSweep = sampleCtaParting(pos);
  var rgb = mix(dh.color, vec3f(1.26, .98, .66), max(loose * .32, ctaSweep * .48));
  rgb += vec3f(1.42, 1.28, 1.02) * (settleFlash * .36 + ctaPressed * .35);
  var energy = .54 * (1. + ctaSweep * .72 + ctaPressed * .42);

  if (isStream) {
    let age = ds.fallProgress;
    let departFade = smoothstep(.01, .10, age) * smoothstep(1.0, .84, age);
    let streamTwinkle = .72 + .28 * sin(u.timing.timeSeconds * 4.4 + dh.seed * 29.);
    rgb = mix(vec3f(.94, .88, .78), vec3f(1.22, .86, .52), smoothstep(.15, .82, age));
    energy = departFade * .34 * streamTwinkle;
  }

  return DustVertex(vec4f(ndc, 0., 1.), c * maxR, rgb, energy, corePx, maxR);
}

@fragment fn dustFragment(input: DustVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  if (r > input.maxR) {
    discard;
  }
  let nr = r / input.corePx;
  let core = exp(-nr * nr);
  let intensity = core * input.energy * u.viewport.opacity;
  return vec4f(input.color * intensity, 0.);
}

// ---------- Pass 2: Anti-aliased HDR Constellation Hairlines ----------

struct EdgeVertex {
  @builtin(position) position: vec4f,
  @location(0) localX: f32,
  @location(1) crossSide: f32,
  @location(2) segLen: f32,
  @location(3) halfWidthPx: f32,
  @location(4) baseOpacity: f32,
  @location(5) kind: f32,
  @location(6) viewPos: vec2f,
}

@vertex fn edgeVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> EdgeVertex {
  let corners = array<vec2f, 6>(
    vec2f(0., -1.),
    vec2f(1., -1.),
    vec2f(0., 1.),
    vec2f(0., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  let ed = edges[iIdx];
  let pA = ed.start;
  let pB = ed.end;
  let segLen = max(length(pB - pA), 1e-4);

  let pxA = u.stage.originPx + (pA / u.view.size) * u.stage.sizePx;
  let pxB = u.stage.originPx + (pB / u.view.size) * u.stage.sizePx;
  let dPx = pxB - pxA;
  let lenPx = max(length(dPx), 1e-4);
  let perpPx = vec2f(-dPx.y, dPx.x) / lenPx;

  let corner = corners[vIdx];
  let isEnd = corner.x > .5;
  let halfW = max((ed.baseWidthPx + .85 * u.viewport.dpr) * 3.2, 5.2);
  let posPx = select(pxA, pxB, isEnd) + perpPx * (corner.y * halfW);
  let ndc = vec2f((posPx.x / u.viewport.sizePx.x) * 2. - 1., 1. - (posPx.y / u.viewport.sizePx.y) * 2.);

  return EdgeVertex(
    vec4f(ndc, 0., 1.),
    select(0., segLen, isEnd),
    corner.y,
    segLen,
    halfW,
    ed.baseOpacity,
    ed.kind,
    select(pA, pB, isEnd),
  );
}

fn shadeEdge(input: EdgeVertex, parting: f32) -> vec4f {
  let distPx = abs(input.crossSide) * input.halfWidthPx;
  let dpr = u.viewport.dpr;
  let isSightline = input.kind > .5;
  let parted = smoothstep(.08, .72, parting);
  let highlight = select(parted, parted * .45, isSightline);

  let lineSigma = (.38 + highlight * .24) * dpr;
  let filterSigma = .78;
  let sigma = sqrt(lineSigma * lineSigma + filterSigma * filterSigma);
  let rawHaloSigma = lineSigma * 2.4;
  let haloSigma = sqrt(rawHaloSigma * rawHaloSigma + filterSigma * filterSigma);
  let energyPreserve = pow(lineSigma / sigma, .65);
  let core = exp(-pow(distPx / sigma, 2.)) * energyPreserve;
  let halo = exp(-pow(distPx / haloSigma, 2.)) * (.08 + highlight * .16) * (rawHaloSigma / haloSigma);

  let t = clamp(input.localX / max(input.segLen, 1e-4), 0., 1.);
  let endLift = .74 + .36 * pow(abs(t * 2. - 1.), 1.6);

  var dashMask = 1.;
  if (isSightline) {
    let phase = fract(input.localX / 8.);
    dashMask = smoothstep(.02, .20, phase) * (1. - smoothstep(.32, .50, phase));
  }

  let baseOp = input.baseOpacity;
  let targetOp = select(1.28, .36, isSightline);
  let edgeOp = mix(baseOp, targetOp, highlight);

  let restTint = vec3f(.96, .91, .84);
  let hoverTint = vec3f(1.28, 1.02, .70);
  let tint = mix(restTint, hoverTint, smoothstep(.05, .7, highlight));

  let intensity = (core + halo) * endLift * dashMask * edgeOp * u.viewport.opacity;
  return vec4f(tint * intensity, 0.);
}

@fragment fn edgeFragment(input: EdgeVertex) -> @location(0) vec4f {
  return shadeEdge(input, sampleTitleParting(input.viewPos));
}

@fragment fn ctaEdgeFragment(input: EdgeVertex) -> @location(0) vec4f {
  return shadeEdge(input, sampleCtaParting(input.viewPos));
}

// ---------- Shared Pass: True HDR Photorealistic Optical Stars ----------

struct StarVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) color: vec3f,
  @location(2) focus: f32,
  @location(3) mag: f32,
  @location(4) maxR: f32,
  @location(5) phase: f32,
  @location(6) spiked: f32,
}

fn buildStarVertex(vIdx: u32, st: StarGpu, parting: f32) -> StarVertex {
  let focus = smoothstep(.08, .72, parting);
  let mag = st.mag;
  let spiked = st.spiked;

  let pxPerView = u.stage.sizePx.x / u.view.size.x;
  let baseViewR = select(select(5.2, 8.0, mag < 2.5), 12.5, mag < 1.5);
  let spikeViewR = select(0., select(11.5, 17.5, mag < 1.5), spiked > .5 || focus > .02);
  let radiusPx = max(baseViewR * (1. + focus * .4), spikeViewR * max(spiked, focus * 1.15)) * pxPerView;

  let centerPx = u.stage.originPx + (st.pos / u.view.size) * u.stage.sizePx;
  let c = quadCorner(vIdx);
  let posPx = centerPx + c * radiusPx;
  let ndc = vec2f((posPx.x / u.viewport.sizePx.x) * 2. - 1., 1. - (posPx.y / u.viewport.sizePx.y) * 2.);

  return StarVertex(
    vec4f(ndc, 0., 1.),
    c * radiusPx,
    st.color,
    focus,
    mag,
    radiusPx,
    st.phase,
    spiked,
  );
}

@vertex fn starVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> StarVertex {
  let st = stars[iIdx];
  return buildStarVertex(vIdx, st, sampleTitleParting(st.pos));
}

@vertex fn ctaStarVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> StarVertex {
  let st = stars[iIdx];
  return buildStarVertex(vIdx, st, sampleCtaParting(st.pos));
}

@fragment fn starFragment(input: StarVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  let maxR = input.maxR;
  if (r > maxR) {
    discard;
  }
  let mag = input.mag;
  let focus = input.focus;
  let phase = input.phase;
  let spiked = input.spiked;
  let pxPerView = u.stage.sizePx.x / u.view.size.x;

  let time = u.timing.timeSeconds;
  let staticMode = u.counts.reducedMotion > .5;
  let scint = select(
    .88 + .08 * sin(time * 1.35 + phase) + .04 * sin(time * 2.9 + phase * 1.7),
    1.,
    staticMode,
  );

  let boost = 1. + focus * .36;
  let coreR = select(select(.36, .54, mag < 2.5), .76, mag < 1.5) * pxPerView * (1. + focus * .28);
  let haloR = select(select(.45, .72, mag < 2.5), 1.08, mag < 1.5) * pxPerView * boost;
  let edgeWindow = smoothstep(maxR, maxR * .3, r);

  let core = exp(-pow(r / max(coreR, .48), 2.));
  let corona = exp(-r / max(haloR, .65)) * edgeWindow;

  let baseCore = select(select(.68, 1.02, mag < 2.5), 1.48, mag < 1.5);
  let coreEnergy = mix(baseCore, max(baseCore * 1.45, 1.65), focus);
  let coronaEnergy = select(select(.045, .085, mag < 2.5), .14, mag < 1.5) * (1. + focus * 1.1);

  var color = vec3f(1., .99, .97) * (core * coreEnergy)
            + input.color * (corona * coronaEnergy);

  if (focus > .02) {
    color += vec3f(1.08, .82, .54) * exp(-pow(r / max(haloR * 2.2, 1.4), 2.)) * edgeWindow * (focus * .14);
  }

  let spikeVis = max(spiked * (.46 + focus * .44), focus * .74);
  if (spikeVis > .01) {
    let spikeLen = select(select(4.8, 7.2, mag < 2.5), 10.6, mag < 1.5) * pxPerView * (1. + focus * .28);
    let rawSpikeW = max(.28 * u.viewport.dpr, .28 * pxPerView);
    let spikeW = sqrt(rawSpikeW * rawSpikeW + .66 * .66);
    let spikeGain = rawSpikeW / spikeW;
    let ax = abs(input.localPx.x);
    let ay = abs(input.localPx.y);
    let hSpike = exp(-pow(ay / spikeW, 2.)) * exp(-ax / max(spikeLen * .34, 1.)) * smoothstep(spikeLen, 0., ax);
    let vSpike = exp(-pow(ax / spikeW, 2.)) * exp(-ay / max(spikeLen * .34, 1.)) * smoothstep(spikeLen, 0., ay);
    let spikeTint = mix(vec3f(1., .97, .92), input.color, .35);
    color += spikeTint * ((hSpike + vSpike) * spikeGain * spikeVis * (.52 + focus * .48));
  }

  return vec4f(color * scint * u.viewport.opacity, 0.);
}
