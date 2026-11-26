// Shared uniforms for both compute and render passes.
struct TitleUniform {
  viewport: vec4f,  // canvasWidthPx, canvasHeightPx, dpr, opacity
  stageRect: vec4f, // stageLeftPx, stageTopPx, stageWidthPx, stageHeightPx
  viewMeta: vec4f,  // viewW, viewH, fieldPad, starCount
  timing: vec4f,    // timeSeconds, dt, outlineShiftX, outlineShiftY
  counts: vec4f,    // edgeCount, particleCount, ctaMode (0 = title, 1 = cta), reducedMotion (0/1)
  slashA: vec4f,    // slashAx, slashAy, slashActive (0/1), brushRadius
  slashB: vec4f,    // slashBx, slashBy, slashVx, slashVy
}

struct StarGpu {
  pos: vec4f,   // x, y (viewBox), z (depth), mag (1/2/3)
  color: vec4f, // r, g, b (linear spectral tint), unused
  extra: vec4f, // phase, spiked (0/1), unused, unused
}

struct EdgeGpu {
  pts: vec4f,   // x1, y1, x2, y2 (trimmed viewBox endpoints)
  style: vec4f, // baseOpacity, baseWidthPx, kind (0 = letter, 1 = sightline), unused
}

struct DustHomeGpu {
  home: vec4f,  // homeX, homeY (viewBox), size (viewBox px), seed (0..1)
  tint: vec4f,  // r, g, b (sculpted rest linear color), kind (0 = outline, 1 = inner bevel)
}

struct DustStateGpu {
  posZ: vec4f, // liveX, liveY (viewBox), liveZ (-1..+1, +toward viewer), settleFlash (0..1)
  velZ: vec4f, // vx, vy (viewBox / s), vz (units / s), maxDisp (peak displacement memory)
}

// ---------- Helpers ----------

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let s = f * f * (3. - 2. * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1., 0.));
  let c = hash21(i + vec2f(0., 1.));
  let d = hash21(i + vec2f(1., 1.));
  return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

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
  return uVal.viewMeta.xy + vec2f(uVal.viewMeta.z * 2.);
}

fn viewToFieldUv(uVal: TitleUniform, viewPos: vec2f) -> vec2f {
  return (viewPos + vec2f(uVal.viewMeta.z)) / fieldSizeOf(uVal);
}

fn viewToNdc(uVal: TitleUniform, viewPos: vec2f) -> vec2f {
  let uv = viewPos / uVal.viewMeta.xy;
  let px = uVal.stageRect.xy + uv * uVal.stageRect.zw;
  return vec2f((px.x / uVal.viewport.x) * 2. - 1., 1. - (px.y / uVal.viewport.y) * 2.);
}

// ============================================================================
// COMPUTE PASS A: Shared 2D Fluid Velocity & Slash Wake Field (172 x 80)
// Stores (vx, vy, wakeEnergy, swirlCurl) in rgba16float ping-pong.
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
  let viewPos = uv * fSize - vec2f(uVel.viewMeta.z);
  let dt = uVel.timing.y;
  let reduced = uVel.counts.w > .5;

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

  // Inject unbroken continuous slash along slashA -> slashB.
  if (uVel.slashA.z > .5) {
    let q = closestOnSeg(viewPos, uVel.slashA.xy, uVel.slashB.xy);
    let diff = viewPos - q;
    let d = length(diff);
    let r = uVel.slashA.w;
    if (d < r * 1.45) {
      let falloff = exp(-pow(d / (r * .62), 2.));
      let outDir = select(vec2f(0., -1.), diff / d, d > .5);
      let slashVel = uVel.slashB.zw;
      let speed = length(slashVel);
      // Multi-scale curl vorticity + directional blade sweep + gentle outward parting (no snowplow ring).
      let tAnim = uVel.timing.x;
      let curlLo = curl2d(viewPos * .032 + vec2f(tAnim * .35, -tAnim * .26));
      let curlHi = curl2d(viewPos * .078 + vec2f(-tAnim * .42, tAnim * .31));
      let curl = curlLo * .65 + curlHi * .45;
      let partStrength = 58. + min(110., speed * .22);
      let injectVel = outDir * (partStrength * falloff) + slashVel * (.56 * falloff) + curl * ((145. + min(140., speed * .28)) * falloff);
      vel += injectVel * min(1., dt * 15.);
      // Clamp peak velocity for numerical stability.
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
// COMPUTE PASS B: Mode 1 — True 24,000-Particle GPU Stardust Lettering
// ============================================================================

@group(0) @binding(0) var<uniform> uDust: TitleUniform;
@group(0) @binding(1) var<storage, read> dustHome: array<DustHomeGpu>;
@group(0) @binding(2) var<storage, read_write> dustState: array<DustStateGpu>;
@group(0) @binding(3) var dustVelTex: texture_2d<f32>;
@group(0) @binding(4) var dustSampler: sampler;

@compute @workgroup_size(128)
fn dustCompute(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let count = u32(uDust.counts.y);
  if (idx >= count) {
    return;
  }
  let h = dustHome[idx].home;
  let homePos = h.xy + uDust.timing.zw;
  let seed = h.w;
  let dt = uDust.timing.y;
  let reduced = uDust.counts.w > .5;

  if (reduced) {
    dustState[idx].posZ = vec4f(homePos, 0., 0.);
    dustState[idx].velZ = vec4f(0.);
    return;
  }

  let stPos = dustState[idx].posZ;
  let stVel = dustState[idx].velZ;
  var pos = stPos.xy;
  var z = stPos.z;
  var settleFlash = max(0., stPos.w * exp(-dt * 5.4));
  var vel = stVel.xy;
  var vz = stVel.z;
  var maxDisp = stVel.w;

  // Sample shared 2D fluid velocity field at the particle's LIVE position.
  let uv = clamp(viewToFieldUv(uDust, pos), vec2f(0.), vec2f(1.));
  let field = textureSampleLevel(dustVelTex, dustSampler, uv, 0.);
  let fluidVel = field.xy;
  let localWake = field.z;

  // Derive independent per-particle random traits so grains disperse as a true 3D cloud, never a rigid sheet.
  let seed2 = fract(seed * 17.317 + .29);
  let seed3 = fract(seed * 53.719 + .61);
  let seed4 = fract(seed * 97.413 + .13);
  let scatterAngle = (seed2 - .5) * 2.15;
  let cs = cos(scatterAngle);
  let sn = sin(scatterAngle);

  // In ctaMode (Explore button):
  // - At rest, sparse fine stardust grains (kind == 1) quietly bleed off the lower strokes of Explore and fall
  //   straight radially toward the true center of the black hole (slashB.xy) with ZERO momentum/inertia.
  //   Any apparent curvature near the bottom comes purely from optical light bending in blackHole.wgsl.
  // - On hover, the user anchors Explore against the pull: new shedding stops immediately while already-escaped
  //   grains finish falling irreversibly into the black hole (never pulled backward), and the button locks solid.
  let isCta = uDust.counts.z > .5;
  if (isCta) {
    let hover = uDust.slashA.z;
    let pressed = uDust.slashA.w;
    let sinkPos = uDust.slashB.xy;
    let horizonViewR = uDust.slashB.z;
    let bottomViewY = uDust.slashB.w;
    let isStream = dustHome[idx].tint.w > .5;
    let time = uDust.timing.x;

    let toSink = sinkPos - homePos;
    let distCenter = length(toSink);
    let sinkDir = toSink / max(distCenter, 1.);

    if (!isStream) {
      // Core Explore letter grains: tiny radial lean toward the black hole center at rest (100% readable),
      // locking completely solid & glowing bright warm starlight on hover/press.
      let tension = (1. - hover) * (.32 + .40 * seed2) * (.65 + .35 * sin(time * 2.4 + seed * 19.));
      let restedPos = homePos + sinkDir * tension;
      let lockedPos = mix(restedPos, homePos, smoothstep(.05, .7, hover));
      let flash = max(settleFlash, max(hover * .30, pressed * .95));
      dustState[idx].posZ = vec4f(lockedPos, 0., flash);
      dustState[idx].velZ = vec4f(0.);
      return;
    }

    // Stream grains: pure zero-momentum, irreversible radial infall straight toward the black hole center.
    // velZ.w stores normalized radial fall progress (0 = anchored inside Explore, >0..1 = falling into void).
    // velZ.z stores a staggered re-release delay (seconds) so shedding resumes gently grain-by-grain after unhover.
    var age = maxDisp;
    var releaseDelay = vz;
    let lifeSpeed = .24 + .18 * seed2; // ~2.4s..4.2s quiet radial drift into the gravity well
    let anchored = hover > .08;

    if (age > .0001) {
      // Already detached: keeps falling irreversibly into the black hole; never rewinds upward.
      age += dt * lifeSpeed;
      if (age >= 1.0) {
        if (anchored) {
          // Hovering halts new drainage once this grain finishes crossing the bottom fold / horizon.
          age = 0.;
          releaseDelay = seed * 2.6;
        } else {
          age = fract(age);
        }
      }
    } else {
      // Held safely at Explore while hovered; softly resumes peeling off after unhovering.
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

    // Gravitational free-fall acceleration (s ~ age^1.55) directly along the instantaneous center ray.
    let fallFrac = pow(age, 1.55);
    let spreadEnv = smoothstep(0., .32, age);
    let lateralJitter = ((seed4 - .5) * 9. + sin(time * 1.6 + seed * 31. + age * 5.5) * 1.8) * spreadEnv;
    let radialPos = homePos + sinkDir * (fallFrac * travelDist) + perp * lateralJitter;

    let speed = 1.55 * pow(max(age, .02), .55) * travelDist * lifeSpeed;
    dustState[idx].posZ = vec4f(radialPos, 0., 0.);
    dustState[idx].velZ = vec4f(sinkDir * speed, releaseDelay, age);
    return;
  }

  // Soft Gaussian slash impulse with wide per-particle speed and angular fan.
  if (uDust.slashA.z > .5) {
    let q = closestOnSeg(pos, uDust.slashA.xy, uDust.slashB.xy);
    let diff = pos - q;
    let d = length(diff);
    let rEff = uDust.slashA.w * (.64 + .74 * seed2);
    if (d < rEff * 1.4) {
      let falloff = exp(-pow(d / (rEff * .58), 2.));
      let angle = seed * 6.28318;
      let fallbackDir = vec2f(cos(angle), sin(angle));
      let rawOut = select(fallbackDir, diff / d, d > .8);
      // Rotate each particle's kick direction uniquely so dust fans out into weightless plumes.
      let fannedDir = vec2f(rawOut.x * cs - rawOut.y * sn, rawOut.x * sn + rawOut.y * cs);
      let slashVel = uDust.slashB.zw;
      let speed = length(slashVel);
      let grainPower = .4 + 1.35 * seed3;
      let push = (72. + min(195., speed * .36)) * grainPower;
      let impScale = falloff * min(1., dt * 13.);
      vel += (fannedDir * push + slashVel * (.26 + .34 * seed2)) * impScale;
      // Very subtle Z drift so dust stays locked to the 2D cursor slash with just a whisper of depth.
      let zBias = (seed4 - .48) * 1.4;
      let zKick = zBias * (.55 + min(.9, speed * .0015));
      vz += zKick * impScale;
    }
  }

  // Fluid drag couples loose stardust into the curling eddies with per-grain inertia variance.
  let prevDisp = length(pos - homePos);
  let coupling = .55 + .85 * seed2;
  vel += (fluidVel * coupling - vel) * min(.36, dt * (1.8 + localWake * 2.4));

  // Organic zero-g curl turbulence while scattered; individually-paced smooth homing spring back to exact homePos.
  let time = uDust.timing.x;
  let swirl = curl2d(pos * .048 + vec2f(time * .36 + seed * 3.1, -time * .28 + seed2 * 2.7)) * (48. + 42. * seed3);
  let calm = 1. - smoothstep(.08, .72, localWake);
  let grainSpring = .68 + .72 * seed;
  let springK = mix(1.4, 21.5, calm * calm) * grainSpring;
  let damping = mix(2.2, 8.2, calm);
  let toHome = homePos - pos;
  let acc = toHome * springK + swirl * smoothstep(1.5, 16., prevDisp) * (1. - calm * .8) - vel * damping;

  // Gentle Z spring pulls the tiny depth shimmer back into the focal plane.
  let zSwirl = (fbm(pos * .05 + vec2f(time * .3, seed4 * 7.1)) - .5) * .55 * smoothstep(2., 18., prevDisp) * (1. - calm * .8);
  let accZ = -z * (springK * .95) + zSwirl - vz * (damping * .9);

  vel += acc * dt;
  pos += vel * dt;
  vz += accZ * dt;
  z = clamp(z + vz * dt, -.35, .35);

  let disp = length(pos - homePos);
  maxDisp = max(maxDisp, disp);

  // Delicate starlight condensation glint as genuinely scattered dust locks back onto the calligraphy contour.
  if (maxDisp > 11. && disp < 3.4 && prevDisp >= 3.0 && localWake < .38) {
    settleFlash = max(settleFlash, clamp((maxDisp - 10.) / 26., .25, 1.));
    maxDisp = 0.;
  }

  // Snap cleanly once settled at rest so zero sub-pixel micro-jitter occurs on static letters.
  if (localWake < .01 && disp < .18 && length(vel) < 1.2 && abs(z) < .015 && abs(vz) < .08) {
    pos = homePos;
    vel = vec2f(0.);
    z = 0.;
    vz = 0.;
    maxDisp = 0.;
  }

  dustState[idx].posZ = vec4f(pos, z, settleFlash);
  dustState[idx].velZ = vec4f(vel, vz, maxDisp);
}

// ============================================================================
// RENDER PASSES (4x MSAA rgba16float HDR, shared bind group)
// ============================================================================

@group(0) @binding(0) var<uniform> u: TitleUniform;
@group(0) @binding(1) var<storage, read> stars: array<StarGpu>;
@group(0) @binding(2) var velSamplerR: sampler;
@group(0) @binding(3) var<storage, read> edges: array<EdgeGpu>;
@group(0) @binding(4) var<storage, read> dustHomeR: array<DustHomeGpu>;
@group(0) @binding(5) var<storage, read> dustStateR: array<DustStateGpu>;
@group(0) @binding(6) var velTexR: texture_2d<f32>;

fn sampleParting(viewPos: vec2f) -> f32 {
  if (u.counts.z > .5) {
    // Cohesive left-to-right ignition sweep across the whole Explore CTA when hovered/focused.
    let hover = u.slashA.z;
    let nx = clamp(viewPos.x / max(u.viewMeta.x, 1.), 0., 1.);
    return clamp(hover * 1.45 - nx * .45, 0., 1.);
  }
  let uv = clamp(viewToFieldUv(u, viewPos), vec2f(0.), vec2f(1.));
  return textureSampleLevel(velTexR, velSamplerR, uv, 0.).z;
}

// ---------- Pass 1: 24,000 GPU Stardust Particles ----------

struct DustVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) color: vec4f,   // rgb: HDR color, a: alpha/energy
  @location(2) params: vec2f,  // x: corePx, y: maxR
}

@vertex fn dustVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> DustVertex {
  let corners = array<vec2f, 6>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(-1., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  let dh = dustHomeR[iIdx];
  let ds = dustStateR[iIdx].posZ;
  let homePos = dh.home.xy + u.timing.zw;
  let pos = ds.xy;
  let z = ds.z; // subtle micro-depth shimmer (-0.35..+0.35)
  let settleFlash = ds.w;
  let disp = length(pos - homePos);
  let loose = smoothstep(1.2, 16., disp);
  let pxPerView = u.stageRect.z / u.viewMeta.x;

  // Pure 2D screen-space position locked to the cursor's motion; only a whisper of Z size/tint shimmer.
  let nearShift = max(0., z);
  let farShift = max(0., -z);
  let baseCore = dh.home.z * pxPerView;
  let scatterSize = mix(1.0, .70 + .36 * dh.home.w, loose);
  let depthSize = 1. + z * .22;
  let corePx = max(.42, baseCore * scatterSize * depthSize);
  let maxR = corePx * 2.35;

  let centerPx = u.stageRect.xy + (pos / u.viewMeta.xy) * u.stageRect.zw;
  let c = corners[vIdx];
  let posPx = centerPx + c * maxR;
  let ndc = vec2f((posPx.x / u.viewport.x) * 2. - 1., 1. - (posPx.y / u.viewport.y) * 2.);

  let twinkle = 1. - loose * .34 + loose * .34 * sin(u.timing.x * 5.2 + dh.home.w * 29.);
  let isCta = u.counts.z > .5;
  let isSecondKind = dh.tint.w > .5;
  let ctaHover = select(0., u.slashA.z, isCta);
  let ctaPressed = select(0., u.slashA.w, isCta);
  let ctaSweep = select(0., sampleParting(pos), isCta);
  let warmLift = vec3f(1.26, .98, .66);
  let coolFar = vec3f(.78, .88, 1.12);
  let condenseGlow = vec3f(1.42, 1.28, 1.02);
  var rgb = mix(dh.tint.rgb, warmLift, max(loose * (.32 + nearShift * .18), ctaSweep * .48));
  rgb = mix(rgb, coolFar, farShift * .35);
  rgb += condenseGlow * (settleFlash * .36 + ctaPressed * .35);

  let baseAlpha = select(.56, .20, isSecondKind);
  let depthFade = 1. - farShift * .24 - nearShift * .12;
  var energy = baseAlpha * mix(1., .66, loose * .65) * depthFade * twinkle + settleFlash * .28;
  if (isCta) {
    if (isSecondKind) {
      // Sparse tidal bleed grains quietly drifting from Explore into the black hole:
      // invisible while anchored at age == 0, soft champagne-ivory warming gently as they fall.
      let age = dustStateR[iIdx].velZ.w;
      let departFade = smoothstep(.01, .10, age) * smoothstep(1.0, .84, age);
      let streamTwinkle = .72 + .28 * sin(u.timing.x * 4.4 + dh.home.w * 29.);
      rgb = mix(vec3f(.94, .88, .78), vec3f(1.22, .86, .52), smoothstep(.15, .82, age));
      energy = departFade * .34 * streamTwinkle;
    } else {
      // Explore letter strokes: clean & steady at rest, locking crisp & luminous when hovered!
      energy = .54 * (1. + ctaSweep * .72 + ctaPressed * .42);
    }
  }

  return DustVertex(
    vec4f(ndc, 0., 1.),
    c * maxR,
    vec4f(rgb, energy),
    vec2f(corePx, maxR),
  );
}

@fragment fn dustFragment(input: DustVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  if (r > input.params.y) {
    discard;
  }
  let nr = r / input.params.x;
  let core = exp(-nr * nr);
  let intensity = core * input.color.a * u.viewport.w;
  return vec4f(input.color.rgb * intensity, 0.);
}

// ---------- Pass 2: Anti-aliased HDR Constellation Hairlines ----------

struct EdgeVertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) metrics: vec4f,
  @location(2) viewPos: vec2f,
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
  let pA = ed.pts.xy;
  let pB = ed.pts.zw;
  let segLen = max(length(pB - pA), 1e-4);

  let pxA = u.stageRect.xy + (pA / u.viewMeta.xy) * u.stageRect.zw;
  let pxB = u.stageRect.xy + (pB / u.viewMeta.xy) * u.stageRect.zw;
  let dPx = pxB - pxA;
  let lenPx = max(length(dPx), 1e-4);
  let perpPx = vec2f(-dPx.y, dPx.x) / lenPx;

  let corner = corners[vIdx];
  let isEnd = corner.x > .5;
  let halfW = max((ed.style.y + .75 * u.viewport.z) * 2.8, 3.6 * u.viewport.z);
  let posPx = select(pxA, pxB, isEnd) + perpPx * (corner.y * halfW);
  let ndc = vec2f((posPx.x / u.viewport.x) * 2. - 1., 1. - (posPx.y / u.viewport.y) * 2.);

  return EdgeVertex(
    vec4f(ndc, 0., 1.),
    vec2f(select(0., segLen, isEnd), corner.y),
    vec4f(segLen, halfW, ed.style.x, ed.style.z),
    select(pA, pB, isEnd),
  );
}

@fragment fn edgeFragment(input: EdgeVertex) -> @location(0) vec4f {
  let distPx = abs(input.local.y) * input.metrics.y;
  let dpr = u.viewport.z;
  let isSightline = input.metrics.w > .5;
  let parted = smoothstep(.08, .72, sampleParting(input.viewPos));
  let highlight = select(parted, parted * .45, isSightline);

  let sigma = (.36 + highlight * .26) * dpr;
  let core = exp(-pow(distPx / sigma, 2.));
  let halo = exp(-pow(distPx / (sigma * 2.8), 2.)) * (.08 + highlight * .18);

  let t = clamp(input.local.x / max(input.metrics.x, 1e-4), 0., 1.);
  let endLift = .74 + .36 * pow(abs(t * 2. - 1.), 1.6);

  var dashMask = 1.;
  if (isSightline) {
    let phase = fract(input.local.x / 8.);
    dashMask = smoothstep(.04, .14, phase) * (1. - smoothstep(.36, .46, phase));
  }

  let baseOp = input.metrics.z;
  let targetOp = select(1.28, .36, isSightline);
  let edgeOp = mix(baseOp, targetOp, highlight);

  let restTint = vec3f(.96, .91, .84);
  let hoverTint = vec3f(1.28, 1.02, .70);
  let tint = mix(restTint, hoverTint, smoothstep(.05, .7, highlight));

  let intensity = (core + halo) * endLift * dashMask * edgeOp * u.viewport.w;
  return vec4f(tint * intensity, 0.);
}

// ---------- Shared Pass: True HDR Photorealistic Optical Stars ----------

struct StarVertex {
  @builtin(position) position: vec4f,
  @location(0) localPx: vec2f,
  @location(1) color: vec4f,
  @location(2) params: vec4f,
}

@vertex fn starVertex(
  @builtin(vertex_index) vIdx: u32,
  @builtin(instance_index) iIdx: u32,
) -> StarVertex {
  let corners = array<vec2f, 6>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(-1., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  let st = stars[iIdx];
  let focus = smoothstep(.08, .72, sampleParting(st.pos.xy));
  let mag = st.pos.w;
  let spiked = st.extra.y;

  let pxPerView = u.stageRect.z / u.viewMeta.x;
  let baseViewR = select(select(5.2, 8.0, mag < 2.5), 12.5, mag < 1.5);
  let spikeViewR = select(0., select(11.5, 17.5, mag < 1.5), spiked > .5 || focus > .02);
  let radiusPx = max(baseViewR * (1. + focus * .4), spikeViewR * max(spiked, focus * 1.15)) * pxPerView;

  let centerPx = u.stageRect.xy + (st.pos.xy / u.viewMeta.xy) * u.stageRect.zw;
  let c = corners[vIdx];
  let posPx = centerPx + c * radiusPx;
  let ndc = vec2f((posPx.x / u.viewport.x) * 2. - 1., 1. - (posPx.y / u.viewport.y) * 2.);

  return StarVertex(
    vec4f(ndc, 0., 1.),
    c * radiusPx,
    vec4f(st.color.rgb, focus),
    vec4f(mag, radiusPx, st.extra.x, spiked),
  );
}

@fragment fn starFragment(input: StarVertex) -> @location(0) vec4f {
  let r = length(input.localPx);
  let maxR = input.params.y;
  if (r > maxR) {
    discard;
  }
  let mag = input.params.x;
  let focus = input.color.a;
  let phase = input.params.z;
  let spiked = input.params.w;
  let pxPerView = u.stageRect.z / u.viewMeta.x;

  let time = u.timing.x;
  let staticMode = u.counts.w > .5;
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
            + input.color.rgb * (corona * coronaEnergy);

  if (focus > .02) {
    color += vec3f(1.08, .82, .54) * exp(-pow(r / max(haloR * 2.2, 1.4), 2.)) * edgeWindow * (focus * .14);
  }

  let spikeVis = max(spiked * (.46 + focus * .44), focus * .74);
  if (spikeVis > .01) {
    let spikeLen = select(select(4.8, 7.2, mag < 2.5), 10.6, mag < 1.5) * pxPerView * (1. + focus * .28);
    let spikeW = max(.28 * u.viewport.z, .28 * pxPerView);
    let ax = abs(input.localPx.x);
    let ay = abs(input.localPx.y);
    let hSpike = exp(-pow(ay / spikeW, 2.)) * exp(-ax / max(spikeLen * .34, 1.)) * smoothstep(spikeLen, 0., ax);
    let vSpike = exp(-pow(ax / spikeW, 2.)) * exp(-ay / max(spikeLen * .34, 1.)) * smoothstep(spikeLen, 0., ay);
    let spikeTint = mix(vec3f(1., .97, .92), input.color.rgb, .35);
    color += spikeTint * ((hSpike + vSpike) * spikeVis * (.42 + focus * .44));
  }

  return vec4f(color * scint * u.viewport.w, 0.);
}
