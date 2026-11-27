// Pre-bloom HDR Schwarzschild gravitational lensing & Interstellar-esque event horizon shader.
// Runs on the resolved rgba16float HDR scene before bloom so stretched starlight and the
// razor-thin critical photon ring naturally acquire true HDR optical bloom.

struct HoleGeometry {
  center: vec2f, // viewport UV (0..1)
  rh: f32,       // event horizon radius in viewport-height units
  aspect: f32,   // viewport aspect ratio (width / height)
}

struct LensParams {
  timeSeconds: f32,
  scrollProgress: f32,
  lensStrength: f32,
  ringIntensity: f32,
}

struct BlackHoleUniform {
  hole: HoleGeometry,
  params: LensParams,
  viewportHeightPx: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<uniform> bh: BlackHoleUniform;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;

fn fbm(p: vec2f) -> f32 {
  var v = 0.;
  var amp = .52;
  var q = p;
  for (var i = 0; i < 4; i++) {
    v += amp * valueNoise(q);
    q = mat2x2f(.8, -.6, .6, .8) * q * 2.03 + vec2f(11.7, 6.4);
    amp *= .5;
  }
  return v;
}

fn applyBoundedOffset(u: f32, delta: f32) -> f32 {
  // Exact identity when delta == 0 (zero step across the gravity boundary!),
  // and asymptotically compresses large deflections inside [0.001, 0.999] with zero hard clamp.
  if (delta > 0.) {
    let room = max(.999 - u, 0.);
    return u + room * (delta / (delta + max(room, .06)));
  }
  if (delta < 0.) {
    let room = max(u - .001, 0.);
    let ad = -delta;
    return u - room * (ad / (ad + max(room, .06)));
  }
  return u;
}

fn sampleLensedScene(uv: vec2f, offset: vec2f) -> vec3f {
  let boundedUv = vec2f(applyBoundedOffset(uv.x, offset.x), applyBoundedOffset(uv.y, offset.y));
  return textureSampleLevel(sceneTex, linearSampler, boundedUv, 0.).rgb;
}

@fragment fn lensFragment(input: ScreenVertex) -> @location(0) vec4f {
  let uv = input.uv;
  let cx = bh.hole.center.x;
  let cy = bh.hole.center.y;
  let rh = bh.hole.rh;
  let aspect = bh.hole.aspect;
  let time = bh.params.timeSeconds;
  let scroll = bh.params.scrollProgress;
  let lensStrength = bh.params.lensStrength;
  let ringIntensity = bh.params.ringIntensity;

  // Aspect-corrected offset from black hole center (units of viewport height).
  let delta = vec2f((uv.x - cx) * aspect, uv.y - cy);
  let r = length(delta);
  let dir = delta / max(r, 1e-5);
  let dr = r - rh;

  // Pure pitch-black event horizon shadow void (#000000) with crisp sub-pixel anti-aliasing.
  let aa = max(1.5 / max(bh.viewportHeightPx, 1.), .0016);
  let outsideMask = smoothstep(-aa, aa * 1.35, dr);
  let voidColor = vec3f(0., 0., 0.);

  if (outsideMask <= .0005) {
    return vec4f(voidColor, 1.);
  }

  // Smooth gravitational influence zone outside the event horizon.
  // Tighter at scroll=0 so bottom-right Explore stays level while bottom-center Mars & Milky Way visibly bend.
  let reach = mix(.33, .54, smoothstep(0., .35, scroll));
  let prox = smoothstep(reach, 0., dr);

  // Fast path outside the gravity well: zero extra texture samples, pixel-exact passthrough.
  if (prox <= .0001) {
    return textureSampleLevel(sceneTex, linearSampler, uv, 0.);
  }

  // Schwarzschild / Einstein-ring radial deflection: background light near the horizon
  // bends outward around the shadow, forming curved Einstein arcs along the unseen limb.
  let soften = .052;
  let baseCurve = (1. / (max(dr, 0.) + soften)) - (1. / (reach + soften));
  let deflect = baseCurve * .027 * prox * lensStrength;

  // Subtle tangential frame-dragging shear right next to the photon sphere (Kerr spin cue).
  let tang = vec2f(-dir.y, dir.x);
  let frameDrag = exp(-max(dr, 0.) / .075) * .014 * lensStrength * prox;

  let toUv = vec2f(1. / max(aspect, .001), 1.);
  // Subtle gravitational chromatic dispersion right near the critical curve.
  let chroma = .032 * exp(-max(dr, 0.) / .12) * prox;
  let offR = -(dir * (deflect * (1. + chroma)) + tang * frameDrag) * toUv;
  let offG = -(dir * deflect + tang * frameDrag) * toUv;
  let offB = -(dir * (deflect * (1. - chroma)) + tang * frameDrag) * toUv;

  var color = vec3f(
    sampleLensedScene(uv, offR).r,
    sampleLensedScene(uv, offG).g,
    sampleLensedScene(uv, offB).b,
  );

  // Subtle gravitational lensing magnification right outside the critical photon sphere
  // (Einstein ring brightening of background stars & Milky Way dust!).
  let einsteinBoost = exp(-max(dr, 0.) / .068) * .42 * lensStrength * prox;
  color *= 1. + einsteinBoost;

  // Subtle gravitational dimming right before the absolute event horizon cliff (photon capture).
  let captureShadow = smoothstep(0., .014, dr);
  color *= mix(.12, 1., captureShadow);

  // ---------- Interstellar-esque Border: Razor-thin HDR Photon Ring & Accretion Rim ----------
  let angle = atan2(dir.y, dir.x);
  // Relativistic Doppler beaming: slightly brighter warm-gold along the upper-left arc (facing Mars),
  // softer champagne-ivory along the opposite limb.
  let doppler = .80 + .28 * cos(angle + 2.25);

  // Polar relativistic accretion wisps swirling slowly around the horizon rim.
  let logR = log(max(dr, .0006) + .010);
  let swirlNoise = fbm(vec2f(angle * 4.8 + time * .14, logR * 4.4 - time * .06));
  let fineStreak = fbm(vec2f(angle * 14.0 + time * .24, logR * 8.5));
  let wispMask = (.42 + .58 * swirlNoise) * (.62 + .38 * fineStreak);

  // 1. Razor-sharp Interstellar critical photon ring (fine incandescent hairline feeding HDR bloom).
  let photonCore = exp(-pow(max(dr - .0018, 0.) / .0036, 1.5)) * smoothstep(-.0015, .0018, dr);
  // 2. Inner lensed accretion halo (delicate swirling golden-amber relativistic filaments hugging the border).
  let innerCorona = exp(-max(dr, 0.) / .022) * wispMask;
  // 3. Subtle extended gravitational sheen (whispers along bottom center at scroll=0 as a diegetic scroll cue).
  let outerGlow = exp(-max(dr, 0.) / .082) * (.38 + .62 * swirlNoise);

  let ringColorCore = vec3f(1.65, 1.38, 1.00);
  let ringColorAmber = vec3f(1.24, .66, .26);
  let ringColorMist = vec3f(.86, .56, .34);

  let ringEmission =
    (ringColorCore * (photonCore * 1.95) +
      ringColorAmber * (innerCorona * .44) +
      ringColorMist * (outerGlow * .11)) *
    doppler *
    ringIntensity *
    prox;

  color += ringEmission;

  // Smoothly transition into the pure pitch-black event horizon interior.
  let finalRgb = mix(voidColor, color, outsideMask);
  return vec4f(finalRgb, 1.);
}
