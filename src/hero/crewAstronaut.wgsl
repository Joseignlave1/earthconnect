// 3D stylized Content-Warning-inspired astronaut undergoing cartoony vortex spaghettification
// (smooth C-infinity tidal pull + spiraling drain twist + flowing noodle curve + amber photon rim)
// on pure #000000.

struct CrewUniform {
  viewProj: mat4x4f,
  pullDir: vec3f,
  timeSeconds: f32,
  lightDir: vec3f,
  transitionAlpha: f32,
  lightingEye: vec3f,
  reducedMotion: f32,
}

@group(0) @binding(0) var<uniform> u: CrewUniform;
@group(0) @binding(1) var noteTex: texture_2d<f32>;
@group(0) @binding(2) var noteSampler: sampler;

struct VertexIn {
  @location(0) pos: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) matId: u32, // 0 suit, 1 metal, 2 pack, 3 visor, 4 stickyNote, 5 tape, 6 prop
  @location(4) tetherMask: f32,
}

struct VertexOut {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) worldNormal: vec3f,
  @location(2) uv: vec2f,
  @location(3) @interpolate(flat) matId: u32,
  @location(4) sWarped: f32,
  @location(5) stretchAmt: f32,
}

struct VortexWarp {
  pos: vec3f,
  stretch: f32,
}

// Orthonormal basis around the singularity pull axis:
// axU lies in the XY plane perpendicular to pull (pointing right/up toward black-hole horizontal center),
// axV points roughly out toward the viewer (+Z).
fn pullBasis(pull: vec3f) -> mat3x3f {
  let axU = normalize(cross(vec3f(0., 0., 1.), pull));
  let axV = cross(pull, axU);
  return mat3x3f(axU, axV, pull);
}

// Smooth C-infinity cartoony vortex spaghettification field:
// Zero threshold / start line — head & visor sticky note stay intact & close to camera,
// while torso, waist, and legs continuously taper, braid/twist in a vortex spiral,
// and plunge away into -Z in a flowing cartoon noodle curve into the black hole throat.
fn warpVortex(pIn: vec3f, tetherMask: f32) -> VortexWarp {
  let time = u.timeSeconds;
  let reduced = u.reducedMotion > .5;
  let pull = normalize(u.pullDir);
  let basis = pullBasis(pull);
  let axU = basis[0];
  let axV = basis[1];

  var p = pIn;

  // Subtle calm zero-G drift (whole-body float, zero jitter).
  if (!reduced) {
    p += vec3f(sin(time * .48) * .020, cos(time * .38) * .026, sin(time * .32) * .012);
    if (tetherMask > .5) {
      let along = dot(p, pull);
      p += (axU * sin(time * 1.15 - along * 1.9) + axV * cos(time * .95 - along * 1.6)) * (.022 * tetherMask);
    }
  }

  let s = dot(p, pull);
  let uCoord = dot(p, axU);
  let vCoord = dot(p, axV);

  // Continuous exponential gravity-well intensity (C-infinity everywhere, no kink).
  let k = 1.32;
  let a = .42;
  let g = exp(k * clamp(s - .04, -2.5, 2.25));
  let sWarped = s + a * g;

  // Longitudinal stretch lambda = dsWarped/ds -> strong point-convergence funnel pinch.
  let lambda = 1. + a * k * g;
  let pinch = pow(lambda, -.94);

  // Vortex spiral twist anchored at head depth (g ≈ 0.23) so the sticky-note face stays square to camera
  // while waist, hips, and legs progressively wring & braid around each other down the drain!
  let twist = .78 * (g - .23);
  let ct = cos(twist);
  let st = sin(twist);
  let uSpin = (uCoord * ct - vCoord * st) * pinch;
  let vSpin = (uCoord * st + vCoord * ct) * pinch;

  // Flowing cartoony noodle S-curve traveling smoothly down into the singularity throat.
  let waveEnv = .17 * (g * g) / (2.3 + g * g);
  let flowPhase = sWarped * 2.05 - select(time * 1.35, 0., reduced);
  let noodle = axU * (sin(flowPhase) * waveEnv) + axV * (cos(flowPhase * .9) * waveEnv * .36);

  // Smooth overarching spacetime-funnel curve sweeping right AND plunging away from camera (-axV -> -Z)
  // to sell true 3D depth & absurd leg length receding into the black hole!
  let funnelCurve = axU * (.26 * a * g) - axV * (.44 * a * g);

  let warped = pull * sWarped + axU * uSpin + axV * vSpin + noodle + funnelCurve;
  return VortexWarp(warped, lambda - 1.);
}

@vertex fn vsMain(input: VertexIn) -> VertexOut {
  let pull = normalize(u.pullDir);
  let w0 = warpVortex(input.pos, input.tetherMask);
  let p = w0.pos;
  let stretchAmt = w0.stretch;

  // Finite-difference Jacobian to keep surface normals crisp under twisting + curved spaghettification.
  let nIn = normalize(input.normal);
  let upHint = select(vec3f(0., 1., 0.), vec3f(1., 0., 0.), abs(nIn.y) > .9);
  let t1 = normalize(cross(upHint, nIn));
  let t2 = cross(nIn, t1);
  let eps = .012;
  let dp1 = warpVortex(input.pos + t1 * eps, input.tetherMask).pos - p;
  let dp2 = warpVortex(input.pos + t2 * eps, input.tetherMask).pos - p;
  let nWarped = normalize(cross(dp1, dp2));

  let sWarped = dot(p, pull);

  var out: VertexOut;
  out.clipPos = u.viewProj * vec4f(p, 1.);
  out.worldPos = p;
  out.worldNormal = nWarped;
  out.uv = input.uv;
  out.matId = input.matId;
  out.sWarped = sWarped;
  out.stretchAmt = stretchAmt;
  return out;
}

@fragment fn fsMain(input: VertexOut) -> @location(0) vec4f {
  let matId = input.matId;
  // Per-pixel smooth fade into pure #000000 abyss well before the canvas bottom edge.
  let abyssFade = (1. - smoothstep(1.65, 3.25, input.sWarped)) * u.transitionAlpha;
  if (abyssFade <= .002) {
    discard;
  }

  let n = normalize(input.worldNormal);
  let v = normalize(u.lightingEye - input.worldPos);
  let keyL = normalize(u.lightDir);
  let pullL = normalize(u.pullDir);

  let ndlKey = max(0., dot(n, keyL));
  let ndlPull = max(0., dot(n, pullL));
  let ndv = max(0., dot(n, v));
  let fresnel = pow(1. - ndv, 2.8);

  var albedo = vec3f(.19, .21, .25);
  var specPower = 18.;
  var specAmt = .14;
  var emissive = vec3f(0.);
  var alpha = 1.;

  if (matId == 0u) {
    // Smooth matte charcoal Content-Warning-style EVA suit
    albedo = vec3f(.20, .21, .24);
    specPower = 12.;
    specAmt = .10;
  } else if (matId == 1u) {
    // Brushed warm-titanium helmet ring, cuffs, joints, tether
    albedo = vec3f(.52, .48, .42);
    specPower = 42.;
    specAmt = .38;
  } else if (matId == 2u) {
    // Industrial backpack & chunky boot soles
    albedo = vec3f(.26, .27, .30);
    specPower = 16.;
    specAmt = .16;
  } else if (matId == 3u) {
    // Deep reflective gold-tinted astronaut helmet visor
    albedo = vec3f(.07, .06, .04);
    specPower = 90.;
    specAmt = .85;
    emissive = vec3f(.24, .15, .05) * fresnel;
  } else if (matId == 4u) {
    // Sticky note stuck directly onto the visor (photo or Content Warning panic doodle!)
    let noteSample = textureSampleLevel(noteTex, noteSampler, input.uv, 0.);
    albedo = noteSample.rgb;
    specPower = 10.;
    specAmt = .05;
    // Crisp Post-it readability on pitch-black background while still taking subtle scene shading
    emissive = noteSample.rgb * .62;
  } else if (matId == 5u) {
    // Translucent strip of matte Scotch tape holding the sticky note to the visor
    albedo = vec3f(.86, .84, .76);
    specPower = 24.;
    specAmt = .18;
    emissive = vec3f(.16, .15, .13);
    alpha = .58;
  } else {
    // Floating mini-laptop / prop glow
    albedo = vec3f(.58, .62, .68);
    specPower = 48.;
    specAmt = .42;
    emissive = vec3f(.18, .32, .52) * smoothstep(.2, .8, input.uv.y);
  }

  // Lighting: cool crisp moonlight key on upper body transitioning into intense warm Interstellar
  // amber photon-ring rim light + subtle gravitational redshift down the stretched lower legs!
  let depthT = smoothstep(-.35, 2.35, input.sWarped);
  let halfKey = normalize(keyL + v);
  let specKey = pow(max(0., dot(n, halfKey)), specPower) * specAmt * mix(1., .42, depthT);
  let halfPull = normalize(pullL + v);
  let specPull = pow(max(0., dot(n, halfPull)), specPower * .7) * specAmt * (.55 + depthT * .95);

  let coolKey = vec3f(.84, .90, 1.02) * (.16 + ndlKey * .90) * mix(1., .36, depthT);
  let rimCurve = pow(1. - ndv, 2.2);
  let photonIntensity = mix(.18, 1.45, depthT);
  let warmPhoton = vec3f(1.28, .66, .24) * (ndlPull * .58 + rimCurve * (.42 + ndlPull * .72)) * photonIntensity;
  let ambient = vec3f(.08, .09, .12) * mix(1., .55, depthT);

  var color = albedo * (ambient + coolKey + warmPhoton * .58)
            + vec3f(.95, .97, 1.) * specKey
            + vec3f(1.15, .74, .36) * specPull
            + warmPhoton * (.22 * rimCurve)
            + emissive;

  // Gravitational redshift bathing the deepest stretched legs in warm amber-crimson before vanishing into #000000.
  if (matId != 4u && matId != 5u) {
    let redshift = smoothstep(.75, 2.45, input.sWarped);
    let stretchRim = smoothstep(.55, 2.3, input.stretchAmt) * (.18 + rimCurve * .62);
    color += vec3f(1.15, .52, .16) * stretchRim * .38;
    color = mix(color, color * vec3f(1.24, .64, .32), redshift * .55);
  }

  // Soft ACES-like filmic roll-off + abyss fade into #000000
  color = color / (color + vec3f(.62));
  color = pow(max(color, vec3f(0.)), vec3f(1. / 1.15));

  // Fade RGB smoothly into pure #000000 while keeping opaque suit/boots depth-solid
  // (avoids seeing interior back-faces of stretched boots through partial alpha).
  let outColor = color * (alpha * abyssFade);
  let outAlpha = alpha * smoothstep(0., .06, abyssFade);
  return vec4f(outColor, outAlpha);
}
