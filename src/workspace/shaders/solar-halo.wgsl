struct HaloParams {
  sunNdc: vec2f,
  visibility: f32,
  aspect: f32,
}

@group(0) @binding(1) var<uniform> halo: HaloParams;

struct Halo {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment fn haloFragment(input: ScreenVertex) -> Halo {
  let ndc = vec2f(input.uv.x * 2. - 1., 1. - input.uv.y * 2.);
  let ray = viewRay(ndc);
  let sun = view.lightPos;
  let sunDist = length(sun);
  if (sunDist <= view.near) { discard; }
  let sunDir = sun / sunDist;
  let cosAngle = dot(ray, sunDir);
  let distance = cosAngle * sunDist;
  if (cosAngle <= 0. || distance <= view.near) { discard; }
  let sunAngRadius = max(695700. / sunDist, view.pixelAngularScale * 2.6);
  let r = length(ray - sunDir * cosAngle) / sunAngRadius;
  if (r >= 24.) { discard; }
  let sunVis = halo.visibility;
  // Pure ray-free multi-scale optical glare with micro R/G/B glass dispersion.
  let rVec = vec3f(r * .95, r, r * 1.055);
  let outerFade = smoothstep(24., 7., r);
  let coreGlow = 1.75 / (vec3f(1.) + rVec * rVec * 1.15);
  let midHalo = .20 / pow(vec3f(1.) + rVec * .55, vec3f(2.35));
  let wideVeil = .024 / (vec3f(1.) + rVec * rVec * .065);
  // Crisp diamond-ring intensity lift when a planet's limb grazes the solar disc.
  let eclipseGraze = 4. * sunVis * (1. - sunVis);
  let diamond = (1.1 / (1. + r * r * 1.8)) * eclipseGraze * .45;
  let color = (
    vec3f(1., .98, .93) * coreGlow +
    vec3f(1., .86, .66) * midHalo +
    vec3f(1., .91, .80) * wideVeil +
    vec3f(1., .96, .88) * diamond
  ) * outerFade;
  let depth = -ray.z * distance;
  if (depth <= view.near) { discard; }
  return Halo(vec4f(color, 0.), view.near / depth);
}

// Mechanical barrel vignetting: intersecting two offset aperture circles squeezes off-axis
// bokeh ghosts into realistic almonds perpendicular to the optical axis.
fn catsEyeGhost(p: vec2f, center: vec2f, axis: vec2f, ecc: f32, radius: f32, clip: f32, rim: f32) -> f32 {
  let v = p - center;
  let shift = axis * (radius * clamp(ecc * clip, 0., .62));
  let d = max(length(v - shift), length(v + shift)) / radius;
  if (d >= 1.12) { return 0.; }
  let mask = smoothstep(1.04, .76, d);
  let core = .38 + .62 * smoothstep(0., .90, d);
  let cT = (d - .90) / .11;
  let caustic = exp(-(cT * cT)) * smoothstep(1.06, .94, d);
  return mask * mix(core, caustic * 1.15 + core * .24, rim);
}

// Lateral R/G/B chromatic separation along the flare axis for multi-coated glass reflections.
fn chromaticGhost(p: vec2f, sun: vec2f, axis: vec2f, ecc: f32, pos: f32, radius: f32, clip: f32, rim: f32, disp: f32, tint: vec3f) -> vec3f {
  let r = catsEyeGhost(p, sun * (pos + disp), axis, ecc, radius * 1.03, clip, rim);
  let g = catsEyeGhost(p, sun * pos, axis, ecc, radius, clip, rim);
  let b = catsEyeGhost(p, sun * (pos - disp), axis, ecc, radius * .97, clip, rim);
  return vec3f(r, g, b) * tint;
}

// Foreground camera-lens pass (no depth test): reflections float in front of planets and
// vanish smoothly via sunVis whenever a planet eclipses the Sun.
@fragment fn lensFlareFragment(input: ScreenVertex) -> @location(0) vec4f {
  let sunVis = halo.visibility;
  let sunNdc = halo.sunNdc;
  let aspect = halo.aspect;
  let sunSensor = sunNdc * vec2f(aspect, 1.);
  let sunEcc = length(sunSensor);
  let onScreen = smoothstep(1.55, .82, max(abs(sunNdc.x), abs(sunNdc.y)));
  let offAxis = smoothstep(.06, .30, sunEcc);
  let weight = sunVis * onScreen * offAxis;
  if (weight <= .001) { discard; }
  let ndc = vec2f(input.uv.x * 2. - 1., 1. - input.uv.y * 2.);
  let p = ndc * vec2f(aspect, 1.);
  let axis = sunSensor / max(sunEcc, 1e-4);
  var flare = vec3f(0.);
  // Near-focus specular pinpoint, warm diffuse reflection, AR-coated caustic bubble, far ghost.
  flare += chromaticGhost(p, sunSensor, axis, sunEcc, -.22, .022, .15, .15, .014, vec3f(1., .92, .78)) * .022;
  flare += chromaticGhost(p, sunSensor, axis, sunEcc, -.44, .068, .46, .24, .022, vec3f(1., .78, .58)) * .011;
  flare += chromaticGhost(p, sunSensor, axis, sunEcc, -.82, .135, .54, .84, .032, vec3f(.52, .94, .88)) * .010;
  flare += chromaticGhost(p, sunSensor, axis, sunEcc, -1.28, .044, .36, .42, .028, vec3f(.66, .76, 1.)) * .014;
  // Whisper-faint curved chromatic diffraction arc from front-element off-axis grazing.
  let pRad = length(p);
  let pDir = p / max(pRad, 1e-4);
  let arcAlign = pow(max(0., dot(pDir, -axis)), 3.5);
  let arcR = vec3f(pRad * .985, pRad, pRad * 1.015);
  let arcT = (arcR - vec3f(.76)) / vec3f(.055);
  let arcRing = exp(-(arcT * arcT));
  flare += arcRing * arcAlign * smoothstep(.35, .95, sunEcc) * .009;
  flare *= weight;
  if (max(flare.r, max(flare.g, flare.b)) <= .0002) { discard; }
  return vec4f(flare, 0.);
}
