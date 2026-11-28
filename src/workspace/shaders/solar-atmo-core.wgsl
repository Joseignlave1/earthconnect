struct PlanetAtmo {
  center: vec3f,
  radius: f32,
  strength: f32,
  isMars: f32,
  isSmall: f32,
  _padParam: f32,
  poleAxis: vec3f,
  poleStretch: f32,
}

@group(0) @binding(1) var<uniform> atmo: PlanetAtmo;

struct AtmoSample {
  radiance: vec3f,
  depth: f32,
  valid: bool,
}

fn evalAtmoB(mapped: vec2f, centerS: vec3f, pAxis: vec3f, stretch: f32) -> f32 {
  let r = rayFromMapped(mapped);
  let rS = normalize(r + pAxis * (dot(r, pAxis) * stretch));
  let tS = dot(rS, centerS);
  return length(rS * tS - centerS);
}

fn computeEarthAtmo(input: ScreenVertex) -> AtmoSample {
  if (atmo.strength <= .001) { return AtmoSample(vec3f(0.), 0., false); }
  let isMars = atmo.isMars > .5;
  let isSmall = atmo.isSmall > .5;
  // Evaluate pixel-center NDC and analytical radial step directly from framebuffer coordinates
  // so radial footprints and shell integrals have no hardware quad-derivative dependence.
  let pxCenter = floor(input.position.xy) + vec2f(.5);
  let ndc = pixelToNdc(pxCenter);
  let mappedNdc = mapCrop(ndc);
  let mappedStep = view.invScreenSize * .5 * view.cropScale;
  let ray = rayFromMapped(mappedNdc);
  let center = atmo.center;
  let radius = atmo.radius;
  // Transform into ellipsoid-normalized space (using Earth's camera-space polar axis and
  // WGS84 flattening; identity for Mars) so bEllip == radius coincides exactly with the
  // solid horizon at every latitude and camera roll angle.
  let pAxis = atmo.poleAxis;
  let stretch = atmo.poleStretch;
  let rayS = normalize(ray + pAxis * (dot(ray, pAxis) * stretch));
  let centerS = center + pAxis * (dot(center, pAxis) * stretch);
  let t = dot(ray, center);
  let tS = dot(rayS, centerS);
  let offset = ray * t - center;
  let offsetS = rayS * tS - centerS;
  let bEllip = length(offsetS);
  let dbX = evalAtmoB(mappedNdc + vec2f(mappedStep.x, 0.), centerS, pAxis, stretch)
    - evalAtmoB(mappedNdc - vec2f(mappedStep.x, 0.), centerS, pAxis, stretch);
  let dbY = evalAtmoB(mappedNdc + vec2f(0., mappedStep.y), centerS, pAxis, stretch)
    - evalAtmoB(mappedNdc - vec2f(0., mappedStep.y), centerS, pAxis, stretch);
  let db = max(length(vec2f(dbX, dbY)), max(t, 0.) * view.pixelAngularScale * .5);
  let maxScale = select(1.058, 1.032, isMars);
  // Displaced 3D terrain can dip below the reference radius (e.g. Mars's northern lowlands
  // and Hellas Basin sit up to ~25 km below MOLA datum with 3x relief). Extend the inner
  // plateau past the deepest depression when not using the smooth analytic small-planet sphere
  // so hardware/4-sample depth testing clips the atmosphere seamlessly against the real mesh
  // silhouette instead of leaving a dark gap between depressed terrain and `radius`.
  let minScale = select(select(.998, .988, isMars), 1., isSmall);
  if (t <= view.near || bEllip - .5 * db >= radius * maxScale || bEllip + .5 * db <= radius * minScale) {
    return AtmoSample(vec3f(0.), 0., false);
  }
  let shellWidth = select(.046, .024, isMars);
  let invShell = 1. / (radius * shellWidth);
  let altSigned = (bEllip - radius) * invShell;
  let w = max(db * invShell, 1e-4);
  let a0 = altSigned - .5 * w;
  let a1 = altSigned + .5 * w;
  let altMin = (minScale - 1.) / shellWidth;
  let altMax = 1.25;
  let flatLo = max(altMin, a0);
  let flatHi = min(0., a1);
  let flatSpan = max(0., flatHi - flatLo);
  let expLo = max(0., a0);
  let expHi = min(altMax, a1);
  let expSpan = max(0., expHi - expLo);
  if (flatSpan + expSpan <= 0.) { return AtmoSample(vec3f(0.), 0., false); }
  // Closed-form integral of (exp(-k*x) - exp(-k*altMax)) / (1 - exp(-k*altMax)) over [expLo, expHi],
  // joined continuously with a unit plateau over [flatLo, flatHi] below the reference datum.
  // Small planets have exact analytical ellipsoid/sphere horizons with zero displaced relief (altMin = 0),
  // so testing depth at the front bounding plane lets all 4 MSAA samples pass on the rim (while still
  // occluding behind closer bodies like the Moon) and dividing by full pixel width w yields the
  // exact continuous integral at every sub-pixel offset without 4-sample visibility mismatch.
  let k = select(4.2, 4.6, isMars);
  let eEnd = exp(-k * altMax);
  let expIntegral = select(
    (exp(-k * expLo) - exp(-k * expHi)) / k,
    exp(-k * .5 * (expLo + expHi)) * expSpan,
    k * expSpan < 1e-4,
  );
  let expContrib = select(0., max(0., (expIntegral - eEnd * expSpan) / (1. - eEnd)), expSpan > 0.);
  let denom = select(max(a1 - max(altMin, a0), 1e-4), w, isSmall);
  let shell = (flatSpan + expContrib) / denom;
  let limbNormal = offset / max(length(offset), 1.e-4);
  let sunDir = normalize(view.lightPos - center);
  let sunDot = dot(limbNormal, sunDir);
  let day = smoothstep(-.14, .26, sunDot);
  let backScatter = pow(max(0., dot(ray, sunDir)), 5.);
  let tw = (sunDot + .01) / .13;
  let twilight = exp(-(tw * tw)) * (1. + 2.2 * backScatter);
  let radiance = select(
    vec3f(.18, .50, 1.0) * .68 * day +
    vec3f(1., .42, .12) * .12 * twilight +
    vec3f(.04, .10, .20) * .20 * (1. - day),
    vec3f(.92, .56, .34) * .34 * day +
    vec3f(.38, .56, .80) * .08 * twilight +
    vec3f(.10, .06, .04) * .06 * (1. - day),
    isMars,
  ) * shell * atmo.strength;
  // For 3D displaced meshes, place the atmosphere depth plane slightly past the planet center plane
  // so steep front-facing limb relief never fails depth test against the atmosphere while still
  // occluding properly behind closer bodies and staying strictly in front of the infinite sky.
  let atmoT = select(t + radius * .05, max(view.near * 1.01, t - radius * 1.01), isSmall);
  let depth = -ray.z * atmoT;
  if (depth <= view.near) { return AtmoSample(vec3f(0.), 0., false); }
  return AtmoSample(radiance, view.near / depth, true);
}
