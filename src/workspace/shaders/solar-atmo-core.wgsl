struct EarthAtmo {
  sphere: vec4f,
  params: vec4f,
  pole: vec4f,
}

@group(0) @binding(1) var<uniform> atmo: EarthAtmo;

struct AtmoSample {
  radiance: vec3f,
  depth: f32,
  valid: bool,
}

fn rayFromMapped(mapped: vec2f) -> vec3f {
  let p = vec2f(mapped.x * view.right.w, mapped.y) * view.up.w;
  return normalize(vec3f(p, -1.));
}

fn evalAtmoB(mapped: vec2f, centerS: vec3f, pAxis: vec3f, stretch: f32) -> f32 {
  let r = rayFromMapped(mapped);
  let rS = normalize(r + pAxis * (dot(r, pAxis) * stretch));
  let tS = dot(rS, centerS);
  return length(rS * tS - centerS);
}

fn computeEarthAtmo(input: ScreenVertex) -> AtmoSample {
  if (atmo.params.x <= .001) { return AtmoSample(vec3f(0.), 0., false); }
  let isMars = atmo.params.y > .5;
  let isSmall = atmo.params.z > .5;
  // Evaluate pixel-center NDC and analytical radial step directly from framebuffer coordinates
  // so radial footprints and shell integrals have no hardware quad-derivative dependence.
  let pxCenter = floor(input.position.xy) + vec2f(.5);
  let ndc = vec2f(pxCenter.x * view.screen.x - 1., 1. - pxCenter.y * view.screen.y);
  let mappedNdc = view.crop.xy + ndc * view.crop.zw;
  let mappedStep = view.screen.xy * .5 * view.crop.zw;
  let ray = rayFromMapped(mappedNdc);
  let center = atmo.sphere.xyz;
  let radius = atmo.sphere.w;
  // Transform into ellipsoid-normalized space (using Earth's camera-space polar axis and
  // WGS84 flattening; identity for Mars) so bEllip == radius coincides exactly with the
  // solid horizon at every latitude and camera roll angle.
  let pAxis = atmo.pole.xyz;
  let stretch = atmo.pole.w;
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
  let db = max(length(vec2f(dbX, dbY)), max(t, 0.) * view.light.w * .5);
  let maxScale = select(1.058, 1.032, isMars);
  if (t <= view.back.w || bEllip - .5 * db >= radius * maxScale || bEllip + .5 * db <= radius) {
    return AtmoSample(vec3f(0.), 0., false);
  }
  let shellWidth = select(.046, .024, isMars);
  let invShell = 1. / (radius * shellWidth);
  let altSigned = (bEllip - radius) * invShell;
  let w = max(db * invShell, 1e-4);
  let a0 = altSigned - .5 * w;
  let a1 = altSigned + .5 * w;
  let altMax = 1.25;
  let lo = max(0., a0);
  let hi = min(altMax, a1);
  if (hi <= lo) { return AtmoSample(vec3f(0.), 0., false); }
  // Closed-form integral of (exp(-k*x) - exp(-k*altMax)) / (1 - exp(-k*altMax)) over [lo, hi].
  // Small planets have exact analytical ellipsoid/sphere horizons with zero displaced relief, so
  // testing depth at the front bounding plane lets all 4 MSAA samples pass on the rim (while still
  // occluding behind closer bodies like the Moon) and dividing by full pixel width w yields the
  // exact continuous integral at every sub-pixel offset without 4-sample visibility mismatch.
  let k = select(4.2, 4.6, isMars);
  let eEnd = exp(-k * altMax);
  let span = hi - lo;
  let expIntegral = select((exp(-k * lo) - exp(-k * hi)) / k, exp(-k * .5 * (lo + hi)) * span, k * span < 1e-4);
  let denom = select(max(a1 - lo, 1e-4), w, isSmall);
  let shell = max(0., (expIntegral - eEnd * span) / (denom * (1. - eEnd)));
  let limbNormal = offset / max(length(offset), 1.e-4);
  let sunDir = normalize(view.light.xyz - center);
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
  ) * shell * atmo.params.x;
  let atmoT = select(t, max(view.back.w * 1.01, t - radius * 1.01), isSmall);
  let depth = -ray.z * atmoT;
  if (depth <= view.back.w) { return AtmoSample(vec3f(0.), 0., false); }
  return AtmoSample(radiance, view.back.w / depth, true);
}
