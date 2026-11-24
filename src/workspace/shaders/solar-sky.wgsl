@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlasSampler: sampler;
const PI = 3.141592653589793;

fn mapGradient(equatorial: vec3f, directionDelta: vec3f) -> vec2f {
  let radiusSquared = max(1.e-8, dot(equatorial.xy, equatorial.xy));
  let raDelta = (equatorial.x * directionDelta.y - equatorial.y * directionDelta.x) / radiusSquared;
  let decDelta = directionDelta.z / sqrt(radiusSquared);
  return -vec2f(raDelta / (2. * PI), decDelta / PI);
}

fn toEquatorial(ray: vec3f) -> vec3f {
  let ecliptic = view.right.xyz * ray.x + view.up.xyz * ray.y + view.back.xyz * ray.z;
  // Rotate the world direction to the J2000 mean equator by the obliquity.
  return vec3f(
    ecliptic.x,
    .917482143 * ecliptic.y - .397776969 * ecliptic.z,
    .397776969 * ecliptic.y + .917482143 * ecliptic.z,
  );
}

@fragment fn skyFragment(input: ScreenVertex) -> @location(0) vec4f {
  let destPos = floor(input.position.xy) + vec2f(.5);
  let ndc = vec2f(destPos.x * view.screen.x - 1., 1. - destPos.y * view.screen.y);
  let mapped = view.crop.xy + ndc * view.crop.zw;
  let tanX = view.right.w * view.up.w;
  let tanY = view.up.w;
  let v = vec3f(mapped.x * tanX, mapped.y * tanY, -1.);
  let invLen = inverseSqrt(dot(v, v));
  let ray = v * invLen;
  let equatorial = toEquatorial(ray);
  let ra = atan2(equatorial.y, equatorial.x);
  let dec = atan2(equatorial.z, length(equatorial.xy));
  // The NASA map has 0h in the middle and increasing RA to the left.
  let uv = vec2f(.5 - ra / (2. * PI), .5 - dec / PI);
  // Analytical per-pixel direction derivatives avoid quad-derivative artifacts across the RA seam.
  let stepX = view.screen.x * view.crop.z;
  let stepY = view.screen.y * view.crop.w;
  let dvX = vec3f(tanX * stepX, 0., 0.);
  let dvY = vec3f(0., -tanY * stepY, 0.);
  let dRayX = (dvX - ray * dot(dvX, ray)) * invLen;
  let dRayY = (dvY - ray * dot(dvY, ray)) * invLen;
  let dx = mapGradient(equatorial, toEquatorial(dRayX));
  let dy = mapGradient(equatorial, toEquatorial(dRayY));
  let raw = textureSampleGrad(atlas, atlasSampler, uv, dx, dy).rgb;
  let luma = dot(raw, vec3f(.2126, .7152, .0722));
  // Gentle Milky Way dust richness on the 8K EXR-derived sky map + restrained bright-star lift.
  let clean = smoothstep(.0006, .0045, luma);
  let nebulaBand = smoothstep(.003, .040, luma) * (1. - smoothstep(.06, .24, luma));
  var sky = raw * mix(.70, 1., clean) + (raw - vec3f(luma)) * (.22 * nebulaBand);
  let starBoost = pow(max(0., luma - .22), 2.0) * 1.6;
  let spectral = mix(vec3f(1.), raw / max(luma, 1e-4), 1.1);
  sky += spectral * starBoost;
  // Subtle exposure adaptation: stars dim gently right inside the Sun's bright corona.
  let sunGlare = smoothstep(.92, .998, dot(ray, normalize(view.light.xyz)));
  sky *= 1. - .55 * sunGlare;
  return vec4f(max(sky, vec3f(0.)), 1.);
}
