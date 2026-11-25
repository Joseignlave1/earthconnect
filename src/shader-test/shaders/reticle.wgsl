fn worldTangent(uv: vec2f, point: vec2f, bodyMorph: f32, unitPerPixel: f32) -> vec2f {
  let angles = (uv - .5) * vec2f(6.283185307, 3.141592654);
  let origin = (point - .5) * vec2f(6.283185307, 3.141592654);
  let direction = vec3f(cos(angles.y) * sin(angles.x), sin(angles.y), cos(angles.y) * cos(angles.x));
  let anchor = vec3f(cos(origin.y) * sin(origin.x), sin(origin.y), cos(origin.y) * cos(origin.x));
  if (bodyMorph > .5 && dot(direction, anchor) < 0.) { return vec2f(1e5, 1e5); }
  let planeDelta = equalEarth(angles.x, angles.y) - equalEarth(origin.x, origin.y);
  let east = vec3f(cos(origin.x), 0., -sin(origin.x));
  let north = cross(anchor, east);
  let offset = direction - anchor;
  return mix(planeDelta, vec2f(dot(offset, east), dot(offset, north)), bodyMorph) / max(unitPerPixel, 1e-9);
}

fn applyPrecisionReticle(base: vec3f, bg: f32, t: vec2f, d: f32, aa: f32, opacity: f32, hdr: bool) -> vec3f {
  if (d > 16. || opacity <= .002) { return base; }
  let radius = 8.4;
  let ringDist = abs(d - radius);
  let ax = abs(t.x);
  let ay = abs(t.y);
  let isNorth = t.y > 0. && ax < ay;
  let tickEnd = select(11.3, 12.6, isNorth);
  let tickBand = smoothstep(radius - .4 - aa, radius - .4 + aa, d) * (1. - smoothstep(tickEnd - aa, tickEnd + aa, d));
  let tickCore = max(1. - smoothstep(.58 - aa, .58 + aa, ax), 1. - smoothstep(.58 - aa, .58 + aa, ay)) * tickBand;
  let tickCase = max(1. - smoothstep(1.45 - aa, 1.45 + aa, ax), 1. - smoothstep(1.45 - aa, 1.45 + aa, ay)) * tickBand;
  let ringCore = 1. - smoothstep(.66 - aa, .66 + aa, ringDist);
  let dotCore = 1. - smoothstep(1.35 - aa, 1.35 + aa, d);
  let core = max(max(ringCore, dotCore), tickCore);
  let casing = max(max(1. - smoothstep(1.65 - aa, 1.65 + aa, ringDist), 1. - smoothstep(2.3 - aa, 2.3 + aa, d)), tickCase);
  let inside = 1. - smoothstep(radius - 1. - aa, radius - 1. + aa, d);
  var col = mix(base, mix(base * 1.08 + select(.012, .003, hdr), base * .92, smoothstep(.25, .7, bg)), inside * .55);
  let caseAlpha = mix(.62, .92, smoothstep(.1, .65, bg));
  let darkCol = select(vec3f(.02, .028, .045), vec3f(.0016, .0022, .0036), hdr);
  let whiteCol = select(vec3f(.96, .98, 1.), vec3f(1.12, 1.16, 1.20), hdr);
  let northCol = select(vec3f(.55, .86, 1.), vec3f(.26, .72, 1.), hdr);
  col = mix(col, darkCol, casing * caseAlpha);
  let accent = mix(whiteCol, northCol, select(0., 1., isNorth) * tickCore * .85);
  let out = mix(col, accent, core * .96);
  return mix(base, out, opacity);
}

fn evalWorldMarkerSample(
  s: vec4f, coordinates: vec2f, selEarth: vec4f, selMoon: vec4f, selMars: vec4f,
  earthMorph: f32, exposure: f32, ratio: f32, hdr: bool,
) -> vec4f {
  if (coordinates.y == -1.) { return s; }
  let bodyId = i32(floor(coordinates.x * .5));
  let sel = select(select(selEarth, selMoon, bodyId == 1), selMars, bodyId == 2);
  if (sel.z <= .005) { return s; }
  let uv = vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y);
  let bodyMorph = select(1., earthMorph, bodyId == 0);
  let t = worldTangent(uv, sel.xy, bodyMorph, sel.w);
  let d = length(t);
  if (d > 16.) { return s; }
  let aa = clamp(.65 / max(ratio, 1.), .15, .75);
  let base = s.rgb / max(s.a, .0001);
  let bg = dot(select(base, toDisplay(acesToneMap(base * exposure)), hdr), vec3f(.2126, .7152, .0722));
  let col = applyPrecisionReticle(base, bg, t, d, aa, sel.z, hdr);
  return vec4f(col * s.a, s.a);
}
