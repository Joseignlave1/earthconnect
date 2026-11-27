struct Body {
  position: vec3f,
  radius: f32,
  color: vec3f,
  isSun: f32,
  visible: f32,
  minDiskRadiusPx: f32,
  glintWeight: f32,
  _pad: f32,
}

@group(0) @binding(1) var<storage, read> bodies: array<Body>;

struct Quad {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) body: u32,
}

@vertex fn bodyVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Quad {
  let corners = array<vec2f, 6>(vec2f(-1., -1.), vec2f(1., -1.), vec2f(-1., 1.), vec2f(-1., 1.), vec2f(1., -1.), vec2f(1., 1.));
  let body = bodies[instance];
  let position = body.position;
  let radius = body.radius;
  let depth = -position.z;
  let tanY = view.tanHalfFovY;
  let aspect = view.aspect;
  let tanX = tanY * aspect;
  if (depth + radius <= 0. || body.visible < .5) { return Quad(vec4f(2., 2., 0., 1.), instance); }
  // The fragment disk test is dist <= max(radius, middle * k), which is the exact union of
  // the physical sphere and an angular cone tan(theta) <= k (equivalent to a sphere of radius
  // |position| * sin(theta)). Project that exact effective sphere conservatively for all aspects.
  let k = view.pixelAngularScale * body.minDiskRadiusPx;
  let rEff = max(radius, length(position) * k * inverseSqrt(1. + k * k));
  if (depth <= rEff * 1.001) {
    return Quad(vec4f(corners[vertex] * 2., 0., 1.), instance);
  }
  let r2 = rEff * rEff;
  let denom = depth * depth - r2;
  let d2x = position.x * position.x + depth * depth;
  let d2y = position.y * position.y + depth * depth;
  let centerNdc = vec2f((position.x * depth) / (denom * tanX), (position.y * depth) / (denom * tanY));
  let halfNdc = vec2f(rEff * sqrt(max(0., d2x - r2)) / (denom * tanX), rEff * sqrt(max(0., d2y - r2)) / (denom * tanY));
  let pxNdc = select(vec2f(view.pixelAngularScale / tanX, view.pixelAngularScale / tanY), view.invScreenSize, view.invScreenSize.x > 0.);
  let pad = 3. * pxNdc;
  let corner = centerNdc + corners[vertex] * (halfNdc + pad) - view.cropCenter;
  return Quad(vec4f(corner, 0., 1.), instance);
}

struct BodyRay {
  dir: vec3f,
  offset: vec3f,
  dist: f32,
  worldRadius: f32,
  gap: f32,
  depth: f32,
  mask: u32,
}

fn evalBodyRay(input: Quad, sampleIdx: u32) -> BodyRay {
  let body = bodies[input.body];
  let position = body.position;
  let radius = body.radius;
  let ndc = pixelToNdc(input.position.xy);
  let dir = rayFromMapped(ndc);
  let middle = dot(position, dir);
  let offset = position - dir * middle;
  let dist = length(offset);
  let worldRadius = middle * view.pixelAngularScale * body.minDiskRadiusPx;
  let effRadius = max(radius, worldRadius);
  if (middle <= 0. || dist > effRadius) { discard; }
  let gap = radius * radius - dot(offset, offset);
  let intersection = middle - sqrt(max(gap, 0.));
  let depth = -dir.z * intersection;
  if (depth <= view.near) { discard; }
  return BodyRay(dir, offset, dist, worldRadius, gap, view.near / depth, 1u << sampleIdx);
}

struct Pixel {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn bodyFragment(input: Quad, @builtin(sample_index) sampleIdx: u32) -> Pixel {
  let body = bodies[input.body];
  let position = body.position;
  let radius = body.radius;
  let hit = evalBodyRay(input, sampleIdx);
  let limbNormal = normalize(-hit.offset - hit.dir * sqrt(max(hit.gap, 0.)));
  let physicalBlend = select(smoothstep(.45, 1., radius / max(hit.worldRadius, 1e-6)), 1., radius >= hit.worldRadius);
  let physical = radius >= hit.worldRadius || hit.gap >= 0.;
  let normal = select(-hit.dir, limbNormal, physical);
  let sunDir = normalize(view.lightPos - position);
  let sunDot = dot(normal, sunDir);
  let daylight = smoothstep(-.035, .045, sunDot);
  let diffuse = daylight * max(0., sunDot);
  let glint = 1. - smoothstep(.16, 1., hit.dist / max(hit.worldRadius, .001));
  let edge = mix(mix(1., glint, body.glintWeight), 1., physicalBlend);
  let litFactor = mix(.60 + .30 * diffuse, .04 + .96 * diffuse, physicalBlend);
  let surface = body.color * (edge * litFactor + .84 * glint * body.glintWeight);
  let sunMu = max(0., -dot(normal, hit.dir));
  let sunLimb = mix(vec3f(1., .74, .44), vec3f(1.), pow(sunMu, .45));
  let sun = body.color * sunLimb * (.45 + .55 * pow(sunMu, .6));
  let radiance = select(surface, sun, body.isSun > .5);
  return Pixel(vec4f(radiance, 1.), hit.depth, hit.mask);
}

struct BodyAnnotation {
  @location(0) geo: vec4f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn bodyAnnotationFragment(input: Quad, @builtin(sample_index) sampleIdx: u32) -> BodyAnnotation {
  let hit = evalBodyRay(input, sampleIdx);
  return BodyAnnotation(vec4f(0., 0., -1., 0.), hit.depth, hit.mask);
}

struct BodyAnnotationCoverage {
  @location(0) geo: vec4f,
  @location(1) coverage: vec2f,
  @builtin(frag_depth) depth: f32,
  @builtin(sample_mask) mask: u32,
}

@fragment fn bodyAnnotationCoverageFragment(input: Quad, @builtin(sample_index) sampleIdx: u32) -> BodyAnnotationCoverage {
  let hit = evalBodyRay(input, sampleIdx);
  return BodyAnnotationCoverage(vec4f(0., 0., -1., 0.), vec2f(0., 0.), hit.depth, hit.mask);
}
