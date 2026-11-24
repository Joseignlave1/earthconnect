struct AnnotationView {
  camera: mat4x4f,
  info: vec4f, // x: body u-offset (2 for moon, 4 for mars), y: reliefFade, z: reference radius (km), w: reserved
}

@group(0) @binding(0) var<uniform> view: AnnotationView;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(perspective, sample) local: vec3f,
}

@vertex fn vertexMain(@location(0) position: vec3f) -> Vertex {
  let refPos = normalize(position) * view.info.z;
  let pos = mix(refPos, position, view.info.y);
  return Vertex(view.camera * vec4f(pos, 1.), pos);
}

fn packSplit16(x: f32) -> vec2f {
  let hi = round(x * 256.) / 256.;
  return vec2f(hi, x - hi);
}

fn evalPlanetAnnotation(input: Vertex) -> vec4f {
  let p = normalize(input.local);
  let lon = atan2(p.x, p.z);
  let lat = atan2(p.y, length(p.xz));
  let pi = 3.141592653589793;
  let u = lon / (2. * pi) + .5 + view.info.x;
  let v = lat / pi + .5;
  return vec4f(packSplit16(u), packSplit16(v));
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  return evalPlanetAnnotation(input);
}

struct CoverageAnnotationOut {
  @location(0) geo: vec4f,
  @location(1) coverage: vec2f,
}

@fragment fn fragmentCoverageMain(input: Vertex) -> CoverageAnnotationOut {
  return CoverageAnnotationOut(evalPlanetAnnotation(input), vec2f(0., 0.));
}
