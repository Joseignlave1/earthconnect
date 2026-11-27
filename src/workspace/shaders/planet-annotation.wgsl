struct AnnotationView {
  camera: mat4x4f,
  bodyUOffset: f32, // 2 for moon, 4 for mars
  reliefFade: f32,
  radiusKm: f32,
  relief: f32,
}

@group(0) @binding(0) var<uniform> view: AnnotationView;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(perspective, sample) local: vec3f,
}

@vertex fn vertexMain(input: CubePatchIn) -> Vertex {
  let pos = displacedCubePatchPos(input, heights, gridSampler, view.radiusKm, view.relief, view.reliefFade);
  return Vertex(view.camera * vec4f(pos, 1.), pos);
}

fn evalPlanetAnnotation(input: Vertex) -> vec4f {
  let p = normalize(input.local);
  let lon = atan2(p.x, p.z);
  let lat = atan2(p.y, length(p.xz));
  let u = lon / (2. * PLANET_PI) + .5 + view.bodyUOffset;
  let v = lat / PLANET_PI + .5;
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
