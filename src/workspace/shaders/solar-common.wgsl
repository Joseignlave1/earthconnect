struct View {
  right: vec3f,
  aspect: f32,
  up: vec3f,
  tanHalfFovY: f32,
  back: vec3f,
  near: f32,
  lightPos: vec3f,
  pixelAngularScale: f32,
  cropCenter: vec2f,
  cropScale: vec2f,
  invScreenSize: vec2f,
  _padScreen0: f32,
  _padScreen1: f32,
}

@group(0) @binding(0) var<uniform> view: View;

fn pixelToNdc(pixelPos: vec2f) -> vec2f {
  return vec2f(pixelPos.x * view.invScreenSize.x - 1., 1. - pixelPos.y * view.invScreenSize.y);
}

fn mapCrop(ndc: vec2f) -> vec2f {
  return view.cropCenter + ndc * view.cropScale;
}

fn unnormalizedRay(mapped: vec2f) -> vec3f {
  return vec3f(vec2f(mapped.x * view.aspect, mapped.y) * view.tanHalfFovY, -1.);
}

fn rayFromMapped(mapped: vec2f) -> vec3f {
  return normalize(unnormalizedRay(mapped));
}

fn viewRay(point: vec2f) -> vec3f {
  return rayFromMapped(mapCrop(point));
}
