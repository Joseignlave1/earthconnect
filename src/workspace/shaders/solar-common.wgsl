struct View {
  right: vec4f,
  up: vec4f,
  back: vec4f,
  light: vec4f,
  crop: vec4f,
  screen: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;

fn viewRay(point: vec2f) -> vec3f {
  let mapped = view.crop.xy + point * view.crop.zw;
  let p = vec2f(mapped.x * view.right.w, mapped.y) * view.up.w;
  return normalize(vec3f(p, -1.));
}
