struct ScreenVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex fn screenVertex(@builtin(vertex_index) index: u32) -> ScreenVertex {
  let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  // Texture coordinates run downwards, while clip-space Y points upwards.
  return ScreenVertex(vec4f(p * 2. - 1., 0., 1.), vec2f(p.x, 1. - p.y));
}
