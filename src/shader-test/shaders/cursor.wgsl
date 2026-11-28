struct CursorUniforms {
  centerPx: vec2f,
  viewportPx: vec2f,
  pixelRatio: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<uniform> cursor: CursorUniforms;

struct CursorVertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
}

@vertex fn cursorVertex(@builtin(vertex_index) index: u32) -> CursorVertex {
  let local = unitQuadCorner(index) * 4.;
  let screen = (cursor.centerPx + local * cursor.pixelRatio) / cursor.viewportPx;
  return CursorVertex(vec4f(screen * vec2f(2., -2.) + vec2f(-1., 1.), 0., 1.), local);
}

@fragment fn cursorFragment(input: CursorVertex) -> @location(0) vec4f {
  let d = length(input.local);
  let mark = 1. - smoothstep(1.15, 1.95, d);
  let outline = 1. - smoothstep(2.1, 2.8, d);
  let alpha = max(mark, outline * .8);
  return vec4f(mix(vec3f(.025, .03, .04), vec3f(.96, .94, 1.), mark / max(.0001, alpha)), alpha);
}
