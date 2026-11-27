fn unitQuadCorner(index: u32) -> vec2f {
  let corners = array<vec2f, 6>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(-1., 1.),
    vec2f(1., -1.),
    vec2f(1., 1.),
  );
  return corners[index];
}
