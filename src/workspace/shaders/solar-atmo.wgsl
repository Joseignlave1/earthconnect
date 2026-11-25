struct AtmoOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment fn earthAtmoFragment(input: ScreenVertex) -> AtmoOut {
  let sample = computeEarthAtmo(input);
  if (!sample.valid) { discard; }
  return AtmoOut(vec4f(sample.radiance, 0.), sample.depth);
}
