@group(0) @binding(0) var orbitAccum: texture_multisampled_2d<f32>;

struct OrbitSampleOut {
  @location(0) color: vec4f,
  @builtin(sample_mask) mask: u32,
}

@fragment fn orbitSceneCompositeFragment(
  input: ScreenVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> OrbitSampleOut {
  // Composite accumulated per-sample orbits into the 4x MSAA HDR target so orbits behind
  // partially covered foreground limbs never darken the foreground and persist cleanly in scene.
  let res = textureLoad(orbitAccum, vec2i(input.position.xy), i32(sampleIdx));
  if (res.a <= .0005) { discard; }
  return OrbitSampleOut(res, 1u << sampleIdx);
}
