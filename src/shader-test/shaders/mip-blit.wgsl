@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

@fragment fn mipFragment(input: ScreenVertex) -> @location(0) vec4f {
  return textureSampleLevel(source, linearSampler, input.uv, 0.);
}
