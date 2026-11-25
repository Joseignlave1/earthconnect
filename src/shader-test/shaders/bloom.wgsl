struct BloomUniforms {
  step: vec4f, // x, y: texel step, z: downsample flag (> 0.5), w: reserved
}

@group(0) @binding(0) var<uniform> p: BloomUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;

fn bright(uv: vec2f) -> vec3f {
  let c = textureSampleLevel(source, linearSampler, uv, 0.).rgb;
  let peak = max(c.r, max(c.g, c.b));
  // C1 quadratic soft knee over [0.80 .. 1.20] avoids hard threshold popping during zoom.
  let soft = clamp(peak - .80, 0., .40);
  let contrib = max(peak - 1., soft * soft / .80);
  return c * contrib / max(peak, .0001);
}

@fragment fn bloomFragment(input: ScreenVertex) -> @location(0) vec4f {
  let uv = input.uv;
  var c = vec3f(0.);
  if (p.step.z > .5) {
    c = (bright(uv + p.step.xy * vec2f(-.5, -.5)) + bright(uv + p.step.xy * vec2f(.5, -.5))
      + bright(uv + p.step.xy * vec2f(-.5, .5)) + bright(uv + p.step.xy * vec2f(.5, .5))) * .25;
  } else {
    c = textureSampleLevel(source, linearSampler, uv, 0.).rgb * .227027;
    c += (textureSampleLevel(source, linearSampler, uv + p.step.xy * 1.384615, 0.).rgb + textureSampleLevel(source, linearSampler, uv - p.step.xy * 1.384615, 0.).rgb) * .316216;
    c += (textureSampleLevel(source, linearSampler, uv + p.step.xy * 3.230769, 0.).rgb + textureSampleLevel(source, linearSampler, uv - p.step.xy * 3.230769, 0.).rgb) * .070270;
  }
  return vec4f(c, 1.);
}
