@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
@group(0) @binding(3) var bloom: texture_2d<f32>;
@group(0) @binding(4) var coverageSource: texture_2d<f32>;

@fragment fn compositeFragment(input: ScreenVertex) -> @location(0) vec4f {
  let outPx = vec2i(input.position.xy);
  let srcSize = vec2i(textureDimensions(source));
  let outSize = vec2i(round(p.viewport.zw));
  var scene: vec4f;
  if (srcSize.x == outSize.x && srcSize.y == outSize.y) {
    scene = textureLoad(source, clamp(outPx, vec2i(0), srcSize - vec2i(1)), 0);
  } else {
    scene = textureSampleLevel(source, linearSampler, input.uv, 0.);
  }
  // MSAA resolves against clear space, so linear edge colors are premultiplied.
  let alpha = scene.a;
  var c = scene.rgb / max(alpha, .0001);
  if (p.hoverScreen.w > .5) {
    c = (c + textureSampleLevel(bloom, linearSampler, input.uv, 0.).rgb * p.effects.y) * p.effects.x;
    c = toDisplay(acesToneMap(c));
  }
  if (p.effects.w > .5) {
    c = mix(c, vec3f(.83, .29, .32), missingDataLine(input.uv));
  }
  return vec4f(c * alpha, alpha);
}
