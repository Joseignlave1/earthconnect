fn terrainLevel(uv: vec2f, lod: f32) -> vec2f {
  return textureSampleLevel(terrain, geographicSampler, uv, lod).rg;
}

fn terrainPair(uv: vec2f, dx: vec2f, dy: vec2f) -> vec2f {
  let sampleUv = vec2f(uv.x, 1. - uv.y);
  return textureSampleGrad(terrain, geographicSampler, sampleUv, vec2f(dx.x, -dx.y), vec2f(dy.x, -dy.y)).rg;
}
