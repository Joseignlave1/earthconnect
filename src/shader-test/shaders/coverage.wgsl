// A cached display-resolution rg8unorm mask carries MSAA-resolved surface
// coverage in R and valid-score coverage in G (space=(0,0), missing=(1,0),
// valid=(1,1)), so partial limb resolves never alias into missing data.
fn validScoreFraction(sample: vec2f) -> f32 {
  return sample.g / max(sample.r, 1e-4);
}

fn missingDataLine(uv: vec2f) -> f32 {
  let maskUv = uv;
  let cov = textureSampleLevel(coverageSource, linearSampler, maskUv, 0.).rg;
  if (cov.r < .9 || cov.g > cov.r * .2) { return 0.; }
  let delta = vec2f(1.25 * p.effects.z) / p.viewport.zw;
  let a = textureSampleLevel(coverageSource, linearSampler, maskUv + vec2f(delta.x, 0.), 0.).rg;
  let b = textureSampleLevel(coverageSource, linearSampler, maskUv - vec2f(delta.x, 0.), 0.).rg;
  let c = textureSampleLevel(coverageSource, linearSampler, maskUv + vec2f(0., delta.y), 0.).rg;
  let d = textureSampleLevel(coverageSource, linearSampler, maskUv - vec2f(0., delta.y), 0.).rg;
  let e = textureSampleLevel(coverageSource, linearSampler, maskUv + delta, 0.).rg;
  let f = textureSampleLevel(coverageSource, linearSampler, maskUv + vec2f(-delta.x, delta.y), 0.).rg;
  let g = textureSampleLevel(coverageSource, linearSampler, maskUv + vec2f(delta.x, -delta.y), 0.).rg;
  let h = textureSampleLevel(coverageSource, linearSampler, maskUv - delta, 0.).rg;
  if (min(min(min(a.r, b.r), min(c.r, d.r)), min(min(e.r, f.r), min(g.r, h.r))) < .8) { return 0.; }
  let available = (validScoreFraction(a) + validScoreFraction(b) + validScoreFraction(c) + validScoreFraction(d)) * .125
    + (validScoreFraction(e) + validScoreFraction(f) + validScoreFraction(g) + validScoreFraction(h)) * .0625;
  let edge = smoothstep(.025, .24, available);
  return .28 * edge;
}
