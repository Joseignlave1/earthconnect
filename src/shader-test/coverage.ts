// A cached display-resolution mask carries coverage: space=0, missing=.5,
// valid=1. The boundary follows native-score validity, not hardcoded
// geography or a second world-sized mask.
export const coverageShader = `
fn missingDataLine(uv: vec2f) -> f32 {
  let maskUv = uv;
  let mask = textureSampleLevel(coverageSource,linearSampler,maskUv,0.).r;
  if (mask < .45 || mask > .6) { return 0.; }
  let delta = vec2f(1.25 * p.effects.z) / p.mask.zw;
  let a = textureSampleLevel(coverageSource,linearSampler,maskUv + vec2f(delta.x,0.),0.).r;
  let b = textureSampleLevel(coverageSource,linearSampler,maskUv - vec2f(delta.x,0.),0.).r;
  let c = textureSampleLevel(coverageSource,linearSampler,maskUv + vec2f(0.,delta.y),0.).r;
  let d = textureSampleLevel(coverageSource,linearSampler,maskUv - vec2f(0.,delta.y),0.).r;
  let e = textureSampleLevel(coverageSource,linearSampler,maskUv + delta,0.).r;
  let f = textureSampleLevel(coverageSource,linearSampler,maskUv + vec2f(-delta.x,delta.y),0.).r;
  let g = textureSampleLevel(coverageSource,linearSampler,maskUv + vec2f(delta.x,-delta.y),0.).r;
  let h = textureSampleLevel(coverageSource,linearSampler,maskUv - delta,0.).r;
  if (min(min(min(a,b),min(c,d)),min(min(e,f),min(g,h))) < .4) { return 0.; }
  let neighbors = (a + b + c + d) * .125 + (e + f + g + h) * .0625;
  let available = max(0.,2. * (neighbors - .375));
  let edge = smoothstep(.025,.24,available);
  return .28 * edge;
}`
