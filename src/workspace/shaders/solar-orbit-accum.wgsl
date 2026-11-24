struct OrbitBatch {
  colors: array<vec4f, 4>,
}

@group(0) @binding(0) var<uniform> batch: OrbitBatch;
@group(0) @binding(1) var orbitCov: texture_multisampled_2d<f32>;

fn blendOrbitSample(cov: vec4f) -> vec4f {
  var rgb = vec3f(0.);
  var alpha = 0.;
  for (var i = 0; i < 4; i++) {
    let c = batch.colors[i];
    let a = c.a * cov[i];
    if (a > .0005) {
      rgb = c.rgb * a + rgb * (1. - a);
      alpha = a + alpha * (1. - a);
    }
  }
  return vec4f(rgb, alpha);
}

struct OrbitSampleOut {
  @location(0) color: vec4f,
  @builtin(sample_mask) mask: u32,
}

@fragment fn orbitAccumFragment(
  input: ScreenVertex,
  @builtin(sample_index) sampleIdx: u32,
) -> OrbitSampleOut {
  // Accumulate per-sample premultiplied orbit layers across all batches into 4x MSAA orbitAccum.
  let res = blendOrbitSample(textureLoad(orbitCov, vec2i(input.position.xy), i32(sampleIdx)));
  if (res.a <= .0005) { discard; }
  return OrbitSampleOut(res, 1u << sampleIdx);
}
