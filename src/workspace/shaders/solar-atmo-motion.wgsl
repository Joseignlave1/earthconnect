@group(0) @binding(2) var sceneDepth: texture_depth_multisampled_2d;

@fragment fn earthAtmoMotionFragment(input: ScreenVertex) -> @location(0) vec4f {
  // Composite crisp planet-attached atmosphere onto the 1x motion-blurred target after encodeMotion
  // with exact 4-sample depth visibility so atmospheres never inherit sky-at-infinity motion smear.
  let sample = computeEarthAtmo(input);
  if (!sample.valid) { discard; }
  let pixel = vec2i(input.position.xy);
  var vis = 0.;
  for (var s = 0; s < 4; s++) {
    if (sample.depth > textureLoad(sceneDepth, pixel, s)) { vis += .25; }
  }
  if (vis <= 0.) { discard; }
  return vec4f(sample.radiance * vis, 0.);
}
