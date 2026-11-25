fn palettePosition(normalized: f32, band: f32, contrast: f32) -> f32 {
  let knee = 1. - band;
  let context = .25 * knee;
  if (normalized < knee) { return context * normalized / knee; }
  return context + (1. - context) * pow(clamp((normalized - knee) / band, 0., 1.), contrast);
}

fn autoPalette(score: f32, cutoff: f32, anchors: vec4f) -> f32 {
  if (score <= cutoff || anchors.w <= cutoff) { return 0.; }
  if (score < anchors.x) { return .22 * clamp((score - cutoff) / max(.0000001, anchors.x - cutoff), 0., 1.); }
  if (score < anchors.y) { return mix(.22, .70, clamp((score - anchors.x) / max(.0000001, anchors.y - anchors.x), 0., 1.)); }
  if (score < anchors.z) { return mix(.70, .90, clamp((score - anchors.y) / max(.0000001, anchors.z - anchors.y), 0., 1.)); }
  return mix(.90, 1., clamp((score - anchors.z) / max(.0000001, anchors.w - anchors.z), 0., 1.));
}
