struct MarkerRecord {
  uv: vec2f,
  opacity: f32,
  unitsPerPixel: f32,
}

struct PostUniforms {
  hoverScreenPx: vec2f,
  hasHoverScreen: f32,
  hdr: f32,
  exposure: f32,
  bloomGlow: f32,
  pixelRatio: f32,
  showDataBoundary: f32,
  displayOnlyHover: f32,
  hoverBody: f32,
  outSizePx: vec2f,
  hover: MarkerRecord,
  earthMorph: f32,
  bakedMode: f32, // 0 = none, 1 = bakedSel, 2 = motion
  hasOrbits: f32,
  bakeMarkers: f32,
  selEarth: MarkerRecord,
  selMoon: MarkerRecord,
  selMars: MarkerRecord,
}
