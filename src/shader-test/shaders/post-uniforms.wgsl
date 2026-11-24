struct PostUniforms {
  hoverScreen: vec4f, // x, y: cursor output px, z: hasHoverScreen, w: hdr
  effects: vec4f,     // x: exposure, y: bloomGlow, z: pixelRatio, w: showDataBoundary
  viewport: vec4f,    // x: displayOnlyHover, y: hoverBody, z: outWidth, w: outHeight
  hover: vec4f,       // x, y: uv, z: active, w: unitPerPixel
  state: vec4f,       // x: earthMorph, y: bakedMode (0=none, 1=bakedSel, 2=motion), z: hasOrbits, w: bakeMarkers
  selEarth: vec4f,
  selMoon: vec4f,
  selMars: vec4f,
}
