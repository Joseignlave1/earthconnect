// Hello, World!
// Drag the globe to follow the greeting around Earth.
//
// Pick Moon or Mars in the upper left, then click a spot on its surface.
// In point actions (⋯), choose Copy as WGSL expression to paste into score(),
// or choose Insert WGSL expression with the cursor inside score().
// Bind the call as `let site = …;` and compare a terrain field such as
// site.terrain.slope_1km_deg with cell.slope_mean_deg.
//
// score() runs for each Earth cell. Higher scores give brighter colors.
// Return NO_SCORE to leave the original globe visible at that cell.
//
// One glyph per slot. Read each five-digit row as a tiny image:
// 1 marks a stroke and 0 leaves the original globe visible.
const H = array(10001u, 10001u, 10001u, 11111u, 10001u, 10001u, 10001u, 0u, 0u);
const e = array(    0u,     0u,  1110u, 11011u, 11111u, 10000u,  1111u, 0u, 0u);
const l = array(  100u,   100u,   100u,   100u,   100u,   100u,   110u, 0u, 0u);
const o = array(    0u,     0u, 11111u, 10001u, 10001u, 10001u, 11111u, 0u, 0u);
const W = array(10001u, 10001u, 10001u, 10101u, 10101u, 11111u,  1010u, 0u, 0u);
const r = array(    0u,     0u, 11110u, 10010u, 10000u, 10000u, 10000u, 0u, 0u);
const d = array(    1u,     1u, 11111u, 10001u, 10001u, 10001u, 11111u, 0u, 0u);
const COMMA = array(0u, 0u, 0u, 0u, 0u, 0u, 11000u, 11000u, 10000u);
const SPACE = array(0u,     0u,     0u,     0u,     0u,     0u,     0u, 0u, 0u);
const BANG = array(100u,  100u,   100u,   100u,   100u,     0u,   100u, 0u, 0u);
const SLOTS = 13u;
const MESSAGE = array(H, e, l, l, o, COMMA, SPACE, W, o, r, l, d, BANG);
const DIGIT = array(10000u, 1000u, 100u, 10u, 1u);

fn score(cell: EarthCell) -> f32 {
  if (!has_data(cell.latitude_deg) || !has_data(cell.longitude_deg)) {
    return NO_SCORE;
  }

  let row = (59.0 - cell.latitude_deg) / 5.5;
  if (row < 0.0 || row >= 9.0) { return NO_SCORE; }

  // Start 4° east of the date line so the flat map cuts through a blank column.
  let east = cell.longitude_deg + 176.0;
  let wrapped = east - floor(east / 360.0) * 360.0;
  let column = u32(wrapped * f32(SLOTS * 6u) / 360.0);
  let pixel = column % 6u;
  if (pixel == 5u) { return NO_SCORE; }

  let glyph = MESSAGE[column / 6u];
  let stroke = glyph[u32(row)] / DIGIT[pixel] % 10u == 1u;

  // Keep the whole phrase bright while increasing scores toward the last glyph.
  let intensity = mix(0.992, 1.0, wrapped / 360.0);
  return select(NO_SCORE, intensity, stroke);
}
