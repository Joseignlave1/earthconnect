export const comparisonExamples: readonly { id: string; name: string; source: string }[] = [
  {
    id: 'example-apollo',
    name: 'apollo-11.wgsl',
    source: /* wgsl */ `// Which exposed Earth plains resemble the terrain around Tranquility Base?
// Compare slope in degrees and elevation range in metres across 5 km.
fn score(cell: EarthCell) -> f32 {
  let site = moon_point(0.674, 23.473);
  let slope = site.terrain.slope_1km_deg;
  let relief = site.terrain.relief_5km_m;
  if (!has_data(slope) || !has_data(relief) ||
      cell.terrain_valid_fraction < 0.9 || cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.slope_mean_deg) || !has_data(cell.relief_5km_m) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }

  // Scale slope differences by 1.5° and relief differences by 85 m.
  let slope_error = (cell.slope_mean_deg - slope) / 1.5;
  let relief_error = (cell.relief_5km_m - relief) / 85.0;
  let terrain = exp(-0.5 * (slope_error * slope_error + relief_error * relief_error));
  // Favor exposed ground as bare cover increases from 20% to 80%.
  return terrain * smoothstep(0.2, 0.8, cell.bare_fraction);
}`,
  },
  {
    id: 'example-jezero',
    name: 'jezero.wgsl',
    source: /* wgsl */ `// Where on Earth does gently rolling ground sit within a rugged basin?
// Sample Perseverance's landing site on the floor of Jezero crater.
fn score(cell: EarthCell) -> f32 {
  let site = mars_point(18.4447, 77.4508);
  let local_relief = site.terrain.relief_5km_m;
  let regional_relief = site.terrain.relief_21km_m;
  let regional_position = site.terrain.tpi_21km_m;
  if (!has_data(local_relief) || !has_data(regional_relief) ||
      !has_data(regional_position) || cell.terrain_valid_fraction < 0.9 ||
      !has_data(cell.relief_5km_m) || !has_data(cell.relief_21km_m) ||
      !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }

  // Relief is elevation range. Topographic position is height above the
  // neighborhood mean, so the negative value favors low-lying terrain.
  // Tolerances in metres are 100 for local relief, 325 for regional relief,
  // and 100 for regional topographic position.
  let difference = vec3f(
    (cell.relief_5km_m - local_relief) / 100.0,
    (cell.relief_21km_m - regional_relief) / 325.0,
    (cell.tpi_21km_m - regional_position) / 100.0,
  );
  return exp(-0.5 * dot(difference, difference));
}`,
  },
  {
    id: 'example-tycho',
    name: 'tycho.wgsl',
    source: /* wgsl */ `// Which Earth ridges rise above their surroundings like Tycho's north rim?
fn score(cell: EarthCell) -> f32 {
  let rim = moon_point(-41.9, -11.2);
  let slope = rim.terrain.slope_1km_deg;
  let relief = rim.terrain.relief_21km_m;
  let position = rim.terrain.tpi_21km_m;
  if (!has_data(slope) || !has_data(relief) || !has_data(position) ||
      cell.terrain_valid_fraction < 0.9 || !has_data(cell.slope_mean_deg) ||
      !has_data(cell.relief_21km_m) || !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }

  // Scale slope differences by 5°. Over 21 km, scale elevation range
  // by 800 m and height above the neighborhood mean by 225 m.
  let difference = vec3f(
    (cell.slope_mean_deg - slope) / 5.0,
    (cell.relief_21km_m - relief) / 800.0,
    (cell.tpi_21km_m - position) / 225.0,
  );
  // Average the three squared differences to show partial matches.
  return exp(-0.5 * dot(difference, difference) / 3.0);
}`,
  },
]
