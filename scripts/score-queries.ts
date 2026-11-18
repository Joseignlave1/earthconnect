// Frozen rendering experiments. All numbers are bound SQL parameters.
const flatExpression = `CASE WHEN bare_fraction IS NULL OR slope_mean_deg IS NULL THEN NULL
  ELSE bare_fraction * exp(-slope_mean_deg / :scale) END`
const basinExpression = `CASE
  WHEN terrain_valid_fraction IS NULL OR terrain_valid_fraction < :coverage
    OR elevation_median_m IS NULL OR slope_mean_deg IS NULL
    OR relief_21km_m IS NULL OR tpi_21km_m IS NULL
    OR annual_precip_mm IS NULL OR coldest_month_air_c IS NULL
    OR warmest_month_air_c IS NULL THEN NULL
  ELSE exp(-0.5 * least(1000.0, (
    power((elevation_median_m - :elevation) / :elevation_tolerance, 2)
    + power(greatest(0.0, slope_mean_deg - :slope_max) / :slope_tolerance, 2)
    + power((relief_21km_m - :relief) / :relief_tolerance, 2)
    + power((tpi_21km_m - :position) / :position_tolerance, 2)
    + power(greatest(0.0, annual_precip_mm - :precip_max) / :precip_tolerance, 2)
    + (power((coldest_month_air_c - :coldest) / :coldest_tolerance, 2)
       + power((warmest_month_air_c - :warmest) / :warmest_tolerance, 2)) / 2.0
  ) / 6.0)) END`

export const scoreQueries = {
  flat: {
    file: 'surface-scores', name: 'Flat, bare terrain', expression: flatExpression,
    description: 'bare_fraction × exp(−slope_mean_deg / 8); missing measurements remain missing.',
    params: { scale: 8 },
  },
  basins: {
    file: 'surface-basins', name: 'High, dry mountain basins', expression: basinExpression,
    description: 'Elevation 3000 ± 1200 m; slope ≤ 4°; 21 km relief 1500 ± 700 m; terrain position −200 ± 300 m; rainfall ≤ 250 mm/year; coldest month −10 ± 12°C, warmest 15 ± 10°C. Soft preferences, ≥90% terrain coverage.',
    params: { coverage: .9, elevation: 3000, elevation_tolerance: 1200, slope_max: 4, slope_tolerance: 3,
      relief: 1500, relief_tolerance: 700, position: -200, position_tolerance: 300,
      precip_max: 250, precip_tolerance: 250, coldest: -10, coldest_tolerance: 12, warmest: 15, warmest_tolerance: 10 },
  },
} as const

export function packedScoreQuery(expression: string) {
  return `WITH scored AS (
  SELECT region_id, pixel, ${expression} AS score FROM measurements
), matrices AS (
  SELECT region_id, list(struct_pack(index := pixel,
    code := CASE WHEN score IS NULL THEN NULL
      ELSE round_even(65534 * greatest(0.0, least(1.0, score)), 0)::INTEGER END
  )) AS codes FROM scored GROUP BY region_id
)
SELECT site_key, width, height, pack_u16_at(codes, width * height) AS scores
FROM matrices JOIN regions USING (region_id) ORDER BY site_key;`
}
