export type BodyId = 'moon' | 'mars'

export type Criterion = {
  key: string
  label: string
  unit: string
  group: string
  enabled: boolean
  target: number | null
  tolerance: number
  weight: number
  min: number | null
  max: number | null
  description?: string
  categorical?: boolean
  qualityKey?: string | null
  origin?: string
  kind?: 'measured' | 'approximate' | 'preference' | 'manual'
  selectionTouched?: boolean
  selectedByUser?: boolean
  toleranceTouched?: boolean
  weightTouched?: boolean
  targetTouched?: boolean
}

export type EarthMatch = {
  id: string
  title: string
  region: string
  country: string
  description: string
  lat: number
  lon: number
  score: number
  areaSemiMajor: number
  areaSemiMinor: number
  tags: string[]
  metrics: { label: string; value: string }[]
  reasons?: { label: string; comparison: string; similarity: number }[]
}

const FIELD_DEFINITIONS = [
  ['slope_mean_deg', 'Mean slope', '°', 'Terrain', 2],
  ['elevation_median_m', 'Elevation', 'm', 'Terrain', 1000],
  ['relief_5km_m', '5 km relief', 'm', 'Terrain', 500],
  ['relief_21km_m', '21 km relief', 'm', 'Terrain', 1500],
  ['tpi_5km_m', '5 km terrain position', 'm', 'Terrain', 300],
  ['tpi_21km_m', '21 km terrain position', 'm', 'Terrain', 1000],
  ['coldest_month_air_c', 'Coldest-month air', '°C', 'Climate', 20],
  ['warmest_month_air_c', 'Warmest-month air', '°C', 'Climate', 20],
  ['annual_precip_mm', 'Annual precipitation', 'mm', 'Climate', 500],
  ['bare_fraction', 'Bare ground', '%', 'Surface', .25],
  ['permanent_snow_ice_fraction', 'Permanent snow / ice', '%', 'Surface', .25],
  ['permafrost_fraction', 'Permafrost', '%', 'Surface', .25],
  ['dominant_lithology_code', 'Lithology code', 'code', 'Geology', 1],
  ['surface_day_c', 'Daytime ground temp.', '°C', 'Thermal', 30],
  ['surface_night_c', 'Nighttime ground temp.', '°C', 'Thermal', 30],
  ['surface_mean_c', 'Mean ground temp.', '°C', 'Thermal', 30],
  ['surface_low_c', 'Coldest surface composite', '°C', 'Thermal', 30],
  ['surface_high_c', 'Hottest surface composite', '°C', 'Thermal', 30],
  ['surface_range_c', 'Ground temperature range', '°C', 'Thermal', 40],
] as const

export function initialCriteria(): Criterion[] {
  return FIELD_DEFINITIONS.map(([key, label, unit, group, tolerance], index) => ({
    key,
    label,
    unit,
    group,
    enabled: index < 8 || index >= 13,
    target: 0,
    tolerance,
    weight: 1,
    min: null,
    max: null,
  }))
}

export function sampleCriteria(body: BodyId, lat: number, lon: number, previous: Criterion[]) {
  const latitude = Math.abs(lat) / 90
  const longitude = Math.abs(lon) / 180
  const mars = body === 'mars'
  const targets: Record<string, number> = {
    slope_mean_deg: 3 + latitude * (mars ? 18 : 12),
    elevation_median_m: (mars ? -2800 : -900) + longitude * (mars ? 6900 : 4200),
    relief_5km_m: 120 + latitude * (mars ? 2100 : 1100),
    relief_21km_m: 450 + longitude * (mars ? 4300 : 2800),
    tpi_5km_m: (lat / 90) * (mars ? 720 : 410),
    tpi_21km_m: (lon / 180) * (mars ? 1600 : 950),
    coldest_month_air_c: mars ? -61 + latitude * 26 : -74 + latitude * 34,
    warmest_month_air_c: mars ? -8 + longitude * 33 : -18 + longitude * 30,
    annual_precip_mm: mars ? 55 + latitude * 75 : 80 + latitude * 180,
    bare_fraction: mars ? .93 : .72,
    permanent_snow_ice_fraction: Math.max(0, latitude - .62),
    permafrost_fraction: Math.max(0, latitude - .48),
    dominant_lithology_code: mars ? 7 : 4,
    surface_day_c: mars ? -2 + longitude * 35 : 26 + longitude * 24,
    surface_night_c: mars ? -74 + latitude * 28 : -115 + latitude * 38,
    surface_mean_c: mars ? -37 + longitude * 24 : -45 + longitude * 28,
    surface_low_c: mars ? -105 + latitude * 20 : -151 + latitude * 31,
    surface_high_c: mars ? 18 + longitude * 42 : 62 + longitude * 39,
    surface_range_c: mars ? 82 + latitude * 32 : 138 + latitude * 37,
  }
  return previous.map((criterion) => ({
    ...criterion,
    target: Number((targets[criterion.key] ?? 0).toFixed(2)),
  }))
}

const SITES = [
  ['atacama-high-plain', 'Atacama High Plain', 'Antofagasta', 'Chile', -23.46, -68.24, 'An exposed high-altitude plateau with exceptional aridity, intense radiation and broad volcanic terrain.'],
  ['mcmurdo-dry-valleys', 'McMurdo Dry Valleys', 'Victoria Land', 'Antarctica', -77.52, 161.74, 'A cold polar desert shaped by katabatic winds, exposed rock and almost no surface precipitation.'],
  ['wadi-rum', 'Wadi Rum', 'Aqaba', 'Jordan', 29.57, 35.42, 'Sandstone massifs and open desert basins create a dry, rocky landscape with strong thermal contrast.'],
  ['danakil', 'Danakil Depression', 'Afar', 'Ethiopia', 14.24, 40.30, 'A low volcanic basin marked by salt flats, extreme heat and active geological processes.'],
  ['altiplano', 'Bolivian Altiplano', 'Potosí', 'Bolivia', -20.10, -67.50, 'A high, cold and sparsely vegetated plateau surrounded by volcanic peaks and saline basins.'],
  ['namib', 'Central Namib', 'Erongo', 'Namibia', -23.35, 15.30, 'Ancient dunes meet gravel plains in one of Earth’s oldest and driest desert systems.'],
  ['iceland-highlands', 'Icelandic Highlands', 'Suðurland', 'Iceland', 64.75, -18.10, 'A young volcanic interior of lava fields, glaciers and low vegetation under cool maritime conditions.'],
  ['sahara-tibesti', 'Tibesti Massif', 'Borkou', 'Chad', 20.75, 17.40, 'Remote volcanic mountains rise from hyper-arid Saharan plains with little biological cover.'],
  ['tinto', 'Río Tinto Basin', 'Andalusia', 'Spain', 37.69, -6.56, 'Mineral-rich acidic waters and weathered terrain form an unusual terrestrial analogue environment.'],
  ['puna', 'Puna de Atacama', 'Catamarca', 'Argentina', -25.72, -67.68, 'A high desert of salt basins, volcanic cones and strong daily temperature swings.'],
  ['gobi', 'Gobi Desert', 'Ömnögovi', 'Mongolia', 43.02, 104.85, 'Cold desert plains combine exposed sediment, sparse grass and severe seasonal temperature changes.'],
  ['simpson', 'Simpson Desert', 'Northern Territory', 'Australia', -24.57, 137.42, 'Long parallel dunes and dry clay pans dominate a vast continental desert interior.'],
  ['death-valley', 'Death Valley', 'California', 'United States', 36.24, -116.82, 'A below-sea-level basin with salt flats, alluvial fans and some of Earth’s highest surface temperatures.'],
  ['mauna-kea', 'Mauna Kea Summit', 'Hawaiʻi', 'United States', 19.82, -155.47, 'A high volcanic summit with thin air, dry conditions and a cinder-covered surface.'],
  ['kamchatka', 'Kamchatka Volcanic Belt', 'Kamchatka Krai', 'Russia', 55.83, 160.33, 'Active volcanoes, snowfields and broad ash deposits shape this remote subarctic peninsula.'],
  ['svalbard', 'Svalbard Interior', 'Svalbard', 'Norway', 78.45, 16.25, 'Polar desert, permafrost and exposed sedimentary rock define the high Arctic environment.'],
  ['tassili', 'Tassili n’Ajjer', 'Illizi', 'Algeria', 25.50, 9.00, 'Eroded sandstone plateaus and dunes form a dry labyrinth across the central Sahara.'],
  ['dallol', 'Dallol Hydrothermal Field', 'Afar', 'Ethiopia', 14.24, 40.30, 'Salt, sulfur and hydrothermal deposits produce an extreme, chemically active landscape.'],
  ['canary', 'Teide Caldera', 'Canary Islands', 'Spain', 28.27, -16.64, 'A high volcanic caldera with lava flows, pumice fields and sharp elevation gradients.'],
  ['ladakh', 'Ladakh Cold Desert', 'Ladakh', 'India', 34.16, 77.58, 'A high-altitude rain shadow combines barren slopes, broad valleys and severe winters.'],
  ['qaidam', 'Qaidam Basin', 'Qinghai', 'China', 37.15, 94.40, 'Saline lakes and wind-eroded terrain occupy a cold, elevated continental basin.'],
  ['utah', 'San Rafael Desert', 'Utah', 'United States', 38.82, -110.58, 'Layered sedimentary rock, mesas and dry washes create a rugged desert analogue.'],
  ['dry-andes', 'Central Andean Volcanic Zone', 'Tarapacá', 'Chile', -19.40, -68.70, 'Young volcanoes and high-elevation plains experience intense sun and limited moisture.'],
  ['devon-island', 'Haughton Crater', 'Nunavut', 'Canada', 75.38, -89.67, 'An impact crater in a cold polar desert, widely studied as a planetary analogue site.'],
  ['bardenas', 'Bardenas Reales', 'Navarre', 'Spain', 42.20, -1.47, 'Eroded clay, gypsum and sandstone create badlands with sparse semi-arid vegetation.'],
] as const

export function createMatches(body: BodyId, lat: number, lon: number, criteria: Criterion[]): EarthMatch[] {
  const enabled = criteria.filter((item) => item.enabled).length
  const bias = body === 'mars' ? 3.5 : 0
  return SITES.map((site, index) => {
    const [, title, region, country, siteLat, siteLon, description] = site
    const angular = Math.abs(lat - siteLat) * .03 + Math.abs(lon - siteLon) * .008
    const score = Math.max(61.5, 97.8 - index * .72 - (angular % 5) + bias - enabled * .015)
    return {
      id: site[0], title, region, country, description, lat: siteLat, lon: siteLon,
      score: Number(score.toFixed(1)),
      areaSemiMajor: 120000,
      areaSemiMinor: 90000,
      tags: index % 3 === 0 ? ['High altitude', 'Low moisture'] : index % 3 === 1 ? ['Thermal analogue', 'Exposed rock'] : ['Sparse cover', 'Remote'],
      metrics: [
        { label: 'Elevation', value: `${Math.round(280 + Math.abs(siteLat) * 47 + index * 31).toLocaleString()} m` },
        { label: 'Mean slope', value: `${(2.4 + index * .41).toFixed(1)}°` },
        { label: 'Annual rain', value: `${Math.round(12 + (index * 19) % 240)} mm` },
        { label: 'Surface range', value: `${Math.round(31 + (index * 7) % 54)}°C` },
      ],
    }
  }).sort((a, b) => b.score - a.score)
}

export const DEFAULT_SQL = `WITH scored AS (
  SELECT region_id, pixel,
         similarity_score(*) AS score
  FROM measurements
  WHERE coverage >= :minimum_coverage
), ranked AS (
  SELECT *, row_number() OVER (ORDER BY score DESC) AS rank
  FROM scored
)
SELECT site_key, score, latitude, longitude
FROM ranked
WHERE rank <= :limit
ORDER BY score DESC;`
