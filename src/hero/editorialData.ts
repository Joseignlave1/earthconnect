import type { CrewAstronautSpec } from './crewAstronaut.ts'

/** Perseverance landing site in Jezero Crater, Mars. */
export const HERO_MARS_SITE = {
  name: 'Jezero Crater',
  latitude: 18.4447,
  longitude: 77.4508,
} as const

/**
 * Self-contained Jezero Crater terrain analogue shader that runs directly on
 * the resident WebGPU EarthScore dataset without needing an external API call.
 * Matches gently rolling arid basin floors enclosed by rugged regional relief.
 */
export const HERO_BRIDGE_SCORE_SHADER = /* wgsl */ `// Jezero Crater arid basin analogue (local WebGPU hero query)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 ||
      cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.relief_5km_m) ||
      !has_data(cell.relief_21km_m) ||
      !has_data(cell.tpi_21km_m) ||
      !has_data(cell.annual_precip_mm) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }

  let local_relief = (cell.relief_5km_m - 95.0) / 110.0;
  let regional_relief = (cell.relief_21km_m - 420.0) / 360.0;
  let basin_position = (cell.tpi_21km_m + 115.0) / 145.0;
  let aridity = max(0.0, cell.annual_precip_mm - 180.0) / 260.0;
  let terrain = exp(-0.5 * (
    local_relief * local_relief +
    regional_relief * regional_relief +
    basin_position * basin_position +
    aridity * aridity
  ) * 0.55);
  return terrain * smoothstep(0.15, 0.78, cell.bare_fraction);
}`

export function formatCoord(lat: number, lon: number, decimals = 2) {
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(decimals)}°${ns}, ${Math.abs(lon).toFixed(decimals)}°${ew}`
}

export type SiteId = 'jezero' | 'apollo' | 'tycho'

export type SiteConfig = {
  id: SiteId
  label: string
  body: 'Mars' | 'Moon'
  coords: string
  question: string
  subline: string
}

export const SITES: readonly SiteConfig[] = [
  {
    id: 'jezero',
    label: HERO_MARS_SITE.name,
    body: 'Mars',
    coords: formatCoord(HERO_MARS_SITE.latitude, HERO_MARS_SITE.longitude),
    question: 'Where on Earth feels like Jezero Crater?',
    subline: 'Gently rolling floor inside a rugged regional basin. Change the scope to watch Earth respond.',
  },
  {
    id: 'apollo',
    label: 'Tranquility Base',
    body: 'Moon',
    coords: '0.67°N, 23.47°E',
    question: 'Which plains resemble Tranquility Base?',
    subline: 'Low-slope mare terrain around the Apollo 11 landing ellipse, compared across 1 km and 5 km.',
  },
  {
    id: 'tycho',
    label: 'Tycho North Rim',
    body: 'Moon',
    coords: '41.90°S, 11.20°W',
    question: "Which ridges rise like Tycho's north rim?",
    subline: 'Steep highland crater wall standing high above the surrounding relief over 21 km.',
  },
]

export type SiteShaderRule = {
  name: string
  description: string
  filename: string
  source: string
}

export function buildSiteShader(
  site: SiteId,
  opts: {
    jezeroBasin: boolean
    jezeroArid: boolean
    apolloBare: boolean
    tychoProminence: boolean
  },
): SiteShaderRule {
  if (site === 'jezero') {
    if (opts.jezeroBasin && opts.jezeroArid) {
      return {
        name: 'Jezero Crater · Arid basin',
        description: 'Gently rolling arid basin floor enclosed by rugged 21 km relief',
        filename: 'jezero-basin.wgsl',
        source: HERO_BRIDGE_SCORE_SHADER,
      }
    }
    if (!opts.jezeroBasin && opts.jezeroArid) {
      return {
        name: 'Jezero Crater · 5 km arid ground',
        description: 'Local 5 km relief and dry bare surface without regional basin constraint',
        filename: 'jezero-local.wgsl',
        source: `// Jezero Crater — nearby 5 km ground only (dry & bare)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.relief_5km_m) || !has_data(cell.annual_precip_mm) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }
  let local_relief = (cell.relief_5km_m - 95.0) / 110.0;
  let aridity = max(0.0, cell.annual_precip_mm - 180.0) / 260.0;
  let terrain = exp(-0.5 * (local_relief * local_relief + aridity * aridity));
  return terrain * smoothstep(0.15, 0.78, cell.bare_fraction);
}`,
      }
    }
    if (opts.jezeroBasin && !opts.jezeroArid) {
      return {
        name: 'Jezero Crater · Basin shape (any climate)',
        description: '5 km floor inside a 21 km regional basin regardless of rainfall or cover',
        filename: 'jezero-terrain.wgsl',
        source: `// Jezero Crater — 5 km floor & 21 km basin shape (any climate)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.relief_5km_m) ||
      !has_data(cell.relief_21km_m) || !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }
  let difference = vec3f(
    (cell.relief_5km_m - 95.0) / 110.0,
    (cell.relief_21km_m - 420.0) / 360.0,
    (cell.tpi_21km_m + 115.0) / 145.0,
  );
  return exp(-0.5 * dot(difference, difference) * 0.65);
}`,
      }
    }
    return {
      name: 'Jezero Crater · 5 km relief (any climate)',
      description: 'Local 5 km rolling terrain anywhere on Earth',
      filename: 'jezero-local-any.wgsl',
      source: `// Jezero Crater — nearby 5 km ground only (any climate)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.relief_5km_m)) {
    return NO_SCORE;
  }
  let local_relief = (cell.relief_5km_m - 95.0) / 110.0;
  return exp(-0.5 * local_relief * local_relief);
}`,
    }
  }

  if (site === 'apollo') {
    if (opts.apolloBare) {
      return {
        name: 'Tranquility Base · Exposed plains',
        description: 'Flat 1 km slope and gentle 5 km relief on bare exposed ground',
        filename: 'apollo-11.wgsl',
        source: `// Apollo 11 · Tranquility Base (exposed bare plains)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || cell.landcover_valid_fraction < 0.9 ||
      !has_data(cell.slope_mean_deg) || !has_data(cell.relief_5km_m) ||
      !has_data(cell.bare_fraction)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 1.4) / 1.5;
  let relief_err = (cell.relief_5km_m - 65.0) / 85.0;
  let terrain = exp(-0.5 * (slope_err * slope_err + relief_err * relief_err));
  return terrain * smoothstep(0.2, 0.8, cell.bare_fraction);
}`,
      }
    }
    return {
      name: 'Tranquility Base · Terrain only',
      description: 'Flat 1 km slope and gentle 5 km relief across any land cover',
      filename: 'apollo-11-terrain.wgsl',
      source: `// Apollo 11 · Tranquility Base (any ground cover)
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 ||
      !has_data(cell.slope_mean_deg) || !has_data(cell.relief_5km_m)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 1.4) / 1.5;
  let relief_err = (cell.relief_5km_m - 65.0) / 85.0;
  return exp(-0.5 * (slope_err * slope_err + relief_err * relief_err));
}`,
    }
  }

  if (opts.tychoProminence) {
    return {
      name: 'Tycho North Rim · High-prominence ridges',
      description: 'Steep slopes rising prominently above surrounding 21 km relief',
      filename: 'tycho.wgsl',
      source: `// Tycho North Rim — steep ridges standing above 21 km surroundings
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.slope_mean_deg) ||
      !has_data(cell.relief_21km_m) || !has_data(cell.tpi_21km_m)) {
    return NO_SCORE;
  }
  let difference = vec3f(
    (cell.slope_mean_deg - 16.5) / 5.5,
    (cell.relief_21km_m - 2100.0) / 800.0,
    (cell.tpi_21km_m - 480.0) / 240.0,
  );
  return exp(-0.5 * dot(difference, difference) / 3.0);
}`,
    }
  }

  return {
    name: 'Tycho North Rim · Steepness only',
    description: 'High 1 km slope matching Tycho crater wall steepness',
    filename: 'tycho-slope.wgsl',
    source: `// Tycho North Rim — 1 km slope steepness only
fn score(cell: EarthCell) -> f32 {
  if (cell.terrain_valid_fraction < 0.9 || !has_data(cell.slope_mean_deg)) {
    return NO_SCORE;
  }
  let slope_err = (cell.slope_mean_deg - 16.5) / 5.0;
  return exp(-0.5 * slope_err * slope_err);
}`,
  }
}

const RICKROLL_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

export type TeamMember = CrewAstronautSpec & {
  role: string
  bio: string
  lat: number
  lon: number
  socials: readonly { label: string; href: string }[]
}

const PLACEHOLDER_SOCIALS = [
  { label: 'GitHub', href: RICKROLL_URL },
  { label: 'LinkedIn', href: RICKROLL_URL },
  { label: 'Website', href: RICKROLL_URL },
] as const

/**
 * Six equal team credits (alphabetical, author last).
 * Swap placeholder roles, bios, hometown/hospital coordinates, photos, and social links when ready.
 */
export const TEAM_MEMBERS: readonly TeamMember[] = [
  {
    id: 'diego',
    name: 'Diego',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Still trying to rescue the laptop before spaghettification.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'laptop',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'joaquin',
    name: 'Joaquín',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Heroically bracing both space boots against empty vacuum.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'brace',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'micaela',
    name: 'Micaela',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Hanging on to absolutely nothing by one EVA glove.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'hang',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'nacho',
    name: 'Nacho',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Tumbling sideways into the event horizon at orbital velocity.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'tumble',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'roy',
    name: 'Roy',
    role: 'Role placeholder',
    bio: 'Short personal note or what you worked on goes here. Has made peace with the curvature of spacetime.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'accept',
    socials: PLACEHOLDER_SOCIALS,
  },
  {
    id: 'cristian',
    name: 'Cristian',
    role: 'Tech Lead',
    bio: 'Short personal note or what you worked on goes here. Currently losing a fight with a gravitational singularity.',
    lat: -34.9011,
    lon: -56.1645,
    pose: 'cling',
    socials: PLACEHOLDER_SOCIALS,
  },
]

export type CopilotExample = {
  id: string
  label: string
  prompt: string
  steps: readonly { call: string; detail: string }[]
  snippet: readonly { added?: boolean; code: string }[]
}

export const COPILOT_EXAMPLES: readonly CopilotExample[] = [
  {
    id: 'elevation',
    label: 'Filter by elevation',
    prompt: "Keep Jezero's arid basin shape, but exclude plateaus above 2,500 m.",
    steps: [
      { call: 'edit_shader("jezero-basin.wgsl")', detail: 'Added elevation cap before basin scoring' },
      { call: 'rank_regions(limit: 3)', detail: 'Namib Escarpment · Lut Basin · Mojave Basin' },
    ],
    snippet: [
      { code: 'if (!has_data(cell.elevation_m)) { return NO_SCORE; }' },
      { added: true, code: 'if (cell.elevation_m > 2500.0) { return NO_SCORE; }' },
      { code: 'let local_relief = (cell.relief_5km_m - 95.0) / 110.0;' },
    ],
  },
  {
    id: 'compare',
    label: 'Sample another site',
    prompt: 'Build a comparison for Shackleton Crater rim on the lunar south pole.',
    steps: [
      { call: 'moon_point(-89.54, 0.0)', detail: 'Fetched LRO 1 km slope & 21 km rim prominence' },
      { call: 'create_shader("shackleton.wgsl")', detail: 'Compiled live · 31.8M Earth cells scored' },
    ],
    snippet: [
      { added: true, code: 'let rim = moon_point(-89.54, 0.0);' },
      { added: true, code: 'let slope_err = (cell.slope_mean_deg - rim.terrain.slope_1km_deg) / 4.5;' },
      { code: 'return exp(-0.5 * slope_err * slope_err);' },
    ],
  },
  {
    id: 'explain',
    label: 'Inspect top match',
    prompt: 'Why did southern Namibia outscore the Atacama on this rule?',
    steps: [
      { call: 'inspect_cell(-26.28, 15.55)', detail: '5 km relief 98 m · 21 km basin 435 m · 62 mm/yr' },
      { call: 'compare_region("Atacama")', detail: 'Higher regional relief (890 m) penalized basin match' },
    ],
    snippet: [
      { code: '// Namib Escarpment: score 0.99 · TPI -118 m · bare 0.94' },
      { code: '// Atacama Foothills: score 0.81 · TPI -240 m · bare 0.98' },
    ],
  },
]
