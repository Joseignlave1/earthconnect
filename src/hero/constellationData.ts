export type StarTint = 'warm' | 'white' | 'blue'

export type ConstellationStar = {
  x: number
  y: number
  z: number // 0.18 (far) .. 0.98 (near) for subtle per-star 3D depth parallax
  mag: 1 | 2 | 3
  tint: StarTint
  delay: number
  label?: string
}

export type ConstellationLetter = {
  id: string
  glyph: string
  cx: number
  baselineY: number
  fontSize: number
  stars: readonly ConstellationStar[]
  edges: readonly (readonly [number, number])[]
}

export const VIEW_W = 616
export const VIEW_H = 246
export const FIELD_PAD = 36
export const FIELD_W = VIEW_W + FIELD_PAD * 2
export const FIELD_H = VIEW_H + FIELD_PAD * 2

// Subtle distant cosmic parallax so stars feel light-years away, not nearby.
export const MAX_PARALLAX_X = 3.8
export const MAX_PARALLAX_Y = 2.6

// Linear HDR spectral starlight tints (K/G amber, A ivory, B sapphire-white).
export const STAR_TINT_RGB: Record<StarTint, readonly [number, number, number]> = {
  warm: [1.0, 0.78, 0.52],
  white: [0.96, 0.98, 1.0],
  blue: [0.68, 0.84, 1.0],
}

// Sparse, breathable 3D constellation asterisms paired with Cinzel Decorative.
// Only the 9 hero anchor stars (mag: 1) shine prominently — placed right where
// the tall initials (E, C) and sweeping flourishes (R, H, N, T) break past the
// baselines to reach them — while 2-4 quieter companion stars per letter form
// open star-chart geometry that the lettering bridges across the dark sky.
export const CONSTELLATION_LETTERS: readonly ConstellationLetter[] = [
  // Row 1: E a R t H  (shared baselineY = 104)
  {
    id: 'E1',
    glyph: 'E',
    cx: 30,
    baselineY: 104,
    fontSize: 96,
    stars: [
      { x: 5, y: 29, z: 0.94, mag: 1, tint: 'warm', delay: 0.2, label: 'α' },
      { x: 52, y: 33, z: 0.36, mag: 3, tint: 'blue', delay: 1.5 },
      { x: 10, y: 71, z: 0.56, mag: 3, tint: 'white', delay: 2.7 },
      { x: 59, y: 100, z: 0.86, mag: 2, tint: 'white', delay: 1.8 },
    ],
    edges: [
      [0, 1],
      [0, 2],
      [2, 3],
    ],
  },
  {
    id: 'A1',
    glyph: 'a',
    cx: 107,
    baselineY: 104,
    fontSize: 82,
    stars: [
      { x: 75, y: 107, z: 0.42, mag: 3, tint: 'white', delay: 0.6 },
      { x: 106, y: 40, z: 0.98, mag: 1, tint: 'warm', delay: 2.1, label: 'β' },
      { x: 140, y: 102, z: 0.64, mag: 2, tint: 'blue', delay: 3.6 },
    ],
    edges: [
      [0, 1],
      [1, 2],
    ],
  },
  {
    id: 'R1',
    glyph: 'R',
    cx: 184,
    baselineY: 104,
    fontSize: 82,
    stars: [
      { x: 162, y: 105, z: 0.72, mag: 3, tint: 'blue', delay: 1.4 },
      { x: 164, y: 46, z: 0.34, mag: 3, tint: 'white', delay: 0.3 },
      { x: 199, y: 54, z: 0.66, mag: 2, tint: 'warm', delay: 2.8 },
      { x: 238, y: 121, z: 0.96, mag: 1, tint: 'warm', delay: 0.8 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  },
  {
    id: 'T1',
    glyph: 't',
    cx: 249,
    baselineY: 104,
    fontSize: 82,
    stars: [
      { x: 223, y: 47, z: 0.82, mag: 3, tint: 'warm', delay: 2.4 },
      { x: 249, y: 42, z: 0.46, mag: 3, tint: 'white', delay: 0.5 },
      { x: 276, y: 40, z: 0.9, mag: 2, tint: 'blue', delay: 3.1 },
      { x: 247, y: 107, z: 0.25, mag: 3, tint: 'warm', delay: 1.6 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [1, 3],
    ],
  },
  {
    id: 'H1',
    glyph: 'H',
    cx: 325,
    baselineY: 104,
    fontSize: 82,
    stars: [
      { x: 294, y: 119, z: 0.9, mag: 1, tint: 'white', delay: 0.4 },
      { x: 302, y: 74, z: 0.56, mag: 3, tint: 'warm', delay: 3.4 },
      { x: 350, y: 72, z: 0.3, mag: 3, tint: 'white', delay: 4.1 },
      { x: 356, y: 30, z: 0.98, mag: 1, tint: 'blue', delay: 2.2, label: 'γ' },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  },

  // Row 2: C o N n e c T  (indented, shared baselineY = 208)
  {
    id: 'C1',
    glyph: 'C',
    cx: 65,
    baselineY: 208,
    fontSize: 96,
    stars: [
      { x: 63, y: 135, z: 0.84, mag: 1, tint: 'white', delay: 2.5 },
      { x: 27, y: 175, z: 0.95, mag: 2, tint: 'warm', delay: 1.3 },
      { x: 99, y: 200, z: 0.72, mag: 2, tint: 'blue', delay: 0.1 },
    ],
    edges: [
      [0, 1],
      [1, 2],
    ],
  },
  {
    id: 'O1',
    glyph: 'o',
    cx: 147,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 145, y: 146, z: 0.9, mag: 2, tint: 'warm', delay: 1.8 },
      { x: 174, y: 194, z: 0.52, mag: 3, tint: 'white', delay: 0.7 },
      { x: 118, y: 184, z: 0.28, mag: 3, tint: 'blue', delay: 4.0 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 0],
    ],
  },
  {
    id: 'N1',
    glyph: 'N',
    cx: 227,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 175, y: 136, z: 0.96, mag: 1, tint: 'warm', delay: 0.4, label: 'δ' },
      { x: 201, y: 210, z: 0.62, mag: 3, tint: 'white', delay: 2.9 },
      { x: 253, y: 149, z: 0.78, mag: 3, tint: 'blue', delay: 3.5 },
      { x: 282, y: 223, z: 0.94, mag: 1, tint: 'warm', delay: 0.7 },
    ],
    edges: [
      [1, 0],
      [0, 3],
      [3, 2],
    ],
  },
  {
    id: 'N2',
    glyph: 'n',
    cx: 317,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 286, y: 210, z: 0.3, mag: 3, tint: 'blue', delay: 1.1 },
      { x: 290, y: 147, z: 0.72, mag: 3, tint: 'white', delay: 2.6 },
      { x: 347, y: 210, z: 0.9, mag: 2, tint: 'warm', delay: 0.2 },
      { x: 346, y: 148, z: 0.46, mag: 3, tint: 'blue', delay: 3.8 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  },
  {
    id: 'E2',
    glyph: 'e',
    cx: 392,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 373, y: 147, z: 0.84, mag: 2, tint: 'white', delay: 2.0 },
      { x: 374, y: 179, z: 0.54, mag: 3, tint: 'blue', delay: 3.1 },
      { x: 416, y: 204, z: 0.92, mag: 2, tint: 'warm', delay: 0.9 },
    ],
    edges: [
      [0, 1],
      [1, 2],
    ],
  },
  {
    id: 'C2',
    glyph: 'c',
    cx: 464,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 462, y: 146, z: 0.86, mag: 3, tint: 'warm', delay: 1.0 },
      { x: 436, y: 180, z: 0.56, mag: 3, tint: 'blue', delay: 2.4 },
      { x: 491, y: 200, z: 0.34, mag: 3, tint: 'white', delay: 3.9 },
    ],
    edges: [
      [0, 1],
      [1, 2],
    ],
  },
  {
    id: 'T2',
    glyph: 'T',
    cx: 542,
    baselineY: 208,
    fontSize: 82,
    stars: [
      { x: 507, y: 162, z: 0.72, mag: 2, tint: 'blue', delay: 1.6 },
      { x: 541, y: 150, z: 0.36, mag: 3, tint: 'white', delay: 3.0 },
      { x: 571, y: 144, z: 0.98, mag: 1, tint: 'warm', delay: 0.3, label: 'ε' },
      { x: 541, y: 211, z: 0.82, mag: 3, tint: 'white', delay: 2.2 },
    ],
    edges: [
      [0, 1],
      [1, 2],
      [1, 3],
    ],
  },
]

// Quiet distant companion field stars gently framing the wordmark without clutter.
export const COMPANION_STARS: readonly ConstellationStar[] = [
  { x: 74, y: 46, z: 0.34, mag: 3, tint: 'warm', delay: 2.1 },
  { x: 146, y: 24, z: 0.22, mag: 3, tint: 'white', delay: 3.4 },
  { x: 282, y: 66, z: 0.26, mag: 3, tint: 'blue', delay: 4.0 },
  { x: 12, y: 196, z: 0.2, mag: 3, tint: 'white', delay: 3.1 },
  { x: 428, y: 160, z: 0.24, mag: 3, tint: 'blue', delay: 2.5 },
]

// Two whisper-faint inter-word sightlines tying EARTH to CONNECT through the flourishes
// (E's lower foot -> C's high crown, and R's plunging tail -> N's upper anchor).
export const CROSS_SIGHTLINES: readonly (readonly [
  { letter: number; star: number } | { companion: number },
  { letter: number; star: number } | { companion: number },
])[] = [
  [
    { letter: 0, star: 3 },
    { letter: 5, star: 0 },
  ],
  [
    { letter: 2, star: 3 },
    { letter: 7, star: 2 },
  ],
]

export type BayerAnchor = {
  id: string
  label: string
  x: number
  y: number
  z: number
  letterIndex: number | null
}

export const BAYER_ANCHORS: readonly BayerAnchor[] = [
  ...CONSTELLATION_LETTERS.flatMap((letter, letterIndex) =>
    letter.stars
      .filter(star => Boolean(star.label))
      .map((star, idx) => ({
        id: `${letter.id}-bayer-${idx}`,
        label: star.label!,
        x: star.x,
        y: star.y,
        z: star.z,
        letterIndex,
      })),
  ),
  ...COMPANION_STARS.filter(star => Boolean(star.label)).map((star, idx) => ({
    id: `comp-bayer-${idx}`,
    label: star.label!,
    x: star.x,
    y: star.y,
    z: star.z,
    letterIndex: null,
  })),
]

export function projectStar(
  star: { x: number; y: number; z: number },
  parallax: { x: number; y: number },
  maxShiftX = MAX_PARALLAX_X,
  maxShiftY = MAX_PARALLAX_Y,
) {
  return {
    x: star.x - parallax.x * star.z * maxShiftX,
    y: star.y - parallax.y * star.z * maxShiftY,
  }
}

/** Trim a segment cleanly short of each star so stars float freely like a real star chart. */
export function trimmedSegment(
  a: { x: number; y: number; mag: 1 | 2 | 3 },
  b: { x: number; y: number; mag: 1 | 2 | 3 },
) {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const dist = Math.hypot(dx, dy)
  const padA = a.mag === 1 ? 5.6 : a.mag === 2 ? 4.0 : 3.0
  const padB = b.mag === 1 ? 5.6 : b.mag === 2 ? 4.0 : 3.0
  if (dist <= padA + padB + 2) return null
  const ux = dx / dist
  const uy = dy / dist
  return {
    x1: a.x + ux * padA,
    y1: a.y + uy * padA,
    x2: b.x - ux * padB,
    y2: b.y - uy * padB,
  }
}

export function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}
