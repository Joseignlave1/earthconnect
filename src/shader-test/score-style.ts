// Manual mapping reserves most of the palette for the upper numeric band.
// Both manual and automatic styles are independent of the visible viewport.
export const HIGHLIGHT_START = 0.22
export const HIGHLIGHT_FULL = 0.32

export function normalizedScore(score: number, cutoff: number, best: number) {
  return best > cutoff ? Math.max(0, Math.min(1, (score - cutoff) / (best - cutoff))) : 0
}

export function palettePosition(normalized: number, topRange: number, contrast: number) {
  const value = Math.max(0, Math.min(1, normalized)),
    band = Math.max(0.01, Math.min(1, topRange))
  const knee = 1 - band,
    context = 0.25 * knee
  return value < knee
    ? (context * value) / knee
    : context + (1 - context) * Math.max(0, Math.min(1, (value - knee) / band)) ** contrast
}

const SCALE = 65534
const clamp = (value: number) => Math.max(0, Math.min(1, value))
export const AUTO_POSITIONS = [0, 0.22, 0.7, 0.9, 1] as const
export type ColorMode = 'auto' | 'manual' | 'locked'
export type StyleControls = { colorMode: ColorMode; base: number; focus: number; topRange: number; emphasis: number }
export type ScoreStyle = {
  kind: 'auto' | 'manual'
  cutoff: number
  best: number
  anchors: [number, number, number, number]
  topRange: number
  contrast: number
  eligible: number
  // A query-wide budget limits glow when many cells tie at the top.
  emissionScale: number
}

// Counts come only from level zero of the equal-area native grid. Missing cells
// never enter the histogram; zero remains a valid score.
export class ScoreDistribution {
  readonly cumulative = new Uint32Array(SCALE + 1)
  readonly best: number
  readonly valid: number
  readonly counts: Uint32Array
  constructor(counts: Uint32Array) {
    this.counts = counts
    let total = 0,
      best = 0
    for (let code = 0; code <= SCALE; code++) {
      total += counts[code] ?? 0
      this.cumulative[code] = total
      if (counts[code]) best = code
    }
    this.best = best / SCALE
    this.valid = total
  }
  private below(cutoff: number) {
    const code = Math.floor(cutoff * SCALE)
    return code < 0 ? 0 : this.cumulative[Math.min(SCALE, code)]
  }
  candidates(cutoff: number) {
    return this.valid - this.below(cutoff)
  }
  // The display filter is inclusive and independent of calibration membership.
  atLeast(minimum: number) {
    const below = Math.ceil(minimum * SCALE) - 1
    return this.valid - (below < 0 ? 0 : this.cumulative[Math.min(SCALE, below)])
  }
  quantile(cutoff: number, fraction: number) {
    const below = this.below(cutoff),
      count = this.valid - below
    if (!count) return cutoff
    const rank = below + Math.max(1, Math.ceil(clamp(fraction) * count))
    let low = Math.max(0, Math.floor(cutoff * SCALE) + 1),
      high = Math.round(this.best * SCALE)
    while (low < high) {
      const middle = (low + high) >>> 1
      if (this.cumulative[middle] >= rank) high = middle
      else low = middle + 1
    }
    return low / SCALE
  }
}

export function scorePalette(score: number, style: ScoreStyle) {
  if (score <= style.cutoff || style.best <= style.cutoff) return 0
  if (style.kind === 'manual')
    return palettePosition(normalizedScore(score, style.cutoff, style.best), style.topRange, style.contrast)
  let prev = style.cutoff
  for (let i = 0; i < 4; i++) {
    const curr = style.anchors[i]
    if (score < curr) {
      const t = clamp((score - prev) / (curr - prev))
      return AUTO_POSITIONS[i] + (AUTO_POSITIONS[i + 1] - AUTO_POSITIONS[i]) * t
    }
    prev = curr
  }
  return 1
}

// The legend labels equal positions along the palette with their actual scores.
export function scoreAtPalette(position: number, style: ScoreStyle) {
  const value = clamp(position)
  if (style.best <= style.cutoff) return style.cutoff
  if (style.kind === 'manual') {
    const knee = 1 - style.topRange,
      context = 0.25 * knee
    const normalized =
      value < context
        ? value / 0.25
        : knee + style.topRange * ((value - context) / (1 - context)) ** (1 / style.contrast)
    return style.cutoff + normalized * (style.best - style.cutoff)
  }
  let prev = style.cutoff
  for (let i = 0; i < 4; i++) {
    const curr = style.anchors[i]
    if (value <= AUTO_POSITIONS[i + 1]) {
      return prev + ((curr - prev) * (value - AUTO_POSITIONS[i])) / (AUTO_POSITIONS[i + 1] - AUTO_POSITIONS[i])
    }
    prev = curr
  }
  return style.best
}

export function resolveScoreStyle(
  distribution: ScoreDistribution | null,
  options: StyleControls,
  previous?: ScoreStyle,
): ScoreStyle {
  if (options.colorMode === 'locked' && previous) return previous
  const cutoff = options.base,
    best = distribution?.best ?? 1
  const style: ScoreStyle = {
    kind: options.colorMode === 'manual' ? 'manual' : 'auto',
    cutoff,
    best,
    anchors: [cutoff, cutoff, cutoff, best],
    topRange: options.topRange,
    contrast: options.emphasis,
    eligible: distribution?.candidates(cutoff) ?? 0,
    emissionScale: 1,
  }
  if (style.kind === 'manual' || !distribution || !style.eligible) return style
  const focus = Math.max(0.01, Math.min(1, options.focus))
  style.anchors = [
    distribution.quantile(cutoff, 1 - focus),
    distribution.quantile(cutoff, 1 - focus * 0.1),
    distribution.quantile(cutoff, 1 - focus * 0.01),
    best,
  ]
  // Separate coincident quantiles and cap amplification: one code step must not
  // become a large color jump. A full ramp spans at least .02 raw score unless
  // the user deliberately selects a narrower cutoff-to-best interval.
  const minimumSpan = Math.min(0.02, best - cutoff)
  for (let i = 2; i >= 0; i--)
    style.anchors[i] = Math.min(
      style.anchors[i],
      style.anchors[i + 1] - minimumSpan * (AUTO_POSITIONS[i + 2] - AUTO_POSITIONS[i + 1]),
    )
  // Boundaries from the percentile distribution are never below the cutoff.
  style.anchors = style.anchors.map((value, i) =>
    Math.max(cutoff + minimumSpan * AUTO_POSITIONS[i + 1], value),
  ) as ScoreStyle['anchors']
  // Auto glow follows the upper color tail rather than the absolute score.
  // Budget equivalent to .5% of candidates at full emission, preserving hue
  // when a broad plateau or tied maximum would otherwise become white.
  let glow = 0
  for (let code = Math.max(0, Math.floor(cutoff * SCALE) + 1); code <= SCALE; code++) {
    if (distribution.counts[code]) glow += distribution.counts[code] * scorePalette(code / SCALE, style) ** 12
  }
  style.emissionScale = glow ? Math.min(1, (0.005 * style.eligible) / glow) : 1
  return style
}
