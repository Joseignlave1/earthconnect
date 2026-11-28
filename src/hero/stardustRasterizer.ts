import {
  COMPANION_STARS,
  CONSTELLATION_LETTERS,
  FIELD_H,
  FIELD_PAD,
  FIELD_W,
  STAR_TINT_RGB,
} from './constellationData.ts'

export const MAX_DUST_PARTICLES = 20000
export const CTA_VIEW_W = 248
export const CTA_VIEW_H = 62
export const CTA_PAD = 24
export const CTA_FIELD_W = CTA_VIEW_W + CTA_PAD * 2
export const CTA_FIELD_H = CTA_VIEW_H + CTA_PAD * 2
export const CTA_STATE_FLOATS = 6 // CtaDustState: pos(2), settleFlash(1), releaseDelay(1), fallProgress(1), _pad(1)
const MAX_CTA_PARTICLES = 3600
const CTA_STREAM_MOTES = 150

export type CtaStar = {
  x: number
  y: number
  z: number
  mag: 1 | 2 | 3
  tint: keyof typeof STAR_TINT_RGB
  delay: number
  spiked: boolean
}

export const CTA_STARS: readonly CtaStar[] = [
  { x: 11, y: 12, z: 0.9, mag: 2, tint: 'warm', delay: 0.4, spiked: false },
  { x: 14, y: 50, z: 0.55, mag: 3, tint: 'white', delay: 1.5, spiked: false },
  { x: 184, y: 50, z: 0.48, mag: 3, tint: 'blue', delay: 2.6, spiked: false },
  { x: 230, y: 34, z: 0.96, mag: 1, tint: 'warm', delay: 0.9, spiked: true },
  { x: 217, y: 29, z: 0.72, mag: 3, tint: 'white', delay: 2.1, spiked: false },
  { x: 222, y: 46, z: 0.72, mag: 3, tint: 'white', delay: 3.1, spiked: false },
]

export const CTA_EDGES: readonly (readonly [number, number])[] = [
  [1, 2],
  [2, 3],
  [4, 3],
  [5, 3],
]

export type TitleDustAssets = {
  dustCount: number
  dustHomeData: Float32Array
  dustInitState: Float32Array
}

// Exact linear-time 1D squared Euclidean Distance Transform (Felzenszwalb & Huttenlocher).
const INF = 1e20

function edt1d(f: Float32Array, n: number, out: Float32Array, v: Int32Array, z: Float32Array) {
  let k = 0
  v[0] = 0
  z[0] = -INF
  z[1] = INF
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    while (s <= z[k]) {
      k--
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    }
    k++
    v[k] = q
    z[k] = s
    z[k + 1] = INF
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++
    const dx = q - v[k]
    out[q] = dx * dx + f[v[k]]
  }
}

function edt2d(grid: Float32Array, width: number, height: number): Float32Array {
  const maxDim = Math.max(width, height)
  const f = new Float32Array(maxDim)
  const out = new Float32Array(maxDim)
  const v = new Int32Array(maxDim)
  const z = new Float32Array(maxDim + 1)
  const tmp = new Float32Array(width * height)
  const res = new Float32Array(width * height)

  for (let y = 0; y < height; y++) {
    const row = y * width
    for (let x = 0; x < width; x++) f[x] = grid[row + x]
    edt1d(f, width, out, v, z)
    for (let x = 0; x < width; x++) tmp[row + x] = out[x]
  }
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = tmp[y * width + x]
    edt1d(f, height, out, v, z)
    for (let y = 0; y < height; y++) res[y * width + x] = Math.sqrt(out[y])
  }
  return res
}

function sampleBilinear(field: Float32Array, width: number, height: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(width - 1.0001, x))
  const cy = Math.max(0, Math.min(height - 1.0001, y))
  const x0 = Math.floor(cx)
  const y0 = Math.floor(cy)
  const x1 = Math.min(width - 1, x0 + 1)
  const y1 = Math.min(height - 1, y0 + 1)
  const tx = cx - x0
  const ty = cy - y0
  const a = field[y0 * width + x0]
  const b = field[y0 * width + x1]
  const c = field[y1 * width + x0]
  const d = field[y1 * width + x1]
  return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty
}

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function buildSignedDistanceField(
  fillPixels: Uint8ClampedArray,
  fieldW: number,
  fieldH: number,
  supersample: number,
  sampleMode: 'average2x2' | 'center',
): Float32Array {
  const texW = fieldW * supersample
  const outsideSeed = new Float32Array(fieldW * fieldH)
  const insideSeed = new Float32Array(fieldW * fieldH)
  const centerOff = Math.floor(supersample / 2)

  for (let y = 0; y < fieldH; y++) {
    for (let x = 0; x < fieldW; x++) {
      const sx = x * supersample
      const sy = y * supersample
      let fAlpha: number
      if (sampleMode === 'average2x2') {
        const i00 = (sy * texW + sx) * 4 + 3
        const i10 = (sy * texW + sx + 1) * 4 + 3
        const i01 = ((sy + 1) * texW + sx) * 4 + 3
        const i11 = ((sy + 1) * texW + sx + 1) * 4 + 3
        fAlpha = (fillPixels[i00] + fillPixels[i10] + fillPixels[i01] + fillPixels[i11]) * 0.25
      } else {
        fAlpha = fillPixels[((sy + centerOff) * texW + sx + centerOff) * 4 + 3]
      }
      const idx = y * fieldW + x
      outsideSeed[idx] = fAlpha >= 128 ? 0 : INF
      insideSeed[idx] = fAlpha < 128 ? 0 : INF
    }
  }

  const dOut = edt2d(outsideSeed, fieldW, fieldH)
  const dIn = edt2d(insideSeed, fieldW, fieldH)
  const signedDist = new Float32Array(fieldW * fieldH)
  for (let i = 0; i < signedDist.length; i++) signedDist[i] = dOut[i] - dIn[i]
  return signedDist
}

type CandidateColumns = {
  x: number[]
  y: number[]
  weight: number[]
}

function createCandidates(): CandidateColumns {
  return { x: [], y: [], weight: [] }
}

function sampleBlueNoiseCandidates(
  primaryCand: CandidateColumns,
  secondaryCand: CandidateColumns,
  opts: {
    fieldW: number
    fieldH: number
    pad: number
    supersample: number
    minDistView: number
    jitter: number
    maxPoints: number
    rand: () => number
  },
): CandidateColumns {
  const { fieldW, fieldH, pad, supersample, minDistView, jitter, maxPoints, rand } = opts
  const pCount = primaryCand.x.length
  const sCount = secondaryCand.x.length
  if (pCount + sCount === 0 || maxPoints <= 0) return createCandidates()

  const shuffleIndices = (n: number) => {
    const arr = new Int32Array(n)
    for (let i = 0; i < n; i++) arr[i] = i
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1))
      const tmp = arr[i]
      arr[i] = arr[j]
      arr[j] = tmp
    }
    return arr
  }
  const pOrder = shuffleIndices(pCount)
  const sOrder = sCount > 0 ? shuffleIndices(sCount) : new Int32Array(0)

  const cell = minDistView / Math.SQRT2
  const gridW = Math.ceil(fieldW / cell)
  const gridH = Math.ceil(fieldH / cell)
  const grid = new Int32Array(gridW * gridH)
  grid.fill(-1)

  const acceptedX: number[] = []
  const acceptedY: number[] = []
  const acceptedW: number[] = []
  const minDist2 = minDistView * minDistView

  const visitTier = (cand: CandidateColumns, order: Int32Array) => {
    for (let k = 0; k < order.length && acceptedX.length < maxPoints; k++) {
      const idx = order[k]
      const fx = (cand.x[idx] + 0.5) / supersample + (rand() - 0.5) * jitter
      const fy = (cand.y[idx] + 0.5) / supersample + (rand() - 0.5) * jitter
      const gx = Math.floor(fx / cell)
      const gy = Math.floor(fy / cell)
      if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) continue

      let blocked = false
      const yMin = Math.max(0, gy - 2)
      const yMax = Math.min(gridH - 1, gy + 2)
      const xMin = Math.max(0, gx - 2)
      const xMax = Math.min(gridW - 1, gx + 2)
      for (let ny = yMin; ny <= yMax && !blocked; ny++) {
        const row = ny * gridW
        for (let nx = xMin; nx <= xMax; nx++) {
          const hit = grid[row + nx]
          if (hit >= 0) {
            const dx = fx - acceptedX[hit]
            const dy = fy - acceptedY[hit]
            if (dx * dx + dy * dy < minDist2) {
              blocked = true
              break
            }
          }
        }
      }
      if (blocked) continue

      grid[gy * gridW + gx] = acceptedX.length
      acceptedX.push(fx)
      acceptedY.push(fy)
      acceptedW.push(cand.weight[idx])
    }
  }

  visitTier(primaryCand, pOrder)
  visitTier(secondaryCand, sOrder)

  for (let i = 0; i < acceptedX.length; i++) {
    acceptedX[i] -= pad
    acceptedY[i] -= pad
  }
  return { x: acceptedX, y: acceptedY, weight: acceptedW }
}

type ShadingStar = {
  x: number
  y: number
  mag: 1 | 2 | 3
  tint: keyof typeof STAR_TINT_RGB
}

function computeStardustColor(
  vx: number,
  vy: number,
  coverage: number,
  signedDist: Float32Array,
  edtW: number,
  edtH: number,
  pad: number,
  stars: readonly ShadingStar[],
  rand: () => number,
  cfg: {
    keyMin: number
    keyRange: number
    maxStarDist2: number
    heroWeight: number
    midWeight: number
    faintWeight: number
    heroRadius: number
    otherRadius: number
    baseMinRgb: readonly [number, number, number]
    baseMaxRgb: readonly [number, number, number]
    baseGain: number
    starGain: number
    covMin: number
  },
): [number, number, number] {
  const gx = Math.max(1, Math.min(edtW - 2, Math.round(vx + pad)))
  const gy = Math.max(1, Math.min(edtH - 2, Math.round(vy + pad)))
  const ddx = signedDist[gy * edtW + gx + 1] - signedDist[gy * edtW + gx - 1]
  const ddy = signedDist[(gy + 1) * edtW + gx] - signedDist[(gy - 1) * edtW + gx]
  const nLen = Math.hypot(ddx, ddy)
  const nx = nLen > 1e-4 ? ddx / nLen : 0
  const ny = nLen > 1e-4 ? ddy / nLen : -1
  const keyDot = nx * -0.48 + ny * -0.88
  const keyShade = cfg.keyMin + cfg.keyRange * Math.max(0, Math.min(1, (keyDot + 0.75) / 1.6))

  let sr = 0
  let sg = 0
  let sb = 0
  for (const st of stars) {
    const dx = vx - st.x
    const dy = vy - st.y
    const d2 = dx * dx + dy * dy
    if (d2 > cfg.maxStarDist2) continue
    const isHero = st.mag === 1
    const w = isHero ? cfg.heroWeight : st.mag === 2 ? cfg.midWeight : cfg.faintWeight
    const rT = isHero ? cfg.heroRadius : cfg.otherRadius
    const tight = Math.exp(-d2 / (2 * rT * rT)) * w
    const rgb = STAR_TINT_RGB[st.tint]
    sr += rgb[0] * tight
    sg += rgb[1] * tight
    sb += rgb[2] * tight
  }

  const grainVar = 0.92 + rand() * 0.16
  const cov = cfg.covMin + (1 - cfg.covMin) * coverage
  const shadeScale = cfg.baseGain * keyShade * grainVar
  return [
    ((cfg.baseMinRgb[0] + (cfg.baseMaxRgb[0] - cfg.baseMinRgb[0]) * keyShade) * shadeScale + sr * cfg.starGain) * cov,
    ((cfg.baseMinRgb[1] + (cfg.baseMaxRgb[1] - cfg.baseMinRgb[1]) * keyShade) * shadeScale + sg * cfg.starGain) * cov,
    ((cfg.baseMinRgb[2] + (cfg.baseMaxRgb[2] - cfg.baseMinRgb[2]) * keyShade) * shadeScale + sb * cfg.starGain) * cov,
  ]
}

async function buildDustAssets(): Promise<TitleDustAssets> {
  try {
    await Promise.all([
      document.fonts.load('400 96px "Cinzel Decorative"'),
      document.fonts.load('400 82px "Cinzel Decorative"'),
    ])
  } catch {
    // Proceed with fallback serif if webfont fails to load.
  }

  const supersample = 2
  const texW = FIELD_W * supersample
  const texH = FIELD_H * supersample

  const strokeCanvas = document.createElement('canvas')
  strokeCanvas.width = texW
  strokeCanvas.height = texH
  const sCtx = strokeCanvas.getContext('2d')!
  sCtx.scale(supersample, supersample)
  sCtx.translate(FIELD_PAD, FIELD_PAD)
  sCtx.textAlign = 'center'
  sCtx.textBaseline = 'alphabetic'
  sCtx.strokeStyle = '#ffffff'
  sCtx.lineWidth = 1.08
  sCtx.lineJoin = 'round'
  sCtx.lineCap = 'round'

  const fillCanvas = document.createElement('canvas')
  fillCanvas.width = texW
  fillCanvas.height = texH
  const fCtx = fillCanvas.getContext('2d')!
  fCtx.scale(supersample, supersample)
  fCtx.translate(FIELD_PAD, FIELD_PAD)
  fCtx.textAlign = 'center'
  fCtx.textBaseline = 'alphabetic'
  fCtx.fillStyle = '#ffffff'

  for (const letter of CONSTELLATION_LETTERS) {
    const font = `400 ${letter.fontSize}px "Cinzel Decorative", Georgia, serif`
    sCtx.font = font
    sCtx.strokeText(letter.glyph, letter.cx, letter.baselineY)
    fCtx.font = font
    fCtx.fillText(letter.glyph, letter.cx, letter.baselineY)
  }

  const strokePixels = sCtx.getImageData(0, 0, texW, texH).data
  const fillPixels = fCtx.getImageData(0, 0, texW, texH).data
  const signedDist = buildSignedDistanceField(fillPixels, FIELD_W, FIELD_H, supersample, 'average2x2')

  const coreOutlineCand = createCandidates()
  const edgeOutlineCand = createCandidates()
  const bevelCand = createCandidates()

  for (let y = 0; y < texH; y++) {
    const ey = (y + 0.5) / supersample - 0.5
    for (let x = 0; x < texW; x++) {
      const ex = (x + 0.5) / supersample - 0.5
      const pIdx = (y * texW + x) * 4
      const sA = strokePixels[pIdx + 3]
      const fA = fillPixels[pIdx + 3]
      if (sA >= 160) {
        coreOutlineCand.x.push(x)
        coreOutlineCand.y.push(y)
        coreOutlineCand.weight.push(Math.min(1, sA / 235))
      } else if (sA > 78) {
        edgeOutlineCand.x.push(x)
        edgeOutlineCand.y.push(y)
        edgeOutlineCand.weight.push((sA - 60) / 195)
      } else if (fA > 150) {
        const sd = sampleBilinear(signedDist, FIELD_W, FIELD_H, ex, ey)
        if (sd < -0.35 && sd > -3.1) {
          bevelCand.x.push(x)
          bevelCand.y.push(y)
          bevelCand.weight.push(Math.exp(sd / 1.9))
        }
      }
    }
  }

  const rand = mulberry32(0x9e3779b9)
  const targetOutline = Math.floor(MAX_DUST_PARTICLES * 0.82)
  const outlinePicked = sampleBlueNoiseCandidates(coreOutlineCand, edgeOutlineCand, {
    fieldW: FIELD_W,
    fieldH: FIELD_H,
    pad: FIELD_PAD,
    supersample,
    minDistView: 0.58,
    jitter: 0.22,
    maxPoints: targetOutline,
    rand,
  })
  const bevelPicked = sampleBlueNoiseCandidates(bevelCand, createCandidates(), {
    fieldW: FIELD_W,
    fieldH: FIELD_H,
    pad: FIELD_PAD,
    supersample,
    minDistView: 1.08,
    jitter: 0.22,
    maxPoints: MAX_DUST_PARTICLES - outlinePicked.x.length,
    rand,
  })

  const allStars = [...CONSTELLATION_LETTERS.flatMap(l => l.stars), ...COMPANION_STARS]
  const oPickedCount = outlinePicked.x.length
  const bPickedCount = bevelPicked.x.length
  const dustCount = Math.max(1, oPickedCount + bPickedCount)

  const dustHomeData = new Float32Array(dustCount * 8)
  const dustInitState = new Float32Array(dustCount * 8)
  const titleShadeCfg = {
    keyMin: 0.58,
    keyRange: 0.42,
    maxStarDist2: 3200,
    heroWeight: 1.0,
    midWeight: 0.48,
    faintWeight: 0.18,
    heroRadius: 16.5,
    otherRadius: 9.5,
    baseMinRgb: [0.62, 0.57, 0.52] as const,
    baseMaxRgb: [0.9, 0.8, 0.66] as const,
    baseGain: 0.45,
    starGain: 0.56,
    covMin: 0.42,
  }

  let outIdx = 0
  const emitParticle = (vx: number, vy: number, coverage: number, kind: 0 | 1) => {
    const [baseR, baseG, baseB] = computeStardustColor(
      vx,
      vy,
      coverage,
      signedDist,
      FIELD_W,
      FIELD_H,
      FIELD_PAD,
      allStars,
      rand,
      titleShadeCfg,
    )
    const hOff = outIdx * 8
    dustHomeData[hOff] = vx
    dustHomeData[hOff + 1] = vy
    dustHomeData[hOff + 2] = kind === 0 ? 0.58 + rand() * 0.18 : 0.56 + rand() * 0.22
    dustHomeData[hOff + 3] = rand()
    dustHomeData[hOff + 4] = baseR
    dustHomeData[hOff + 5] = baseG
    dustHomeData[hOff + 6] = baseB
    dustHomeData[hOff + 7] = kind

    const sOff = outIdx * 8
    dustInitState[sOff] = vx
    dustInitState[sOff + 1] = vy
    outIdx++
  }

  for (let i = 0; i < oPickedCount; i++) {
    emitParticle(outlinePicked.x[i], outlinePicked.y[i], outlinePicked.weight[i], 0)
  }
  for (let i = 0; i < bPickedCount; i++) {
    emitParticle(bevelPicked.x[i], bevelPicked.y[i], bevelPicked.weight[i], 1)
  }

  return { dustCount, dustHomeData, dustInitState }
}

async function buildCtaDustAssets(): Promise<TitleDustAssets> {
  try {
    await Promise.all([
      document.fonts.load('400 36px "Cinzel Decorative"'),
      document.fonts.load('400 30px "Cinzel Decorative"'),
    ])
  } catch {
    // Proceed with fallback serif if webfont fails to load.
  }

  const supersample = 3
  const texW = CTA_FIELD_W * supersample
  const texH = CTA_FIELD_H * supersample

  const strokeCanvas = document.createElement('canvas')
  strokeCanvas.width = texW
  strokeCanvas.height = texH
  const sCtx = strokeCanvas.getContext('2d')!
  sCtx.scale(supersample, supersample)
  sCtx.translate(CTA_PAD, CTA_PAD)
  sCtx.textAlign = 'left'
  sCtx.textBaseline = 'alphabetic'
  sCtx.strokeStyle = '#ffffff'
  sCtx.lineWidth = 0.72
  sCtx.lineJoin = 'round'
  sCtx.lineCap = 'round'

  const fillCanvas = document.createElement('canvas')
  fillCanvas.width = texW
  fillCanvas.height = texH
  const fCtx = fillCanvas.getContext('2d')!
  fCtx.scale(supersample, supersample)
  fCtx.translate(CTA_PAD, CTA_PAD)
  fCtx.textAlign = 'left'
  fCtx.textBaseline = 'alphabetic'
  fCtx.fillStyle = '#ffffff'

  const letters: readonly { glyph: string; size: number }[] = [
    { glyph: 'E', size: 36 },
    { glyph: 'x', size: 30 },
    { glyph: 'p', size: 30 },
    { glyph: 'l', size: 30 },
    { glyph: 'o', size: 30 },
    { glyph: 'r', size: 30 },
    { glyph: 'e', size: 30 },
  ]
  const tracking = 5.2
  const baselineY = 41
  let cursorX = 14
  for (const item of letters) {
    const font = `400 ${item.size}px "Cinzel Decorative", Georgia, serif`
    sCtx.font = font
    fCtx.font = font
    sCtx.strokeText(item.glyph, cursorX, baselineY)
    fCtx.fillText(item.glyph, cursorX, baselineY)
    cursorX += sCtx.measureText(item.glyph).width + tracking
  }

  const strokePixels = sCtx.getImageData(0, 0, texW, texH).data
  const fillPixels = fCtx.getImageData(0, 0, texW, texH).data
  const signedDist = buildSignedDistanceField(fillPixels, CTA_FIELD_W, CTA_FIELD_H, supersample, 'center')

  const coreCand = createCandidates()
  const edgeCand = createCandidates()
  for (let y = 0; y < texH; y++) {
    for (let x = 0; x < texW; x++) {
      const sA = strokePixels[(y * texW + x) * 4 + 3]
      if (sA >= 155) {
        coreCand.x.push(x)
        coreCand.y.push(y)
        coreCand.weight.push(Math.min(1, sA / 235))
      } else if (sA > 72) {
        edgeCand.x.push(x)
        edgeCand.y.push(y)
        edgeCand.weight.push((sA - 55) / 195)
      }
    }
  }

  const rand = mulberry32(0x517cc1b7)
  const picked = sampleBlueNoiseCandidates(coreCand, edgeCand, {
    fieldW: CTA_FIELD_W,
    fieldH: CTA_FIELD_H,
    pad: CTA_PAD,
    supersample,
    minDistView: 0.44,
    jitter: 0.14,
    maxPoints: MAX_CTA_PARTICLES,
    rand,
  })
  const pickedCount = picked.x.length
  const letterCount = Math.max(1, pickedCount)
  const streamCount = pickedCount > 0 ? Math.min(CTA_STREAM_MOTES, pickedCount) : 0
  const dustCount = letterCount + streamCount
  const dustHomeData = new Float32Array(dustCount * 8)
  const dustInitState = new Float32Array(dustCount * CTA_STATE_FLOATS)
  const ctaShadeCfg = {
    keyMin: 0.62,
    keyRange: 0.38,
    maxStarDist2: 2400,
    heroWeight: 0.9,
    midWeight: 0.45,
    faintWeight: 0.16,
    heroRadius: 18,
    otherRadius: 10,
    baseMinRgb: [0.64, 0.59, 0.54] as const,
    baseMaxRgb: [0.92, 0.82, 0.68] as const,
    baseGain: 0.46,
    starGain: 0.48,
    covMin: 0.45,
  }

  let minY = Infinity
  let maxY = -Infinity
  for (let i = 0; i < pickedCount; i++) {
    const vx = picked.x[i]
    const vy = picked.y[i]
    const cov = picked.weight[i]
    if (vy < minY) minY = vy
    if (vy > maxY) maxY = vy

    const [baseR, baseG, baseB] = computeStardustColor(
      vx,
      vy,
      cov,
      signedDist,
      CTA_FIELD_W,
      CTA_FIELD_H,
      CTA_PAD,
      CTA_STARS,
      rand,
      ctaShadeCfg,
    )

    const hOff = i * 8
    dustHomeData[hOff] = vx
    dustHomeData[hOff + 1] = vy
    dustHomeData[hOff + 2] = 0.46 + rand() * 0.14
    dustHomeData[hOff + 3] = rand()
    dustHomeData[hOff + 4] = baseR
    dustHomeData[hOff + 5] = baseG
    dustHomeData[hOff + 6] = baseB
    dustHomeData[hOff + 7] = 0

    const sOff = i * CTA_STATE_FLOATS
    dustInitState[sOff] = vx
    dustInitState[sOff + 1] = vy
  }

  const ySpan = Math.max(1, maxY - minY)
  const lowerPool: number[] = []
  for (let i = 0; i < pickedCount; i++) {
    const ny = (picked.y[i] - minY) / ySpan
    if (ny >= 0.48) lowerPool.push(i)
  }
  const srcPool = lowerPool.length >= streamCount ? lowerPool : null

  for (let m = 0; m < streamCount; m++) {
    const srcIdx = srcPool
      ? srcPool[Math.floor((m / streamCount) * srcPool.length)]
      : Math.floor((m / streamCount) * pickedCount)
    const vx = picked.x[srcIdx]
    const vy = picked.y[srcIdx]
    const seed = rand()
    const age = (m + rand()) / streamCount

    const idx = letterCount + m
    const hOff = idx * 8
    dustHomeData[hOff] = vx
    dustHomeData[hOff + 1] = vy
    dustHomeData[hOff + 2] = 0.34 + rand() * 0.16
    dustHomeData[hOff + 3] = seed
    dustHomeData[hOff + 4] = 0.92
    dustHomeData[hOff + 5] = 0.86
    dustHomeData[hOff + 6] = 0.74
    dustHomeData[hOff + 7] = 1

    const sOff = idx * CTA_STATE_FLOATS
    dustInitState[sOff] = vx
    dustInitState[sOff + 1] = vy
    dustInitState[sOff + 4] = age
  }

  return { dustCount, dustHomeData, dustInitState }
}

let cachedTitleDustAssets: Promise<TitleDustAssets> | undefined
let cachedCtaDustAssets: Promise<TitleDustAssets> | undefined

export function getTitleDustAssets(): Promise<TitleDustAssets> {
  if (!cachedTitleDustAssets) {
    cachedTitleDustAssets = buildDustAssets().catch(err => {
      cachedTitleDustAssets = undefined
      throw err
    })
  }
  return cachedTitleDustAssets
}

export function getCtaDustAssets(): Promise<TitleDustAssets> {
  if (!cachedCtaDustAssets) {
    cachedCtaDustAssets = buildCtaDustAssets().catch(err => {
      cachedCtaDustAssets = undefined
      throw err
    })
  }
  return cachedCtaDustAssets
}
