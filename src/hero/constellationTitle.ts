import { shaderModule } from '../shader-test/gpu.ts'
import { type BlackHoleGeometry, unlensBlackHoleUv } from './blackHoleLens.ts'
import {
  COMPANION_STARS,
  CONSTELLATION_LETTERS,
  CROSS_SIGHTLINES,
  FIELD_H,
  FIELD_PAD,
  FIELD_W,
  MAX_PARALLAX_X,
  MAX_PARALLAX_Y,
  projectStar,
  STAR_TINT_RGB,
  trimmedSegment,
  VIEW_H,
  VIEW_W,
} from './constellationData.ts'
import titleWgsl from './constellationTitle.wgsl?raw'

export type TitleStageRect = {
  leftPx: number
  topPx: number
  widthPx: number
  heightPx: number
}

export const WAKE_RADIUS = 72

const VEL_W = Math.ceil(FIELD_W / 4) // 172
const VEL_H = Math.ceil(FIELD_H / 4) // 80
const MAX_DUST_PARTICLES = 20000

type TitlePipelines = {
  velCompute: GPUComputePipeline
  velLayout: GPUBindGroupLayout
  dustCompute: GPUComputePipeline
  dustLayout: GPUBindGroupLayout
  dustRender: GPURenderPipeline
  edgeRender: GPURenderPipeline
  starRender: GPURenderPipeline
  renderLayout: GPUBindGroupLayout
}

export type TitlePipelineTarget = {
  format?: GPUTextureFormat
  sampleCount?: 1 | 4
  hasDepth?: boolean
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<TitlePipelines>>>()

function getTitlePipelines(device: GPUDevice, target: TitlePipelineTarget = {}): Promise<TitlePipelines> {
  const format = target.format ?? 'rgba16float'
  const sampleCount = target.sampleCount ?? 4
  const hasDepth = target.hasDepth ?? true
  const key = `${format}:${sampleCount}:${hasDepth ? 'd' : 'nd'}`

  let byKey = pipelineCache.get(device)
  if (!byKey) {
    byKey = new Map()
    pipelineCache.set(device, byKey)
  }
  let pending = byKey.get(key)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'Hero constellation title shader', titleWgsl)

      const velLayout = device.createBindGroupLayout({
        label: 'Hero title velocity compute layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
          },
        ],
      })
      const dustLayout = device.createBindGroupLayout({
        label: 'Hero title dust compute layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        ],
      })
      const renderLayout = device.createBindGroupLayout({
        label: 'Hero title render layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'read-only-storage' },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            sampler: { type: 'filtering' },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'read-only-storage' },
          },
          { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          {
            binding: 6,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            texture: { sampleType: 'float' },
          },
        ],
      })

      const additive: GPUBlendState = {
        color: { srcFactor: 'one', dstFactor: 'one' },
        alpha: { srcFactor: 'zero', dstFactor: 'one' },
      }
      const depthStencil: GPUDepthStencilState | undefined = hasDepth
        ? {
            format: 'depth32float',
            depthWriteEnabled: false,
            depthCompare: 'always',
          }
        : undefined
      const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] })

      const [velCompute, dustCompute, dustRender, edgeRender, starRender] = await Promise.all([
        device.createComputePipelineAsync({
          label: 'Hero title fluid velocity compute',
          layout: device.createPipelineLayout({ bindGroupLayouts: [velLayout] }),
          compute: { module, entryPoint: 'velCompute' },
        }),
        device.createComputePipelineAsync({
          label: 'Hero title stardust particle compute',
          layout: device.createPipelineLayout({ bindGroupLayouts: [dustLayout] }),
          compute: { module, entryPoint: 'dustCompute' },
        }),
        device.createRenderPipelineAsync({
          label: 'Hero title stardust particle render',
          layout: renderPipelineLayout,
          multisample: { count: sampleCount },
          vertex: { module, entryPoint: 'dustVertex' },
          fragment: {
            module,
            entryPoint: 'dustFragment',
            targets: [{ format, blend: additive }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        }),
        device.createRenderPipelineAsync({
          label: 'Hero title constellation edges render',
          layout: renderPipelineLayout,
          multisample: { count: sampleCount },
          vertex: { module, entryPoint: 'edgeVertex' },
          fragment: {
            module,
            entryPoint: 'edgeFragment',
            targets: [{ format, blend: additive }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        }),
        device.createRenderPipelineAsync({
          label: 'Hero title HDR stars render',
          layout: renderPipelineLayout,
          multisample: { count: sampleCount },
          vertex: { module, entryPoint: 'starVertex' },
          fragment: {
            module,
            entryPoint: 'starFragment',
            targets: [{ format, blend: additive }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        }),
      ])

      return {
        velCompute,
        velLayout,
        dustCompute,
        dustLayout,
        dustRender,
        edgeRender,
        starRender,
        renderLayout,
      }
    })()
    byKey.set(key, pending)
    void pending.catch(() => {
      if (byKey?.get(key) === pending) byKey.delete(key)
    })
  }
  return pending
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

async function buildDustAssets(): Promise<{
  dustCount: number
  dustHomeData: Float32Array
  dustInitState: Float32Array
}> {
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

  const edtW = FIELD_W
  const edtH = FIELD_H
  const outsideSeed = new Float32Array(edtW * edtH)
  const insideSeed = new Float32Array(edtW * edtH)

  for (let y = 0; y < edtH; y++) {
    for (let x = 0; x < edtW; x++) {
      const sx = x * supersample
      const sy = y * supersample
      const i00 = (sy * texW + sx) * 4 + 3
      const i10 = (sy * texW + sx + 1) * 4 + 3
      const i01 = ((sy + 1) * texW + sx) * 4 + 3
      const i11 = ((sy + 1) * texW + sx + 1) * 4 + 3
      const fAvg = (fillPixels[i00] + fillPixels[i10] + fillPixels[i01] + fillPixels[i11]) * 0.25
      const idx = y * edtW + x
      outsideSeed[idx] = fAvg >= 128 ? 0 : INF
      insideSeed[idx] = fAvg < 128 ? 0 : INF
    }
  }

  const dOut = edt2d(outsideSeed, edtW, edtH)
  const dIn = edt2d(insideSeed, edtW, edtH)
  const signedDist = new Float32Array(edtW * edtH)
  for (let i = 0; i < signedDist.length; i++) signedDist[i] = dOut[i] - dIn[i]

  // Blue-noise stratified Poisson-disk sampling across 2x raster candidates:
  // eliminates additive overlap hot-spots & horizontal/vertical clumping so letters read crisp at rest.
  const rand = mulberry32(0x9e3779b9)

  const sampleBlueNoise = (
    primaryCand: number[], // packed triples: [px2x, py2x, weight] (prioritized first)
    secondaryCand: number[],
    minDistView: number,
    maxPoints: number,
  ): number[] => {
    const pCount = Math.floor(primaryCand.length / 3)
    const sCount = Math.floor(secondaryCand.length / 3)
    const total = pCount + sCount
    if (total === 0 || maxPoints <= 0) return []

    // Shuffle primary and secondary tiers independently so stroke cores always win priority over edge fringes.
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
    const sOrder = shuffleIndices(sCount)

    const cell = minDistView / Math.SQRT2
    const gridW = Math.ceil(FIELD_W / cell)
    const gridH = Math.ceil(FIELD_H / cell)
    const grid = new Int32Array(gridW * gridH)
    grid.fill(-1)

    const acceptedX: number[] = []
    const acceptedY: number[] = []
    const acceptedW: number[] = []
    const minDist2 = minDistView * minDistView

    const visitTier = (cand: number[], order: Int32Array) => {
      for (let k = 0; k < order.length && acceptedX.length < maxPoints; k++) {
        const base = order[k] * 3
        const fx = (cand[base] + 0.5) / supersample + (rand() - 0.5) * 0.22
        const fy = (cand[base + 1] + 0.5) / supersample + (rand() - 0.5) * 0.22
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

        const id = acceptedX.length
        grid[gy * gridW + gx] = id
        acceptedX.push(fx)
        acceptedY.push(fy)
        acceptedW.push(cand[base + 2])
      }
    }

    visitTier(primaryCand, pOrder)
    visitTier(secondaryCand, sOrder)

    const out: number[] = []
    for (let i = 0; i < acceptedX.length; i++) {
      out.push(acceptedX[i] - FIELD_PAD, acceptedY[i] - FIELD_PAD, acceptedW[i])
    }
    return out
  }

  // Collect stroke-core (high alpha) and stroke-fringe candidates separately so centerlines get first priority.
  const coreOutlineCand: number[] = []
  const edgeOutlineCand: number[] = []
  const bevelCand: number[] = []

  for (let y = 0; y < texH; y++) {
    const ey = (y + 0.5) / supersample - 0.5
    for (let x = 0; x < texW; x++) {
      const ex = (x + 0.5) / supersample - 0.5
      const pIdx = (y * texW + x) * 4
      const sA = strokePixels[pIdx + 3]
      const fA = fillPixels[pIdx + 3]
      if (sA >= 160) {
        coreOutlineCand.push(x, y, Math.min(1, sA / 235))
      } else if (sA > 78) {
        edgeOutlineCand.push(x, y, (sA - 60) / 195)
      } else if (fA > 150) {
        const sd = sampleBilinear(signedDist, edtW, edtH, ex, ey)
        if (sd < -0.35 && sd > -3.1) {
          const bevelFalloff = Math.exp(sd / 1.9)
          bevelCand.push(x, y, bevelFalloff)
        }
      }
    }
  }

  // Prioritize stroke cores before edge fringes inside the same blue-noise Poisson grid.
  const targetOutline = Math.floor(MAX_DUST_PARTICLES * 0.82)
  const outlinePicked = sampleBlueNoise(coreOutlineCand, edgeOutlineCand, 0.58, targetOutline)
  const bevelPicked = sampleBlueNoise(bevelCand, [], 1.08, MAX_DUST_PARTICLES - outlinePicked.length / 3)

  // Precompute all constellation stars so each stardust particle bakes in its sculpted starlight rest color.
  const allStars = [...CONSTELLATION_LETTERS.flatMap(l => l.stars), ...COMPANION_STARS]

  const oPickedCount = Math.floor(outlinePicked.length / 3)
  const bPickedCount = Math.floor(bevelPicked.length / 3)
  const dustCount = Math.max(1, oPickedCount + bPickedCount)

  const dustHomeData = new Float32Array(dustCount * 8)
  const dustInitState = new Float32Array(dustCount * 8) // 2 x vec4f: posZ + velZ

  let outIdx = 0
  const emitParticle = (vx: number, vy: number, coverage: number, kind: 0 | 1) => {
    // Compute smooth outward normal from signedDist for upper-left celestial key shading.
    const gx = Math.max(1, Math.min(edtW - 2, Math.round(vx + FIELD_PAD)))
    const gy = Math.max(1, Math.min(edtH - 2, Math.round(vy + FIELD_PAD)))
    const ddx = signedDist[gy * edtW + gx + 1] - signedDist[gy * edtW + gx - 1]
    const ddy = signedDist[(gy + 1) * edtW + gx] - signedDist[(gy - 1) * edtW + gx]
    const nLen = Math.hypot(ddx, ddy)
    const nx = nLen > 1e-4 ? ddx / nLen : 0
    const ny = nLen > 1e-4 ? ddy / nLen : -1
    const keyDot = nx * -0.48 + ny * -0.88
    const keyShade = 0.58 + 0.42 * Math.max(0, Math.min(1, (keyDot + 0.75) / 1.6))

    // Accumulate localized hero-star light so flourish tips glow warmly/coolly right where they touch anchor stars.
    let sr = 0
    let sg = 0
    let sb = 0
    for (const st of allStars) {
      const dx = vx - st.x
      const dy = vy - st.y
      const d2 = dx * dx + dy * dy
      if (d2 > 3200) continue
      const isHero = st.mag === 1
      const w = isHero ? 1.0 : st.mag === 2 ? 0.48 : 0.18
      const rT = isHero ? 16.5 : 9.5
      const tight = Math.exp(-d2 / (2 * rT * rT)) * w
      const rgb = STAR_TINT_RGB[st.tint]
      sr += rgb[0] * tight
      sg += rgb[1] * tight
      sb += rgb[2] * tight
    }

    const grainVar = 0.92 + rand() * 0.16
    const cov = 0.42 + 0.58 * coverage
    const baseR = ((0.62 + (0.9 - 0.62) * keyShade) * 0.45 * keyShade * grainVar + sr * 0.56) * cov
    const baseG = ((0.57 + (0.8 - 0.57) * keyShade) * 0.45 * keyShade * grainVar + sg * 0.56) * cov
    const baseB = ((0.52 + (0.66 - 0.52) * keyShade) * 0.45 * keyShade * grainVar + sb * 0.56) * cov

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
    dustInitState[sOff + 2] = 0
    dustInitState[sOff + 3] = 0
    dustInitState[sOff + 4] = 0
    dustInitState[sOff + 5] = 0
    dustInitState[sOff + 6] = 0
    dustInitState[sOff + 7] = 0
    outIdx++
  }

  for (let i = 0; i < oPickedCount; i++) {
    emitParticle(outlinePicked[i * 3], outlinePicked[i * 3 + 1], outlinePicked[i * 3 + 2], 0)
  }
  for (let i = 0; i < bPickedCount; i++) {
    emitParticle(bevelPicked[i * 3], bevelPicked[i * 3 + 1], bevelPicked[i * 3 + 2], 1)
  }

  return { dustCount, dustHomeData, dustInitState }
}

export type TitleDustAssets = {
  dustCount: number
  dustHomeData: Float32Array
  dustInitState: Float32Array
}

let cachedTitleDustAssets: Promise<TitleDustAssets> | undefined

export function getTitleDustAssets(): Promise<TitleDustAssets> {
  if (!cachedTitleDustAssets) {
    cachedTitleDustAssets = buildDustAssets().catch(err => {
      cachedTitleDustAssets = undefined
      throw err
    })
  }
  return cachedTitleDustAssets
}

const TOTAL_STARS = CONSTELLATION_LETTERS.reduce((sum, letter) => sum + letter.stars.length, 0) + COMPANION_STARS.length
const MAX_EDGES = CONSTELLATION_LETTERS.reduce((sum, letter) => sum + letter.edges.length, 0) + CROSS_SIGHTLINES.length
const EDGE_FLOATS = 8

export class ConstellationTitleOverlay {
  private readonly uniformData = new Float32Array(28) // 7 x vec4f
  private readonly uniformBuffer: GPUBuffer
  private readonly starData = new Float32Array(TOTAL_STARS * 12)
  private readonly starBuffer: GPUBuffer
  private readonly edgeData = new Float32Array(MAX_EDGES * EDGE_FLOATS)
  private readonly edgeBuffer: GPUBuffer
  private readonly dustHomeBuffer: GPUBuffer
  private readonly dustStateBuffer: GPUBuffer
  private readonly dustCount: number
  private readonly velTextures: readonly [GPUTexture, GPUTexture]
  private readonly sampler: GPUSampler

  private readonly velBindGroups: readonly [GPUBindGroup, GPUBindGroup]
  private readonly dustBindGroups: readonly [GPUBindGroup, GPUBindGroup]
  private readonly renderBindGroups: readonly [GPUBindGroup, GPUBindGroup]

  private ping = 0
  private prepared = false
  private destroyed = false

  private readonly projectedLetterStars = CONSTELLATION_LETTERS.map(letter =>
    letter.stars.map(star => ({
      x: star.x,
      y: star.y,
      z: star.z,
      mag: star.mag,
      tint: star.tint,
      delay: star.delay,
      label: star.label,
    })),
  )
  private readonly projectedCompanionStars = COMPANION_STARS.map(star => ({
    x: star.x,
    y: star.y,
    z: star.z,
    mag: star.mag,
    tint: star.tint,
    delay: star.delay,
    label: star.label,
  }))

  private pointerClient: { x: number; y: number } | null = null
  private smoothParallax = { x: 0, y: 0 }
  private prevCursorView: { x: number; y: number } | null = null
  private lastEdgeCount = 0
  private lastTime = -1

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: TitlePipelines,
    assets: {
      dustCount: number
      dustHomeData: Float32Array
      dustInitState: Float32Array
    },
  ) {
    this.dustCount = assets.dustCount

    this.uniformBuffer = device.createBuffer({
      label: 'Hero title uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.starBuffer = device.createBuffer({
      label: 'Hero title stars storage',
      size: this.starData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.edgeBuffer = device.createBuffer({
      label: 'Hero title edges storage',
      size: this.edgeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.dustHomeBuffer = device.createBuffer({
      label: 'Hero title dust home storage',
      size: assets.dustHomeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustHomeBuffer, 0, assets.dustHomeData)

    this.dustStateBuffer = device.createBuffer({
      label: 'Hero title dust state storage',
      size: assets.dustInitState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustStateBuffer, 0, assets.dustInitState)

    this.velTextures = [
      device.createTexture({
        label: 'Hero title fluid velocity A',
        size: [VEL_W, VEL_H, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }),
      device.createTexture({
        label: 'Hero title fluid velocity B',
        size: [VEL_W, VEL_H, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }),
    ]

    this.sampler = device.createSampler({
      label: 'Hero title linear sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const velViews = [this.velTextures[0].createView(), this.velTextures[1].createView()] as const

    // For ping = 0: read from 0 -> write to 1; then render reads 1.
    // For ping = 1: read from 1 -> write to 0; then render reads 0.
    this.velBindGroups = [
      device.createBindGroup({
        layout: pipelines.velLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: velViews[0] },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: velViews[1] },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.velLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: velViews[1] },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: velViews[0] },
        ],
      }),
    ]

    this.dustBindGroups = [
      device.createBindGroup({
        layout: pipelines.dustLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.dustHomeBuffer } },
          { binding: 2, resource: { buffer: this.dustStateBuffer } },
          { binding: 3, resource: velViews[1] },
          { binding: 4, resource: this.sampler },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.dustLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.dustHomeBuffer } },
          { binding: 2, resource: { buffer: this.dustStateBuffer } },
          { binding: 3, resource: velViews[0] },
          { binding: 4, resource: this.sampler },
        ],
      }),
    ]

    this.renderBindGroups = [
      device.createBindGroup({
        layout: pipelines.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.starBuffer } },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.edgeBuffer } },
          { binding: 4, resource: { buffer: this.dustHomeBuffer } },
          { binding: 5, resource: { buffer: this.dustStateBuffer } },
          { binding: 6, resource: velViews[1] },
        ],
      }),
      device.createBindGroup({
        layout: pipelines.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.starBuffer } },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.edgeBuffer } },
          { binding: 4, resource: { buffer: this.dustHomeBuffer } },
          { binding: 5, resource: { buffer: this.dustStateBuffer } },
          { binding: 6, resource: velViews[0] },
        ],
      }),
    ]
  }

  static async fromAssets(device: GPUDevice, assets: TitleDustAssets, target?: TitlePipelineTarget) {
    const pipelines = await getTitlePipelines(device, target)
    return new ConstellationTitleOverlay(device, pipelines, assets)
  }

  static async create(device: GPUDevice, target?: TitlePipelineTarget) {
    const [pipelines, assets] = await Promise.all([getTitlePipelines(device, target), getTitleDustAssets()])
    return new ConstellationTitleOverlay(device, pipelines, assets)
  }

  setPointer(clientX: number | null, clientY: number | null) {
    if (clientX === null || clientY === null) {
      this.pointerClient = null
    } else {
      this.pointerClient = { x: clientX, y: clientY }
    }
  }

  /** Run GPU compute passes (fluid velocity + 24k stardust particles) before the HDR render pass. */
  prepare(
    encoder: GPUCommandEncoder,
    stageRect: TitleStageRect,
    canvasRect: { left: number; top: number; width: number; height: number },
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    reducedMotion: boolean,
    opacity = 1,
    blackHole?: BlackHoleGeometry,
  ) {
    this.prepared = false
    if (this.destroyed || opacity <= 0.001 || stageRect.widthPx < 2 || stageRect.heightPx < 2) return

    const dt = this.lastTime < 0 ? 1 / 60 : Math.min(0.05, Math.max(0.001, timeSeconds - this.lastTime))
    this.lastTime = timeSeconds
    const parallaxLerp = reducedMotion ? 1 : 1 - Math.exp(-dt * 3.8)

    let targetParX = 0
    let targetParY = 0
    let cursorView: { x: number; y: number } | null = null

    if (this.pointerClient && canvasRect.width > 0 && canvasRect.height > 0) {
      const screenU = (this.pointerClient.x - canvasRect.left) / canvasRect.width
      const screenV = (this.pointerClient.y - canvasRect.top) / canvasRect.height
      if (!reducedMotion) {
        targetParX = screenU * 2 - 1
        targetParY = screenV * 2 - 1
      }
      // Map screen cursor through the exact inverse black hole lensing equation so hovering
      // distorted letters slashes pixel-exact under the visible cursor (and never inside the void).
      const sceneUv = blackHole ? unlensBlackHoleUv(screenU, screenV, blackHole) : { u: screenU, v: screenV }
      if (sceneUv) {
        const localPxX = sceneUv.u * width - stageRect.leftPx
        const localPxY = sceneUv.v * height - stageRect.topPx
        const viewX = (localPxX / stageRect.widthPx) * VIEW_W
        const viewY = (localPxY / stageRect.heightPx) * VIEW_H
        const margin = WAKE_RADIUS + 42
        if (viewX > -margin && viewX < VIEW_W + margin && viewY > -margin && viewY < VIEW_H + margin) {
          cursorView = { x: viewX, y: viewY }
        }
      }
    }

    this.smoothParallax.x += (targetParX - this.smoothParallax.x) * parallaxLerp
    this.smoothParallax.y += (targetParY - this.smoothParallax.y) * parallaxLerp
    const parallax = reducedMotion ? { x: 0, y: 0 } : this.smoothParallax

    let slashAx = 0
    let slashAy = 0
    let slashBx = 0
    let slashBy = 0
    let slashVx = 0
    let slashVy = 0
    let slashActive = 0
    let brushRadius = WAKE_RADIUS

    if (cursorView) {
      const prev = this.prevCursorView ?? cursorView
      slashAx = prev.x
      slashAy = prev.y
      slashBx = cursorView.x
      slashBy = cursorView.y
      const dx = slashBx - slashAx
      const dy = slashBy - slashAy
      const dist = Math.hypot(dx, dy)
      const speed = Math.min(1800, dist / dt)
      if (dist > 0.01) {
        const clamped = Math.min(720, speed)
        slashVx = (dx / dist) * clamped
        slashVy = (dy / dist) * clamped
      }
      brushRadius = WAKE_RADIUS + Math.min(18, speed * 0.022)
      slashActive = 1
      this.prevCursorView = cursorView
    } else {
      this.prevCursorView = null
    }

    // Project stars & trimmed edges into preallocated scratch records (zero per-frame JS allocations).
    let sOff = 0
    const pushStar = (star: {
      x: number
      y: number
      z: number
      mag: 1 | 2 | 3
      tint: keyof typeof STAR_TINT_RGB
      delay: number
      label?: string
    }) => {
      const rgb = STAR_TINT_RGB[star.tint]
      this.starData[sOff++] = star.x
      this.starData[sOff++] = star.y
      this.starData[sOff++] = star.z
      this.starData[sOff++] = star.mag
      this.starData[sOff++] = rgb[0]
      this.starData[sOff++] = rgb[1]
      this.starData[sOff++] = rgb[2]
      this.starData[sOff++] = 0
      this.starData[sOff++] = star.delay * 1.37
      this.starData[sOff++] = star.label ? 1 : 0
      this.starData[sOff++] = 0
      this.starData[sOff++] = 0
    }

    for (let li = 0; li < CONSTELLATION_LETTERS.length; li++) {
      const srcStars = CONSTELLATION_LETTERS[li].stars
      const dstStars = this.projectedLetterStars[li]
      for (let si = 0; si < srcStars.length; si++) {
        const pos = projectStar(srcStars[si], parallax)
        const dst = dstStars[si]
        dst.x = pos.x
        dst.y = pos.y
        pushStar(dst)
      }
    }
    for (let ci = 0; ci < COMPANION_STARS.length; ci++) {
      const pos = projectStar(COMPANION_STARS[ci], parallax)
      const dst = this.projectedCompanionStars[ci]
      dst.x = pos.x
      dst.y = pos.y
      pushStar(dst)
    }
    this.device.queue.writeBuffer(this.starBuffer, 0, this.starData)

    let eOff = 0
    let edgeCount = 0
    const pushEdge = (
      x1: number,
      y1: number,
      x2: number,
      y2: number,
      baseOp: number,
      baseWidthPx: number,
      kind: number,
    ) => {
      this.edgeData[eOff++] = x1
      this.edgeData[eOff++] = y1
      this.edgeData[eOff++] = x2
      this.edgeData[eOff++] = y2
      this.edgeData[eOff++] = baseOp
      this.edgeData[eOff++] = baseWidthPx
      this.edgeData[eOff++] = kind
      this.edgeData[eOff++] = 0
      edgeCount++
    }

    for (let li = 0; li < CONSTELLATION_LETTERS.length; li++) {
      const letter = CONSTELLATION_LETTERS[li]
      const stars = this.projectedLetterStars[li]
      for (const [i, j] of letter.edges) {
        const a = stars[i]
        const b = stars[j]
        const seg = trimmedSegment(a, b)
        if (!seg) continue
        const avgDepth = (a.z + b.z) * 0.5
        pushEdge(seg.x1, seg.y1, seg.x2, seg.y2, 0.09 + avgDepth * 0.08, (0.65 + avgDepth * 0.22) * ratio, 0)
      }
    }

    for (const [refA, refB] of CROSS_SIGHTLINES) {
      const a =
        'companion' in refA
          ? this.projectedCompanionStars[refA.companion]
          : this.projectedLetterStars[refA.letter][refA.star]
      const b =
        'companion' in refB
          ? this.projectedCompanionStars[refB.companion]
          : this.projectedLetterStars[refB.letter][refB.star]
      const seg = trimmedSegment(a, b)
      if (!seg) continue
      pushEdge(seg.x1, seg.y1, seg.x2, seg.y2, 0.1, 0.7 * ratio, 1)
    }
    this.lastEdgeCount = edgeCount
    if (edgeCount > 0) {
      this.device.queue.writeBuffer(this.edgeBuffer, 0, this.edgeData.subarray(0, edgeCount * EDGE_FLOATS))
    }

    const outlineShiftX = -parallax.x * 0.35 * MAX_PARALLAX_X
    const outlineShiftY = -parallax.y * 0.35 * MAX_PARALLAX_Y
    this.uniformData.set([width, height, ratio, opacity], 0)
    this.uniformData.set([stageRect.leftPx, stageRect.topPx, stageRect.widthPx, stageRect.heightPx], 4)
    this.uniformData.set([VIEW_W, VIEW_H, FIELD_PAD, TOTAL_STARS], 8)
    this.uniformData.set([timeSeconds, dt, outlineShiftX, outlineShiftY], 12)
    this.uniformData.set([edgeCount, this.dustCount, 0, reducedMotion ? 1 : 0], 16)
    this.uniformData.set([slashAx, slashAy, slashActive, brushRadius], 20)
    this.uniformData.set([slashBx, slashBy, slashVx, slashVy], 24)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const compute = encoder.beginComputePass({ label: 'Hero title stardust simulation' })

    // 1. Update 2D fluid velocity & slash wake field.
    compute.setPipeline(this.pipelines.velCompute)
    compute.setBindGroup(0, this.velBindGroups[this.ping])
    compute.dispatchWorkgroups(Math.ceil(VEL_W / 8), Math.ceil(VEL_H / 8))

    // 2. Advance 24k GPU stardust particles forming the typography.
    compute.setPipeline(this.pipelines.dustCompute)
    compute.setBindGroup(0, this.dustBindGroups[this.ping])
    compute.dispatchWorkgroups(Math.ceil(this.dustCount / 128))

    compute.end()
    this.prepared = true
  }

  /** Draw the 24k GPU stardust typography + underlying constellations inside the HDR pass. */
  draw(pass: GPURenderPassEncoder) {
    if (this.destroyed || !this.prepared) return
    this.prepared = false
    const bindGroup = this.renderBindGroups[this.ping]
    pass.setBindGroup(0, bindGroup)

    // 1. 24k GPU stardust particles forming the Cinzel Decorative letterforms.
    pass.setPipeline(this.pipelines.dustRender)
    pass.draw(6, this.dustCount)

    // 2. Constellation hairlines & cross-sightlines emerging cleanly inside the parted rift.
    if (this.lastEdgeCount > 0) {
      pass.setPipeline(this.pipelines.edgeRender)
      pass.draw(6, this.lastEdgeCount)
    }

    // 3. True HDR optical stars with scintillation and diffraction spikes.
    pass.setPipeline(this.pipelines.starRender)
    pass.draw(6, TOTAL_STARS)

    // Swap ping-pong index for next frame's compute pass.
    this.ping = 1 - this.ping
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.uniformBuffer.destroy()
    this.starBuffer.destroy()
    this.edgeBuffer.destroy()
    this.dustHomeBuffer.destroy()
    this.dustStateBuffer.destroy()
    this.velTextures[0].destroy()
    this.velTextures[1].destroy()
  }
}

// ============================================================================
// Companion Bottom-Right Stardust CTA ("Explore" + miniature star-chevron)
// Shares the exact same WebGPU HDR stardust pipelines & material as the title.
// ============================================================================

export const CTA_VIEW_W = 248
export const CTA_VIEW_H = 62
const CTA_PAD = 24
const CTA_FIELD_W = CTA_VIEW_W + CTA_PAD * 2
const CTA_FIELD_H = CTA_VIEW_H + CTA_PAD * 2
const MAX_CTA_PARTICLES = 3600
const CTA_STREAM_MOTES = 150

type CtaStar = {
  x: number
  y: number
  z: number
  mag: 1 | 2 | 3
  tint: keyof typeof STAR_TINT_RGB
  delay: number
  spiked: boolean
}

const CTA_STARS: readonly CtaStar[] = [
  { x: 11, y: 12, z: 0.9, mag: 2, tint: 'warm', delay: 0.4, spiked: false },
  { x: 14, y: 50, z: 0.55, mag: 3, tint: 'white', delay: 1.5, spiked: false },
  { x: 184, y: 50, z: 0.48, mag: 3, tint: 'blue', delay: 2.6, spiked: false },
  { x: 230, y: 34, z: 0.96, mag: 1, tint: 'warm', delay: 0.9, spiked: true },
  { x: 217, y: 29, z: 0.72, mag: 3, tint: 'white', delay: 2.1, spiked: false },
  { x: 222, y: 46, z: 0.72, mag: 3, tint: 'white', delay: 3.1, spiked: false },
]

const CTA_EDGES: readonly (readonly [number, number])[] = [
  [1, 2],
  [2, 3],
  [4, 3],
  [5, 3],
]

async function buildCtaDustAssets(): Promise<{
  dustCount: number
  dustHomeData: Float32Array
  dustInitState: Float32Array
}> {
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

  const edtW = CTA_FIELD_W
  const edtH = CTA_FIELD_H
  const outsideSeed = new Float32Array(edtW * edtH)
  const insideSeed = new Float32Array(edtW * edtH)
  for (let y = 0; y < edtH; y++) {
    for (let x = 0; x < edtW; x++) {
      const sx = x * supersample + 1
      const sy = y * supersample + 1
      const fA = fillPixels[(sy * texW + sx) * 4 + 3]
      const idx = y * edtW + x
      outsideSeed[idx] = fA >= 128 ? 0 : INF
      insideSeed[idx] = fA < 128 ? 0 : INF
    }
  }
  const dOut = edt2d(outsideSeed, edtW, edtH)
  const dIn = edt2d(insideSeed, edtW, edtH)
  const signedDist = new Float32Array(edtW * edtH)
  for (let i = 0; i < signedDist.length; i++) signedDist[i] = dOut[i] - dIn[i]

  const coreCand: number[] = []
  const edgeCand: number[] = []
  for (let y = 0; y < texH; y++) {
    for (let x = 0; x < texW; x++) {
      const sA = strokePixels[(y * texW + x) * 4 + 3]
      if (sA >= 155) coreCand.push(x, y, Math.min(1, sA / 235))
      else if (sA > 72) edgeCand.push(x, y, (sA - 55) / 195)
    }
  }

  const rand = mulberry32(0x517cc1b7)
  const minDistView = 0.44
  const cell = minDistView / Math.SQRT2
  const gridW = Math.ceil(CTA_FIELD_W / cell)
  const gridH = Math.ceil(CTA_FIELD_H / cell)
  const grid = new Int32Array(gridW * gridH)
  grid.fill(-1)

  const acceptedX: number[] = []
  const acceptedY: number[] = []
  const acceptedW: number[] = []
  const minDist2 = minDistView * minDistView

  const visit = (cand: number[]) => {
    const count = Math.floor(cand.length / 3)
    const order = new Int32Array(count)
    for (let i = 0; i < count; i++) order[i] = i
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1))
      const tmp = order[i]
      order[i] = order[j]
      order[j] = tmp
    }
    for (let k = 0; k < count && acceptedX.length < MAX_CTA_PARTICLES; k++) {
      const base = order[k] * 3
      const fx = (cand[base] + 0.5) / supersample + (rand() - 0.5) * 0.14
      const fy = (cand[base + 1] + 0.5) / supersample + (rand() - 0.5) * 0.14
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
      acceptedW.push(cand[base + 2])
    }
  }
  visit(coreCand)
  visit(edgeCand)

  const letterCount = Math.max(1, acceptedX.length)
  const streamCount = Math.min(CTA_STREAM_MOTES, letterCount)
  const dustCount = letterCount + streamCount
  const dustHomeData = new Float32Array(dustCount * 8)
  const dustInitState = new Float32Array(dustCount * 8)

  for (let i = 0; i < acceptedX.length; i++) {
    const vx = acceptedX[i] - CTA_PAD
    const vy = acceptedY[i] - CTA_PAD
    const gx = Math.max(1, Math.min(edtW - 2, Math.round(vx + CTA_PAD)))
    const gy = Math.max(1, Math.min(edtH - 2, Math.round(vy + CTA_PAD)))
    const ddx = signedDist[gy * edtW + gx + 1] - signedDist[gy * edtW + gx - 1]
    const ddy = signedDist[(gy + 1) * edtW + gx] - signedDist[(gy - 1) * edtW + gx]
    const nLen = Math.hypot(ddx, ddy)
    const nx = nLen > 1e-4 ? ddx / nLen : 0
    const ny = nLen > 1e-4 ? ddy / nLen : -1
    const keyDot = nx * -0.48 + ny * -0.88
    const keyShade = 0.62 + 0.38 * Math.max(0, Math.min(1, (keyDot + 0.75) / 1.6))

    let sr = 0
    let sg = 0
    let sb = 0
    for (const st of CTA_STARS) {
      const dx = vx - st.x
      const dy = vy - st.y
      const d2 = dx * dx + dy * dy
      if (d2 > 2400) continue
      const w = st.mag === 1 ? 0.9 : st.mag === 2 ? 0.45 : 0.16
      const rT = st.mag === 1 ? 18 : 10
      const tight = Math.exp(-d2 / (2 * rT * rT)) * w
      const rgb = STAR_TINT_RGB[st.tint]
      sr += rgb[0] * tight
      sg += rgb[1] * tight
      sb += rgb[2] * tight
    }

    const grainVar = 0.92 + rand() * 0.16
    const cov = 0.45 + 0.55 * acceptedW[i]
    const baseR = ((0.64 + (0.92 - 0.64) * keyShade) * 0.46 * keyShade * grainVar + sr * 0.48) * cov
    const baseG = ((0.59 + (0.82 - 0.59) * keyShade) * 0.46 * keyShade * grainVar + sg * 0.48) * cov
    const baseB = ((0.54 + (0.68 - 0.54) * keyShade) * 0.46 * keyShade * grainVar + sb * 0.48) * cov

    const hOff = i * 8
    dustHomeData[hOff] = vx
    dustHomeData[hOff + 1] = vy
    dustHomeData[hOff + 2] = 0.46 + rand() * 0.14
    dustHomeData[hOff + 3] = rand()
    dustHomeData[hOff + 4] = baseR
    dustHomeData[hOff + 5] = baseG
    dustHomeData[hOff + 6] = baseB
    dustHomeData[hOff + 7] = 0

    const sOff = i * 8
    dustInitState[sOff] = vx
    dustInitState[sOff + 1] = vy
  }

  // Sparse, delicate tidal bleed motes (kind = 1) quietly peeling mostly off the lower strokes of Explore
  // and falling radially toward the black hole center. Pre-seeded across staggered ages (0..1).
  let minY = Infinity
  let maxY = -Infinity
  for (let i = 0; i < acceptedY.length; i++) {
    if (acceptedY[i] < minY) minY = acceptedY[i]
    if (acceptedY[i] > maxY) maxY = acceptedY[i]
  }
  const ySpan = Math.max(1, maxY - minY)
  const lowerPool: number[] = []
  for (let i = 0; i < acceptedY.length; i++) {
    const ny = (acceptedY[i] - minY) / ySpan
    if (ny >= 0.48) lowerPool.push(i)
  }
  const srcPool = lowerPool.length >= streamCount ? lowerPool : null

  for (let m = 0; m < streamCount; m++) {
    const srcIdx = srcPool
      ? srcPool[Math.floor((m / streamCount) * srcPool.length)]
      : Math.floor((m / streamCount) * acceptedX.length)
    const vx = acceptedX[srcIdx] - CTA_PAD
    const vy = acceptedY[srcIdx] - CTA_PAD
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

    const sOff = idx * 8
    dustInitState[sOff] = vx
    dustInitState[sOff + 1] = vy
    dustInitState[sOff + 7] = age
  }

  return { dustCount, dustHomeData, dustInitState }
}

export class StardustCtaOverlay {
  private readonly uniformData = new Float32Array(28)
  private readonly uniformBuffer: GPUBuffer
  private readonly starData = new Float32Array(CTA_STARS.length * 12)
  private readonly starBuffer: GPUBuffer
  private readonly edgeData = new Float32Array(CTA_EDGES.length * EDGE_FLOATS)
  private readonly edgeBuffer: GPUBuffer
  private readonly dustHomeBuffer: GPUBuffer
  private readonly dustStateBuffer: GPUBuffer
  private readonly dustCount: number
  private readonly velTexture: GPUTexture
  private readonly sampler: GPUSampler

  private readonly dustBindGroup: GPUBindGroup
  private readonly renderBindGroup: GPUBindGroup

  private prepared = false
  private destroyed = false
  private pointerClient: { x: number; y: number } | null = null
  private smoothParallax = { x: 0, y: 0 }
  private lastEdgeCount = 0
  private lastTime = -1
  private hovered = false
  private focused = false
  private pressed = false
  private hoverLevel = 0
  private pressedLevel = 0

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: TitlePipelines,
    assets: { dustCount: number; dustHomeData: Float32Array; dustInitState: Float32Array },
  ) {
    this.dustCount = assets.dustCount

    this.uniformBuffer = device.createBuffer({
      label: 'Hero CTA uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.starBuffer = device.createBuffer({
      label: 'Hero CTA stars storage',
      size: this.starData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.edgeBuffer = device.createBuffer({
      label: 'Hero CTA edges storage',
      size: this.edgeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.dustHomeBuffer = device.createBuffer({
      label: 'Hero CTA dust home storage',
      size: assets.dustHomeData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustHomeBuffer, 0, assets.dustHomeData)

    this.dustStateBuffer = device.createBuffer({
      label: 'Hero CTA dust state storage',
      size: assets.dustInitState.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.dustStateBuffer, 0, assets.dustInitState)

    this.velTexture = device.createTexture({
      label: 'Hero CTA dummy velocity texture',
      size: [1, 1, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    })

    this.sampler = device.createSampler({
      label: 'Hero CTA linear sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const velView = this.velTexture.createView()
    this.dustBindGroup = device.createBindGroup({
      layout: pipelines.dustLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.dustHomeBuffer } },
        { binding: 2, resource: { buffer: this.dustStateBuffer } },
        { binding: 3, resource: velView },
        { binding: 4, resource: this.sampler },
      ],
    })
    this.renderBindGroup = device.createBindGroup({
      layout: pipelines.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.starBuffer } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: { buffer: this.edgeBuffer } },
        { binding: 4, resource: { buffer: this.dustHomeBuffer } },
        { binding: 5, resource: { buffer: this.dustStateBuffer } },
        { binding: 6, resource: velView },
      ],
    })
  }

  static async create(device: GPUDevice) {
    const [pipelines, assets] = await Promise.all([getTitlePipelines(device), buildCtaDustAssets()])
    return new StardustCtaOverlay(device, pipelines, assets)
  }

  setPointer(clientX: number | null, clientY: number | null) {
    this.pointerClient = clientX === null || clientY === null ? null : { x: clientX, y: clientY }
  }

  setHovered(hovered: boolean) {
    this.hovered = hovered
  }

  setFocused(focused: boolean) {
    this.focused = focused
  }

  setPressed(pressed: boolean) {
    this.pressed = pressed
  }

  prepare(
    encoder: GPUCommandEncoder,
    stageRect: TitleStageRect,
    canvasRect: { left: number; top: number; width: number; height: number },
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    reducedMotion: boolean,
    opacity = 1,
    blackHole?: BlackHoleGeometry,
  ) {
    this.prepared = false
    if (this.destroyed || opacity <= 0.001 || stageRect.widthPx < 2 || stageRect.heightPx < 2) return

    const dt = this.lastTime < 0 ? 1 / 60 : Math.min(0.05, Math.max(0.001, timeSeconds - this.lastTime))
    this.lastTime = timeSeconds
    const parallaxLerp = reducedMotion ? 1 : 1 - Math.exp(-dt * 3.8)

    let targetParX = 0
    let targetParY = 0
    let pointerInside = false

    if (this.pointerClient && canvasRect.width > 0 && canvasRect.height > 0) {
      const screenU = (this.pointerClient.x - canvasRect.left) / canvasRect.width
      const screenV = (this.pointerClient.y - canvasRect.top) / canvasRect.height
      if (!reducedMotion) {
        targetParX = screenU * 2 - 1
        targetParY = screenV * 2 - 1
      }
      const sceneUv = blackHole ? unlensBlackHoleUv(screenU, screenV, blackHole) : { u: screenU, v: screenV }
      if (sceneUv) {
        const localPxX = sceneUv.u * width - stageRect.leftPx
        const localPxY = sceneUv.v * height - stageRect.topPx
        const viewX = (localPxX / stageRect.widthPx) * CTA_VIEW_W
        const viewY = (localPxY / stageRect.heightPx) * CTA_VIEW_H
        if (viewX >= -6 && viewX <= CTA_VIEW_W + 6 && viewY >= -6 && viewY <= CTA_VIEW_H + 6) {
          pointerInside = true
        }
      }
    }

    this.smoothParallax.x += (targetParX - this.smoothParallax.x) * parallaxLerp
    this.smoothParallax.y += (targetParY - this.smoothParallax.y) * parallaxLerp
    const parallax = reducedMotion ? { x: 0, y: 0 } : this.smoothParallax

    // Smooth cohesive whole-button hover & press charge (never flees from cursor!).
    // When pointer coordinates are tracked, use the lensed-aware `pointerInside` check so
    // hovering inside the black hole void never triggers the hidden CTA behind it.
    const pointerHover = this.pointerClient ? pointerInside : this.hovered
    const targetHover = pointerHover || this.focused ? 1 : 0
    const hoverRate = targetHover > this.hoverLevel ? 10.5 : 5.5
    this.hoverLevel += (targetHover - this.hoverLevel) * (reducedMotion ? 1 : 1 - Math.exp(-dt * hoverRate))
    const targetPressed = this.pressed ? 1 : 0
    this.pressedLevel += (targetPressed - this.pressedLevel) * (reducedMotion ? 1 : 1 - Math.exp(-dt * 18))

    // Gently nudge the ↗ constellation arrow tip forward along its travel vector on hover for tactile button affordance.
    const arrowAdvance = this.hoverLevel * 6.5 + this.pressedLevel * 2.5
    const arrowDx = arrowAdvance * 0.94
    const arrowDy = arrowAdvance * -0.34

    const projected: { x: number; y: number; z: number }[] = []
    let sOff = 0
    for (let idx = 0; idx < CTA_STARS.length; idx++) {
      const st = CTA_STARS[idx]
      const isArrowStar = idx >= 3
      const px = st.x + parallax.x * (st.z - 0.5) * 2.2 + (isArrowStar ? arrowDx : 0)
      const py = st.y + parallax.y * (st.z - 0.5) * 1.6 + (isArrowStar ? arrowDy : 0)
      projected.push({ x: px, y: py, z: st.z })
      const rgb = STAR_TINT_RGB[st.tint]
      this.starData[sOff++] = px
      this.starData[sOff++] = py
      this.starData[sOff++] = st.z
      this.starData[sOff++] = st.mag
      this.starData[sOff++] = rgb[0]
      this.starData[sOff++] = rgb[1]
      this.starData[sOff++] = rgb[2]
      this.starData[sOff++] = 0
      this.starData[sOff++] = st.delay * 1.37
      this.starData[sOff++] = st.spiked ? 1 : 0
      this.starData[sOff++] = 0
      this.starData[sOff++] = 0
    }
    this.device.queue.writeBuffer(this.starBuffer, 0, this.starData)

    let eOff = 0
    let edgeCount = 0
    for (const [i, j] of CTA_EDGES) {
      const a = projected[i]
      const b = projected[j]
      const dx = b.x - a.x
      const dy = b.y - a.y
      const len = Math.hypot(dx, dy)
      if (len < 6) continue
      const pad = Math.min(3.5, len * 0.14)
      const ux = dx / len
      const uy = dy / len
      this.edgeData[eOff++] = a.x + ux * pad
      this.edgeData[eOff++] = a.y + uy * pad
      this.edgeData[eOff++] = b.x - ux * pad
      this.edgeData[eOff++] = b.y - uy * pad
      this.edgeData[eOff++] = i >= 2 ? 0.24 : 0.14
      this.edgeData[eOff++] = 0.72 * ratio
      this.edgeData[eOff++] = 0
      this.edgeData[eOff++] = 0
      edgeCount++
    }
    this.lastEdgeCount = edgeCount
    if (edgeCount > 0) {
      this.device.queue.writeBuffer(this.edgeBuffer, 0, this.edgeData.subarray(0, edgeCount * EDGE_FLOATS))
    }

    // Pass the true black hole center and event horizon radius in CTA view coordinates.
    // Dust falls radially straight toward the black hole center with zero inertia; any visual
    // curvature near the horizon comes purely from optical light bending in blackHole.wgsl.
    const sinkPxX = (blackHole ? blackHole.cx : 0.5) * width
    const sinkPxY = (blackHole ? blackHole.cy : 2.155) * height
    const sinkViewX = ((sinkPxX - stageRect.leftPx) / stageRect.widthPx) * CTA_VIEW_W
    const sinkViewY = ((sinkPxY - stageRect.topPx) / stageRect.heightPx) * CTA_VIEW_H
    const horizonViewR = (((blackHole ? blackHole.rh : 1.12) * height) / stageRect.heightPx) * CTA_VIEW_H
    const bottomViewY = ((height - stageRect.topPx) / stageRect.heightPx) * CTA_VIEW_H

    const outlineShiftX = -parallax.x * 0.22 * MAX_PARALLAX_X
    const outlineShiftY = -parallax.y * 0.22 * MAX_PARALLAX_Y
    this.uniformData.set([width, height, ratio, opacity], 0)
    this.uniformData.set([stageRect.leftPx, stageRect.topPx, stageRect.widthPx, stageRect.heightPx], 4)
    this.uniformData.set([CTA_VIEW_W, CTA_VIEW_H, CTA_PAD, CTA_STARS.length], 8)
    this.uniformData.set([timeSeconds, dt, outlineShiftX, outlineShiftY], 12)
    this.uniformData.set([edgeCount, this.dustCount, 1, reducedMotion ? 1 : 0], 16)
    this.uniformData.set([0, 0, this.hoverLevel, this.pressedLevel], 20)
    this.uniformData.set([sinkViewX, sinkViewY, horizonViewR, bottomViewY], 24)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const compute = encoder.beginComputePass({ label: 'Hero CTA stardust simulation' })
    compute.setPipeline(this.pipelines.dustCompute)
    compute.setBindGroup(0, this.dustBindGroup)
    compute.dispatchWorkgroups(Math.ceil(this.dustCount / 128))
    compute.end()
    this.prepared = true
  }

  draw(pass: GPURenderPassEncoder) {
    if (this.destroyed || !this.prepared) return
    this.prepared = false
    pass.setBindGroup(0, this.renderBindGroup)

    pass.setPipeline(this.pipelines.dustRender)
    pass.draw(6, this.dustCount)

    if (this.lastEdgeCount > 0) {
      pass.setPipeline(this.pipelines.edgeRender)
      pass.draw(6, this.lastEdgeCount)
    }

    pass.setPipeline(this.pipelines.starRender)
    pass.draw(6, CTA_STARS.length)
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.uniformBuffer.destroy()
    this.starBuffer.destroy()
    this.edgeBuffer.destroy()
    this.dustHomeBuffer.destroy()
    this.dustStateBuffer.destroy()
    this.velTexture.destroy()
  }
}
