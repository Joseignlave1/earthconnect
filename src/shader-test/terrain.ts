export type Terrain = {
  pixels: Uint8ClampedArray
  width: number
  height: number
  stride?: 2 | 4
  polarSmoothed?: boolean
}
type TerrainZstdDecompressor = (frame: ArrayBuffer, outputBytes: number, signal?: AbortSignal) => Promise<Uint8Array<ArrayBuffer>>
export type TerrainPackFormat = 'rg32float' | 'rg8unorm'
export type PackedTerrainLevel<T extends Float32Array<ArrayBuffer> | Uint8Array<ArrayBuffer> = Float32Array<ArrayBuffer> | Uint8Array<ArrayBuffer>> = {
  data: T
  width: number
  height: number
}
export type TerrainWorkerRequest = { url: string; format: TerrainPackFormat }
export type TerrainWorkerResponse =
  | { kind: 'ok'; terrain: Terrain; poles: [number, number]; levels: PackedTerrainLevel[] }
  | { kind: 'error'; message: string }

export function packRg32FloatLevels(levels: ReturnType<typeof elevationPyramid>): PackedTerrainLevel<Float32Array<ArrayBuffer>>[] {
  return levels.map(({ values, coast, width, height }) => {
    const data = new Float32Array(values.length * 2)
    for (let i = 0; i < values.length; i++) {
      data[i * 2] = values[i]
      data[i * 2 + 1] = coast[i] ?? 0
    }
    return { data, width, height }
  })
}

export function packRg8Levels(levels: ReturnType<typeof elevationPyramid>): PackedTerrainLevel<Uint8Array<ArrayBuffer>>[] {
  return levels.map(({ values, width, height }) => {
    const data = new Uint8Array(width * height * 2)
    for (let i = 0; i < values.length; i++) {
      const code = Math.max(0, Math.min(65535, Math.round(values[i] + 32768)))
      data[i * 2] = code >> 8
      data[i * 2 + 1] = code & 255
    }
    return { data, width, height }
  })
}

// Low-memory rg8unorm mip builder for offline-smoothed ECT2 DEMs: reuses level 0's
// packed 2-byte pixels directly without allocating a 128 MiB Float32Array at full 8K.
export function packRg8PyramidDirect(terrain: Terrain): PackedTerrainLevel<Uint8Array<ArrayBuffer>>[] {
  if ((terrain.stride ?? 4) !== 2 || !terrain.polarSmoothed) {
    return packRg8Levels(elevationPyramid(terrain, true))
  }
  let width = terrain.width, height = terrain.height
  const level0 = new Uint8Array(terrain.pixels.buffer as ArrayBuffer, terrain.pixels.byteOffset, terrain.pixels.byteLength)
  const levels: PackedTerrainLevel<Uint8Array<ArrayBuffer>>[] = [{ data: level0, width, height }]
  let prev = level0
  while (width > 1 || height > 1) {
    const w = Math.max(1, width >> 1), h = Math.max(1, height >> 1)
    const next = new Uint8Array(w * h * 2)
    for (let y = 0; y < h; y++) {
      const sy0 = Math.floor(y * height / h), sy1 = Math.floor((y + 1) * height / h)
      for (let x = 0; x < w; x++) {
        const sx0 = Math.floor(x * width / w), sx1 = Math.floor((x + 1) * width / w)
        let sum = 0, count = 0
        for (let sy = sy0; sy < sy1; sy++) {
          const rowOff = sy * width
          for (let sx = sx0; sx < sx1; sx++) {
            const i = (rowOff + sx) * 2
            sum += prev[i] * 256 + prev[i + 1]
            count++
          }
        }
        const code = Math.max(0, Math.min(65535, Math.round(sum / count)))
        const o = (y * w + x) * 2
        next[o] = code >> 8
        next[o + 1] = code & 255
      }
    }
    // Close polar rows to their average code.
    for (const row of new Set([0, h - 1])) {
      let sum = 0
      for (let x = 0; x < w; x++) {
        const i = (row * w + x) * 2
        sum += next[i] * 256 + next[i + 1]
      }
      const code = Math.max(0, Math.min(65535, Math.round(sum / w)))
      const hi = code >> 8, lo = code & 255
      for (let x = 0; x < w; x++) {
        const i = (row * w + x) * 2
        next[i] = hi
        next[i + 1] = lo
      }
    }
    levels.push({ data: next, width: w, height: h })
    prev = next; width = w; height = h
  }
  return levels
}

// ECT1: magic, uint32 little-endian width/height, then packed RGBA data bytes.
// Decode numeric data without canvas readback, color conversion or privacy noise.
export function decodeTerrain(buffer: ArrayBuffer): Terrain {
  const view = new DataView(buffer)
  if (buffer.byteLength < 12 || view.getUint32(0) !== 0x45435431) throw new Error('Invalid terrain data header.')
  const width = view.getUint32(4,true), height = view.getUint32(8,true)
  if (!width || !height || width > 16384 || height > 16384 || buffer.byteLength !== 12 + width * height * 4) {
    throw new Error('Invalid terrain data dimensions or length.')
  }
  return { pixels: new Uint8ClampedArray(buffer,12), width, height, stride: 4 }
}

// ECT2: big-endian magic 0x45435432, LE uint32 width/height/flags, then Zstd-compressed
// byte-shuffled zigzag 2D-predicted uint16 metres (2 bytes/px), plus optional 2D-predicted
// uint8 signed coast distance channel when flags & 4 (3 bytes/px -> stride 4 RGBA).
export function decodeTerrainEct2Payload(width: number, height: number, flags: number, payload: Uint8Array): Terrain {
  const count = width * height
  const hasCoast = Boolean(flags & 4)
  const payloadStride = hasCoast ? 3 : 2
  if (!width || !height || width > 16384 || height > 16384 || payload.byteLength !== count * payloadStride) {
    throw new Error('Invalid ECT2 terrain payload dimensions or length.')
  }
  const stride: 2 | 4 = hasCoast ? 4 : 2
  const pixels = new Uint8ClampedArray(count * stride)
  if (flags & 1) {
    const prevRow = new Uint16Array(width)
    const prevCoast = hasCoast ? new Uint8Array(width) : null
    const coastOffset = count * 2
    for (let y = 0; y < height; y++) {
      const rowOff = y * width
      let left = 32768
      let leftCoast = 128
      for (let x = 0; x < width; x++) {
        const i = rowOff + x
        const zz = (payload[i] << 8) | payload[count + i]
        const signed = (zz >>> 1) ^ -(zz & 1)
        const pred = y === 0 ? (x === 0 ? 32768 : left) : (x === 0 ? prevRow[0] : (left + prevRow[x]) >> 1)
        const code = (pred + signed) & 0xffff
        left = code
        prevRow[x] = code
        const o = i * stride
        pixels[o] = code >> 8
        pixels[o + 1] = code & 255
        if (prevCoast) {
          const czz = payload[coastOffset + i]
          const cs = (czz >>> 1) ^ -(czz & 1)
          const cpred = y === 0 ? (x === 0 ? 128 : leftCoast) : (x === 0 ? prevCoast[0] : (leftCoast + prevCoast[x]) >> 1)
          const ccode = (cpred + cs) & 0xff
          leftCoast = ccode
          prevCoast[x] = ccode
          pixels[o + 2] = ccode
          pixels[o + 3] = 255
        }
      }
    }
  } else if (hasCoast) {
    for (let i = 0; i < count; i++) {
      pixels[i * 4] = payload[i * 3]
      pixels[i * 4 + 1] = payload[i * 3 + 1]
      pixels[i * 4 + 2] = payload[i * 3 + 2]
      pixels[i * 4 + 3] = 255
    }
  } else {
    pixels.set(payload)
  }
  return { pixels, width, height, stride, polarSmoothed: Boolean(flags & 2) }
}

export async function decodeTerrainEct2(buffer: ArrayBuffer, decompress: TerrainZstdDecompressor, signal?: AbortSignal): Promise<Terrain> {
  const view = new DataView(buffer)
  if (buffer.byteLength < 16 || view.getUint32(0) !== 0x45435432) throw new Error('Invalid ECT2 terrain header.')
  const width = view.getUint32(4, true), height = view.getUint32(8, true), flags = view.getUint32(12, true)
  if (!width || !height || width > 16384 || height > 16384) throw new Error('Invalid ECT2 terrain dimensions.')
  const payloadStride = (flags & 4) ? 3 : 2
  const payload = await decompress(buffer.slice(16), width * height * payloadStride, signal)
  return decodeTerrainEct2Payload(width, height, flags, payload)
}

export async function loadTerrain(url: string, signal: AbortSignal, decompress?: TerrainZstdDecompressor): Promise<Terrain> {
  const response = await fetch(url,{ signal })
  if (!response.ok || !response.body) throw new Error('Could not load terrain elevation. Reload to retry.')
  const raw = await response.arrayBuffer()
  signal.throwIfAborted()
  if (raw.byteLength >= 16 && new DataView(raw).getUint32(0) === 0x45435432) {
    if (!decompress) throw new Error('Zstd decompressor is required for ECT2 terrain.')
    return decodeTerrainEct2(raw, decompress, signal)
  }
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))
  return decodeTerrain(await new Response(stream).arrayBuffer())
}

export function poleElevations(terrain: Terrain): [number, number] {
  const stride = terrain.stride ?? 4
  const average = (row: number) => {
    let sum = 0
    for (let x = 0; x < terrain.width; x++) {
      const i = (row * terrain.width + x) * stride
      sum += terrain.pixels[i] * 256 + terrain.pixels[i + 1] - 32768
    }
    return sum / terrain.width
  }
  return [average(terrain.height - 1),average(0)]
}

function smoothPolarLongitudes(data: Float32Array, width: number, height: number) {
  if (width < 4 || height < 4) return
  const prefix = new Float64Array(width * 3 + 1)
  const scratch = new Float32Array(width)
  const maxHalf = (width - 1) * .5
  for (let y = 1; y < height - 1; y++) {
    const latitude = (.5 - (y + .5) / height) * Math.PI
    const ratio = 1 / Math.max(1e-4, Math.cos(latitude))
    if (ratio < 1.25) continue
    // Three periodic fractional box passes match a Gaussian of sigma ~= half,
    // equalizing east-west resolution to the north-south texel pitch so polar
    // orbital cross-track noise does not form radial spokes.
    const half = Math.min(maxHalf, .5 * Math.sqrt(ratio * ratio - 1))
    const radius = Math.floor(half)
    const frac = half - radius
    const weight = 2 * half + 1
    const offset = y * width
    for (let pass = 0; pass < 3; pass++) {
      let acc = 0
      prefix[0] = 0
      for (let i = 0; i < width * 3; i++) {
        acc += data[offset + (i % width)]
        prefix[i + 1] = acc
      }
      for (let x = 0; x < width; x++) {
        const center = x + width
        let sum = prefix[center + radius + 1] - prefix[center - radius]
        if (frac > 0) {
          const right = (x + radius + 1) % width
          const left = (x - radius - 1 + width * 2) % width
          sum += frac * (data[offset + right] + data[offset + left])
        }
        scratch[x] = sum / weight
      }
      data.set(scratch, offset)
    }
  }
}

// Numeric averages, not averages of the packed PNG color channels. Uploading
// these levels also avoids requiring float render-target extensions for mipmaps.
export function elevationPyramid(terrain: Terrain, isotropicPoles = false) {
  let width = terrain.width, height = terrain.height
  const stride = terrain.stride ?? 4
  const hasCoast = stride === 4
  let values = new Float32Array(width * height)
  let coast = hasCoast ? new Float32Array(width * height) : new Float32Array(0)
  for (let i = 0; i < values.length; i++) {
    const o = i * stride
    values[i] = terrain.pixels[o] * 256 + terrain.pixels[o + 1] - 32768
    if (hasCoast) coast[i] = (terrain.pixels[o + 2] - 128) / 4
  }
  const closePoles = (data: Float32Array, columns: number, rows: number) => {
    if (!data.length) return
    // Every longitude converges to one point. A longitude-dependent last row
    // otherwise remains visible as radial wedges, even with correct coordinates.
    for (const row of new Set([0,rows - 1])) {
      let sum = 0
      for (let x = 0; x < columns; x++) sum += data[row * columns + x]
      data.fill(sum / columns,row * columns,(row + 1) * columns)
    }
  }
  const smoothRuntime = isotropicPoles && !terrain.polarSmoothed
  if (smoothRuntime) smoothPolarLongitudes(values,width,height)
  closePoles(values,width,height); closePoles(coast,width,height)
  const levels = [{ values, coast, width, height }]
  while (width > 1 || height > 1) {
    const w = Math.max(1,width >> 1), h = Math.max(1,height >> 1)
    const next = new Float32Array(w * h)
    const nextCoast = hasCoast ? new Float32Array(w * h) : new Float32Array(0)
    if (width === w * 2 && height === h * 2) {
      for (let y = 0; y < h; y++) {
        const r0 = (y * 2) * width
        const r1 = r0 + width
        const dstRow = y * w
        if (hasCoast) {
          for (let x = 0; x < w; x++) {
            const x0 = x * 2
            const i0 = r0 + x0, i1 = r1 + x0
            next[dstRow + x] = (values[i0] + values[i0 + 1] + values[i1] + values[i1 + 1]) * .25
            nextCoast[dstRow + x] = (coast[i0] + coast[i0 + 1] + coast[i1] + coast[i1 + 1]) * .25
          }
        } else {
          for (let x = 0; x < w; x++) {
            const x0 = x * 2
            const i0 = r0 + x0, i1 = r1 + x0
            next[dstRow + x] = (values[i0] + values[i0 + 1] + values[i1] + values[i1 + 1]) * .25
          }
        }
      }
    } else {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let sum = 0, coastSum = 0, count = 0
        for (let sy = Math.floor(y * height / h); sy < Math.floor((y + 1) * height / h); sy++) {
          for (let sx = Math.floor(x * width / w); sx < Math.floor((x + 1) * width / w); sx++) {
            sum += values[sy * width + sx]
            if (hasCoast) coastSum += coast[sy * width + sx]
            count++
          }
        }
        next[y * w + x] = sum / count
        if (hasCoast) nextCoast[y * w + x] = coastSum / count
      }
    }
    if (smoothRuntime) smoothPolarLongitudes(next,w,h)
    closePoles(next,w,h); closePoles(nextCoast,w,h)
    levels.push({ values: next, coast: nextCoast, width: w, height: h })
    values = next; coast = nextCoast; width = w; height = h
  }
  return levels
}

// Match normalized GPU linear sampling, including the dateline and pole clamps.
export function elevationAt(terrain: Terrain, longitude: number, latitude: number) {
  const { pixels, width, height } = terrain
  const stride = terrain.stride ?? 4
  const x = (longitude / (2 * Math.PI) + .5) * width - .5
  const y = Math.max(0, Math.min(height - 1, (.5 - latitude / Math.PI) * height - .5))
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0
  const read = (column: number, row: number) => {
    const i = (Math.min(height - 1, row) * width + ((column % width) + width) % width) * stride
    return pixels[i] * 256 + pixels[i + 1] - 32768
  }
  const top = read(x0, y0) * (1 - fx) + read(x0 + 1, y0) * fx
  const bottom = read(x0, y0 + 1) * (1 - fx) + read(x0 + 1, y0 + 1) * fx
  return top * (1 - fy) + bottom * fy
}
