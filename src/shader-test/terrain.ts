export type Terrain = { pixels: Uint8ClampedArray; width: number; height: number }
export type TerrainPackFormat = 'rg32float' | 'rgba8unorm'
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
      data[i * 2 + 1] = coast[i]
    }
    return { data, width, height }
  })
}

export function packRgba8Levels(levels: ReturnType<typeof elevationPyramid>): PackedTerrainLevel<Uint8Array<ArrayBuffer>>[] {
  return levels.map(({ values, width, height }) => {
    // Normalized linear filtering commutes with the affine two-channel decode.
    // Generate each mip from real metres before packing.
    const data = new Uint8Array(width * height * 4)
    for (let i = 0; i < values.length; i++) {
      const code = Math.round(values[i] + 32768)
      data[i * 4] = code >> 8
      data[i * 4 + 1] = code & 255
    }
    return { data, width, height }
  })
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
  return { pixels: new Uint8ClampedArray(buffer,12), width, height }
}

export async function loadTerrain(url: string, signal: AbortSignal): Promise<Terrain> {
  const response = await fetch(url,{ signal })
  if (!response.ok || !response.body) throw new Error('Could not load terrain elevation. Reload to retry.')
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'))
  return decodeTerrain(await new Response(stream).arrayBuffer())
}

export function poleElevations(terrain: Terrain): [number, number] {
  const average = (row: number) => {
    let sum = 0
    for (let x = 0; x < terrain.width; x++) {
      const i = (row * terrain.width + x) * 4
      sum += terrain.pixels[i] * 256 + terrain.pixels[i + 1] - 32768
    }
    return sum / terrain.width
  }
  return [average(terrain.height - 1),average(0)]
}

// Numeric averages, not averages of the packed PNG color channels. Uploading
// these levels also avoids requiring float render-target extensions for mipmaps.
export function elevationPyramid(terrain: Terrain) {
  let width = terrain.width, height = terrain.height
  let values = new Float32Array(width * height)
  let coast = new Float32Array(width * height)
  for (let i = 0; i < values.length; i++) {
    values[i] = terrain.pixels[i * 4] * 256 + terrain.pixels[i * 4 + 1] - 32768
    coast[i] = (terrain.pixels[i * 4 + 2] - 128) / 4
  }
  const closePoles = (data: Float32Array, columns: number, rows: number) => {
    // Every longitude converges to one point. A longitude-dependent last row
    // otherwise remains visible as radial wedges, even with correct coordinates.
    for (const row of new Set([0,rows - 1])) {
      let sum = 0
      for (let x = 0; x < columns; x++) sum += data[row * columns + x]
      data.fill(sum / columns,row * columns,(row + 1) * columns)
    }
  }
  closePoles(values,width,height); closePoles(coast,width,height)
  const levels = [{ values, coast, width, height }]
  while (width > 1 || height > 1) {
    const w = Math.max(1,width >> 1), h = Math.max(1,height >> 1)
    const next = new Float32Array(w * h)
    const nextCoast = new Float32Array(w * h)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let sum = 0, coastSum = 0, count = 0
      for (let sy = Math.floor(y * height / h); sy < Math.floor((y + 1) * height / h); sy++) {
        for (let sx = Math.floor(x * width / w); sx < Math.floor((x + 1) * width / w); sx++) {
          sum += values[sy * width + sx]; coastSum += coast[sy * width + sx]; count++
        }
      }
      next[y * w + x] = sum / count
      nextCoast[y * w + x] = coastSum / count
    }
    closePoles(next,w,h); closePoles(nextCoast,w,h)
    levels.push({ values: next, coast: nextCoast, width: w, height: h })
    values = next; coast = nextCoast; width = w; height = h
  }
  return levels
}

// Match normalized GPU linear sampling, including the dateline and pole clamps.
export function elevationAt(terrain: Terrain, longitude: number, latitude: number) {
  const { pixels, width, height } = terrain
  const x = (longitude / (2 * Math.PI) + .5) * width - .5
  const y = Math.max(0, Math.min(height - 1, (.5 - latitude / Math.PI) * height - .5))
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0
  const read = (column: number, row: number) => {
    const i = (Math.min(height - 1, row) * width + ((column % width) + width) % width) * 4
    return pixels[i] * 256 + pixels[i + 1] - 32768
  }
  const top = read(x0, y0) * (1 - fx) + read(x0 + 1, y0) * fx
  const bottom = read(x0, y0 + 1) * (1 - fx) + read(x0 + 1, y0 + 1) * fx
  return top * (1 - fy) + bottom * fy
}
