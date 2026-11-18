function permutation(seed: number) {
  const values = Array.from({ length: 256 }, (_, index) => index)
  let state = seed >>> 0
  for (let i = 255; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    const j = state % (i + 1)
    ;[values[i], values[j]] = [values[j], values[i]]
  }
  return Uint16Array.from([...values, ...values])
}

export function perlin(seed: number) {
  const p = permutation(seed)
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10)
  const mix = (a: number, b: number, t: number) => a + (b - a) * t
  const grad = (hash: number, x: number, y: number, z: number) => {
    const h = hash & 15, u = h < 8 ? x : y, v = h < 4 ? y : h === 12 || h === 14 ? x : z
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v)
  }
  return (x: number, y: number, z: number) => {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z)
    const X = ix & 255, Y = iy & 255, Z = iz & 255
    x -= ix; y -= iy; z -= iz
    const u = fade(x), v = fade(y), w = fade(z)
    const a = p[X] + Y, aa = p[a] + Z, ab = p[a + 1] + Z
    const b = p[X + 1] + Y, ba = p[b] + Z, bb = p[b + 1] + Z
    return mix(
      mix(mix(grad(p[aa], x, y, z), grad(p[ba], x - 1, y, z), u), mix(grad(p[ab], x, y - 1, z), grad(p[bb], x - 1, y - 1, z), u), v),
      mix(mix(grad(p[aa + 1], x, y, z - 1), grad(p[ba + 1], x - 1, y, z - 1), u), mix(grad(p[ab + 1], x, y - 1, z - 1), grad(p[bb + 1], x - 1, y - 1, z - 1), u), v), w)
  }
}

// Sample 3D Perlin noise on the unit sphere: no seam at ±180° or polar discontinuity.
export function noiseTexture(width: number, height: number, frequency: number, seed: number) {
  const noise = perlin(seed), scores = new Uint16Array(width * height)
  let best = 0
  for (let y = 0; y < height; y++) {
    const lat = ((y + .5) / height - .5) * Math.PI, cos = Math.cos(lat)
    for (let x = 0; x < width; x++) {
      const lon = ((x + .5) / width - .5) * 2 * Math.PI
      const px = cos * Math.sin(lon), py = Math.sin(lat), pz = cos * Math.cos(lon)
      let sum = 0, amplitude = 1, normalization = 0, scale = frequency
      for (let octave = 0; octave < 5; octave++) {
        sum += amplitude * noise(px * scale + 17.3, py * scale + 31.7, pz * scale + 8.9)
        normalization += amplitude; amplitude *= .5; scale *= 2
      }
      const value = Math.max(0, Math.min(1, .5 + sum / normalization * .85))
      const code = Math.round(value * 65534)
      scores[y * width + x] = code
      best = Math.max(best, code)
    }
  }
  return { scores, best }
}

export function maxPyramid(scores: Uint16Array, width: number, height: number) {
  const levels = [{ scores, width, height }]
  while (width > 1 || height > 1) {
    const w = Math.max(1, width >> 1), h = Math.max(1, height >> 1)
    const next = new Uint16Array(w * h)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let best = 0, valid = false
      // Proportional footprints also cover odd source dimensions.
      for (let sy = Math.floor(y * height / h); sy < Math.floor((y + 1) * height / h); sy++) {
        for (let sx = Math.floor(x * width / w); sx < Math.floor((x + 1) * width / w); sx++) {
          const value = scores[sy * width + sx]
          if (value !== 65535) { best = Math.max(best, value); valid = true }
        }
      }
      next[y * w + x] = valid ? best : 65535
    }
    levels.push({ scores: next, width: w, height: h })
    scores = next; width = w; height = h
  }
  return levels
}
