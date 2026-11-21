// WebGPU owns resource lifetime explicitly; uploads never read numeric data
// through canvas or image color conversion.
export async function shaderModule(device: GPUDevice, label: string, code: string) {
  const module = device.createShaderModule({ label, code })
  const info = await module.getCompilationInfo()
  const errors = info.messages.filter(message => message.type === 'error')
  if (errors.length) throw new Error(`${label}: ${errors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')}`)
  return module
}

export function uploadBuffer(device: GPUDevice, label: string, data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>, usage: GPUBufferUsageFlags) {
  const buffer = device.createBuffer({ label, size: data.byteLength, usage, mappedAtCreation: true })
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength))
  buffer.unmap()
  return buffer
}

// Upload large texture mip levels in bounded row strips so Dawn does not retain
// a single multi-tens-of-MiB IPC transfer buffer in the renderer process.
export function writeTextureChunked(
  queue: GPUQueue,
  texture: GPUTexture,
  mipLevel: number,
  data: ArrayBufferView<ArrayBuffer>,
  width: number,
  height: number,
  bytesPerPixel: number,
  maxChunkBytes = 8 * 1024 * 1024,
) {
  const bytesPerRow = width * bytesPerPixel
  const rowsPerChunk = Math.max(1, Math.floor(maxChunkBytes / bytesPerRow))
  if (height <= rowsPerChunk) {
    queue.writeTexture({ texture, mipLevel }, data, { bytesPerRow }, [width, height])
    return
  }
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  for (let y = 0; y < height; y += rowsPerChunk) {
    const rows = Math.min(rowsPerChunk, height - y)
    const slice = bytes.subarray(y * bytesPerRow, (y + rows) * bytesPerRow)
    queue.writeTexture({ texture, mipLevel, origin: [0, y, 0] }, slice, { bytesPerRow }, [width, rows])
  }
}

export const fullscreenVertex = `
struct ScreenVertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}
@vertex fn screenVertex(@builtin(vertex_index) index: u32) -> ScreenVertex {
  let p = vec2f(f32((index << 1u) & 2u),f32(index & 2u));
  // Texture coordinates run downwards, while clip-space Y points upwards.
  return ScreenVertex(vec4f(p * 2. - 1.,0.,1.),vec2f(p.x,1. - p.y));
}`
