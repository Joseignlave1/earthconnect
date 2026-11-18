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
