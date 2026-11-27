import { shaderModule } from '../shader-test/gpu.ts'
import type { createSurfaceDevicePool } from '../shader-test/surface-device.ts'
import { vec3Cross, vec3Sub, type MutableVec3 } from '../shader-test/vec3.ts'
import crewAstronautWgsl from './crewAstronaut.wgsl?raw'
import { buildAstronautGeometry, normOrUp } from './crewMesh.ts'
import { renderStickyNoteCanvas } from './crewStickyNote.ts'

export type AstronautPoseId = 'cling' | 'laptop' | 'brace' | 'hang' | 'tumble' | 'accept'

export type CrewAstronautSpec = {
  id: string
  name: string
  pose: AstronautPoseId
  /** Optional portrait image URL placed directly on the visor sticky note. */
  photo?: string
}

type Vec3 = MutableVec3

const sub = vec3Sub
const cross = vec3Cross
const norm = normOrUp

function perspectiveMatrix(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovRad / 2)
  const nf = 1 / (near - far)
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0])
}

function lookAtMatrix(eye: Vec3, target: Vec3, up: Vec3): Float32Array {
  const z = norm(sub(eye, target))
  const x = norm(cross(up, z))
  const y = cross(z, x)
  return new Float32Array([
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
    1,
  ])
}

function mulMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3]
    }
  }
  return out
}

type CrewPipeline = {
  pipeline: GPURenderPipeline
  layout: GPUBindGroupLayout
}

const crewPipelineCache = new WeakMap<GPUDevice, Map<GPUTextureFormat, Promise<CrewPipeline>>>()

function getCrewPipeline(device: GPUDevice, format: GPUTextureFormat): Promise<CrewPipeline> {
  let byFormat = crewPipelineCache.get(device)
  if (!byFormat) {
    byFormat = new Map()
    crewPipelineCache.set(device, byFormat)
  }
  let pending = byFormat.get(format)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'Crew astronaut spaghettification shader', crewAstronautWgsl)
      const layout = device.createBindGroupLayout({
        label: 'Crew astronaut layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      })
      const pipeline = await device.createRenderPipelineAsync({
        label: 'Crew astronaut 3D MSAA',
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: {
          module,
          entryPoint: 'vsMain',
          buffers: [
            {
              arrayStride: 40,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 12, format: 'float32x3' },
                { shaderLocation: 2, offset: 24, format: 'float32x2' },
                { shaderLocation: 3, offset: 32, format: 'uint32' },
                { shaderLocation: 4, offset: 36, format: 'float32' },
              ],
            },
          ],
        },
        fragment: {
          module,
          entryPoint: 'fsMain',
          targets: [
            {
              format,
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
        multisample: { count: 4 },
      })
      return { pipeline, layout }
    })()
    byFormat.set(format, pending)
    void pending.catch(() => {
      if (byFormat?.get(format) === pending) byFormat.delete(format)
    })
  }
  return pending
}

// Pull vector aims down-and-right AND plunges away into -Z so perspective shrinking sells depth & length.
const PULL_DIR = norm([0.38, -0.78, -0.49])
const KEY_LIGHT_DIR = norm([-0.68, 0.72, 0.55])
const LIGHTING_EYE: Vec3 = [0.14, -0.14, 6.65]

type PreparedCrewMesh = {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexFormat: GPUIndexFormat
  indexCount: number
  noteTexture: GPUTexture
  bindGroup: GPUBindGroup
}

export class CrewAstronautStage {
  private readonly uniformData = new Float32Array(28)
  private readonly uniformBuffer: GPUBuffer
  private readonly sampler: GPUSampler
  private readonly meshes = new Map<string, PreparedCrewMesh>()
  private msaaColor: GPUTexture | null = null
  private msaaColorView: GPUTextureView | null = null
  private depthTex: GPUTexture | null = null
  private depthTexView: GPUTextureView | null = null
  private projMatrix: Float32Array | null = null
  private texW = 0
  private texH = 0
  private cssW = 0
  private cssH = 0
  private activeId: string
  private active = true
  private hovered = false
  private pointerNorm = { x: 0, y: 0 }
  private smoothPointer = { x: 0, y: 0 }
  private raf = 0
  private lastTime = performance.now() * 0.001
  private startedAt = performance.now() * 0.001
  private fadeIn = 1
  private readySignaled = false
  private destroyed = false
  private readonly reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  private readonly onReducedMotionChange = () => {
    if (this.destroyed || !this.active) return
    if (this.reducedMotion.matches) {
      if (this.raf) {
        cancelAnimationFrame(this.raf)
        this.raf = 0
      }
      this.drawOnce()
    } else if (!this.raf) {
      this.lastTime = performance.now() * 0.001
      this.schedule()
    }
  }
  private readonly observer: ResizeObserver

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly context: GPUCanvasContext,
    private readonly device: GPUDevice,
    private readonly lease: { release: () => void },
    private readonly format: GPUTextureFormat,
    private readonly gpu: CrewPipeline,
    specs: readonly CrewAstronautSpec[],
    initialId: string,
  ) {
    this.activeId = initialId
    this.uniformBuffer = device.createBuffer({
      label: 'Crew astronaut uniforms',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Constant directional vectors & fixed lighting eye stay invariant across frames.
    this.uniformData.set(PULL_DIR, 16)
    this.uniformData.set(KEY_LIGHT_DIR, 20)
    this.uniformData.set(LIGHTING_EYE, 24)
    this.sampler = device.createSampler({
      label: 'Crew sticky note sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    for (const spec of specs) {
      const { vertexData, indexData, indexFormat, indexCount } = buildAstronautGeometry(spec.pose)
      const vertexBuffer = device.createBuffer({
        label: `Crew astronaut verts ${spec.id}`,
        size: vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(vertexBuffer, 0, vertexData)

      const alignedIndexBytes = (indexData.byteLength + 3) & ~3
      const indexBuffer = device.createBuffer({
        label: `Crew astronaut indices ${spec.id}`,
        size: alignedIndexBytes,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      if (alignedIndexBytes === indexData.byteLength) {
        device.queue.writeBuffer(indexBuffer, 0, indexData.buffer, indexData.byteOffset, indexData.byteLength)
      } else {
        const padded = new Uint8Array(alignedIndexBytes)
        padded.set(new Uint8Array(indexData.buffer, indexData.byteOffset, indexData.byteLength))
        device.queue.writeBuffer(indexBuffer, 0, padded)
      }

      const noteTexture = device.createTexture({
        label: `Crew sticky note ${spec.id}`,
        size: [512, 512, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })

      const bindGroup = device.createBindGroup({
        layout: gpu.layout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: noteTexture.createView() },
          { binding: 2, resource: this.sampler },
        ],
      })

      this.meshes.set(spec.id, {
        vertexBuffer,
        indexBuffer,
        indexFormat,
        indexCount,
        noteTexture,
        bindGroup,
      })

      void renderStickyNoteCanvas(spec).then(noteCanvas => {
        if (this.destroyed) return
        device.queue.copyExternalImageToTexture({ source: noteCanvas }, { texture: noteTexture }, [512, 512])
        if (this.reducedMotion.matches) this.drawOnce()
      })
    }

    const syncCssSize = () => {
      const rect = canvas.getBoundingClientRect()
      this.cssW = rect.width
      this.cssH = rect.height
    }
    syncCssSize()
    this.observer = new ResizeObserver(() => {
      syncCssSize()
      if (!this.destroyed && this.reducedMotion.matches) this.drawOnce()
    })
    this.observer.observe(canvas)
    this.reducedMotion.addEventListener('change', this.onReducedMotionChange)
    this.schedule()
  }

  static async create(
    canvas: HTMLCanvasElement,
    acquireDevice: ReturnType<typeof createSurfaceDevicePool>,
    specs: readonly CrewAstronautSpec[],
    initialId: string,
    signal?: AbortSignal,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('WebGPU unavailable for crew stage.')
    const lease = acquireDevice()
    let configured = false
    try {
      const { device } = await lease.ready
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      const format = navigator.gpu.getPreferredCanvasFormat()
      context.configure({ device, format, alphaMode: 'premultiplied' })
      configured = true
      const gpu = await getCrewPipeline(device, format)
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      return new CrewAstronautStage(canvas, context, device, lease, format, gpu, specs, initialId)
    } catch (error) {
      if (configured) context.unconfigure()
      lease.release()
      throw error
    }
  }

  setMember(id: string) {
    if (this.activeId === id || !this.meshes.has(id)) return
    this.activeId = id
    this.fadeIn = this.reducedMotion.matches ? 1 : 0.12
    if (this.reducedMotion.matches) this.drawOnce()
  }

  setActive(active: boolean) {
    if (this.active === active) return
    this.active = active
    if (active) {
      this.lastTime = performance.now() * 0.001
      this.schedule()
    } else if (this.raf) {
      cancelAnimationFrame(this.raf)
      this.raf = 0
    }
  }

  setPointer(nx: number, ny: number, hovered: boolean) {
    this.hovered = hovered
    this.pointerNorm.x = nx
    this.pointerNorm.y = ny
  }

  private ensureTargets(w: number, h: number) {
    if (this.msaaColor && this.depthTex && this.texW === w && this.texH === h) return
    this.msaaColor?.destroy()
    this.depthTex?.destroy()
    this.msaaColor = this.device.createTexture({
      label: 'Crew astronaut MSAA color',
      size: [w, h, 1],
      format: this.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.msaaColorView = this.msaaColor.createView()
    this.depthTex = this.device.createTexture({
      label: 'Crew astronaut MSAA depth',
      size: [w, h, 1],
      format: 'depth24plus',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.depthTexView = this.depthTex.createView()
    this.projMatrix = perspectiveMatrix((42 * Math.PI) / 180, w / h, 0.1, 40)
    this.texW = w
    this.texH = h
  }

  private drawOnce() {
    if (this.destroyed) return
    if (this.cssW < 2 || this.cssH < 2) return
    const w = Math.max(2, Math.round(this.cssW))
    const h = Math.max(2, Math.round(this.cssH))
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.ensureTargets(w, h)

    const now = performance.now() * 0.001
    const dt = Math.min(0.05, Math.max(0.001, now - this.lastTime))
    this.lastTime = now
    const time = now - this.startedAt

    const reduced = this.reducedMotion.matches
    const lerp = reduced ? 1 : 1 - Math.exp(-dt * 7)
    this.smoothPointer.x += ((this.hovered ? this.pointerNorm.x : 0) - this.smoothPointer.x) * lerp
    this.smoothPointer.y += ((this.hovered ? this.pointerNorm.y : 0) - this.smoothPointer.y) * lerp
    this.fadeIn = Math.min(1, this.fadeIn + dt * 4.5)

    // Camera framed generously so raised hands, floating laptop, sticky note, and swirling vortex noodle + dust depth all show cleanly
    const camX = 0.14 + this.smoothPointer.x * 0.52 + (reduced ? 0 : Math.sin(time * 0.45) * 0.08)
    const camY = -0.14 - this.smoothPointer.y * 0.32 + (reduced ? 0 : Math.cos(time * 0.38) * 0.05)
    const view = lookAtMatrix([camX, camY, 6.65], [0.24, -0.28, -0.35], [0, 1, 0])
    const viewProj = mulMat4(this.projMatrix!, view)

    this.uniformData.set(viewProj, 0)
    this.uniformData[19] = time
    this.uniformData[23] = this.fadeIn
    this.uniformData[27] = reduced ? 1 : 0
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    const mesh = this.meshes.get(this.activeId)
    if (!mesh) return

    const encoder = this.device.createCommandEncoder({ label: 'Crew astronaut frame' })
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaaColorView!,
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexView!,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
    pass.setPipeline(this.gpu.pipeline)
    pass.setBindGroup(0, mesh.bindGroup)
    pass.setVertexBuffer(0, mesh.vertexBuffer)
    pass.setIndexBuffer(mesh.indexBuffer, mesh.indexFormat)
    pass.drawIndexed(mesh.indexCount)
    pass.end()
    this.device.queue.submit([encoder.finish()])
    if (!this.readySignaled) {
      this.readySignaled = true
      this.canvas.dataset.crewReady = 'true'
    }
  }

  private schedule() {
    if (this.destroyed || !this.active) return
    if (this.reducedMotion.matches) {
      this.drawOnce()
      return
    }
    this.raf = requestAnimationFrame(() => {
      this.raf = 0
      this.drawOnce()
      this.schedule()
    })
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    if (this.raf) cancelAnimationFrame(this.raf)
    this.reducedMotion.removeEventListener('change', this.onReducedMotionChange)
    this.observer.disconnect()
    this.msaaColor?.destroy()
    this.depthTex?.destroy()
    this.uniformBuffer.destroy()
    for (const mesh of this.meshes.values()) {
      mesh.vertexBuffer.destroy()
      mesh.indexBuffer.destroy()
      mesh.noteTexture.destroy()
    }
    this.meshes.clear()
    this.context.unconfigure()
    this.lease.release()
  }
}
