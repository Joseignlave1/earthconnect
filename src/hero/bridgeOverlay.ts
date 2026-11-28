import { FOV } from '../shader-test/geometry.ts'
import { shaderModule } from '../shader-test/gpu.ts'
import {
  add,
  dot,
  length,
  normalized,
  relativeToEye,
  scale,
  subtract,
  type SolarFrame,
} from '../workspace/solarCamera.ts'
import type { Vec3 } from '../workspace/solarSystem.ts'
import bridgeWgsl from './bridge.wgsl?raw'

type BridgePipelines = {
  arc: GPURenderPipeline
  anchor: GPURenderPipeline
  layout: GPUBindGroupLayout
}

const bridgePipelineCache = new WeakMap<GPUDevice, Promise<BridgePipelines>>()

function getBridgePipelines(device: GPUDevice): Promise<BridgePipelines> {
  let pending = bridgePipelineCache.get(device)
  if (!pending) {
    pending = (async () => {
      const module = await shaderModule(device, 'The Bridge telemetry filament', bridgeWgsl)
      const layout = device.createBindGroupLayout({
        label: 'Bridge filament uniforms',
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
      })
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })
      const additive: GPUBlendState = {
        color: { srcFactor: 'one', dstFactor: 'one' },
        alpha: { srcFactor: 'zero', dstFactor: 'one' },
      }
      const depthStencil: GPUDepthStencilState = {
        format: 'depth32float',
        depthWriteEnabled: false,
        depthCompare: 'greater',
      }
      const [arc, anchor] = await Promise.all([
        device.createRenderPipelineAsync({
          label: 'Bridge 3D filament arc',
          layout: pipelineLayout,
          multisample: { count: 4 },
          vertex: {
            module,
            entryPoint: 'arcVertex',
            buffers: [
              {
                arrayStride: 32,
                stepMode: 'instance',
                attributes: [
                  { shaderLocation: 0, offset: 0, format: 'float32x3' },
                  { shaderLocation: 1, offset: 12, format: 'float32' },
                  { shaderLocation: 2, offset: 16, format: 'float32x3' },
                  { shaderLocation: 3, offset: 28, format: 'float32' },
                ],
              },
            ],
          },
          fragment: {
            module,
            entryPoint: 'arcFragment',
            targets: [{ format: 'rgba16float', blend: additive }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        }),
        device.createRenderPipelineAsync({
          label: 'Bridge surface anchors',
          layout: pipelineLayout,
          multisample: { count: 4 },
          vertex: {
            module,
            entryPoint: 'anchorVertex',
            buffers: [
              {
                arrayStride: 20,
                stepMode: 'instance',
                attributes: [
                  { shaderLocation: 0, offset: 0, format: 'float32x3' },
                  { shaderLocation: 1, offset: 12, format: 'float32' },
                  { shaderLocation: 2, offset: 16, format: 'float32' },
                ],
              },
            ],
          },
          fragment: {
            module,
            entryPoint: 'anchorFragment',
            targets: [{ format: 'rgba16float', blend: additive }],
          },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil,
        }),
      ])
      return { arc, anchor, layout }
    })()
    bridgePipelineCache.set(device, pending)
    void pending.catch(() => {
      if (bridgePipelineCache.get(device) === pending) bridgePipelineCache.delete(device)
    })
  }
  return pending
}

export type BridgeEndpoints = {
  marsPoint: Vec3
  marsNormal: Vec3
  earthPoint: Vec3
  earthNormal: Vec3
}

const ARC_SEGMENTS = 96
const ARC_FLOATS_PER_INSTANCE = 8
const ANCHOR_INSTANCE_COUNT = 2
const ANCHOR_FLOATS_PER_INSTANCE = 5

function cubicBezier(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3 {
  const u = 1 - t
  const b0 = u * u * u
  const b1 = 3 * u * u * t
  const b2 = 3 * u * t * t
  const b3 = t * t * t
  return [
    b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
    b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2],
  ]
}

export class BridgeOverlay {
  private readonly uniformData = new Float32Array(16)
  private readonly uniformBuffer: GPUBuffer
  private readonly arcData = new Float32Array(ARC_SEGMENTS * ARC_FLOATS_PER_INSTANCE)
  private readonly arcBuffer: GPUBuffer
  private readonly anchorData = new Float32Array(ANCHOR_INSTANCE_COUNT * ANCHOR_FLOATS_PER_INSTANCE)
  private readonly anchorBuffer: GPUBuffer
  private readonly bindGroup: GPUBindGroup
  private destroyed = false

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: BridgePipelines,
  ) {
    this.uniformBuffer = device.createBuffer({
      label: 'Bridge filament uniform buffer',
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.arcBuffer = device.createBuffer({
      label: 'Bridge filament ribbon vertices',
      size: this.arcData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.anchorBuffer = device.createBuffer({
      label: 'Bridge surface anchor vertices',
      size: this.anchorData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.bindGroup = device.createBindGroup({
      layout: pipelines.layout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    })
  }

  static async create(device: GPUDevice) {
    const pipelines = await getBridgePipelines(device)
    return new BridgeOverlay(device, pipelines)
  }

  draw(
    pass: GPURenderPassEncoder,
    frame: SolarFrame,
    endpoints: BridgeEndpoints,
    width: number,
    height: number,
    ratio: number,
    timeSeconds: number,
    reducedMotion: boolean,
    opacity = 1,
  ) {
    if (this.destroyed || opacity <= 0.001) return
    const tanY = Math.tan(FOV / 2)
    const tanX = tanY * frame.aspect
    const halfWidthPx = 5.5 * ratio
    this.uniformData.set([width, height, tanX, tanY], 0)
    this.uniformData.set([frame.near, timeSeconds, reducedMotion ? 1 : 0, opacity], 4)
    // Warm Martian amber-ivory at origin -> luminous Magma gold-coral at Earth peak.
    this.uniformData.set([1.28, 0.72, 0.44, halfWidthPx], 8)
    this.uniformData.set([1.55, 0.62, 0.36, 0], 12)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData)

    // Compute a single graceful, non-hooking 3D cubic Bezier arch sweeping across
    // space from Mars (p0) into Earth's glowing analogue peak (p3).
    const p0 = add(endpoints.marsPoint, scale(endpoints.marsNormal, 22))
    const p3 = add(endpoints.earthPoint, scale(endpoints.earthNormal, 40))
    const chord = subtract(p3, p0)
    const span = length(chord)
    const chordDir = normalized(chord)
    // On wide desktop, bow gently up-and-left across the Milky Way; on portrait mobile,
    // use a tighter gentle upward-right bow staying inside the narrow viewport while
    // lifting cleanly off Mars's foreground surface.
    const narrow = frame.aspect < 0.95
    const rawLift = narrow
      ? add(scale(frame.right, 0.48), scale(frame.up, 0.32))
      : add(scale(frame.up, 0.88), scale(frame.right, -0.34))
    const liftDir = normalized(subtract(rawLift, scale(chordDir, dot(rawLift, chordDir))))
    const archHeight = span * (narrow ? 0.048 : 0.125)
    const p1 = add(
      add(p0, scale(chordDir, span * (narrow ? 0.12 : 0.3))),
      add(scale(liftDir, archHeight), scale(endpoints.marsNormal, narrow ? 4600 : 1200)),
    )
    const p2 = add(
      add(p3, scale(chordDir, -span * (narrow ? 0.26 : 0.28))),
      add(scale(liftDir, archHeight * 0.92), scale(endpoints.earthNormal, narrow ? 1400 : 1600)),
    )

    let prev = relativeToEye(frame, p0)
    let offset = 0
    for (let i = 0; i < ARC_SEGMENTS; i++) {
      const t0 = i / ARC_SEGMENTS
      const t1 = (i + 1) / ARC_SEGMENTS
      const next = relativeToEye(frame, cubicBezier(p0, p1, p2, p3, t1))
      this.arcData[offset++] = prev[0]
      this.arcData[offset++] = prev[1]
      this.arcData[offset++] = prev[2]
      this.arcData[offset++] = t0
      this.arcData[offset++] = next[0]
      this.arcData[offset++] = next[1]
      this.arcData[offset++] = next[2]
      this.arcData[offset++] = t1
      prev = next
    }
    this.device.queue.writeBuffer(this.arcBuffer, 0, this.arcData)

    const marsAnchor = relativeToEye(frame, add(endpoints.marsPoint, scale(endpoints.marsNormal, 26)))
    const earthAnchor = relativeToEye(frame, add(endpoints.earthPoint, scale(endpoints.earthNormal, 48)))
    this.anchorData.set([marsAnchor[0], marsAnchor[1], marsAnchor[2], 0, 10.5 * ratio], 0)
    this.anchorData.set([earthAnchor[0], earthAnchor[1], earthAnchor[2], 1, 9.5 * ratio], 5)
    this.device.queue.writeBuffer(this.anchorBuffer, 0, this.anchorData)

    pass.setPipeline(this.pipelines.arc)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.arcBuffer)
    pass.draw(6, ARC_SEGMENTS)

    pass.setPipeline(this.pipelines.anchor)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.anchorBuffer)
    pass.draw(6, ANCHOR_INSTANCE_COUNT)
  }

  destroy() {
    if (this.destroyed) return
    this.destroyed = true
    this.uniformBuffer.destroy()
    this.arcBuffer.destroy()
    this.anchorBuffer.destroy()
  }
}
