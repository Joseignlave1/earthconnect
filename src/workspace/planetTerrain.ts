import { createMesh, type Vec3 } from '../shader-test/geometry.ts'
import { shaderModule, uploadBuffer } from '../shader-test/gpu.ts'
import { bodyRadius } from './solarSystem.ts'
import { elevationAt, elevationPyramid, loadTerrain, poleElevations, type Terrain } from '../shader-test/terrain.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'

export type TerrainBody = 'moon' | 'mars'
export type TerrainPoint = { longitude: number; latitude: number }

const settings = {
  moon: { relief: 3, interval: 1500, low: [.16, .17, .19], high: [.38, .38, .41] },
  mars: { relief: 4, interval: 2400, low: [.16, .078, .048], high: [.45, .26, .17] },
}

const degrees = Math.PI / 180
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

function normalAt(longitude: number, latitude: number): Vec3 {
  if (Math.abs(latitude) === Math.PI / 2) return [0, Math.sign(latitude), 0]
  const c = Math.cos(latitude)
  return [c * Math.sin(longitude), Math.sin(latitude), c * Math.cos(longitude)]
}

export type PlanetTerrain = ReturnType<typeof buildPlanetTerrain>

export function buildPlanetTerrain(body: TerrainBody, terrain: Terrain) {
  const { relief } = settings[body]
  const radius = bodyRadius(body)
  const columns = 512, rows = 256
  const { uv, indices } = createMesh(columns, rows)
  const [south, north] = poleElevations(terrain)
  const readElevation = (longitude: number, latitude: number) => {
    if (latitude <= -Math.PI / 2) return south
    if (latitude >= Math.PI / 2) return north
    return elevationAt(terrain, longitude, latitude)
  }
  const positions = new Float32Array(uv.length / 2 * 3)
  for (let vertex = 0; vertex < uv.length / 2; vertex++) {
    const longitude = (uv[vertex * 2] - .5) * 2 * Math.PI
    const latitude = (uv[vertex * 2 + 1] - .5) * Math.PI
    const magnitude = radius + readElevation(longitude, latitude) * relief / 1000
    const [x, y, z] = normalAt(longitude, latitude)
    positions.set([magnitude * x, magnitude * y, magnitude * z], vertex * 3)
  }

  const positionAt = (longitude: number, latitude: number): Vec3 => {
    const x = clamp((longitude / (2 * Math.PI) + .5) * columns, 0, columns)
    const y = clamp((latitude / Math.PI + .5) * rows, 0, rows)
    const column = Math.min(columns - 1, Math.floor(x))
    const row = Math.min(rows - 1, Math.floor(y))
    const dx = x - column, dy = y - row
    const bottomLeft = row * (columns + 1) + column
    const topLeft = bottomLeft + columns + 1
    // These weights use the same bottom-right to top-left diagonal as createMesh.
    const interpolate = (axis: number) => {
      const bottomRight = positions[(bottomLeft + 1) * 3 + axis]
      const top = positions[topLeft * 3 + axis]
      return dx + dy <= 1
        ? positions[bottomLeft * 3 + axis] * (1 - dx - dy) + bottomRight * dx + top * dy
        : positions[(topLeft + 1) * 3 + axis] * (dx + dy - 1) + bottomRight * (1 - dy) + top * (1 - dx)
    }
    return [interpolate(0), interpolate(1), interpolate(2)]
  }
  return { radius, columns, rows, positions, uv, indices, elevationAt: readElevation, positionAt }
}

const surfaceShader = /* wgsl */ `
struct View {
  camera: mat4x4f,
  sun: vec4f,
  terrain: vec4f, // reference radius metres, vertical exaggeration, major contours, DPR
  low: vec4f,
  high: vec4f,
  marker: vec4f, // local km, present
  viewport: vec4f,
}
@group(0) @binding(0) var<uniform> u: View;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec3f,
}

@vertex fn vertexMain(@location(0) position: vec3f) -> Vertex {
  return Vertex(u.camera * vec4f(position, 1.), position);
}
fn wrappedGradient(delta: vec2f) -> vec2f {
  return vec2f(delta.x - round(delta.x), delta.y);
}
fn decode(sample: vec4f) -> f32 {
  return dot(sample.rg, vec2f(65280., 255.)) - 32768.;
}
fn sampleHeight(uv: vec2f, lod: f32) -> f32 {
  return decode(textureSampleLevel(heights, gridSampler, uv, lod));
}
fn contourLine(height: f32, gradient: f32, interval: f32, width: f32) -> f32 {
  let level = round(height / interval);
  let distance = abs(height - level * interval) / max(.001, gradient);
  let radius = width * u.terrain.w * .5;
  let visible = 1. - smoothstep(.20, .52, gradient * u.terrain.w / interval);
  return (1. - smoothstep(radius - .35, radius + .35, distance)) * visible;
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  let p = normalize(input.local);
  let parallel = length(p.xz);
  let lon = atan2(p.x, p.z);
  let lat = atan2(p.y, parallel);
  let pi = 3.141592653589793;
  let uv = vec2f(lon / (2. * pi) + .5, .5 - lat / pi);
  let dx = wrappedGradient(dpdx(uv));
  let dy = wrappedGradient(dpdy(uv));
  let size = vec2f(textureDimensions(heights));
  let height = decode(textureSampleGrad(heights, gridSampler, uv, dx, dy));
  let footprint = max(1., max(length(dx * size), length(dy * size)));
  let lod = min(f32(textureNumLevels(heights) - 1u), log2(footprint));
  let step = exp2(lod) / size;
  let east = vec3f(cos(lon), 0., -sin(lon));
  let north = cross(p, east);
  let slopes = vec2f(
    (sampleHeight(uv + vec2f(step.x, 0.), lod) - sampleHeight(uv - vec2f(step.x, 0.), lod))
      / (2. * step.x * 2. * pi * max(.025, parallel)),
    (sampleHeight(uv - vec2f(0., step.y), lod) - sampleHeight(uv + vec2f(0., step.y), lod))
      / (2. * step.y * pi),
  ) * u.terrain.y / u.terrain.x;
  let hillNormal = normalize(p - east * slopes.x - north * slopes.y);
  let sun = normalize(u.sun.xyz);
  let daylight = smoothstep(-.085, .10, dot(p, sun));
  let illumination = .21 + daylight * (.18 + .75 * max(0., dot(hillNormal, sun)));
  let contourHeight = sampleHeight(uv, max(.8, lod));
  let gradient = length(vec2f(dpdx(contourHeight), dpdy(contourHeight)));
  let interval = u.terrain.z;
  let detail = 1. - smoothstep(3., 12., footprint);
  let contours = max(contourLine(contourHeight, gradient, interval, 1.1),
    detail * .56 * contourLine(contourHeight, gradient, interval * .5, .86));
  let tone = clamp((height + 6000.) / 16500., .0, 1.);
  var color = mix(u.low.rgb, u.high.rgb, tone) * illumination;
  let lineColor = mix(color * .42, color + vec3f(.055, .060, .068), .68);
  color = mix(color, lineColor, .60 * contours * (.4 + .6 * daylight));

  if (u.marker.w > .5) {
    let projection = u.camera * vec4f(u.marker.xyz, 1.);
    let markerNormal = normalize(u.marker.xyz);
    if (projection.w > 0. && dot(markerNormal, p) > .975) {
      let ndc = projection.xy / projection.w;
      let pixel = (ndc * vec2f(.5, -.5) + .5) * u.viewport.xy;
      let distance = length(input.position.xy - pixel) / u.terrain.w;
      color = mix(color, vec3f(.018, .025, .035), 1. - smoothstep(6.2, 7.5, distance));
      color = mix(color, vec3f(.55, .79, 1.), 1. - smoothstep(4.8, 6.2, distance));
      color = mix(color, vec3f(.04, .44, .97), 1. - smoothstep(2.8, 4.2, distance));
    }
  }
  return vec4f(color, 1.);
}`

type GpuTerrain = {
  mesh: PlanetTerrain
  parameters: Float32Array<ArrayBuffer>
  vertices: GPUBuffer
  indices: GPUBuffer
  uniform: GPUBuffer
  heights: GPUTexture
  bindGroup: GPUBindGroup
}

export class PlanetTerrainRenderer {
  private readonly bodies = new Map<TerrainBody, GpuTerrain>()
  private readonly device: GPUDevice
  private readonly pipeline: GPURenderPipeline

  private constructor(device: GPUDevice, pipeline: GPURenderPipeline) {
    this.device = device
    this.pipeline = pipeline
  }

  static async create(device: GPUDevice, signal: AbortSignal, assetUrl: WorkspaceAssets['url']) {
    const bodies: TerrainBody[] = ['moon', 'mars']
    const [module, ...terrain] = await Promise.all([
      shaderModule(device, 'Planetary DEM terrain shader', surfaceShader),
      ...bodies.map(async body => loadTerrain(await assetUrl(`textures/${body}-terrain.bin`, signal), signal)),
    ])
    signal.throwIfAborted()
    const pipeline = await device.createRenderPipelineAsync({
      label: 'Planetary DEM terrain',
      layout: 'auto',
      vertex: {
        module, entryPoint: 'vertexMain',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      multisample: { count: 4 },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    })
    signal.throwIfAborted()
    const renderer = new PlanetTerrainRenderer(device, pipeline)
    try {
      for (const [index, body] of bodies.entries()) {
        renderer.upload(body, terrain[index])
      }
      return renderer
    } catch (error) {
      renderer.destroy()
      throw error
    }
  }

  private upload(body: TerrainBody, terrain: Terrain) {
    const device = this.device
    const mesh = buildPlanetTerrain(body, terrain)
    const levels = elevationPyramid(terrain)
    const heights = device.createTexture({
      label: `${body} numeric terrain grid`,
      size: [terrain.width, terrain.height], mipLevelCount: levels.length,
      format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    for (const [mipLevel, { values, width, height }] of levels.entries()) {
      // Normalized linear filtering commutes with the affine two-channel
      // decode. Generate each mip from real metres before packing.
      const pixels = new Uint8Array(width * height * 4)
      for (let i = 0; i < values.length; i++) {
        const code = Math.round(values[i] + 32768)
        pixels[i * 4] = code >> 8
        pixels[i * 4 + 1] = code & 255
      }
      device.queue.writeTexture({ texture: heights, mipLevel }, pixels, { bytesPerRow: width * 4 }, [width, height])
    }
    const parameters = new Float32Array(40)
    const { relief, interval, low, high } = settings[body]
    parameters.set([mesh.radius * 1000, relief, interval, 1], 20)
    parameters.set(low, 24)
    parameters.set(high, 28)
    const uniform = device.createBuffer({
      label: `${body} terrain parameters`,
      size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const vertices = uploadBuffer(device, `${body} displaced terrain vertices`, mesh.positions, GPUBufferUsage.VERTEX)
    const indices = uploadBuffer(device, `${body} displaced terrain indices`, mesh.indices, GPUBufferUsage.INDEX)
    const sampler = device.createSampler({
      addressModeU: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8,
    })
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: heights.createView() },
        { binding: 2, resource: sampler },
      ],
    })
    this.bodies.set(body, { mesh, parameters, vertices, indices, uniform, heights, bindGroup })
  }

  getTerrain(body: TerrainBody) {
    return this.bodies.get(body)?.mesh
  }

  draw(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    point?: TerrainPoint | null, viewport?: readonly [number, number],
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const { mesh, parameters, vertices, indices, uniform, bindGroup } = data
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)
    parameters.set(matrix, 0)
    parameters.set(sunlight, 16)
    parameters[23] = pixelRatio
    if (point) {
      parameters.set(mesh.positionAt(point.longitude * degrees, point.latitude * degrees), 32)
      parameters[35] = 1
    } else {
      parameters[35] = 0
    }
    parameters.set(viewport ?? [window.innerWidth * pixelRatio, window.innerHeight * pixelRatio], 36)
    this.device.queue.writeBuffer(uniform, 0, parameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  destroy() {
    for (const { vertices, indices, uniform, heights } of this.bodies.values()) {
      vertices.destroy()
      indices.destroy()
      uniform.destroy()
      heights.destroy()
    }
    this.bodies.clear()
  }
}
