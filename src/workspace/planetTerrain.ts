import { colorShader } from '../shader-test/color.ts'
import { contourDetail, contourShader } from '../shader-test/contours.ts'
import { createMesh, type Vec3 } from '../shader-test/geometry.ts'
import { fullscreenVertex, shaderModule, uploadBuffer, writeTextureChunked } from '../shader-test/gpu.ts'
import { bodyRadius } from './solarSystem.ts'
import { elevationAt, elevationPyramid, packRg8Levels, poleElevations, type Terrain } from '../shader-test/terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from '../shader-test/terrain-loader.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'

export type TerrainBody = 'moon' | 'mars'
export type TerrainPoint = { longitude: number; latitude: number }

// sRGB fallback albedo stops linearized in the fragment shader when photographic albedo is absent.
const settings = {
  moon: { relief: 2.2, interval: 1500, low: [.24, .25, .27], high: [.64, .65, .68] },
  mars: { relief: 3.0, interval: 2400, low: [.28, .12, .07], high: [.72, .39, .24] },
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

function normalAt(longitude: number, latitude: number): Vec3 {
  if (Math.abs(latitude) === Math.PI / 2) return [0, Math.sign(latitude), 0]
  const c = Math.cos(latitude)
  return [c * Math.sin(longitude), Math.sin(latitude), c * Math.cos(longitude)]
}

const planetColumns = 512, planetRows = 256
let sharedPlanetGrid: ReturnType<typeof createMesh> | undefined

function getPlanetGrid() {
  return (sharedPlanetGrid ??= createMesh(planetColumns, planetRows))
}

export type PlanetTerrain = ReturnType<typeof buildPlanetTerrain>

export function buildPlanetTerrain(body: TerrainBody, terrain: Terrain) {
  const { relief } = settings[body]
  const radius = bodyRadius(body)
  const columns = planetColumns, rows = planetRows
  const { uv, indices } = getPlanetGrid()
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
  const blocks: { x: number; y: number; endX: number; endY: number; min: Vec3; max: Vec3 }[] = []
  for (let y = 0; y < rows; y += 8) for (let x = 0; x < columns; x += 8) {
    const endX = Math.min(x + 8, columns), endY = Math.min(y + 8, rows)
    const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity]
    for (let row = y; row <= endY; row++) for (let col = x; col <= endX; col++) {
      const i = (row * (columns + 1) + col) * 3
      for (let axis = 0; axis < 3; axis++) {
        const value = positions[i + axis]
        if (value < min[axis]) min[axis] = value
        if (value > max[axis]) max[axis] = value
      }
    }
    blocks.push({ x, y, endX, endY, min, max })
  }

  return { radius, columns, rows, positions, indices, blocks, elevationAt: readElevation, positionAt }
}

const mipShader = /* wgsl */ `${fullscreenVertex}
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@fragment fn mipFragment(input: ScreenVertex) -> @location(0) vec4f {
  return textureSampleLevel(source, linearSampler, input.uv, 0.);
}`

const surfaceShader = /* wgsl */ `
struct View {
  camera: mat4x4f,
  sun: vec4f, // body-fixed sun direction, HDR flag
  terrain: vec4f, // reference radius metres, vertical exaggeration, major contours, DPR
  low: vec4f, // sRGB low albedo, minor contour detail
  high: vec4f, // sRGB high albedo, fine contour detail
}
@group(0) @binding(0) var<uniform> u: View;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var albedoSampler: sampler;
${colorShader}
${contourShader}

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec3f,
}

@vertex fn vertexMain(@location(0) position: vec3f) -> Vertex {
  return Vertex(u.camera * vec4f(position, 1.), position);
}
fn dirToUv(dir: vec3f) -> vec2f {
  let pi = 3.141592653589793;
  let lon = atan2(dir.x, dir.z);
  let lat = atan2(dir.y, length(dir.xz));
  return vec2f(lon / (2. * pi) + .5, .5 - lat / pi);
}
fn decode(sample: vec4f) -> f32 {
  return dot(sample.rg, vec2f(65280., 255.)) - 32768.;
}
// 5-tap Catmull-Rom bicubic height filter for smooth zoomed-in crater rims & contours.
fn sampleHeightBicubic(uv: vec2f) -> f32 {
  let size = vec2f(textureDimensions(heights));
  let invSize = 1. / size;
  let pos = uv * size - .5;
  let f = fract(pos);
  let tc1 = floor(pos) + .5;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = f2 - .5 * (f3 + f);
  let w1 = 1.5 * f3 - 2.5 * f2 + 1.;
  let w3 = .5 * (f3 - f2);
  let w2 = 1. - w0 - w1 - w3;
  let w12 = w1 + w2;
  let tc0 = (tc1 - 1.) * invSize;
  let tc3 = (tc1 + 2.) * invSize;
  let tc12 = (tc1 + w2 / w12) * invSize;
  let s0 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc0.y), 0.));
  let s1 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc0.x, tc12.y), 0.));
  let s2 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc12.y), 0.));
  let s3 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc3.x, tc12.y), 0.));
  let s4 = decode(textureSampleLevel(heights, gridSampler, vec2f(tc12.x, tc3.y), 0.));
  let sum = s0 * (w12.x * w0.y) + s1 * (w0.x * w12.y) + s2 * (w12.x * w12.y)
    + s3 * (w3.x * w12.y) + s4 * (w12.x * w3.y);
  let total = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return sum / total;
}
fn sampleHeight(dir: vec3f, lod: f32) -> f32 {
  let uv = dirToUv(dir);
  let linear = decode(textureSampleLevel(heights, gridSampler, uv, lod));
  if (lod < 1.) {
    return mix(sampleHeightBicubic(uv), linear, smoothstep(0., 1., lod));
  }
  return linear;
}

@fragment fn fragmentMain(input: Vertex) -> @location(0) vec4f {
  let p = normalize(input.local);
  let pi = 3.141592653589793;
  let size = vec2f(textureDimensions(heights));
  // Measure pixel footprint in 3D radians on the sphere rather than UV derivatives,
  // avoiding longitude 1/cos(lat) mip explosion and anisotropy falloff near the poles.
  let texelAngle = pi / size.y;
  let pixelAngle = max(length(dpdx(p)), length(dpdy(p)));
  let lod = clamp(log2(max(1., pixelAngle / texelAngle)), 0., f32(textureNumLevels(heights) - 1u));
  let height = sampleHeight(p, lod);
  // Compute the surface gradient via 3D Cartesian perturbations projected onto the
  // sphere so finite differences stay constant-angle and singularity-free at both poles.
  let s = min(.25, texelAngle * exp2(lod));
  let grad = vec3f(
    sampleHeight(p + vec3f(s, 0., 0.), lod) - sampleHeight(p - vec3f(s, 0., 0.), lod),
    sampleHeight(p + vec3f(0., s, 0.), lod) - sampleHeight(p - vec3f(0., s, 0.), lod),
    sampleHeight(p + vec3f(0., 0., s), lod) - sampleHeight(p - vec3f(0., 0., s), lod),
  ) / (2. * s);
  let tangentGrad = (grad - p * dot(grad, p)) * (u.terrain.y / u.terrain.x);
  let hillNormal = normalize(p - tangentGrad);
  let sun = normalize(u.sun.xyz);
  let daylight = smoothstep(-.035, .045, dot(p, sun));
  // Match Earth's ambient + soft fill + directional sun using a pole-free 3D fill vector.
  let fillKey = normalize(sun * .55 + vec3f(0., .72, 0.) + cross(vec3f(0., 1., 0.), sun) * .42);
  let fill = max(0., dot(hillNormal, normalize(p + .7 * fillKey)));
  let illumination = .32 + .18 * fill + .56 * daylight * max(0., dot(hillNormal, sun));
  let contourHeight = sampleHeight(p, max(.4, lod));
  let contours = terrainContours(
    contourHeight,
    vec2f(dpdx(contourHeight), dpdy(contourHeight)),
    u.terrain.z,
    vec2f(u.low.w, u.high.w),
    u.terrain.w,
    0.,
  );
  let tone = clamp((height + 6000.) / 16500., 0., 1.);
  var surface = mix(u.low.rgb, u.high.rgb, tone);
  let albedoSize = vec2f(textureDimensions(albedo));
  let albedoLod = clamp(log2(max(1., pixelAngle / (pi / albedoSize.y))), 0., f32(textureNumLevels(albedo) - 1u));
  let albedoSample = textureSampleLevel(albedo, albedoSampler, dirToUv(p), albedoLod);
  let hasAlbedo = albedoSample.a > .5;
  if (hasAlbedo) {
    // Convert linear rgba8unorm-srgb sample into display space so contours and toLinear match Earth.
    surface = pow(max(albedoSample.rgb, vec3f(0.)), vec3f(1. / 2.2));
  }
  let lineColor = mix(surface * .52, surface + vec3f(.12, .125, .135), .64);
  surface = mix(surface, lineColor, select(.45, .32, hasAlbedo) * contours);
  var color = surface * illumination;
  if (u.sun.w > .5) { color = toLinear(color); }
  return vec4f(color, 1.);
}`

const annotationShader = /* wgsl */ `
struct AnnotationView {
  camera: mat4x4f,
  info: vec4f, // x: body u-offset (2 for moon, 4 for mars)
}
@group(0) @binding(0) var<uniform> view: AnnotationView;
struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec3f,
}
@vertex fn vertexMain(@location(0) position: vec3f) -> Vertex {
  return Vertex(view.camera * vec4f(position, 1.), position);
}
@fragment fn fragmentMain(input: Vertex) -> @location(0) vec2f {
  let p = normalize(input.local);
  let lon = atan2(p.x, p.z);
  let lat = atan2(p.y, length(p.xz));
  let pi = 3.141592653589793;
  return vec2f(lon / (2. * pi) + .5 + view.info.x, lat / pi + .5);
}`

type SharedBodyTerrain = {
  mesh: PlanetTerrain
  vertices: GPUBuffer
  indices: GPUBuffer
  heights: GPUTexture
  sampler: GPUSampler
}

type GpuTerrain = {
  shared: SharedBodyTerrain
  albedo: GPUTexture
  albedoSampler: GPUSampler
  parameters: Float32Array<ArrayBuffer>
  uniform: GPUBuffer
  bindGroup: GPUBindGroup
  loupeParameters: Float32Array<ArrayBuffer>
  loupeUniform: GPUBuffer
  loupeBindGroup: GPUBindGroup
  annotationUniform: GPUBuffer
  annotationBindGroup: GPUBindGroup
}

type PlanetPipelines = {
  surface: GPURenderPipeline
  annotation: GPURenderPipeline
  mip: GPURenderPipeline
}

type DevicePlanetCache = {
  controller: AbortController
  pipelines?: Promise<PlanetPipelines>
  indices?: GPUBuffer
  albedoSampler?: GPUSampler
  fallbackAlbedo?: GPUTexture
  bodies: WeakMap<Terrain, SharedBodyTerrain>
  albedos: Map<string, Promise<GPUTexture>>
  created: SharedBodyTerrain[]
  createdAlbedos: Set<GPUTexture>
}

const planetCache = new WeakMap<GPUDevice, DevicePlanetCache>()

function getPlanetCache(device: GPUDevice): DevicePlanetCache {
  let entry = planetCache.get(device)
  if (!entry) {
    const controller = new AbortController()
    const created: SharedBodyTerrain[] = []
    const createdAlbedos = new Set<GPUTexture>()
    entry = { controller, bodies: new WeakMap(), albedos: new Map(), created, createdAlbedos }
    planetCache.set(device, entry)
    void device.lost.then(() => {
      controller.abort()
      entry!.indices?.destroy()
      entry!.fallbackAlbedo?.destroy()
      for (const item of created) {
        item.vertices.destroy()
        item.heights.destroy()
      }
      for (const texture of createdAlbedos) texture.destroy()
      created.length = 0
      createdAlbedos.clear()
      planetCache.delete(device)
    })
  }
  return entry
}

function getPlanetPipelines(device: GPUDevice): Promise<PlanetPipelines> {
  const cache = getPlanetCache(device)
  if (!cache.pipelines) {
    const vertexBuffers: GPUVertexBufferLayout[] = [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] },
    ]
    const pending = Promise.all([
      shaderModule(device, 'Planetary DEM terrain shader', surfaceShader).then(module =>
        device.createRenderPipelineAsync({
          label: 'Planetary DEM terrain',
          layout: 'auto',
          vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
          fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }] },
          primitive: { topology: 'triangle-list', cullMode: 'back' },
          multisample: { count: 4 },
          depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
        })),
      shaderModule(device, 'Planetary DEM annotation shader', annotationShader).then(module =>
        device.createRenderPipelineAsync({
          label: 'Planetary DEM annotations',
          layout: 'auto',
          vertex: { module, entryPoint: 'vertexMain', buffers: vertexBuffers },
          fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rg32float' }] },
          primitive: { topology: 'triangle-list', cullMode: 'back' },
          depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
        })),
      shaderModule(device, 'Planetary albedo mipmaps', mipShader).then(module =>
        device.createRenderPipelineAsync({
          label: 'Planetary albedo linear mipmaps',
          layout: 'auto',
          vertex: { module, entryPoint: 'screenVertex' },
          fragment: { module, entryPoint: 'mipFragment', targets: [{ format: 'rgba8unorm-srgb' }] },
        })),
    ]).then(([surface, annotation, mip]) => ({ surface, annotation, mip }))
    cache.pipelines = pending
    void pending.catch(() => { if (cache.pipelines === pending) cache.pipelines = undefined })
  }
  return cache.pipelines
}

function getFallbackAlbedo(device: GPUDevice): GPUTexture {
  const cache = getPlanetCache(device)
  if (!cache.fallbackAlbedo) {
    const texture = device.createTexture({
      label: 'Planetary albedo fallback',
      size: [1, 1],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture({ texture }, new Uint8Array([255, 255, 255, 0]), {}, [1, 1])
    cache.fallbackAlbedo = texture
  }
  return cache.fallbackAlbedo
}

async function buildBodyAlbedo(device: GPUDevice, cache: DevicePlanetCache, body: TerrainBody, url: string, mipPipeline: GPURenderPipeline): Promise<GPUTexture> {
  const { signal } = cache.controller
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`${body} albedo download failed (${response.status}).`)
  const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none' })
  let texture: GPUTexture | undefined
  try {
    signal.throwIfAborted()
    const mipLevelCount = Math.floor(Math.log2(Math.max(bitmap.width, bitmap.height))) + 1
    texture = device.createTexture({
      label: `${body} global albedo`,
      size: [bitmap.width, bitmap.height],
      mipLevelCount,
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height])
    const sampler = (cache.albedoSampler ??= device.createSampler({
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    }))
    const encoder = device.createCommandEncoder({ label: `Build ${body} albedo mipmaps` })
    for (let level = 1; level < mipLevelCount; level++) {
      const group = device.createBindGroup({
        layout: mipPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: sampler },
        ],
      })
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }],
      })
      pass.setPipeline(mipPipeline)
      pass.setBindGroup(0, group)
      pass.draw(3)
      pass.end()
    }
    device.queue.submit([encoder.finish()])
    cache.createdAlbedos.add(texture)
    return texture
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

function loadSharedBodyAlbedo(device: GPUDevice, body: TerrainBody, url: string, mipPipeline: GPURenderPipeline, signal: AbortSignal): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  let pending = cache.albedos.get(url)
  if (!pending) {
    pending = buildBodyAlbedo(device, cache, body, url, mipPipeline)
    cache.albedos.set(url, pending)
    void pending.catch(() => { if (cache.albedos.get(url) === pending) cache.albedos.delete(url) })
  }
  return new Promise<GPUTexture>((resolve, reject) => {
    const abort = () => reject(signal.reason)
    signal.addEventListener('abort', abort, { once: true })
    if (signal.aborted) return abort()
    void pending.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort))
  })
}

function getSharedBodyTerrain(device: GPUDevice, body: TerrainBody, prepared: PreparedTerrain<'rg8unorm'>): SharedBodyTerrain {
  const cache = getPlanetCache(device)
  let shared = cache.bodies.get(prepared.terrain)
  if (!shared) {
    const { terrain } = prepared
    const levels = prepared.levels ?? packRg8Levels(elevationPyramid(terrain, true))
    const mesh = buildPlanetTerrain(body, terrain)
    const heights = device.createTexture({
      label: `${body} numeric terrain grid`,
      size: [terrain.width, terrain.height], mipLevelCount: levels.length,
      format: 'rg8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    for (const [mipLevel, { data, width, height }] of levels.entries()) {
      writeTextureChunked(device.queue, heights, mipLevel, data, width, height, 2)
    }
    const vertices = uploadBuffer(device, `${body} displaced terrain vertices`, mesh.positions, GPUBufferUsage.VERTEX)
    const indices = (cache.indices ??= uploadBuffer(device, 'Planetary displaced terrain indices', mesh.indices, GPUBufferUsage.INDEX))
    const sampler = device.createSampler({
      addressModeU: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    })
    shared = { mesh, heights, vertices, indices, sampler }
    cache.bodies.set(terrain, shared)
    cache.created.push(shared)
  }
  // Mip levels are only needed for GPU upload; drop CPU copy once resident on GPU.
  prepared.levels = null
  return shared
}

export class PlanetTerrainRenderer {
  private readonly bodies = new Map<TerrainBody, GpuTerrain>()
  private readonly device: GPUDevice
  private readonly pipeline: GPURenderPipeline
  private readonly annotationPipeline: GPURenderPipeline
  private readonly annotationScratch = new Float32Array(20)

  private constructor(device: GPUDevice, pipelines: PlanetPipelines) {
    this.device = device
    this.pipeline = pipelines.surface
    this.annotationPipeline = pipelines.annotation
  }

  static async create(device: GPUDevice, signal: AbortSignal, assetUrl: WorkspaceAssets['url'], loadTerrain?: TerrainLoader) {
    const bodies: TerrainBody[] = ['moon', 'mars']
    const pipelinesPromise = getPlanetPipelines(device)
    const [pipelines, terrain, albedos] = await Promise.all([
      pipelinesPromise,
      Promise.all(bodies.map(async body => {
        const path = `textures/${body}-terrain.bin`
        return loadTerrain
          ? loadTerrain(path, 'rg8unorm', s => assetUrl(path, s), signal)
          : loadPreparedTerrain(await assetUrl(path, signal), 'rg8unorm', signal)
      })),
      pipelinesPromise.then(({ mip }) => Promise.all(bodies.map(async body => {
        const path = `textures/${body}-albedo.webp`
        try {
          const url = await assetUrl(path, signal)
          return await loadSharedBodyAlbedo(device, body, url, mip, signal)
        } catch {
          if (signal.aborted) throw signal.reason
          return getFallbackAlbedo(device)
        }
      }))),
    ])
    signal.throwIfAborted()
    const renderer = new PlanetTerrainRenderer(device, pipelines)
    try {
      for (const [index, body] of bodies.entries()) {
        renderer.attach(body, terrain[index], albedos[index])
      }
      return renderer
    } catch (error) {
      renderer.destroy()
      throw error
    }
  }

  private attach(body: TerrainBody, prepared: PreparedTerrain<'rg8unorm'>, albedo: GPUTexture) {
    const device = this.device
    const cache = getPlanetCache(device)
    const shared = getSharedBodyTerrain(device, body, prepared)
    const albedoSampler = (cache.albedoSampler ??= device.createSampler({
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    }))
    const parameters = new Float32Array(32)
    const { relief, interval, low, high } = settings[body]
    parameters.set([shared.mesh.radius * 1000, relief, interval, 1], 20)
    parameters.set(low, 24)
    parameters.set(high, 28)
    const uniform = device.createBuffer({
      label: `${body} terrain parameters`,
      size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: shared.heights.createView() },
        { binding: 2, resource: shared.sampler },
        { binding: 3, resource: albedo.createView() },
        { binding: 4, resource: albedoSampler },
      ],
    })
    const loupeParameters = new Float32Array(parameters)
    const loupeUniform = device.createBuffer({
      label: `${body} loupe terrain parameters`,
      size: loupeParameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const loupeBindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: loupeUniform } },
        { binding: 1, resource: shared.heights.createView() },
        { binding: 2, resource: shared.sampler },
        { binding: 3, resource: albedo.createView() },
        { binding: 4, resource: albedoSampler },
      ],
    })
    const annotationUniform = device.createBuffer({
      label: `${body} annotation camera`,
      size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const annotationBindGroup = device.createBindGroup({
      layout: this.annotationPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: annotationUniform } }],
    })
    this.bodies.set(body, { shared, albedo, albedoSampler, parameters, uniform, bindGroup, loupeParameters, loupeUniform, loupeBindGroup, annotationUniform, annotationBindGroup })
  }

  getTerrain(body: TerrainBody) {
    return this.bodies.get(body)?.shared.mesh
  }

  drawAnnotation(pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array<ArrayBuffer>) {
    const data = this.bodies.get(body)
    if (!data) return
    const { shared: { mesh, vertices, indices }, annotationUniform, annotationBindGroup } = data
    this.annotationScratch.set(matrix, 0)
    this.annotationScratch[16] = body === 'moon' ? 2 : 4
    this.device.queue.writeBuffer(annotationUniform, 0, this.annotationScratch)
    pass.setPipeline(this.annotationPipeline)
    pass.setBindGroup(0, annotationBindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  draw(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    metresPerPixel: number, hdr: boolean, pixelRatio: number,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const { shared: { mesh, vertices, indices }, parameters, uniform, bindGroup } = data
    const [minor, fine] = contourDetail(metresPerPixel)
    parameters.set(matrix, 0)
    parameters.set(sunlight, 16)
    parameters[19] = Number(hdr)
    parameters[23] = pixelRatio
    parameters[27] = minor
    parameters[31] = fine
    this.device.queue.writeBuffer(uniform, 0, parameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  drawLoupe(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    metresPerPixel: number, hdr: boolean, pixelRatio: number,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    const { shared: { mesh, vertices, indices }, loupeParameters, loupeUniform, loupeBindGroup } = data
    const [minor, fine] = contourDetail(metresPerPixel)
    loupeParameters.set(matrix, 0)
    loupeParameters.set(sunlight, 16)
    loupeParameters[19] = Number(hdr)
    loupeParameters[23] = pixelRatio
    loupeParameters[27] = minor
    loupeParameters[31] = fine
    this.device.queue.writeBuffer(loupeUniform, 0, loupeParameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, loupeBindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  destroy() {
    for (const { uniform, loupeUniform, annotationUniform } of this.bodies.values()) {
      uniform.destroy()
      loupeUniform.destroy()
      annotationUniform.destroy()
    }
    this.bodies.clear()
  }
}
