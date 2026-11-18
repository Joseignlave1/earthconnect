import type { WorkspaceAssets } from './workspaceAssets.ts'
import { FOV } from '../shader-test/geometry.ts'
import { fullscreenVertex, shaderModule } from '../shader-test/gpu.ts'
import { PlanetTerrainRenderer, type PlanetTerrain } from './planetTerrain.ts'
import { orbitPolyline, solarSystemAt, type BodyId, type Vec3 } from './solarSystem.ts'
import { loadSkyMap } from './skyMap.ts'
import {
  add, bodyMatrix, dot, length, normalized, raySphere, relativeToEye, scale, sceneRay, SolarCamera,
  subtract, worldToGeographic, type SolarFrame, type SourceBody,
} from './solarCamera.ts'

const ids: readonly BodyId[] = ['sun','mercury','venus','earth','moon','mars','jupiter','saturn','uranus','neptune']
const sources: readonly SourceBody[] = ['moon','mars']
const colors: Record<BodyId, Vec3> = {
  sun: [4,2.59,1.29],
  mercury: [.17,.14,.12],
  venus: [.57,.46,.27],
  earth: [.10,.19,.39],
  moon: [.32,.30,.32],
  mars: [.53,.26,.16],
  jupiter: [.62,.46,.35],
  saturn: [.59,.49,.36],
  uranus: [.24,.45,.50],
  neptune: [.18,.27,.57],
}

const commonShader = `
struct View { right: vec4f, up: vec4f, back: vec4f, light: vec4f, }
@group(0) @binding(0) var<uniform> view: View;
fn viewRay(point: vec2f) -> vec3f {
  let p = vec2f(point.x * view.right.w, point.y) * view.up.w;
  return normalize(vec3f(p,-1.));
}
`
const skyShader = `${fullscreenVertex}${commonShader}
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlasSampler: sampler;
const PI = 3.141592653589793;
fn mapGradient(equatorial: vec3f,directionDelta: vec3f) -> vec2f {
  let radiusSquared = max(1.e-8,dot(equatorial.xy,equatorial.xy));
  let raDelta = (equatorial.x * directionDelta.y - equatorial.y * directionDelta.x) / radiusSquared;
  let decDelta = directionDelta.z / sqrt(radiusSquared);
  return -vec2f(raDelta / (2. * PI),decDelta / PI);
}
@fragment fn skyFragment(input: ScreenVertex) -> @location(0) vec4f {
  let ray = viewRay(vec2f(input.uv.x * 2. - 1.,1. - input.uv.y * 2.));
  let ecliptic = view.right.xyz * ray.x + view.up.xyz * ray.y + view.back.xyz * ray.z;
  // Rotate the world direction to the J2000 mean equator by the obliquity.
  let equatorial = vec3f(ecliptic.x,.917482143 * ecliptic.y - .397776969 * ecliptic.z,
    .397776969 * ecliptic.y + .917482143 * ecliptic.z);
  let ra = atan2(equatorial.y,equatorial.x);
  let dec = atan2(equatorial.z,length(equatorial.xy));
  // The NASA map has 0h in the middle and increasing RA to the left.
  let uv = vec2f(.5 - ra / (2. * PI),.5 - dec / PI);
  // Differentiate the direction to preserve the footprint at the RA seam.
  let dx = mapGradient(equatorial,dpdx(equatorial));
  let dy = mapGradient(equatorial,dpdy(equatorial));
  return textureSampleGrad(atlas,atlasSampler,uv,dx,dy);
}
`

const bodyShader = `${commonShader}
struct Body { position: vec4f, color: vec4f, flags: vec4f, }
@group(0) @binding(1) var<storage,read> bodies: array<Body>;
struct Quad { @builtin(position) position: vec4f, @location(0) ndc: vec2f,
  @location(1) @interpolate(flat) body: u32, }
@vertex fn bodyVertex(@builtin(vertex_index) vertex: u32,@builtin(instance_index) instance: u32) -> Quad {
  let corners = array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.));
  let body = bodies[instance];
  let position = body.position.xyz;
  let radius = body.position.w;
  let depth = -position.z;
  let tan = view.up.w;
  let aspect = view.right.w;
  let projection = vec2f(tan * aspect,tan);
  if (depth + radius <= 0. || body.flags.x < .5) { return Quad(vec4f(2.,2.,0.,1.),vec2f(2.),instance); }
  let center = position.xy / max(depth,.001) / projection;
  let extent = radius * (1. + abs(position.xy) / max(depth,.001)) / max(depth - radius,1.) / projection;
  let minSize = body.flags.y * view.light.w / tan;
  let corner = center + corners[vertex] * max(extent,vec2f(minSize / aspect,minSize));
  return Quad(vec4f(corner,0.,1.),corner,instance);
}
struct Pixel { @location(0) color: vec4f, @builtin(frag_depth) depth: f32, }
@fragment fn bodyFragment(input: Quad) -> Pixel {
  let body = bodies[input.body];
  let position = body.position.xyz;
  let radius = body.position.w;
  let dir = viewRay(input.ndc);
  let middle = dot(position,dir);
  let offset = position - dir * middle;
  let gap = radius * radius - dot(offset,offset);
  let worldRadius = middle * view.light.w * body.flags.y;
  if (middle <= 0. || dot(offset,offset) > max(radius * radius,worldRadius * worldRadius)) { discard; }
  let intersection = middle - sqrt(max(gap,0.));
  let depth = -dir.z * intersection;
  if (depth <= view.back.w) { discard; }
  let physicalNormal = (dir * (intersection - middle) - offset) / radius;
  let normal = select(-dir,physicalNormal,gap >= 0.);
  let diffuse = clamp(dot(normal,normalize(view.light.xyz - position)),0.,1.);
  let glint = 1. - smoothstep(.16,1.,length(offset) / max(worldRadius,.001));
  let edge = select(mix(1.,glint,body.flags.z),1.,gap >= 0.);
  let surface = body.color.xyz * (edge * select(.60 + .30 * diffuse,.18 + .90 * diffuse,gap >= 0.) + .84 * glint * body.flags.z);
  let sun = body.color.xyz * (.63 + .37 * max(0.,-dot(normal,dir)));
  let radiance = select(surface,sun,body.color.w > .5);
  return Pixel(vec4f(radiance,1.),view.back.w / depth);
}
`

const haloShader = `${fullscreenVertex}${commonShader}
struct Halo { @location(0) color: vec4f, @builtin(frag_depth) depth: f32, }
@fragment fn haloFragment(input: ScreenVertex) -> Halo {
  let ray = viewRay(vec2f(input.uv.x * 2. - 1.,1. - input.uv.y * 2.));
  let sun = view.light.xyz;
  let distance = dot(ray,sun);
  if (distance <= view.back.w) { discard; }
  let offset = length(sun - ray * distance);
  let width = max(695700. * 1.25,distance * view.light.w * 2.6);
  let radius = offset / width;
  if (radius >= 4.) { discard; }
  let intensity = .17 * exp(-radius * radius) + .12 * exp(-5. * radius * radius);
  return Halo(vec4f(vec3f(1.,.57,.24) * intensity,0.),view.back.w / (-ray.z * distance));
}
`

const orbitShader = `
struct Vertex { @builtin(position) position: vec4f, @location(0) color: vec4f, }
@vertex fn orbitVertex(@location(0) position: vec4f,@location(1) color: vec4f) -> Vertex {
  return Vertex(position,color);
}
@fragment fn orbitFragment(input: Vertex) -> @location(0) vec4f { return input.color; }
`

export type SourcePoint = { latitude: number; longitude: number }
export type SolarLabel = { id: BodyId; x: number; y: number }

export class SolarScene {
  readonly bodies
  private terrain?: PlanetTerrainRenderer
  private loadingTerrain = false
  skySettled = false
  private readonly controller = new AbortController()
  private readonly viewBuffer: GPUBuffer
  private readonly bodyBuffer: GPUBuffer
  private readonly orbitBuffer: GPUBuffer
  private readonly orbits: { id: BodyId; vertices: readonly Vec3[] }[]
  private readonly bodyData = new Float32Array(ids.length * 12)
  private readonly orbitData: Float32Array<ArrayBuffer>
  private skyGroup: GPUBindGroup
  private skyTexture: GPUTexture
  private readonly skySampler: GPUSampler
  private readonly haloGroup: GPUBindGroup
  private readonly bodyGroup: GPUBindGroup

  static async create(device: GPUDevice,date: Date,invalidate: () => void,onError: (error: Error) => void,camera: SolarCamera | undefined, assetUrl: WorkspaceAssets['url']) {
    const [sky,body,orbit,halo] = await Promise.all([
      shaderModule(device,'Inertial sky',skyShader),
      shaderModule(device,'Solar bodies',bodyShader),
      shaderModule(device,'Planetary paths',orbitShader),
      shaderModule(device,'Sunlight halo',haloShader),
    ])
    const color: GPUColorTargetState = { format: 'rgba16float' }
    const depth: GPUDepthStencilState = { format: 'depth32float',depthWriteEnabled: true,depthCompare: 'greater' }
    const common = { layout: 'auto' as const,multisample: { count: 4 } }
    const [skyPipeline,bodyPipeline,orbitPipeline,haloPipeline] = await Promise.all([
      device.createRenderPipelineAsync({ ...common,label: 'Inertial sky',
        vertex: { module: sky,entryPoint: 'screenVertex' },fragment: { module: sky,entryPoint: 'skyFragment',targets: [color] },
        depthStencil: { ...depth,depthWriteEnabled: false,depthCompare: 'always' },
      }),
      device.createRenderPipelineAsync({ ...common,label: 'Sun and planet disks',
        vertex: { module: body,entryPoint: 'bodyVertex' },fragment: { module: body,entryPoint: 'bodyFragment',targets: [color] },depthStencil: depth,
      }),
      device.createRenderPipelineAsync({ ...common,label: 'Ephemeris orbit paths',
        vertex: { module: orbit,entryPoint: 'orbitVertex',buffers: [{
          arrayStride: 32,attributes: [{ shaderLocation: 0,offset: 0,format: 'float32x4' },{ shaderLocation: 1,offset: 16,format: 'float32x4' }],
        }] },fragment: { module: orbit,entryPoint: 'orbitFragment',targets: [{ ...color,blend: {
          color: { srcFactor: 'one',dstFactor: 'one-minus-src-alpha' },alpha: { srcFactor: 'one',dstFactor: 'one-minus-src-alpha' },
        } }] },
        primitive: { topology: 'line-list' },depthStencil: { ...depth,depthWriteEnabled: false },
      }),
      device.createRenderPipelineAsync({ ...common,label: 'Solar halo with occlusion',
        vertex: { module: halo,entryPoint: 'screenVertex' },fragment: { module: halo,entryPoint: 'haloFragment',targets: [{ ...color,blend: {
          color: { srcFactor: 'one',dstFactor: 'one' },alpha: { srcFactor: 'zero',dstFactor: 'one' },
        } }] },
        depthStencil: { ...depth,depthWriteEnabled: false },
      }),
    ])
    return new SolarScene(device,date,skyPipeline,bodyPipeline,orbitPipeline,haloPipeline,invalidate,onError,camera ?? new SolarCamera(),assetUrl)
  }

  private constructor(private readonly device: GPUDevice,date: Date,private readonly skyPipeline: GPURenderPipeline,
    private readonly bodyPipeline: GPURenderPipeline,private readonly orbitPipeline: GPURenderPipeline,
    private readonly haloPipeline: GPURenderPipeline,private readonly invalidate: () => void,private readonly onError: (error: Error) => void,
    readonly camera: SolarCamera, private readonly assetUrl: WorkspaceAssets['url']) {
    this.bodies = solarSystemAt(date)
    this.orbits = ids.flatMap(id => {
      const vertices = orbitPolyline(id,date)
      return vertices.length ? [{ id,vertices }] : []
    })
    this.orbitData = new Float32Array(this.orbits.reduce((count,path) => count + (path.vertices.length - 1) * 16,0))
    this.orbitBuffer = device.createBuffer({ label: 'Camera-relative orbit lines',size: this.orbitData.byteLength,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX })
    this.viewBuffer = device.createBuffer({ label: 'Heliocentric camera basis',size: 64,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.bodyBuffer = device.createBuffer({ label: 'Solar system centers in camera space',size: this.bodyData.byteLength,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE })
    this.skySampler = device.createSampler({
      addressModeU: 'repeat',addressModeV: 'clamp-to-edge',minFilter: 'linear',magFilter: 'linear',mipmapFilter: 'linear',maxAnisotropy: 4,
    })
    this.skyTexture = device.createTexture({
      label: 'Dark sky placeholder',size: [1,1],format: 'rgba8unorm-srgb',usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    device.queue.writeTexture({ texture: this.skyTexture },new Uint8Array([6,6,7,255]),{},[1,1])
    this.skyGroup = this.bindSky(this.skyTexture)
    this.haloGroup = device.createBindGroup({ layout: haloPipeline.getBindGroupLayout(0),entries: [{ binding: 0,resource: { buffer: this.viewBuffer } }] })
    this.bodyGroup = device.createBindGroup({ layout: bodyPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: { buffer: this.bodyBuffer } },
    ] })
    this.loadTerrain()
    const { signal } = this.controller
    void this.assetUrl('textures/nasa-deep-star-map.webp', signal)
      .then(url => loadSkyMap(device,signal,url)).then(texture => {
      if (signal.aborted) { texture.destroy(); return }
      const group = this.bindSky(texture)
      this.skyTexture.destroy()
      this.skyTexture = texture
      this.skyGroup = group
    }).catch(() => {
      // Keep rendering the planets over dark space if the atlas is unavailable.
    }).finally(() => {
      if (signal.aborted) return
      this.skySettled = true
      this.invalidate()
    })
  }

  private bindSky(texture: GPUTexture) {
    return this.device.createBindGroup({ layout: this.skyPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: texture.createView() },{ binding: 2,resource: this.skySampler },
    ] })
  }

  get terrainReady() { return Boolean(this.terrain) }

  loadTerrain() {
    if (this.loadingTerrain || this.terrain || this.controller.signal.aborted) return
    this.loadingTerrain = true
    const { signal } = this.controller
    void PlanetTerrainRenderer.create(this.device,signal,this.assetUrl).then(terrain => {
      if (signal.aborted) { terrain.destroy(); return }
      this.terrain = terrain
      this.invalidate()
    },error => {
      if (!signal.aborted) this.onError(error instanceof Error ? error : new Error(String(error)))
    }).finally(() => { this.loadingTerrain = false })
  }

  sunlight(id: BodyId) {
    const body = this.bodies[id]
    return normalized(worldToGeographic(body,subtract(this.bodies.sun.center,body.center)))
  }

  private detailed(frame: SolarFrame,id: BodyId,height: number) {
    const body = this.bodies[id]
    const distance = length(subtract(body.center,frame.eye))
    return body.radius / distance * height / (2 * Math.tan(FOV / 2)) >= 4
  }

  drawEarth(frame: SolarFrame,height: number) { return this.detailed(frame,'earth',height) }

  earthMatrix(frame: SolarFrame) { return bodyMatrix(frame,this.bodies.earth,6378.137) }

  draw(pass: GPURenderPassEncoder,frame: SolarFrame,width: number,height: number,points: Partial<Record<SourceBody,SourcePoint>>,overview: boolean,flightTarget?: SourceBody | 'earth') {
    const queue = this.device.queue
    const tangent = Math.tan(FOV / 2)
    queue.writeBuffer(this.viewBuffer,0,new Float32Array([
      ...frame.right,frame.aspect,...frame.up,tangent,...frame.back,frame.near,
      ...relativeToEye(frame,this.bodies.sun.center),tangent * 2 / height,
    ]))
    pass.setPipeline(this.skyPipeline)
    pass.setBindGroup(0,this.skyGroup)
    pass.draw(3)
    for (const [index,id] of ids.entries()) {
      const body = this.bodies[id]
      const mesh = (id === 'earth' || (id === 'moon' || id === 'mars') && this.terrain?.getTerrain(id)) && this.detailed(frame,id,height)
      const color = colors[id]
      const position = relativeToEye(frame,body.center)
      let glint = 0
      if (!mesh && id === flightTarget && -position[2] > frame.near) {
        const [x,y,z] = position
        const radiusPixels = body.radius * height / (-2 * z * tangent)
        const nx = x / (-z * tangent * frame.aspect),ny = y / (-z * tangent)
        if (Math.abs(nx) < 1 && Math.abs(ny) < 1 &&
            this.visiblePoint(frame,nx,ny,id,length(subtract(body.center,frame.eye)) - body.radius)) {
          glint = Math.max(0,Math.min(1,(3.5 - radiusPixels) / 2.5))
        }
      }
      this.bodyData.set([
        ...position,body.radius,
        ...color,Number(id === 'sun'),
        Number(!mesh),id === 'sun' ? 0 : overview ? 1.9 : 1 + 2.2 * glint,glint,0,
      ],index * 12)
    }
    queue.writeBuffer(this.bodyBuffer,0,this.bodyData)
    pass.setPipeline(this.bodyPipeline)
    pass.setBindGroup(0,this.bodyGroup)
    pass.draw(6,ids.length)

    for (const id of sources) {
      if (this.terrain?.getTerrain(id) && this.detailed(frame,id,height)) {
        this.terrain.draw(pass,id,bodyMatrix(frame,this.bodies[id],1),this.sunlight(id),points[id],[width,height])
      }
    }
  }

  drawHalo(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.haloPipeline)
    pass.setBindGroup(0,this.haloGroup)
    pass.draw(3)
  }

  labels(frame: SolarFrame): SolarLabel[] {
    const tan = Math.tan(FOV / 2)
    return ids.flatMap(id => {
      if (id === 'moon') return []
      const body = this.bodies[id]
      const [x,y,z] = relativeToEye(frame,body.center)
      const px = .5 - x / (2 * z * tan * frame.aspect)
      const py = .5 + y / (2 * z * tan)
      if (z >= -frame.near || px < 0 || px > 1 || py < 0 || py > 1) return []
      const distance = length(subtract(frame.eye,body.center)) - body.radius
      if (!this.visiblePoint(frame,px * 2 - 1,1 - py * 2,id,distance)) return []
      return [{ id,x: px,y: py }]
    })
  }

  drawOrbits(pass: GPURenderPassEncoder,frame: SolarFrame,enabled: boolean) {
    if (!enabled) return
    const f = 1 / Math.tan(FOV / 2)
    let offset = 0
    for (const { id,vertices } of this.orbits) {
      const color = colors[id]
      const alpha = .38
      const style = [...scale(color,alpha),alpha]
      const addVertex = (world: Vec3) => {
        const [x,y,z] = relativeToEye(frame,world)
        this.orbitData.set([x * f / frame.aspect,y * f,frame.near,-z,...style],offset)
        offset += 8
      }
      for (let i = 1; i < vertices.length; i++) {
        addVertex(vertices[i - 1])
        addVertex(vertices[i])
      }
    }
    this.device.queue.writeBuffer(this.orbitBuffer,0,this.orbitData)
    pass.setPipeline(this.orbitPipeline)
    pass.setVertexBuffer(0,this.orbitBuffer)
    pass.draw(offset / 8)
  }

  visiblePoint(frame: SolarFrame,nx: number,ny: number,focus: BodyId,distance: number) {
    const ray = sceneRay(frame,nx,ny)
    return !ids.some(id => {
      if (id === focus) return false
      const body = this.bodies[id]
      return raySphere(frame.eye,ray,body.center,body.radius) < distance
    })
  }

  pickSource(frame: SolarFrame,body: SourceBody,nx: number,ny: number): SourcePoint | null {
    const terrain = this.terrain?.getTerrain(body)
    if (!terrain) return null
    const state = this.bodies[body]
    const ray = sceneRay(frame,nx,ny)
    if (!Number.isFinite(raySphere(frame.eye,ray,state.center,terrain.radius + 100))) return null
    const eye = worldToGeographic(state,subtract(frame.eye,state.center))
    const direction = worldToGeographic(state,ray)
    const distance = pickTerrain(terrain,eye,direction)
    if (!Number.isFinite(distance) || !this.visiblePoint(frame,nx,ny,body,distance)) return null
    const [x,y,z] = add(eye,scale(direction,distance))
    return { longitude: Math.atan2(x,z) * 180 / Math.PI,latitude: Math.atan2(y,Math.hypot(x,z)) * 180 / Math.PI }
  }

  destroy() {
    this.controller.abort()
    this.terrain?.destroy()
    this.skyTexture.destroy()
    this.viewBuffer.destroy()
    this.bodyBuffer.destroy()
    this.orbitBuffer.destroy()
  }
}

// Ray/triangle intersection on the exact indexed vertex positions in the pass.
function pickTerrain({ indices,positions }: PlanetTerrain,eye: Vec3,ray: Vec3) {
  const [ox,oy,oz] = eye, [dx,dy,dz] = ray
  let nearest = Infinity
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3,b = indices[i + 1] * 3,c = indices[i + 2] * 3
    const ax = positions[a],ay = positions[a + 1],az = positions[a + 2]
    const ex = positions[b] - ax,ey = positions[b + 1] - ay,ez = positions[b + 2] - az
    const fx = positions[c] - ax,fy = positions[c + 1] - ay,fz = positions[c + 2] - az
    const px = dy * fz - dz * fy,py = dz * fx - dx * fz,pz = dx * fy - dy * fx
    const determinant = ex * px + ey * py + ez * pz
    if (Math.abs(determinant) < 1e-8) continue
    const sx = ox - ax,sy = oy - ay,sz = oz - az
    const u = (sx * px + sy * py + sz * pz) / determinant
    if (u < -1e-8 || u > 1 + 1e-8) continue
    const qx = sy * ez - sz * ey,qy = sz * ex - sx * ez,qz = sx * ey - sy * ex
    const v = (dx * qx + dy * qy + dz * qz) / determinant
    if (v < -1e-8 || u + v > 1 + 1e-8) continue
    const distance = dot([fx,fy,fz],[qx,qy,qz]) / determinant
    if (distance > 0 && distance < nearest) nearest = distance
  }
  return nearest
}
