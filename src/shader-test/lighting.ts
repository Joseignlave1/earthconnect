import { fullscreenVertex, shaderModule } from './gpu.ts'

// Compute great-circle terrain gradients inline during solar illumination baking,
// avoiding a persistent 64 MiB rg32float intermediate slopes texture in VRAM.
const lightBindings = `
struct Lighting { sun: vec4f, } // Earth-fixed XYZ, terrain exaggeration
@group(0) @binding(0) var<uniform> light: Lighting;
@group(0) @binding(1) var source: texture_2d<f32>;
fn sunLight(normal: vec3f,up: vec3f) -> f32 {
  // Match Moon, Mars, and distant disks with a soft twilight terminator transition.
  let daylight = smoothstep(-.035,.045,dot(up,light.sun.xyz));
  return daylight * max(0.,dot(normal,light.sun.xyz));
}
fn polarLight(north: bool) -> f32 {
  let up = vec3f(0.,select(-1.,1.,north),0.);
  return .38 + .16 * inverseSqrt(2.) + .46 * sunLight(up,up);
}
`
const bakeShader = `${fullscreenVertex}${lightBindings}
fn heightTexel(p: vec2i) -> f32 {
  let size = vec2i(textureDimensions(source));
  return textureLoad(source,vec2i(((p.x % size.x) + size.x) % size.x,clamp(p.y,0,size.y - 1)),0).r;
}
fn heightAt(direction: vec3f) -> f32 {
  let size = vec2f(textureDimensions(source));
  let lon = atan2(direction.x,direction.z);
  let lat = atan2(direction.y,length(direction.xz));
  let p = vec2f(lon / 6.283185307 + .5,.5 - lat / 3.141592654) * size - .5;
  let base = vec2i(floor(p));
  let f = fract(p);
  return mix(mix(heightTexel(base),heightTexel(base + vec2i(1,0)),f.x),
    mix(heightTexel(base + vec2i(0,1)),heightTexel(base + vec2i(1,1)),f.x),f.y);
}
@fragment fn bake(input: ScreenVertex) -> @location(0) f32 {
  let size = vec2i(textureDimensions(source));
  let p = vec2i(input.position.xy);
  if (size.y > 1 && p.y == 0) { return polarLight(true); }
  if (size.y > 1 && p.y == size.y - 1) { return polarLight(false); }
  let latitude = (.5 - (f32(p.y) + .5) / f32(size.y)) * 3.141592654;
  let lon = ((f32(p.x) + .5) / f32(size.x) - .5) * 6.283185307;
  let up = vec3f(cos(latitude) * sin(lon),sin(latitude),cos(latitude) * cos(lon));
  let east = vec3f(cos(lon),0.,-sin(lon));
  let north = vec3f(-sin(latitude) * sin(lon),cos(latitude),-sin(latitude) * cos(lon));
  let e2 = .0066943799901413165;
  let denominator = 1. - e2 * sin(latitude) * sin(latitude);
  let eastRadius = 6378137. * inverseSqrt(denominator);
  let northRadius = 6378137. * (1. - e2) / pow(denominator,1.5);
  let angle = 3.141592654 / f32(size.y);
  let center = up * cos(angle);
  let dx = east * sin(angle);
  let dy = north * sin(angle);
  // Keep seafloor bathymetry relief sculpted and readable (continental slopes, mid-ocean ridges, trenches).
  let slopeScale = select(.62,1.,textureLoad(source,p,0).g >= 0.);
  let gradient = vec2f(
    (heightAt(center + dx) - heightAt(center - dx)) / (2. * angle * eastRadius),
    (heightAt(center + dy) - heightAt(center - dy)) / (2. * angle * northRadius),
  ) * slopeScale * light.sun.w;
  let localNormal = normalize(vec3f(-gradient,1.));
  let normal = localNormal.x * east + localNormal.y * north + localNormal.z * up;
  // Sculpted day/night relief while keeping night continents legible for score inspection.
  let fill = max(0.,dot(localNormal,normalize(vec3f(-.6,.8,1.))));
  return .38 + .16 * fill + .46 * sunLight(normal,up);
}`
const mipShader = `${fullscreenVertex}${lightBindings}
@fragment fn mip(input: ScreenVertex) -> @location(0) f32 {
  let sourceSize = vec2i(textureDimensions(source));
  let size = max(vec2i(1),sourceSize / 2);
  let p = vec2i(input.position.xy);
  if (size.y > 1 && p.y == 0) { return polarLight(true); }
  if (size.y > 1 && p.y == size.y - 1) { return polarLight(false); }
  let start = p * sourceSize / size;
  let end = (p + 1) * sourceSize / size;
  var sum = 0.;
  var count = 0.;
  for (var y = start.y; y < end.y; y++) { for (var x = start.x; x < end.x; x++) {
    sum += textureLoad(source,vec2i(x,y),0).r; count += 1.;
  } }
  return sum / count;
}`

type LightingPipelines = {
  bakePipeline: GPURenderPipeline
  mipPipeline: GPURenderPipeline
}

type SharedIllumination = { texture: GPUTexture; built: boolean }

type DeviceLightingCache = {
  pipelines?: Promise<LightingPipelines>
  illuminations: WeakMap<GPUTexture, Map<string, SharedIllumination>>
  created: GPUTexture[]
}

const lightingCache = new WeakMap<GPUDevice, DeviceLightingCache>()

function getLightingCache(device: GPUDevice): DeviceLightingCache {
  let entry = lightingCache.get(device)
  if (!entry) {
    const created: GPUTexture[] = []
    entry = { illuminations: new WeakMap(), created }
    lightingCache.set(device, entry)
    void device.lost.then(() => {
      for (const texture of created) texture.destroy()
      created.length = 0
      lightingCache.delete(device)
    })
  }
  return entry
}

function getLightingPipelines(device: GPUDevice): Promise<LightingPipelines> {
  const cache = getLightingCache(device)
  if (!cache.pipelines) {
    const pending = (async () => {
      const [bake,mip] = await Promise.all([
        shaderModule(device,'Solar lighting WGSL',bakeShader),shaderModule(device,'Lighting mipmaps WGSL',mipShader),
      ])
      const bakeLayout = device.createBindGroupLayout({ entries: [
        { binding: 0,visibility: GPUShaderStage.FRAGMENT,buffer: { type: 'uniform' } },
        { binding: 1,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'unfilterable-float' } },
      ] })
      const [bakePipeline,mipPipeline] = await Promise.all([
        device.createRenderPipelineAsync({ label: 'Bake frozen solar lighting',layout: device.createPipelineLayout({ bindGroupLayouts: [bakeLayout] }),vertex: { module: bake,entryPoint: 'screenVertex' },fragment: { module: bake,entryPoint: 'bake',targets: [{ format: 'r8unorm' }] } }),
        device.createRenderPipelineAsync({ label: 'Filter lighting mipmaps',layout: 'auto',vertex: { module: mip,entryPoint: 'screenVertex' },fragment: { module: mip,entryPoint: 'mip',targets: [{ format: 'r8unorm' }] } }),
      ])
      return { bakePipeline,mipPipeline }
    })()
    cache.pipelines = pending
    void pending.catch(() => { if (cache.pipelines === pending) cache.pipelines = undefined })
  }
  return cache.pipelines
}

export class TerrainLighting {
  texture: GPUTexture
  private textureOwned = true
  private terrain: GPUTexture | null = null
  private illumination: SharedIllumination | null = null
  private sunKey: string
  private uniformScratch: Float32Array<ArrayBuffer>
  private uniforms: GPUBuffer
  private bakePipeline!: GPURenderPipeline
  private mipPipeline!: GPURenderPipeline
  private bakeGroup!: GPUBindGroup
  private mipGroups: GPUBindGroup[] = []
  private exaggeration: number | null = null

  constructor(private device: GPUDevice,sun: readonly number[]) {
    this.sunKey = sun.join(',')
    this.uniformScratch = new Float32Array([sun[0],sun[1],sun[2],0])
    this.uniforms = device.createBuffer({ label: 'Frozen solar lighting',size: 16,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.texture = device.createTexture({ label: 'Lighting placeholder',size: [1,1],format: 'r8unorm',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    device.queue.writeTexture({ texture: this.texture },new Uint8Array([220]),{},[1,1])
  }

  async initialize() {
    const { bakePipeline,mipPipeline } = await getLightingPipelines(this.device)
    this.bakePipeline = bakePipeline; this.mipPipeline = mipPipeline
  }

  private bindIllumination(exaggeration: number) {
    if (!this.terrain) return false
    const cache = getLightingCache(this.device)
    let byKey = cache.illuminations.get(this.terrain)
    if (!byKey) {
      byKey = new Map()
      cache.illuminations.set(this.terrain,byKey)
    }
    const key = `${this.sunKey}|${exaggeration}`
    let shared = byKey.get(key)
    if (!shared) {
      const { width,height } = this.terrain
      const texture = this.device.createTexture({ label: 'Baked solar illumination',size: [width,height],format: 'r8unorm',mipLevelCount: Math.floor(Math.log2(Math.max(width,height))) + 1,usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
      shared = { texture,built: false }
      byKey.set(key,shared)
      cache.created.push(texture)
    }
    if (this.illumination === shared) return false
    if (this.textureOwned) this.texture.destroy()
    this.texture = shared.texture
    this.textureOwned = false
    this.illumination = shared
    this.mipGroups = shared.built ? [] : Array.from({ length: shared.texture.mipLevelCount - 1 },(_,i) => this.device.createBindGroup({
      layout: this.mipPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0,resource: { buffer: this.uniforms } },
        { binding: 1,resource: shared.texture.createView({ baseMipLevel: i,mipLevelCount: 1 }) },
      ],
    }))
    return true
  }

  setTerrain(terrain: GPUTexture,exaggeration = 20) {
    this.terrain = terrain
    this.bakeGroup = this.device.createBindGroup({ layout: this.bakePipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.uniforms } },{ binding: 1,resource: terrain.createView({ baseMipLevel: 0,mipLevelCount: 1 }) },
    ] })
    this.illumination = null
    this.exaggeration = null
    this.bindIllumination(exaggeration)
  }

  encode(encoder: GPUCommandEncoder,exaggeration: number) {
    if (!this.terrain) return { gradients: false,lighting: false,rebound: false }
    const rebound = this.bindIllumination(exaggeration)
    if (this.exaggeration === exaggeration && this.illumination?.built) return { gradients: false,lighting: false,rebound }
    const lighting = !this.illumination!.built
    if (lighting) {
      this.uniformScratch[3] = exaggeration
      this.device.queue.writeBuffer(this.uniforms,0,this.uniformScratch)
      for (let level = 0; level < this.texture.mipLevelCount; level++) {
        const pass = encoder.beginRenderPass({ label: level === 0 ? 'Bake solar illumination' : `Lighting mip ${level}`,colorAttachments: [{
          view: this.texture.createView({ baseMipLevel: level,mipLevelCount: 1 }),loadOp: 'clear',storeOp: 'store',
        }] })
        pass.setPipeline(level === 0 ? this.bakePipeline : this.mipPipeline)
        pass.setBindGroup(0,level === 0 ? this.bakeGroup : this.mipGroups[level - 1]); pass.draw(3); pass.end()
      }
      this.illumination!.built = true
      this.mipGroups = []
    }
    this.exaggeration = exaggeration
    return { gradients: false,lighting,rebound }
  }

  destroy() {
    if (this.textureOwned) this.texture.destroy()
    this.uniforms.destroy()
  }
}
