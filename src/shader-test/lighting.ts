import { fullscreenVertex, shaderModule } from './gpu.ts'

// Slopes are built once from the numeric height mip, in metres per metre.
// Water remains at sea level, matching the displaced rendering geometry.
const gradientShader = `
@group(0) @binding(0) var heights: texture_2d<f32>;
@group(0) @binding(1) var slopes: texture_storage_2d<rg32float,write>;
fn heightTexel(p: vec2i) -> f32 {
  let size = vec2i(textureDimensions(heights));
  return textureLoad(heights,vec2i(((p.x % size.x) + size.x) % size.x,clamp(p.y,0,size.y - 1)),0).r;
}
fn heightAt(direction: vec3f) -> f32 {
  let size = vec2f(textureDimensions(heights));
  let lon = atan2(direction.x,direction.z);
  let lat = atan2(direction.y,length(direction.xz));
  let p = vec2f(lon / 6.283185307 + .5,.5 - lat / 3.141592654) * size - .5;
  let base = vec2i(floor(p));
  let f = fract(p);
  return max(0.,mix(mix(heightTexel(base),heightTexel(base + vec2i(1,0)),f.x),
    mix(heightTexel(base + vec2i(0,1)),heightTexel(base + vec2i(1,1)),f.x),f.y));
}
@compute @workgroup_size(8,8) fn gradients(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(heights));
  let p = vec2i(id.xy);
  if (any(p >= size)) { return; }
  var gradient = vec2f(0.);
  // At each pole all longitudes meet. A single radial normal closes the cap.
  if (p.y > 0 && p.y < size.y - 1 && textureLoad(heights,p,0).g >= 0.) {
    let latitude = (0.5 - (f32(p.y) + .5) / f32(size.y)) * 3.141592654;
    let e2 = .0066943799901413165;
    let denominator = 1. - e2 * sin(latitude) * sin(latitude);
    let eastRadius = 6378137. * inverseSqrt(denominator);
    let northRadius = 6378137. * (1. - e2) / pow(denominator,1.5);
    let lon = ((f32(p.x) + .5) / f32(size.x) - .5) * 6.283185307;
    let up = vec3f(cos(latitude) * sin(lon),sin(latitude),cos(latitude) * cos(lon));
    let east = vec3f(cos(lon),0.,-sin(lon));
    let north = vec3f(-sin(latitude) * sin(lon),cos(latitude),-sin(latitude) * cos(lon));
    // Constant angular offsets along great circles maintain a physical
    // footprint near the poles, rather than differentiating sub-metre columns.
    let angle = 3.141592654 / f32(size.y);
    let center = up * cos(angle);
    let dx = east * sin(angle);
    let dy = north * sin(angle);
    gradient = vec2f((heightAt(center + dx) - heightAt(center - dx)) / (2. * angle * eastRadius),
      (heightAt(center + dy) - heightAt(center - dy)) / (2. * angle * northRadius));
  }
  textureStore(slopes,p,vec4f(gradient,0.,0.));
}`
const lightBindings = `
struct Lighting { sun: vec4f, } // Earth-fixed XYZ, terrain exaggeration
@group(0) @binding(0) var<uniform> light: Lighting;
@group(0) @binding(1) var source: texture_2d<f32>;
fn sunLight(normal: vec3f,up: vec3f) -> f32 {
  // Earth's horizon hides the sun on the night side, even for exaggerated
  // slopes. A transition about one solar diameter wide softens the terminator.
  let daylight = smoothstep(-.00465,.00465,dot(up,light.sun.xyz));
  return daylight * max(0.,dot(normal,light.sun.xyz));
}
fn polarLight(north: bool) -> f32 {
  let up = vec3f(0.,select(-1.,1.,north),0.);
  return .5 + .2 * inverseSqrt(2.) + .3 * sunLight(up,up);
}
`
const bakeShader = `${fullscreenVertex}${lightBindings}
@fragment fn bake(input: ScreenVertex) -> @location(0) f32 {
  let size = vec2i(textureDimensions(source));
  let p = vec2i(input.position.xy);
  if (size.y > 1 && p.y == 0) { return polarLight(true); }
  if (size.y > 1 && p.y == size.y - 1) { return polarLight(false); }
  let lon = (input.uv.x - .5) * 6.283185307;
  let lat = (.5 - input.uv.y) * 3.141592654;
  let up = vec3f(cos(lat) * sin(lon),sin(lat),cos(lat) * cos(lon));
  let east = vec3f(cos(lon),0.,-sin(lon));
  let north = vec3f(-sin(lat) * sin(lon),cos(lat),-sin(lat) * cos(lon));
  let gradient = textureLoad(source,p,0).rg * light.sun.w;
  let localNormal = normalize(vec3f(-gradient,1.));
  let normal = localNormal.x * east + localNormal.y * north + localNormal.z * up;
  // A real directional sun plus ambient and a soft cartographic fill.
  let fill = max(0.,dot(localNormal,normalize(vec3f(-.6,.8,1.))));
  return .5 + .2 * fill + .3 * sunLight(normal,up);
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

export class TerrainLighting {
  texture: GPUTexture
  private slopes: GPUTexture | null = null
  private uniforms: GPUBuffer
  private gradientPipeline!: GPUComputePipeline
  private bakePipeline!: GPURenderPipeline
  private mipPipeline!: GPURenderPipeline
  private gradientGroup!: GPUBindGroup
  private bakeGroup!: GPUBindGroup
  private mipGroups: GPUBindGroup[] = []
  private gradientPending = false
  private exaggeration: number | null = null

  constructor(private device: GPUDevice,private sun: readonly number[]) {
    this.uniforms = device.createBuffer({ label: 'Frozen solar lighting',size: 16,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.texture = device.createTexture({ label: 'Lighting placeholder',size: [1,1],format: 'r8unorm',usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    device.queue.writeTexture({ texture: this.texture },new Uint8Array([220]),{},[1,1])
  }

  async initialize() {
    const d = this.device
    const [gradient,bake,mip] = await Promise.all([
      shaderModule(d,'Terrain gradients WGSL',gradientShader),shaderModule(d,'Solar lighting WGSL',bakeShader),shaderModule(d,'Lighting mipmaps WGSL',mipShader),
    ])
    const gradientLayout = d.createBindGroupLayout({ entries: [
      { binding: 0,visibility: GPUShaderStage.COMPUTE,texture: { sampleType: 'unfilterable-float' } },
      { binding: 1,visibility: GPUShaderStage.COMPUTE,storageTexture: { access: 'write-only',format: 'rg32float' } },
    ] })
    const bakeLayout = d.createBindGroupLayout({ entries: [
      { binding: 0,visibility: GPUShaderStage.FRAGMENT,buffer: { type: 'uniform' } },
      { binding: 1,visibility: GPUShaderStage.FRAGMENT,texture: { sampleType: 'unfilterable-float' } },
    ] })
    const [gp,bp,mp] = await Promise.all([
      d.createComputePipelineAsync({ label: 'Precompute terrain gradients',layout: d.createPipelineLayout({ bindGroupLayouts: [gradientLayout] }),compute: { module: gradient,entryPoint: 'gradients' } }),
      d.createRenderPipelineAsync({ label: 'Bake frozen solar lighting',layout: d.createPipelineLayout({ bindGroupLayouts: [bakeLayout] }),vertex: { module: bake,entryPoint: 'screenVertex' },fragment: { module: bake,entryPoint: 'bake',targets: [{ format: 'r8unorm' }] } }),
      d.createRenderPipelineAsync({ label: 'Filter lighting mipmaps',layout: 'auto',vertex: { module: mip,entryPoint: 'screenVertex' },fragment: { module: mip,entryPoint: 'mip',targets: [{ format: 'r8unorm' }] } }),
    ])
    this.gradientPipeline = gp; this.bakePipeline = bp; this.mipPipeline = mp
  }

  setTerrain(terrain: GPUTexture) {
    // One terrain mip of smoothing retains mountain relief without baking
    // native height-grid noise into lighting. Contours use full-precision data.
    const level = Math.min(1,terrain.mipLevelCount - 1)
    const width = Math.max(1,terrain.width >> level),height = Math.max(1,terrain.height >> level)
    this.slopes?.destroy(); this.texture.destroy()
    this.slopes = this.device.createTexture({ label: 'Cached east/north terrain gradients',size: [width,height],format: 'rg32float',usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC })
    this.texture = this.device.createTexture({ label: 'Baked solar illumination',size: [width,height],format: 'r8unorm',mipLevelCount: Math.floor(Math.log2(Math.max(width,height))) + 1,usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
    this.gradientGroup = this.device.createBindGroup({ layout: this.gradientPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: terrain.createView({ baseMipLevel: level,mipLevelCount: 1 }) },{ binding: 1,resource: this.slopes.createView() },
    ] })
    const group = (pipeline: GPURenderPipeline,view: GPUTextureView) => this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.uniforms } },{ binding: 1,resource: view },
    ] })
    this.bakeGroup = group(this.bakePipeline,this.slopes.createView())
    this.mipGroups = Array.from({ length: this.texture.mipLevelCount - 1 },(_,i) => group(this.mipPipeline,this.texture.createView({ baseMipLevel: i,mipLevelCount: 1 })))
    this.gradientPending = true; this.exaggeration = null
  }

  encode(encoder: GPUCommandEncoder,exaggeration: number) {
    if (!this.slopes || this.exaggeration === exaggeration) return { gradients: false,lighting: false }
    const gradients = this.gradientPending
    if (gradients) {
      const pass = encoder.beginComputePass({ label: 'Cache terrain gradients' })
      pass.setPipeline(this.gradientPipeline); pass.setBindGroup(0,this.gradientGroup)
      pass.dispatchWorkgroups(Math.ceil(this.slopes.width / 8),Math.ceil(this.slopes.height / 8)); pass.end()
      this.gradientPending = false
    }
    this.device.queue.writeBuffer(this.uniforms,0,new Float32Array([...this.sun,exaggeration]))
    for (let level = 0; level < this.texture.mipLevelCount; level++) {
      const pass = encoder.beginRenderPass({ label: level === 0 ? 'Bake solar illumination' : `Lighting mip ${level}`,colorAttachments: [{
        view: this.texture.createView({ baseMipLevel: level,mipLevelCount: 1 }),loadOp: 'clear',storeOp: 'store',
      }] })
      pass.setPipeline(level === 0 ? this.bakePipeline : this.mipPipeline)
      pass.setBindGroup(0,level === 0 ? this.bakeGroup : this.mipGroups[level - 1]); pass.draw(3); pass.end()
    }
    this.exaggeration = exaggeration
    return { gradients,lighting: true }
  }

  destroy() { this.texture.destroy(); this.slopes?.destroy(); this.uniforms.destroy() }
}
