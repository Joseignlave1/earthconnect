import { colorShader } from './color.ts'
import { coverageShader } from './coverage.ts'
import { fullscreenVertex, shaderModule } from './gpu.ts'
import type { GpuTimer } from './gpu-timing.ts'
import { motionBlurShader, type MotionBlurUniforms } from './motion-blur.ts'

const bindings = `
struct PostUniforms { step: vec4f, effects: vec4f, mask: vec4f, points: vec4f, markers: vec4f, }
@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
`
const bloomShader = `${fullscreenVertex}${bindings}
fn bright(uv: vec2f) -> vec3f {
  let c = textureSampleLevel(source,linearSampler,uv,0.).rgb;
  let peak = max(c.r,max(c.g,c.b));
  return c * max(0.,peak - 1.) / max(peak,.0001);
}
@fragment fn bloomFragment(input: ScreenVertex) -> @location(0) vec4f {
  let uv = input.uv;
  var c = vec3f(0.);
  if (p.step.z > .5) {
    c = (bright(uv + p.step.xy * vec2f(-.5,-.5)) + bright(uv + p.step.xy * vec2f(.5,-.5))
      + bright(uv + p.step.xy * vec2f(-.5,.5)) + bright(uv + p.step.xy * vec2f(.5,.5))) * .25;
  } else {
    c = textureSampleLevel(source,linearSampler,uv,0.).rgb * .227027;
    c += (textureSampleLevel(source,linearSampler,uv + p.step.xy * 1.384615,0.).rgb + textureSampleLevel(source,linearSampler,uv - p.step.xy * 1.384615,0.).rgb) * .316216;
    c += (textureSampleLevel(source,linearSampler,uv + p.step.xy * 3.230769,0.).rgb + textureSampleLevel(source,linearSampler,uv - p.step.xy * 3.230769,0.).rgb) * .070270;
  }
  return vec4f(c,1.);
}`
const compositionShader = `${fullscreenVertex}${bindings}
@group(0) @binding(3) var bloom: texture_2d<f32>;
@group(0) @binding(4) var coverageSource: texture_2d<f32>;
${colorShader}
${coverageShader}
@fragment fn compositeFragment(input: ScreenVertex) -> @location(0) vec4f {
  let scene = textureSampleLevel(source,linearSampler,input.uv,0.);
  // MSAA resolves against clear space, so linear edge colors are premultiplied.
  let alpha = scene.a;
  var c = scene.rgb / max(alpha,.0001);
  if (p.step.w > .5) {
    c = (c + textureSampleLevel(bloom,linearSampler,input.uv,0.).rgb * p.effects.y) * p.effects.x;
    c = clamp((c * (2.51 * c + .03)) / (c * (2.43 * c + .59) + .14),vec3f(0.),vec3f(1.));
    c = toDisplay(c);
  }
  if (p.effects.w > .5) { c = mix(c,vec3f(.83,.29,.32),missingDataLine(input.uv)); }
  return vec4f(c * alpha,alpha);
}`
const presentationShader = `${fullscreenVertex}${bindings}
@group(0) @binding(3) var geography: texture_2d<f32>;
fn surfaceMarker(uv: vec2f,point: vec2f) -> f32 {
  let angles = (uv - .5) * vec2f(6.283185307,3.141592654);
  let origin = (point - .5) * vec2f(6.283185307,3.141592654);
  let direction = vec3f(cos(angles.y) * sin(angles.x),sin(angles.y),cos(angles.y) * cos(angles.x));
  let anchor = vec3f(cos(origin.y) * sin(origin.x),sin(origin.y),cos(origin.y) * cos(origin.x));
  let delta = uv.x - point.x;
  let planeDelta = vec2f((delta - floor(delta + .5)) * 6.283185307,direction.y - anchor.y);
  let distance = mix(length(planeDelta),length(direction - anchor),p.markers.z) / p.markers.w;
  let aa = clamp(.5 * fwidth(distance),.01,.75);
  return max(1. - smoothstep(.8 - aa,.8 + aa,abs(distance - 7.)),1. - smoothstep(1.5 - aa,1.5 + aa,distance));
}
@fragment fn presentFragment(input: ScreenVertex) -> @location(0) vec4f {
  let scene = textureSampleLevel(source,linearSampler,input.uv,0.);
  var c = scene.rgb;
  let pixel = vec2i(input.position.xy);
  let coordinates = textureLoad(geography,pixel,0).rg;
  // The geographic attachment clears to a latitude outside the surface UV range.
  let onSurface = coordinates.y != -1.;
  let uv = select(vec2f(0.),coordinates,onSurface);
  var marker = 0.;
  if (p.markers.x > .5) { marker = surfaceMarker(uv,p.points.xy); }
  if (p.markers.y > .5) { marker = max(marker,surfaceMarker(uv,p.points.zw)); }
  if (onSurface) { c = mix(c,vec3f(scene.a),marker); }
  return vec4f(c,scene.a);
}`
const cursorShader = `
struct CursorUniforms { position: vec4f, scale: vec4f, }
@group(0) @binding(0) var<uniform> cursor: CursorUniforms;
struct CursorVertex { @builtin(position) position: vec4f, @location(0) local: vec2f, }
@vertex fn cursorVertex(@builtin(vertex_index) index: u32) -> CursorVertex {
  let corners = array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.),vec2f(-1.,1.));
  let local = corners[index] * 4.;
  let screen = (cursor.position.xy + local * cursor.scale.x) / cursor.position.zw;
  return CursorVertex(vec4f(screen * vec2f(2.,-2.) + vec2f(-1.,1.),0.,1.),local);
}
@fragment fn cursorFragment(input: CursorVertex) -> @location(0) vec4f {
  let d = length(input.local);
  let mark = 1. - smoothstep(1.15,1.95,d);
  let outline = 1. - smoothstep(2.1,2.8,d);
  let alpha = max(mark,outline * .8);
  return vec4f(mix(vec3f(.025,.03,.04),vec3f(.96,.94,1.),mark / max(.0001,alpha)),alpha);
}`

export type Presentation = { hdr: boolean; exposure: number; glow: number; ratio: number; dataBoundary: boolean; cursor: [number,number] | null; markers: number[]; motion?: MotionBlurUniforms }

export class PostProcess {
  private scene!: GPUTexture
  private stableCoverage: GPUTexture | null = null
  private readonly emptyCoverage: GPUTexture
  private coverageValid = false
  private geography!: GPUTexture
  private annotationDepth!: GPUTexture
  private bloom: GPUTexture[] = []
  private depth!: GPUTexture
  private multisample!: GPUTexture
  private display!: GPUTexture
  private width = 0
  private height = 0
  private key = ''
  private bloomValid = false
  private displayedMotion = false
  private motionPipeline?: GPURenderPipeline
  private motionUniform?: GPUBuffer
  private motionTarget?: GPUTexture
  private motionGroup?: GPUBindGroup
  private motionBloomGroup?: GPUBindGroup
  private motionCompositionGroup?: GPUBindGroup
  private bloomPipeline!: GPURenderPipeline
  private compositionPipeline!: GPURenderPipeline
  private presentationPipeline!: GPURenderPipeline
  private cursorPipeline!: GPURenderPipeline
  private uniforms: GPUBuffer[]
  private compositeUniform: GPUBuffer
  private cursorUniform: GPUBuffer
  private sampler: GPUSampler
  private bloomGroups: GPUBindGroup[] = []
  private compositionGroup!: GPUBindGroup
  private presentationGroup!: GPUBindGroup
  private cursorGroup!: GPUBindGroup

  constructor(private device: GPUDevice,private format: GPUTextureFormat,private transparent: boolean,private reverseDepth = false) {
    this.sampler = device.createSampler({ minFilter: 'linear',magFilter: 'linear' })
    this.emptyCoverage = device.createTexture({ label: 'Unused coverage binding',size: [1,1],format: 'r8unorm',usage: GPUTextureUsage.TEXTURE_BINDING })
    const uniform = (label: string) => device.createBuffer({ label,size: 80,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.uniforms = [0,1,2].map(i => uniform(`Bloom parameters ${i}`))
    this.compositeUniform = uniform('Display parameters')
    this.cursorUniform = uniform('Space cursor parameters')
  }

  async initialize() {
    const device = this.device
    const [bloom,composition,presentation,cursor,motion] = await Promise.all([
      shaderModule(device,'Bloom WGSL',bloomShader),shaderModule(device,'Composition WGSL',compositionShader),
      shaderModule(device,'Presentation WGSL',presentationShader),shaderModule(device,'Cursor WGSL',cursorShader),
      this.reverseDepth ? shaderModule(device,'Camera motion blur WGSL',motionBlurShader) : undefined,
    ])
    const vertex = (module: GPUShaderModule,entryPoint = 'screenVertex') => ({ module,entryPoint })
    const [bp,cp,pp,sp,mp] = await Promise.all([
      device.createRenderPipelineAsync({ label: 'HDR bloom',layout: 'auto',vertex: vertex(bloom),fragment: { module: bloom,entryPoint: 'bloomFragment',targets: [{ format: 'rgba16float' }] } }),
      device.createRenderPipelineAsync({ label: 'Cached tone map and coverage',layout: 'auto',vertex: vertex(composition),fragment: { module: composition,entryPoint: 'compositeFragment',targets: [{ format: this.format }] } }),
      device.createRenderPipelineAsync({ label: 'Cached image and surface markers',layout: 'auto',vertex: vertex(presentation),fragment: { module: presentation,entryPoint: 'presentFragment',targets: [{ format: this.format }] } }),
      device.createRenderPipelineAsync({ label: 'Space cursor',layout: 'auto',vertex: vertex(cursor,'cursorVertex'),fragment: { module: cursor,entryPoint: 'cursorFragment',targets: [{ format: this.format,blend: { color: { srcFactor: 'src-alpha',dstFactor: 'one-minus-src-alpha' },alpha: { srcFactor: 'one',dstFactor: 'one-minus-src-alpha' } } }] } }),
      motion ? device.createRenderPipelineAsync({ label: 'Depth-aware camera motion blur',layout: 'auto',vertex: vertex(motion),fragment: { module: motion,entryPoint: 'blurFragment',targets: [{ format: 'rgba16float' }] } }) : undefined,
    ])
    this.motionPipeline = mp
    if (mp) this.motionUniform = device.createBuffer({ label: 'Camera motion parameters',size: 96,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.bloomPipeline = bp; this.compositionPipeline = cp; this.presentationPipeline = pp; this.cursorPipeline = sp
    this.cursorGroup = device.createBindGroup({ layout: sp.getBindGroupLayout(0),entries: [{ binding: 0,resource: { buffer: this.cursorUniform } }] })
  }

  private target(label: string,width: number,height: number,format: GPUTextureFormat = 'rgba16float',sampleCount = 1,sampled = sampleCount === 1) {
    const usage = GPUTextureUsage.RENDER_ATTACHMENT |
      (sampled ? GPUTextureUsage.TEXTURE_BINDING : 0) |
      (sampleCount === 1 ? GPUTextureUsage.COPY_SRC : 0)
    return this.device.createTexture({ label,size: [width,height],format,sampleCount,usage })
  }

  get hasCurrentCoverage() { return this.coverageValid }

  private compositionBinding(source: GPUTexture,coverage: GPUTexture) {
    return this.device.createBindGroup({ layout: this.compositionPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.compositeUniform } },{ binding: 1,resource: source.createView() },
      { binding: 2,resource: this.sampler },{ binding: 3,resource: this.bloom[0].createView() },{ binding: 4,resource: coverage.createView() },
    ] })
  }

  private bindComposition(coverage: GPUTexture) {
    this.compositionGroup = this.compositionBinding(this.scene,coverage)
    this.motionCompositionGroup = this.motionTarget && this.compositionBinding(this.motionTarget,coverage)
  }

  private encodeMotion(encoder: GPUCommandEncoder,motion: MotionBlurUniforms | undefined) {
    if (!motion || !this.motionPipeline || !this.motionUniform) return false
    if (!this.motionTarget) {
      this.motionTarget = this.target('Motion-filtered HDR scene',this.width,this.height)
      this.motionGroup = this.device.createBindGroup({ layout: this.motionPipeline.getBindGroupLayout(0),entries: [
        { binding: 0,resource: { buffer: this.motionUniform } },{ binding: 1,resource: this.scene.createView() },
        { binding: 2,resource: this.depth.createView({ aspect: 'depth-only' }) },
      ] })
      this.motionBloomGroup = this.device.createBindGroup({ layout: this.bloomPipeline.getBindGroupLayout(0),entries: [
        { binding: 0,resource: { buffer: this.uniforms[0] } },{ binding: 1,resource: this.motionTarget.createView() },
        { binding: 2,resource: this.sampler },
      ] })
      this.motionCompositionGroup = this.compositionBinding(this.motionTarget,this.stableCoverage ?? this.emptyCoverage)
    }
    this.device.queue.writeBuffer(this.motionUniform,0,motion)
    const pass = encoder.beginRenderPass({ label: 'Camera motion blur',colorAttachments: [
      { view: this.motionTarget.createView(),loadOp: 'clear',storeOp: 'store' },
    ] })
    pass.setPipeline(this.motionPipeline); pass.setBindGroup(0,this.motionGroup!); pass.draw(3); pass.end()
    return true
  }

  configure(width: number,height: number,ratio: number) {
    const key = `${width},${height},${ratio}`
    if (key === this.key) return false
    this.releaseTargets()
    this.width = width; this.height = height; this.bloomValid = false
    this.scene = this.target('HDR scene',width,height)
    this.multisample = this.target('4x MSAA HDR scene',width,height,'rgba16float',4)
    const depthFormat = this.reverseDepth ? 'depth32float' : 'depth24plus'
    this.depth = this.target('4x MSAA surface depth',width,height,depthFormat,4,Boolean(this.motionPipeline))
    this.geography = this.target('Geographic coordinates',width,height,'rg32float')
    this.annotationDepth = this.target('Annotation depth',width,height,depthFormat)
    this.display = this.target('Cached display image',width,height,this.format)
    const scale = Math.min(.5,1024 / Math.max(width,height))
    this.bloom = [0,1].map(i => this.target(`Bloom ${i}`,Math.max(1,Math.ceil(width * scale)),Math.max(1,Math.ceil(height * scale))))
    this.bloomGroups = [this.scene,this.bloom[0],this.bloom[1]].map((texture,i) => this.device.createBindGroup({
      layout: this.bloomPipeline.getBindGroupLayout(0),entries: [
        { binding: 0,resource: { buffer: this.uniforms[i] } },{ binding: 1,resource: texture.createView() },{ binding: 2,resource: this.sampler },
      ],
    }))
    this.bindComposition(this.emptyCoverage)
    this.presentationGroup = this.device.createBindGroup({ layout: this.presentationPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.compositeUniform } },{ binding: 1,resource: this.display.createView() },
      { binding: 2,resource: this.sampler },{ binding: 3,resource: this.geography.createView() },
    ] })
    this.key = key
    return true
  }

  beginAnnotations(encoder: GPUCommandEncoder,includeCoverage: boolean,timestampWrites?: GPURenderPassTimestampWrites) {
    const colorAttachments: GPURenderPassColorAttachment[] = [{ view: this.geography.createView(),clearValue: [0,-1,0,0],loadOp: 'clear',storeOp: 'store' }]
    if (includeCoverage) {
      if (!this.stableCoverage) {
        this.stableCoverage = this.target('Score coverage',this.width,this.height,'r8unorm')
        this.bindComposition(this.stableCoverage)
      }
      colorAttachments.push({ view: this.stableCoverage.createView(),clearValue: [0,0,0,0],loadOp: 'clear',storeOp: 'store' })
    }
    this.coverageValid = includeCoverage
    return encoder.beginRenderPass({ label: 'Geographic annotations',timestampWrites,colorAttachments,
      depthStencilAttachment: { view: this.annotationDepth.createView(),depthClearValue: this.reverseDepth ? 0 : 1,depthLoadOp: 'clear',depthStoreOp: 'discard' } })
  }

  begin(encoder: GPUCommandEncoder,hdr: boolean,timestampWrites?: GPURenderPassTimestampWrites,storeDepth = false) {
    return encoder.beginRenderPass({ label: '4x MSAA Earth surface',timestampWrites,colorAttachments: [{
      view: this.multisample.createView(),resolveTarget: this.scene.createView(),
      clearValue: this.transparent ? [0,0,0,0] : hdr ? [.018 / 12.92,.026 / 12.92,.039 / 12.92,1] : [.018,.026,.039,1],loadOp: 'clear',storeOp: 'discard',
    }],depthStencilAttachment: { view: this.depth.createView(),depthClearValue: this.reverseDepth ? 0 : 1,depthLoadOp: 'clear',depthStoreOp: storeDepth ? 'store' : 'discard' } })
  }

  finish(encoder: GPUCommandEncoder,output: GPUTextureView,p: Presentation,redrawSurface: boolean,rebuildDisplay: boolean,timer?: GpuTimer) {
    const motion = redrawSurface && this.encodeMotion(encoder,p.motion)
    const motionChanged = motion !== this.displayedMotion
    if (redrawSurface || motionChanged) this.bloomValid = false
    const rebuildBloom = p.hdr && p.glow > 0 && !this.bloomValid
    if (rebuildBloom) {
      for (let i = 0; i < 3; i++) {
        this.device.queue.writeBuffer(this.uniforms[i],0,new Float32Array([
          i === 0 ? 1 / this.width : i === 1 ? 2 * p.ratio / this.width : 0,
          i === 0 ? 1 / this.height : i === 2 ? 2 * p.ratio / this.height : 0,Number(i === 0),0,
        ]))
        const pass = encoder.beginRenderPass({ label: `Bloom pass ${i}`,timestampWrites: i === 1 ? undefined : timer?.writes('bloom',i === 0 ? 'begin' : 'end'),colorAttachments: [{ view: this.bloom[i % 2].createView(),loadOp: 'clear',storeOp: 'store' }] })
        pass.setPipeline(this.bloomPipeline); pass.setBindGroup(0,i === 0 && motion ? this.motionBloomGroup! : this.bloomGroups[i]); pass.draw(3); pass.end()
      }
      this.bloomValid = true
    }
    this.device.queue.writeBuffer(this.compositeUniform,0,new Float32Array([
      0,0,0,Number(p.hdr),p.exposure,p.hdr ? p.glow : 0,p.ratio,Number(p.dataBoundary),0,0,this.width,this.height,...p.markers,
    ]))
    const composition = rebuildDisplay || redrawSurface || rebuildBloom || motionChanged
    if (composition) {
      const pass = encoder.beginRenderPass({ label: 'Cache tone map and coverage',timestampWrites: timer?.writes('composition'),colorAttachments: [{ view: this.display.createView(),loadOp: 'clear',storeOp: 'store' }] })
      pass.setPipeline(this.compositionPipeline); pass.setBindGroup(0,motion ? this.motionCompositionGroup! : this.compositionGroup); pass.draw(3); pass.end()
    }
    const pass = encoder.beginRenderPass({ label: 'Present cached surface and markers',timestampWrites: timer?.writes('presentation'),colorAttachments: [{ view: output,loadOp: 'clear',storeOp: 'store' }] })
    pass.setPipeline(this.presentationPipeline); pass.setBindGroup(0,this.presentationGroup); pass.draw(3)
    if (p.cursor) {
      this.device.queue.writeBuffer(this.cursorUniform,0,new Float32Array([...p.cursor,this.width,this.height,p.ratio,0,0,0]))
      pass.setPipeline(this.cursorPipeline); pass.setBindGroup(0,this.cursorGroup); pass.draw(6)
    }
    pass.end()
    this.displayedMotion = motion
    return { bloom: rebuildBloom,composition,motion }
  }

  private releaseTargets() {
    this.scene?.destroy(); this.stableCoverage?.destroy(); this.geography?.destroy(); this.annotationDepth?.destroy()
    this.stableCoverage = null; this.coverageValid = false; this.displayedMotion = false
    this.depth?.destroy(); this.multisample?.destroy(); this.display?.destroy(); this.bloom.forEach(texture => texture.destroy())
    this.motionTarget?.destroy()
    this.motionTarget = undefined; this.motionGroup = undefined; this.motionBloomGroup = undefined; this.motionCompositionGroup = undefined
  }

  suspend() {
    if (!this.key) return
    this.releaseTargets()
    this.key = ''
  }

  destroy() {
    this.suspend()
    this.emptyCoverage.destroy()
    this.motionUniform?.destroy()
    this.uniforms.forEach(buffer => buffer.destroy()); this.compositeUniform.destroy(); this.cursorUniform.destroy()
  }
}
