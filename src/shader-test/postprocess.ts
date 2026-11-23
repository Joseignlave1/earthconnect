import { colorShader } from './color.ts'
import { coverageShader } from './coverage.ts'
import { equalEarthShader } from './geometry.ts'
import { fullscreenVertex, shaderModule } from './gpu.ts'
import { motionBlurShader, type MotionBlurUniforms } from './motion-blur.ts'

const bindings = `
struct PostUniforms {
  step: vec4f,
  effects: vec4f,
  mask: vec4f,
  hover: vec4f,
  morph: vec4f,
  selEarth: vec4f,
  selMoon: vec4f,
  selMars: vec4f,
}
@group(0) @binding(0) var<uniform> p: PostUniforms;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
`
const bloomShader = `${fullscreenVertex}${bindings}
fn bright(uv: vec2f) -> vec3f {
  let c = textureSampleLevel(source,linearSampler,uv,0.).rgb;
  let peak = max(c.r,max(c.g,c.b));
  // C1 quadratic soft knee over [0.80 .. 1.20] avoids hard threshold popping during zoom.
  let soft = clamp(peak - .80,0.,.40);
  let contrib = max(peak - 1.,soft * soft / .80);
  return c * contrib / max(peak,.0001);
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
${equalEarthShader}
fn surfaceTangent(uv: vec2f, point: vec2f, bodyMorph: f32, unitPerPixel: f32) -> vec2f {
  let angles = (uv - .5) * vec2f(6.283185307, 3.141592654);
  let origin = (point - .5) * vec2f(6.283185307, 3.141592654);
  let direction = vec3f(cos(angles.y) * sin(angles.x), sin(angles.y), cos(angles.y) * cos(angles.x));
  let anchor = vec3f(cos(origin.y) * sin(origin.x), sin(origin.y), cos(origin.y) * cos(origin.x));
  if (bodyMorph > .5 && dot(direction, anchor) < 0.) { return vec2f(1e5, 1e5); }
  let planeDelta = equalEarth(angles.x, angles.y) - equalEarth(origin.x, origin.y);
  let east = vec3f(cos(origin.x), 0., -sin(origin.x));
  let north = cross(anchor, east);
  let offset = direction - anchor;
  return mix(planeDelta, vec2f(dot(offset, east), dot(offset, north)), bodyMorph) / max(unitPerPixel, 1e-9);
}
fn sampleLuma(uv: vec2f) -> f32 {
  let s = textureSampleLevel(source, linearSampler, clamp(uv, vec2f(.001), vec2f(.999)), 0.);
  let rgb = s.rgb / max(s.a, .0001);
  return dot(rgb, vec3f(.2126, .7152, .0722));
}
fn localLuma(screenUv: vec2f) -> f32 {
  let step = 6. * p.effects.z / max(p.mask.zw, vec2f(1.));
  var sum = sampleLuma(screenUv);
  sum += sampleLuma(screenUv + vec2f(step.x, 0.));
  sum += sampleLuma(screenUv - vec2f(step.x, 0.));
  sum += sampleLuma(screenUv + vec2f(0., step.y));
  sum += sampleLuma(screenUv - vec2f(0., step.y));
  sum += sampleLuma(screenUv + vec2f(step.x, step.y) * .707);
  sum += sampleLuma(screenUv + vec2f(-step.x, step.y) * .707);
  sum += sampleLuma(screenUv + vec2f(step.x, -step.y) * .707);
  sum += sampleLuma(screenUv - vec2f(step.x, step.y) * .707);
  return sum / 9.;
}
fn applyPrecisionReticle(base: vec3f, screenUv: vec2f, t: vec2f, d: f32, aa: f32, opacity: f32) -> vec3f {
  if (d > 16. || opacity <= .002) { return base; }
  let bg = localLuma(screenUv);
  let radius = 8.4;
  let ringDist = abs(d - radius);
  let ax = abs(t.x);
  let ay = abs(t.y);
  let isNorth = t.y > 0. && ax < ay;
  let tickEnd = select(11.3, 12.6, isNorth);
  let tickBand = smoothstep(radius - .4 - aa, radius - .4 + aa, d) * (1. - smoothstep(tickEnd - aa, tickEnd + aa, d));
  let tickCore = max(1. - smoothstep(.58 - aa, .58 + aa, ax), 1. - smoothstep(.58 - aa, .58 + aa, ay)) * tickBand;
  let tickCase = max(1. - smoothstep(1.45 - aa, 1.45 + aa, ax), 1. - smoothstep(1.45 - aa, 1.45 + aa, ay)) * tickBand;
  let ringCore = 1. - smoothstep(.66 - aa, .66 + aa, ringDist);
  let dotCore = 1. - smoothstep(1.35 - aa, 1.35 + aa, d);
  let core = max(max(ringCore, dotCore), tickCore);
  let casing = max(max(1. - smoothstep(1.65 - aa, 1.65 + aa, ringDist), 1. - smoothstep(2.3 - aa, 2.3 + aa, d)), tickCase);
  let inside = 1. - smoothstep(radius - 1. - aa, radius - 1. + aa, d);
  var col = mix(base, mix(base * 1.07 + .012, base * .93, smoothstep(.25, .7, bg)), inside * .55);
  let caseAlpha = mix(.62, .92, smoothstep(.1, .65, bg));
  col = mix(col, vec3f(.02, .028, .045), casing * caseAlpha);
  let accent = mix(vec3f(.96, .98, 1.), vec3f(.55, .86, 1.), select(0., 1., isNorth) * tickCore * .85);
  let out = mix(col, accent, core * .96);
  return mix(base, out, opacity);
}
@fragment fn presentFragment(input: ScreenVertex) -> @location(0) vec4f {
  let scene = textureSampleLevel(source, linearSampler, input.uv, 0.);
  let pixel = vec2i(input.position.xy);
  let coordinates = textureLoad(geography, pixel, 0).rg;
  // The geographic attachment clears to a latitude outside the surface UV range.
  // Body IDs are packed in the integer part of x: Earth [0,1], Moon [2,3], Mars [4,5].
  let onSurface = coordinates.y != -1.;
  let bodyId = select(-1, i32(floor(coordinates.x * .5)), onSurface);
  let uv = select(vec2f(0.), vec2f(coordinates.x - f32(bodyId) * 2., coordinates.y), onSurface);
  let bodyMorph = select(1., p.morph.x, bodyId == 0);
  let sel = select(select(p.selEarth, p.selMoon, bodyId == 1), p.selMars, bodyId == 2);
  let tSel = surfaceTangent(uv, sel.xy, bodyMorph, sel.w);
  let dSel = length(tSel);
  let aaSel = clamp(.5 * fwidth(dSel), .02, .75);
  let tHov = surfaceTangent(uv, p.hover.xy, bodyMorph, p.hover.w);
  let dHov = length(tHov);
  let aaHov = clamp(.5 * fwidth(dHov), .02, .75);
  var c = scene.rgb / max(scene.a, .0001);
  let hoverBody = i32(round(p.mask.y));
  let motionBlurred = p.morph.y > .5;
  if (onSurface) {
    if (!motionBlurred && sel.z > .005) { c = applyPrecisionReticle(c, input.uv, tSel, dSel, aaSel, sel.z); }
    if (p.hover.z > .5 && bodyId == hoverBody) { c = applyPrecisionReticle(c, input.uv, tHov, dHov, aaHov, 1.); }
  }
  return vec4f(c * scene.a, scene.a);
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

const loupeShader = `
struct LoupeUniforms { rect: vec4f, viewport: vec4f, state: vec4f, selected: vec4f, }
@group(0) @binding(0) var<uniform> loupe: LoupeUniforms;
@group(0) @binding(1) var loupeTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;
${colorShader}
struct LoupeVertex { @builtin(position) position: vec4f, @location(0) local: vec2f, }
@vertex fn loupeVertex(@builtin(vertex_index) index: u32) -> LoupeVertex {
  let corners = array<vec2f,6>(vec2f(-1.,-1.),vec2f(1.,-1.),vec2f(-1.,1.),vec2f(1.,-1.),vec2f(1.,1.),vec2f(-1.,1.));
  let radiusCss = loupe.rect.z;
  let extent = radiusCss + 12.;
  let local = corners[index] * extent;
  let screen = (loupe.rect.xy + local * loupe.rect.w) / loupe.viewport.xy;
  return LoupeVertex(vec4f(screen * vec2f(2.,-2.) + vec2f(-1.,1.),0.,1.),local);
}
fn toneMapLoupe(uv: vec2f) -> vec3f {
  let s = textureSampleLevel(loupeTex, linearSampler, clamp(uv, vec2f(.002), vec2f(.998)), 0.);
  var c = s.rgb / max(s.a, .0001);
  if (loupe.viewport.w > .5) {
    c = c * loupe.viewport.z;
    c = clamp((c * (2.51 * c + .03)) / (c * (2.43 * c + .59) + .14), vec3f(0.), vec3f(1.));
    c = toDisplay(c);
  }
  return c;
}
@fragment fn loupeFragment(input: LoupeVertex) -> @location(0) vec4f {
  let p = input.local;
  let r = loupe.rect.z;
  let d = length(p);
  let aa = max(.35, .6 / max(loupe.rect.w, 1.));
  let inside = 1. - smoothstep(r - aa, r + aa, d);
  let uv = p / (2. * r) + .5;
  var lens = toneMapLoupe(uv);
  // Subtle optical glass vignette near the bezel edge.
  let edgeVig = smoothstep(r - 7., r, d);
  lens = mix(lens, lens * .82, edgeVig * .65);

  let innerMask = 1. - smoothstep(r - 1.5 - aa, r - 1.5 + aa, d);

  // Selected Earth marker projected into loupe space (so clicked spots stay visible inside the lens).
  if (loupe.selected.z > .5) {
    let sd = length(p - loupe.selected.xy);
    let selRingDist = abs(sd - 6.8);
    let selCase = max(1. - smoothstep(1.75 - aa, 1.75 + aa, selRingDist), 1. - smoothstep(2.4 - aa, 2.4 + aa, sd)) * innerMask;
    let selCore = max(1. - smoothstep(.8 - aa, .8 + aa, selRingDist), 1. - smoothstep(1.45 - aa, 1.45 + aa, sd)) * innerMask;
    lens = mix(lens, vec3f(.02, .028, .045), selCase * .88);
    lens = mix(lens, vec3f(.97, .98, 1.), selCore * .96);
  }

  let locked = loupe.state.z > .5;
  let aim = select(vec2f(0.), loupe.state.xy, locked);
  let ap = p - aim;
  let ad = length(ap);
  let aimUv = aim / (2. * r) + .5;
  let aimCol = toneMapLoupe(aimUv);
  let aimLuma = dot(aimCol, vec3f(.2126, .7152, .0722));
  let aax = abs(ap.x);
  let aay = abs(ap.y);
  let armBand = smoothstep(3.4 - aa, 3.4 + aa, ad) * (1. - smoothstep(10. - aa, 10. + aa, ad));
  let armCore = max(1. - smoothstep(.52 - aa, .52 + aa, aax), 1. - smoothstep(.52 - aa, .52 + aa, aay)) * armBand;
  let armCase = max(1. - smoothstep(1.35 - aa, 1.35 + aa, aax), 1. - smoothstep(1.35 - aa, 1.35 + aa, aay)) * armBand;
  let dotCore = 1. - smoothstep(1.15 - aa, 1.15 + aa, ad);
  let dotCase = 1. - smoothstep(2.05 - aa, 2.05 + aa, ad);
  let crossCase = max(armCase, dotCase) * innerMask;
  let crossCore = max(armCore, dotCore) * innerMask;
  let caseAlpha = mix(.62, .92, smoothstep(.1, .65, aimLuma));
  lens = mix(lens, vec3f(.02, .028, .045), crossCase * caseAlpha);
  lens = mix(lens, vec3f(.97, .98, 1.), crossCore * .96);

  // Precision dual-stroke bezel + 4 cardinal ticks + soft floating shadow.
  let ax = abs(p.x);
  let ay = abs(p.y);
  let shadow = exp(-max(0., d - r) * .36) * .48 * (1. - inside);
  let bezelDist = abs(d - r);
  let bezelCase = 1. - smoothstep(2.1 - aa, 2.1 + aa, bezelDist);
  let bezelCore = 1. - smoothstep(.85 - aa, .85 + aa, bezelDist);
  let tickBand = smoothstep(r + .4 - aa, r + .4 + aa, d) * (1. - smoothstep(r + 4.4 - aa, r + 4.4 + aa, d));
  let tickCase = max(1. - smoothstep(1.4 - aa, 1.4 + aa, ax), 1. - smoothstep(1.4 - aa, 1.4 + aa, ay)) * tickBand;
  let tickCore = max(1. - smoothstep(.58 - aa, .58 + aa, ax), 1. - smoothstep(.58 - aa, .58 + aa, ay)) * tickBand;
  let ringCase = max(bezelCase, tickCase);
  let ringCore = max(bezelCore, tickCore);
  let isTopTick = p.y < 0. && ax < ay;
  let bezelTint = select(vec3f(.95, .97, 1.), vec3f(.68, .90, 1.), locked);
  let coreColor = mix(bezelTint, vec3f(.45, .84, 1.), select(0., 1., isTopTick) * tickCore * .85);

  let alpha = max(inside, max(shadow, max(ringCase * .88, ringCore)));
  if (alpha < .002) { discard; }
  var outCol = mix(vec3f(.015, .02, .035), lens, inside);
  outCol = mix(outCol, vec3f(.02, .028, .045), ringCase * .88);
  outCol = mix(outCol, coreColor, ringCore * .96);
  return vec4f(outCol, alpha);
}`

type PostPipelines = {
  bloom: GPURenderPipeline
  composition: GPURenderPipeline
  presentation: GPURenderPipeline
  cursor: GPURenderPipeline
  loupe: GPURenderPipeline
  motion?: GPURenderPipeline
}

const postPipelines = new WeakMap<GPUDevice, Map<string, Promise<PostPipelines>>>()

function getPostPipelines(device: GPUDevice, format: GPUTextureFormat, reverseDepth: boolean): Promise<PostPipelines> {
  let byKey = postPipelines.get(device)
  if (!byKey) {
    byKey = new Map()
    postPipelines.set(device, byKey)
  }
  const key = `${format}:${reverseDepth}`
  let pending = byKey.get(key)
  if (!pending) {
    pending = (async () => {
      const [bloom,composition,presentation,cursor,loupe,motion] = await Promise.all([
        shaderModule(device,'Bloom WGSL',bloomShader),shaderModule(device,'Composition WGSL',compositionShader),
        shaderModule(device,'Presentation WGSL',presentationShader),shaderModule(device,'Cursor WGSL',cursorShader),
        shaderModule(device,'Optical loupe WGSL',loupeShader),
        reverseDepth ? shaderModule(device,'Camera motion blur WGSL',motionBlurShader) : undefined,
      ])
      const vertex = (module: GPUShaderModule,entryPoint = 'screenVertex') => ({ module,entryPoint })
      const alphaBlend: GPUBlendState = {
        color: { srcFactor: 'src-alpha',dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one',dstFactor: 'one-minus-src-alpha' },
      }
      const [bp,cp,pp,sp,lp,mp] = await Promise.all([
        device.createRenderPipelineAsync({ label: 'HDR bloom',layout: 'auto',vertex: vertex(bloom),fragment: { module: bloom,entryPoint: 'bloomFragment',targets: [{ format: 'rgba16float' }] } }),
        device.createRenderPipelineAsync({ label: 'Cached tone map and coverage',layout: 'auto',vertex: vertex(composition),fragment: { module: composition,entryPoint: 'compositeFragment',targets: [{ format }] } }),
        device.createRenderPipelineAsync({ label: 'Cached image and surface markers',layout: 'auto',vertex: vertex(presentation),fragment: { module: presentation,entryPoint: 'presentFragment',targets: [{ format }] } }),
        device.createRenderPipelineAsync({ label: 'Space cursor',layout: 'auto',vertex: vertex(cursor,'cursorVertex'),fragment: { module: cursor,entryPoint: 'cursorFragment',targets: [{ format,blend: alphaBlend }] } }),
        device.createRenderPipelineAsync({ label: 'Optical loupe',layout: 'auto',vertex: vertex(loupe,'loupeVertex'),fragment: { module: loupe,entryPoint: 'loupeFragment',targets: [{ format,blend: alphaBlend }] } }),
        motion ? device.createRenderPipelineAsync({ label: 'Depth-aware camera motion blur',layout: 'auto',vertex: vertex(motion),fragment: { module: motion,entryPoint: 'blurFragment',targets: [{ format: 'rgba16float' }] } }) : undefined,
      ])
      return { bloom: bp,composition: cp,presentation: pp,cursor: sp,loupe: lp,motion: mp }
    })()
    byKey.set(key, pending)
    void pending.catch(() => { if (byKey!.get(key) === pending) byKey!.delete(key) })
  }
  return pending
}

export const LOUPE_TEXTURE_SIZE = 320
export type LoupeOverlay = { x: number; y: number; radiusCss: number; aimX: number; aimY: number; locked: boolean; zoom: number; selected?: [number, number] | null }
export type Presentation = { hdr: boolean; exposure: number; glow: number; ratio: number; dataBoundary: boolean; cursor: [number,number] | null; hoverBody: number; markers: number[]; loupe?: LoupeOverlay | null; motion?: MotionBlurUniforms }

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
  private loupePipeline!: GPURenderPipeline
  private loupeScene?: GPUTexture
  private loupeMultisample?: GPUTexture
  private loupeDepth?: GPUTexture
  private loupeUniform: GPUBuffer
  private loupeGroup?: GPUBindGroup
  private uniforms: GPUBuffer[]
  private compositeUniform: GPUBuffer
  private cursorUniform: GPUBuffer
  private bloomScratch = new Float32Array(4)
  private compositeScratch = new Float32Array(32)
  private cursorScratch = new Float32Array(8)
  private loupeScratch = new Float32Array(16)
  private sampler: GPUSampler
  private bloomGroups: GPUBindGroup[] = []
  private compositionGroup!: GPUBindGroup
  private presentationGroup!: GPUBindGroup
  private cursorGroup!: GPUBindGroup

  constructor(private device: GPUDevice,private format: GPUTextureFormat,private transparent: boolean,private reverseDepth = false) {
    this.sampler = device.createSampler({ minFilter: 'linear',magFilter: 'linear' })
    this.emptyCoverage = device.createTexture({ label: 'Unused coverage binding',size: [1,1],format: 'r8unorm',usage: GPUTextureUsage.TEXTURE_BINDING })
    const uniform = (label: string) => device.createBuffer({ label,size: 128,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.uniforms = [0,1,2].map(i => uniform(`Bloom parameters ${i}`))
    this.compositeUniform = uniform('Display parameters')
    this.cursorUniform = uniform('Space cursor parameters')
    this.loupeUniform = device.createBuffer({ label: 'Optical loupe parameters',size: 64,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  }

  async initialize() {
    const device = this.device
    const { bloom,composition,presentation,cursor,loupe,motion } = await getPostPipelines(device,this.format,this.reverseDepth)
    this.motionPipeline = motion
    if (motion) this.motionUniform = device.createBuffer({ label: 'Camera motion parameters',size: 96,usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.bloomPipeline = bloom; this.compositionPipeline = composition; this.presentationPipeline = presentation; this.cursorPipeline = cursor; this.loupePipeline = loupe
    this.cursorGroup = device.createBindGroup({ layout: cursor.getBindGroupLayout(0),entries: [{ binding: 0,resource: { buffer: this.cursorUniform } }] })
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
        { binding: 3,resource: { buffer: this.compositeUniform } },{ binding: 4,resource: this.geography.createView() },
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

  beginAnnotations(encoder: GPUCommandEncoder,includeCoverage: boolean) {
    const colorAttachments: GPURenderPassColorAttachment[] = [{ view: this.geography.createView(),clearValue: [0,-1,0,0],loadOp: 'clear',storeOp: 'store' }]
    if (includeCoverage) {
      if (!this.stableCoverage) {
        this.stableCoverage = this.target('Score coverage',this.width,this.height,'r8unorm')
        this.bindComposition(this.stableCoverage)
      }
      colorAttachments.push({ view: this.stableCoverage.createView(),clearValue: [0,0,0,0],loadOp: 'clear',storeOp: 'store' })
    }
    this.coverageValid = includeCoverage
    return encoder.beginRenderPass({ label: 'Geographic annotations',colorAttachments,
      depthStencilAttachment: { view: this.annotationDepth.createView(),depthClearValue: this.reverseDepth ? 0 : 1,depthLoadOp: 'clear',depthStoreOp: 'discard' } })
  }

  beginLoupe(encoder: GPUCommandEncoder,hdr: boolean) {
    if (!this.loupeScene) {
      const size = LOUPE_TEXTURE_SIZE
      const depthFormat = this.reverseDepth ? 'depth32float' : 'depth24plus'
      this.loupeScene = this.target('Optical loupe HDR scene',size,size)
      this.loupeMultisample = this.target('Optical loupe 4x MSAA HDR',size,size,'rgba16float',4)
      this.loupeDepth = this.target('Optical loupe depth',size,size,depthFormat,4,false)
      this.loupeGroup = this.device.createBindGroup({
        layout: this.loupePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0,resource: { buffer: this.loupeUniform } },
          { binding: 1,resource: this.loupeScene.createView() },
          { binding: 2,resource: this.sampler },
        ],
      })
    }
    return encoder.beginRenderPass({
      label: 'Optical loupe mini-pass',
      colorAttachments: [{
        view: this.loupeMultisample!.createView(),
        resolveTarget: this.loupeScene.createView(),
        clearValue: hdr ? [.018 / 12.92,.026 / 12.92,.039 / 12.92,1] : [.018,.026,.039,1],
        loadOp: 'clear',storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this.loupeDepth!.createView(),
        depthClearValue: this.reverseDepth ? 0 : 1,
        depthLoadOp: 'clear',depthStoreOp: 'discard',
      },
    })
  }

  begin(encoder: GPUCommandEncoder,hdr: boolean,storeDepth = false) {
    return encoder.beginRenderPass({ label: '4x MSAA Earth surface',colorAttachments: [{
      view: this.multisample.createView(),resolveTarget: this.scene.createView(),
      clearValue: this.transparent ? [0,0,0,0] : hdr ? [.018 / 12.92,.026 / 12.92,.039 / 12.92,1] : [.018,.026,.039,1],loadOp: 'clear',storeOp: 'discard',
    }],depthStencilAttachment: { view: this.depth.createView(),depthClearValue: this.reverseDepth ? 0 : 1,depthLoadOp: 'clear',depthStoreOp: storeDepth ? 'store' : 'discard' } })
  }

  finish(encoder: GPUCommandEncoder,output: GPUTextureView,p: Presentation,redrawSurface: boolean,rebuildDisplay: boolean) {
    const willBlur = redrawSurface && Boolean(p.motion && this.motionPipeline && this.motionUniform)
    const c = this.compositeScratch
    c[0] = 0; c[1] = 0; c[2] = 0; c[3] = Number(p.hdr)
    c[4] = p.exposure; c[5] = p.hdr ? p.glow : 0; c[6] = p.ratio; c[7] = Number(p.dataBoundary)
    c[8] = 0; c[9] = p.hoverBody; c[10] = this.width; c[11] = this.height
    for (let i = 0; i < p.markers.length; i++) c[12 + i] = p.markers[i]
    c[17] = Number(willBlur)
    this.device.queue.writeBuffer(this.compositeUniform,0,c)
    const motion = redrawSurface && this.encodeMotion(encoder,p.motion)
    const motionChanged = motion !== this.displayedMotion
    if (redrawSurface || motionChanged) this.bloomValid = false
    const rebuildBloom = p.hdr && p.glow > 0 && !this.bloomValid
    if (rebuildBloom) {
      for (let i = 0; i < 3; i++) {
        this.bloomScratch[0] = i === 0 ? 1 / this.width : i === 1 ? 2 * p.ratio / this.width : 0
        this.bloomScratch[1] = i === 0 ? 1 / this.height : i === 2 ? 2 * p.ratio / this.height : 0
        this.bloomScratch[2] = Number(i === 0)
        this.bloomScratch[3] = 0
        this.device.queue.writeBuffer(this.uniforms[i],0,this.bloomScratch)
        const pass = encoder.beginRenderPass({ label: `Bloom pass ${i}`,colorAttachments: [{ view: this.bloom[i % 2].createView(),loadOp: 'clear',storeOp: 'store' }] })
        pass.setPipeline(this.bloomPipeline); pass.setBindGroup(0,i === 0 && motion ? this.motionBloomGroup! : this.bloomGroups[i]); pass.draw(3); pass.end()
      }
      this.bloomValid = true
    }
    const composition = rebuildDisplay || redrawSurface || rebuildBloom || motionChanged
    if (composition) {
      const pass = encoder.beginRenderPass({ label: 'Cache tone map and coverage',colorAttachments: [{ view: this.display.createView(),loadOp: 'clear',storeOp: 'store' }] })
      pass.setPipeline(this.compositionPipeline); pass.setBindGroup(0,motion ? this.motionCompositionGroup! : this.compositionGroup); pass.draw(3); pass.end()
    }
    const pass = encoder.beginRenderPass({ label: 'Present cached surface and markers',colorAttachments: [{ view: output,loadOp: 'clear',storeOp: 'store' }] })
    pass.setPipeline(this.presentationPipeline); pass.setBindGroup(0,this.presentationGroup); pass.draw(3)
    if (p.loupe && this.loupeGroup) {
      const sel = p.loupe.selected
      const l = this.loupeScratch
      l[0] = p.loupe.x; l[1] = p.loupe.y; l[2] = p.loupe.radiusCss; l[3] = p.ratio
      l[4] = this.width; l[5] = this.height; l[6] = p.exposure; l[7] = Number(p.hdr)
      l[8] = p.loupe.aimX; l[9] = p.loupe.aimY; l[10] = Number(p.loupe.locked); l[11] = p.loupe.zoom
      l[12] = sel ? sel[0] : 0; l[13] = sel ? sel[1] : 0; l[14] = Number(Boolean(sel)); l[15] = 0
      this.device.queue.writeBuffer(this.loupeUniform,0,l)
      pass.setPipeline(this.loupePipeline); pass.setBindGroup(0,this.loupeGroup); pass.draw(6)
    } else if (p.cursor) {
      const cur = this.cursorScratch
      cur[0] = p.cursor[0]; cur[1] = p.cursor[1]; cur[2] = this.width; cur[3] = this.height
      cur[4] = p.ratio; cur[5] = 0; cur[6] = 0; cur[7] = 0
      this.device.queue.writeBuffer(this.cursorUniform,0,cur)
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
    this.loupeScene?.destroy(); this.loupeMultisample?.destroy(); this.loupeDepth?.destroy()
    this.loupeScene = this.loupeMultisample = this.loupeDepth = undefined; this.loupeGroup = undefined
    this.emptyCoverage.destroy()
    this.motionUniform?.destroy()
    this.uniforms.forEach(buffer => buffer.destroy()); this.compositeUniform.destroy(); this.cursorUniform.destroy(); this.loupeUniform.destroy()
  }
}
