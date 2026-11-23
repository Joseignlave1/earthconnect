import type { WorkspaceAssets } from './workspaceAssets.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import { EARTH_AXIS, FOV } from '../shader-test/geometry.ts'
import { fullscreenVertex, shaderModule } from '../shader-test/gpu.ts'
import { LOUPE_TEXTURE_SIZE } from '../shader-test/postprocess.ts'
import { PlanetTerrainRenderer, type PlanetTerrain } from './planetTerrain.ts'
import type { PlanetTilePackLoader } from './planetTiles.ts'
import { orbitPolyline, solarSystemAt, type BodyId, type Vec3 } from './solarSystem.ts'
import { loadSkyMap } from './skyMap.ts'
import {
  add, bodyMatrix, dot, length, normalized, raySphere, relativeToEye, scale, sceneRay, SolarCamera,
  subtract, worldToGeographic, type SolarFrame, type SourceBody,
} from './solarCamera.ts'

const EARTH_EQUATORIAL_KM = EARTH_AXIS / 1000
const ids: readonly BodyId[] = ['sun','mercury','venus','earth','moon','mars','jupiter','saturn','uranus','neptune']
const sources: readonly SourceBody[] = ['moon','mars']
const colors: Record<BodyId, Vec3> = {
  sun: [10.0,9.4,8.6],
  mercury: [.17,.14,.12],
  venus: [.57,.46,.27],
  earth: [.10,.19,.39],
  moon: [.18,.185,.20],
  mars: [.24,.075,.032],
  jupiter: [.62,.46,.35],
  saturn: [.59,.49,.36],
  uranus: [.24,.45,.50],
  neptune: [.18,.27,.57],
}

const commonShader = `
struct View { right: vec4f, up: vec4f, back: vec4f, light: vec4f, crop: vec4f, }
@group(0) @binding(0) var<uniform> view: View;
fn viewRay(point: vec2f) -> vec3f {
  let mapped = view.crop.xy + point * view.crop.zw;
  let p = vec2f(mapped.x * view.right.w, mapped.y) * view.up.w;
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
  let raw = textureSampleGrad(atlas,atlasSampler,uv,dx,dy).rgb;
  let luma = dot(raw,vec3f(.2126,.7152,.0722));
  // Gentle Milky Way dust richness on the 8K EXR-derived sky map + restrained bright-star lift.
  let clean = smoothstep(.0006,.0045,luma);
  let nebulaBand = smoothstep(.003,.040,luma) * (1. - smoothstep(.06,.24,luma));
  var sky = raw * mix(.70,1.,clean) + (raw - vec3f(luma)) * (.22 * nebulaBand);
  let starBoost = pow(max(0.,luma - .22),2.0) * 1.6;
  let spectral = mix(vec3f(1.),raw / max(luma,1e-4),1.1);
  sky += spectral * starBoost;
  // Subtle exposure adaptation: stars dim gently right inside the Sun's bright corona.
  let sunGlare = smoothstep(.92,.998,dot(ray,normalize(view.light.xyz)));
  sky *= 1. - .55 * sunGlare;
  return vec4f(max(sky,vec3f(0.)),1.);
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
  let pad = vec2f(1.5 * view.light.w / (tan * aspect),1.5 * view.light.w / tan);
  let corner = center + corners[vertex] * (max(extent,vec2f(minSize / aspect,minSize)) + pad);
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
  let dist = length(offset);
  let pxWorld = max(length(vec2f(dpdx(dist),dpdy(dist))),max(middle,0.) * view.light.w * .5);
  let worldRadius = middle * view.light.w * body.flags.y;
  let effRadius = max(radius,worldRadius);
  let aa = smoothstep(-.65,.65,(effRadius - dist) / max(pxWorld,1e-6));
  let coverage = aa * sqrt(aa);
  if (middle <= 0. || coverage <= .002) { discard; }
  let gap = radius * radius - dot(offset,offset);
  let intersection = middle - sqrt(max(gap,0.));
  let depth = -dir.z * intersection;
  if (depth <= view.back.w) { discard; }
  let limbNormal = normalize(-offset - dir * sqrt(max(gap,0.)));
  let physical = radius >= worldRadius || gap >= 0.;
  let normal = select(-dir,limbNormal,physical);
  let sunDir = normalize(view.light.xyz - position);
  let sunDot = dot(normal,sunDir);
  let daylight = smoothstep(-.035,.045,sunDot);
  let diffuse = daylight * max(0.,sunDot);
  let glint = 1. - smoothstep(.16,1.,dist / max(worldRadius,.001));
  let edge = select(mix(1.,glint,body.flags.z),1.,physical);
  let surface = body.color.xyz * (edge * select(.60 + .30 * diffuse,.06 + .94 * diffuse,physical) + .84 * glint * body.flags.z);
  let sunMu = max(0.,-dot(normal,dir));
  let sunLimb = mix(vec3f(1.,.74,.44),vec3f(1.),pow(sunMu,.45));
  let sun = body.color.xyz * sunLimb * (.45 + .55 * pow(sunMu,.6));
  let radiance = select(surface,sun,body.color.w > .5);
  return Pixel(vec4f(radiance,coverage),view.back.w / depth);
}
`

const haloShader = `${fullscreenVertex}${commonShader}
struct HaloParams { sun: vec4f, } // ndcX, ndcY, visibility [0..1], aspect
@group(0) @binding(1) var<uniform> halo: HaloParams;
struct Halo { @location(0) color: vec4f, @builtin(frag_depth) depth: f32, }
@fragment fn haloFragment(input: ScreenVertex) -> Halo {
  let ndc = vec2f(input.uv.x * 2. - 1.,1. - input.uv.y * 2.);
  let ray = viewRay(ndc);
  let sun = view.light.xyz;
  let sunDist = length(sun);
  if (sunDist <= view.back.w) { discard; }
  let sunDir = sun / sunDist;
  let cosAngle = dot(ray,sunDir);
  let distance = cosAngle * sunDist;
  if (cosAngle <= 0. || distance <= view.back.w) { discard; }
  let sunAngRadius = max(695700. / sunDist,view.light.w * 2.6);
  let r = length(ray - sunDir * cosAngle) / sunAngRadius;
  if (r >= 24.) { discard; }
  let sunVis = halo.sun.z;
  // Pure ray-free multi-scale optical glare with micro R/G/B glass dispersion.
  let rVec = vec3f(r * .95,r,r * 1.055);
  let outerFade = smoothstep(24.,7.,r);
  let coreGlow = 1.75 / (vec3f(1.) + rVec * rVec * 1.15);
  let midHalo = .20 / pow(vec3f(1.) + rVec * .55,vec3f(2.35));
  let wideVeil = .024 / (vec3f(1.) + rVec * rVec * .065);
  // Crisp diamond-ring intensity lift when a planet's limb grazes the solar disc.
  let eclipseGraze = 4. * sunVis * (1. - sunVis);
  let diamond = (1.1 / (1. + r * r * 1.8)) * eclipseGraze * .45;
  let color = (
    vec3f(1.,.98,.93) * coreGlow +
    vec3f(1.,.86,.66) * midHalo +
    vec3f(1.,.91,.80) * wideVeil +
    vec3f(1.,.96,.88) * diamond
  ) * outerFade;
  let depth = -ray.z * distance;
  if (depth <= view.back.w) { discard; }
  return Halo(vec4f(color,0.),view.back.w / depth);
}
// Mechanical barrel vignetting: intersecting two offset aperture circles squeezes off-axis
// bokeh ghosts into realistic almonds perpendicular to the optical axis.
fn catsEyeGhost(p: vec2f, center: vec2f, axis: vec2f, ecc: f32, radius: f32, clip: f32, rim: f32) -> f32 {
  let v = p - center;
  let shift = axis * (radius * clamp(ecc * clip,0.,.62));
  let d = max(length(v - shift),length(v + shift)) / radius;
  if (d >= 1.12) { return 0.; }
  let mask = smoothstep(1.04,.76,d);
  let core = .38 + .62 * smoothstep(0.,.90,d);
  let caustic = exp(-pow((d - .90) / .11,2.)) * smoothstep(1.06,.94,d);
  return mask * mix(core,caustic * 1.15 + core * .24,rim);
}
// Lateral R/G/B chromatic separation along the flare axis for multi-coated glass reflections.
fn chromaticGhost(p: vec2f, sun: vec2f, axis: vec2f, ecc: f32, pos: f32, radius: f32, clip: f32, rim: f32, disp: f32, tint: vec3f) -> vec3f {
  let r = catsEyeGhost(p,sun * (pos + disp),axis,ecc,radius * 1.03,clip,rim);
  let g = catsEyeGhost(p,sun * pos,axis,ecc,radius,clip,rim);
  let b = catsEyeGhost(p,sun * (pos - disp),axis,ecc,radius * .97,clip,rim);
  return vec3f(r,g,b) * tint;
}
// Foreground camera-lens pass (no depth test): reflections float in front of planets and
// vanish smoothly via sunVis whenever a planet eclipses the Sun.
@fragment fn lensFlareFragment(input: ScreenVertex) -> @location(0) vec4f {
  let sunVis = halo.sun.z;
  let sunNdc = halo.sun.xy;
  let aspect = halo.sun.w;
  let sunSensor = sunNdc * vec2f(aspect,1.);
  let sunEcc = length(sunSensor);
  let onScreen = smoothstep(1.55,.82,max(abs(sunNdc.x),abs(sunNdc.y)));
  let offAxis = smoothstep(.06,.30,sunEcc);
  let weight = sunVis * onScreen * offAxis;
  if (weight <= .001) { discard; }
  let ndc = vec2f(input.uv.x * 2. - 1.,1. - input.uv.y * 2.);
  let p = ndc * vec2f(aspect,1.);
  let axis = sunSensor / max(sunEcc,1e-4);
  var flare = vec3f(0.);
  // Near-focus specular pinpoint, warm diffuse reflection, AR-coated caustic bubble, far ghost.
  flare += chromaticGhost(p,sunSensor,axis,sunEcc,-.22,.022,.15,.15,.014,vec3f(1.,.92,.78)) * .022;
  flare += chromaticGhost(p,sunSensor,axis,sunEcc,-.44,.068,.46,.24,.022,vec3f(1.,.78,.58)) * .011;
  flare += chromaticGhost(p,sunSensor,axis,sunEcc,-.82,.135,.54,.84,.032,vec3f(.52,.94,.88)) * .010;
  flare += chromaticGhost(p,sunSensor,axis,sunEcc,-1.28,.044,.36,.42,.028,vec3f(.66,.76,1.)) * .014;
  // Whisper-faint curved chromatic diffraction arc from front-element off-axis grazing.
  let pRad = length(p);
  let pDir = p / max(pRad,1e-4);
  let arcAlign = pow(max(0.,dot(pDir,-axis)),3.5);
  let arcR = vec3f(pRad * .985,pRad,pRad * 1.015);
  let arcRing = exp(-pow((arcR - vec3f(.76)) / vec3f(.055),vec3f(2.)));
  flare += arcRing * arcAlign * smoothstep(.35,.95,sunEcc) * .009;
  flare *= weight;
  if (max(flare.r,max(flare.g,flare.b)) <= .0002) { discard; }
  return vec4f(flare,0.);
}
`

const earthAtmoShader = `${fullscreenVertex}${commonShader}
struct EarthAtmo { sphere: vec4f, params: vec4f, }
@group(0) @binding(1) var<uniform> atmo: EarthAtmo;
struct AtmoOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32, }
@fragment fn earthAtmoFragment(input: ScreenVertex) -> AtmoOut {
  if (atmo.params.x <= .001) { discard; }
  let isMars = atmo.params.y > .5;
  let ray = viewRay(vec2f(input.uv.x * 2. - 1.,1. - input.uv.y * 2.));
  let center = atmo.sphere.xyz;
  let radius = atmo.sphere.w;
  let t = dot(ray,center);
  if (t <= view.back.w) { discard; }
  let offset = ray * t - center;
  let b = length(offset);
  let maxScale = select(1.058, 1.032, isMars);
  if (b >= radius * maxScale) { discard; }
  // Start right inside the planetary polar radius so reverse-Z depth cleanly clips the solid interior.
  let innerScale = select(.996, .994, isMars);
  let shellWidth = select(.046, .024, isMars);
  let alt = max(0.,(b - radius * innerScale) / (radius * shellWidth));
  let shell = exp(-alt * select(4.2, 4.6, isMars)) * smoothstep(1.25,0.,alt);
  let limbNormal = offset / max(b,1.e-4);
  let sunDir = normalize(view.light.xyz - center);
  let sunDot = dot(limbNormal,sunDir);
  let day = smoothstep(-.14,.26,sunDot);
  let backScatter = pow(max(0.,dot(ray,sunDir)),5.);
  let twilight = exp(-pow((sunDot + .01) / .13,2.)) * (1. + 2.2 * backScatter);
  let radiance = select(
    vec3f(.18,.50,1.0) * .68 * day +
    vec3f(1.,.42,.12) * .12 * twilight +
    vec3f(.04,.10,.20) * .20 * (1. - day),
    vec3f(.92,.56,.34) * .34 * day +
    vec3f(.38,.56,.80) * .08 * twilight +
    vec3f(.10,.06,.04) * .06 * (1. - day),
    isMars,
  ) * shell * atmo.params.x;
  let depth = -ray.z * t;
  if (depth <= view.back.w) { discard; }
  return AtmoOut(vec4f(radiance,0.),view.back.w / depth);
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

type SolarPipelines = {
  skyPipeline: GPURenderPipeline
  bodyPipeline: GPURenderPipeline
  orbitPipeline: GPURenderPipeline
  haloPipeline: GPURenderPipeline
  lensFlarePipeline: GPURenderPipeline
  earthAtmoPipeline: GPURenderPipeline
}

const solarPipelines = new WeakMap<GPUDevice, Promise<SolarPipelines>>()

function getSolarPipelines(device: GPUDevice): Promise<SolarPipelines> {
  let pending = solarPipelines.get(device)
  if (!pending) {
    pending = (async () => {
      const [sky,body,orbit,halo,earthAtmo] = await Promise.all([
        shaderModule(device,'Inertial sky',skyShader),
        shaderModule(device,'Solar bodies',bodyShader),
        shaderModule(device,'Planetary paths',orbitShader),
        shaderModule(device,'Sunlight halo',haloShader),
        shaderModule(device,'Earth Rayleigh atmosphere',earthAtmoShader),
      ])
      const color: GPUColorTargetState = { format: 'rgba16float' }
      const depth: GPUDepthStencilState = { format: 'depth32float',depthWriteEnabled: true,depthCompare: 'greater' }
      const common = { layout: 'auto' as const,multisample: { count: 4 } }
      const additive: GPUBlendState = {
        color: { srcFactor: 'one',dstFactor: 'one' },alpha: { srcFactor: 'zero',dstFactor: 'one' },
      }
      const alphaBlend: GPUBlendState = {
        color: { srcFactor: 'src-alpha',dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one',dstFactor: 'one-minus-src-alpha' },
      }
      const [skyPipeline,bodyPipeline,orbitPipeline,haloPipeline,lensFlarePipeline,earthAtmoPipeline] = await Promise.all([
        device.createRenderPipelineAsync({ ...common,label: 'Inertial sky',
          vertex: { module: sky,entryPoint: 'screenVertex' },fragment: { module: sky,entryPoint: 'skyFragment',targets: [color] },
          depthStencil: { ...depth,depthWriteEnabled: false,depthCompare: 'greater-equal' },
        }),
        device.createRenderPipelineAsync({ ...common,label: 'Sun and planet disks',
          vertex: { module: body,entryPoint: 'bodyVertex' },fragment: { module: body,entryPoint: 'bodyFragment',targets: [{ ...color,blend: alphaBlend }] },depthStencil: depth,
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
          vertex: { module: halo,entryPoint: 'screenVertex' },fragment: { module: halo,entryPoint: 'haloFragment',targets: [{ ...color,blend: additive }] },
          depthStencil: { ...depth,depthWriteEnabled: false },
        }),
        device.createRenderPipelineAsync({ ...common,label: 'Camera lens flare',
          vertex: { module: halo,entryPoint: 'screenVertex' },fragment: { module: halo,entryPoint: 'lensFlareFragment',targets: [{ ...color,blend: additive }] },
          depthStencil: { ...depth,depthWriteEnabled: false,depthCompare: 'always' },
        }),
        device.createRenderPipelineAsync({ ...common,label: 'Earth Rayleigh atmosphere',
          vertex: { module: earthAtmo,entryPoint: 'screenVertex' },fragment: { module: earthAtmo,entryPoint: 'earthAtmoFragment',targets: [{ ...color,blend: additive }] },
          depthStencil: { ...depth,depthWriteEnabled: false },
        }),
      ])
      return { skyPipeline,bodyPipeline,orbitPipeline,haloPipeline,lensFlarePipeline,earthAtmoPipeline }
    })()
    solarPipelines.set(device, pending)
    void pending.catch(() => { if (solarPipelines.get(device) === pending) solarPipelines.delete(device) })
  }
  return pending
}

export class SolarScene {
  readonly bodies
  private terrain?: PlanetTerrainRenderer
  private loadingTerrain = false
  skySettled = false
  private readonly controller = new AbortController()
  private readonly viewBuffer: GPUBuffer
  private readonly loupeViewBuffer: GPUBuffer
  private readonly earthAtmoBuffer: GPUBuffer
  private readonly marsAtmoBuffer: GPUBuffer
  private readonly haloBuffer: GPUBuffer
  private readonly viewScratch = new Float32Array(20)
  private readonly loupeViewScratch = new Float32Array(20)
  private readonly earthAtmoScratch = new Float32Array(8)
  private readonly marsAtmoScratch = new Float32Array(8)
  private readonly haloScratch = new Float32Array(4)
  private readonly bodyBuffer: GPUBuffer
  private readonly orbitBuffer: GPUBuffer
  private readonly orbits: { id: BodyId; vertices: readonly Vec3[] }[]
  private readonly bodyData = new Float32Array(ids.length * 12)
  private readonly orbitData: Float32Array<ArrayBuffer>
  private skyGroup: GPUBindGroup
  private loupeSkyGroup: GPUBindGroup
  private skyTexture: GPUTexture
  private skyOwned = true
  private readonly skySampler: GPUSampler
  private readonly haloGroup: GPUBindGroup
  private readonly lensFlareGroup: GPUBindGroup
  private readonly earthAtmoGroup: GPUBindGroup
  private readonly loupeEarthAtmoGroup: GPUBindGroup
  private readonly marsAtmoGroup: GPUBindGroup
  private readonly loupeMarsAtmoGroup: GPUBindGroup
  private readonly bodyGroup: GPUBindGroup

  static async create(device: GPUDevice,date: Date,invalidate: () => void,onError: (error: Error) => void,camera: SolarCamera | undefined, assetUrl: WorkspaceAssets['url'], terrainLoader?: TerrainLoader, loadPlanetTilePack?: PlanetTilePackLoader) {
    const { skyPipeline,bodyPipeline,orbitPipeline,haloPipeline,lensFlarePipeline,earthAtmoPipeline } = await getSolarPipelines(device)
    return new SolarScene(device,date,skyPipeline,bodyPipeline,orbitPipeline,haloPipeline,lensFlarePipeline,earthAtmoPipeline,invalidate,onError,camera ?? new SolarCamera(),assetUrl,terrainLoader,loadPlanetTilePack)
  }

  private constructor(private readonly device: GPUDevice,date: Date,private readonly skyPipeline: GPURenderPipeline,
    private readonly bodyPipeline: GPURenderPipeline,private readonly orbitPipeline: GPURenderPipeline,
    private readonly haloPipeline: GPURenderPipeline,private readonly lensFlarePipeline: GPURenderPipeline,
    private readonly earthAtmoPipeline: GPURenderPipeline,
    private readonly invalidate: () => void,private readonly onError: (error: Error) => void,
    readonly camera: SolarCamera, private readonly assetUrl: WorkspaceAssets['url'], private readonly terrainLoader?: TerrainLoader, private readonly loadPlanetTilePack?: PlanetTilePackLoader) {
    this.bodies = solarSystemAt(date)
    this.orbits = ids.flatMap(id => {
      const vertices = orbitPolyline(id,date)
      return vertices.length ? [{ id,vertices }] : []
    })
    this.orbitData = new Float32Array(this.orbits.reduce((count,path) => count + (path.vertices.length - 1) * 16,0))
    this.orbitBuffer = device.createBuffer({ label: 'Camera-relative orbit lines',size: this.orbitData.byteLength,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX })
    this.viewBuffer = device.createBuffer({ label: 'Heliocentric camera basis',size: 80,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.loupeViewBuffer = device.createBuffer({ label: 'Loupe sky camera basis',size: 80,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.earthAtmoBuffer = device.createBuffer({ label: 'Earth atmosphere sphere',size: 32,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.marsAtmoBuffer = device.createBuffer({ label: 'Mars atmosphere sphere',size: 32,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.haloBuffer = device.createBuffer({ label: 'Solar glare and lens ghosts',size: 16,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM })
    this.bodyBuffer = device.createBuffer({ label: 'Solar system centers in camera space',size: this.bodyData.byteLength,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE })
    this.skySampler = device.createSampler({
      addressModeU: 'repeat',addressModeV: 'clamp-to-edge',minFilter: 'linear',magFilter: 'linear',mipmapFilter: 'linear',maxAnisotropy: 4,
    })
    this.skyTexture = device.createTexture({
      label: 'Dark sky placeholder',size: [1,1],format: 'rgba8unorm-srgb',usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    device.queue.writeTexture({ texture: this.skyTexture },new Uint8Array([6,6,7,255]),{},[1,1])
    this.skyGroup = this.bindSky(this.viewBuffer,this.skyTexture)
    this.loupeSkyGroup = this.bindSky(this.loupeViewBuffer,this.skyTexture)
    this.haloGroup = device.createBindGroup({ layout: haloPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: { buffer: this.haloBuffer } },
    ] })
    this.lensFlareGroup = device.createBindGroup({ layout: lensFlarePipeline.getBindGroupLayout(0),entries: [
      { binding: 1,resource: { buffer: this.haloBuffer } },
    ] })
    this.earthAtmoGroup = device.createBindGroup({ layout: earthAtmoPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: { buffer: this.earthAtmoBuffer } },
    ] })
    this.loupeEarthAtmoGroup = device.createBindGroup({ layout: earthAtmoPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.loupeViewBuffer } },{ binding: 1,resource: { buffer: this.earthAtmoBuffer } },
    ] })
    this.marsAtmoGroup = device.createBindGroup({ layout: earthAtmoPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: { buffer: this.marsAtmoBuffer } },
    ] })
    this.loupeMarsAtmoGroup = device.createBindGroup({ layout: earthAtmoPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.loupeViewBuffer } },{ binding: 1,resource: { buffer: this.marsAtmoBuffer } },
    ] })
    this.bodyGroup = device.createBindGroup({ layout: bodyPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer: this.viewBuffer } },{ binding: 1,resource: { buffer: this.bodyBuffer } },
    ] })
    const { signal } = this.controller
    void this.assetUrl('textures/nasa-deep-star-map.webp', signal)
      .then(url => loadSkyMap(device,signal,url)).then(texture => {
      if (signal.aborted) return
      const group = this.bindSky(this.viewBuffer,texture)
      const loupeGroup = this.bindSky(this.loupeViewBuffer,texture)
      if (this.skyOwned) this.skyTexture.destroy()
      this.skyTexture = texture
      this.skyOwned = false
      this.skyGroup = group
      this.loupeSkyGroup = loupeGroup
    }).catch(() => {
      // Keep rendering the planets over dark space if the atlas is unavailable.
    }).finally(() => {
      if (signal.aborted) return
      this.skySettled = true
      this.invalidate()
    })
  }

  private bindSky(buffer: GPUBuffer,texture: GPUTexture) {
    return this.device.createBindGroup({ layout: this.skyPipeline.getBindGroupLayout(0),entries: [
      { binding: 0,resource: { buffer } },{ binding: 1,resource: texture.createView() },{ binding: 2,resource: this.skySampler },
    ] })
  }

  get terrainReady() { return Boolean(this.terrain) }

  loadTerrain() {
    if (this.loadingTerrain || this.terrain || this.controller.signal.aborted) return
    this.loadingTerrain = true
    const { signal } = this.controller
    void PlanetTerrainRenderer.create(this.device,signal,this.assetUrl,this.terrainLoader,this.invalidate,this.loadPlanetTilePack).then(terrain => {
      if (signal.aborted) { terrain.destroy(); return }
      this.terrain = terrain
      this.invalidate()
    },error => {
      if (!signal.aborted) this.onError(error instanceof Error ? error : new Error(String(error)))
    }).finally(() => { this.loadingTerrain = false })
  }

  updatePlanetTiles(
    frame: SolarFrame,
    body: SourceBody,
    clientHeight: number,
    pixelRatio: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    if (!this.terrain) return
    const state = this.bodies[body]
    const eyeUnit = scale(worldToGeographic(state, subtract(frame.eye, state.center)), 1 / state.radius)
    const right = worldToGeographic(state, frame.right)
    const up = worldToGeographic(state, frame.up)
    const back = worldToGeographic(state, frame.back)
    const metrics = this.bodyMarkerMetrics(frame, body, clientHeight)
    const physicalMpp = (metrics.scale * state.radius * 1000) / Math.max(1, pixelRatio)
    this.terrain.updateTiles(body, eyeUnit, right, up, back, frame.aspect, physicalMpp, loupe)
  }

  planetTileStatus() {
    return this.terrain?.tileStatus() ?? ''
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

  earthMatrix(frame: SolarFrame) { return bodyMatrix(frame,this.bodies.earth,EARTH_EQUATORIAL_KM) }

  earthEye(frame: SolarFrame): Vec3 {
    const earth = this.bodies.earth
    return scale(worldToGeographic(earth,subtract(frame.eye,earth.center)),1 / EARTH_EQUATORIAL_KM)
  }

  private writeView(frame: SolarFrame,height: number) {
    const queue = this.device.queue
    const tangent = Math.tan(FOV / 2)
    const sunRel = relativeToEye(frame,this.bodies.sun.center)
    this.viewScratch.set(frame.right,0); this.viewScratch[3] = frame.aspect
    this.viewScratch.set(frame.up,4); this.viewScratch[7] = tangent
    this.viewScratch.set(frame.back,8); this.viewScratch[11] = frame.near
    this.viewScratch.set(sunRel,12); this.viewScratch[15] = tangent * 2 / height
    this.viewScratch.set([0,0,1,1],16)
    queue.writeBuffer(this.viewBuffer,0,this.viewScratch)
    let sunVis = 0
    let sunNdcX = 10, sunNdcY = 10
    if (-sunRel[2] > frame.near) {
      sunNdcX = sunRel[0] / (-sunRel[2] * tangent * frame.aspect)
      sunNdcY = sunRel[1] / (-sunRel[2] * tangent)
      const toSun = subtract(this.bodies.sun.center,frame.eye)
      const sunDist = length(toSun)
      const sunDir = scale(toSun,1 / Math.max(1,sunDist))
      const sunAng = this.bodies.sun.radius / Math.max(1,sunDist)
      sunVis = 1
      for (const id of ids) {
        if (id === 'sun') continue
        const body = this.bodies[id]
        const rel = subtract(body.center,frame.eye)
        const proj = dot(rel,sunDir)
        if (proj <= 0 || proj >= sunDist) continue
        const perpDist = length(subtract(rel,scale(sunDir,proj)))
        const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
        const clearAng = (perpDist - radius) / proj
        const t = Math.max(0,Math.min(1,(clearAng + sunAng) / Math.max(1e-6,2 * sunAng)))
        sunVis = Math.min(sunVis,t * t * (3 - 2 * t))
      }
    }
    this.haloScratch.set([sunNdcX,sunNdcY,sunVis,frame.aspect])
    queue.writeBuffer(this.haloBuffer,0,this.haloScratch)
  }

  drawSky(pass: GPURenderPassEncoder,frame: SolarFrame,height: number) {
    this.writeView(frame,height)
    pass.setPipeline(this.skyPipeline)
    pass.setBindGroup(0,this.skyGroup)
    pass.draw(3)
  }

  draw(
    pass: GPURenderPassEncoder,frame: SolarFrame,height: number,clientHeight: number,hdr: boolean,
    pixelRatio: number,overview: boolean,flightTarget?: SourceBody | 'earth',drawEarth?: () => void,
  ) {
    const queue = this.device.queue
    const tangent = Math.tan(FOV / 2)
    // Sort bodies farthest-to-nearest so antialiased limbs blend cleanly over space and farther bodies.
    const ordered = ids.map(id => {
      const delta = subtract(this.bodies[id].center,frame.eye)
      return { id, distSq: dot(delta,delta) }
    }).sort((a,b) => b.distSq - a.distSq)
    const meshFlags: boolean[] = []
    for (const [index,{ id }] of ordered.entries()) {
      const body = this.bodies[id]
      const mesh = (id === 'earth' || (id === 'moon' || id === 'mars') && Boolean(this.terrain?.getTerrain(id))) && this.detailed(frame,id,height)
      meshFlags.push(mesh)
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
    let runStart = -1
    const flushBodies = (end: number) => {
      if (runStart < 0) return
      pass.setPipeline(this.bodyPipeline)
      pass.setBindGroup(0,this.bodyGroup)
      pass.draw(6,end - runStart,0,runStart)
      runStart = -1
    }
    for (let i = 0; i < ordered.length; i++) {
      const { id } = ordered[i]
      if (!meshFlags[i]) {
        if (runStart < 0) runStart = i
        continue
      }
      flushBodies(i)
      if (id === 'earth') {
        drawEarth?.()
      } else if ((id === 'moon' || id === 'mars') && this.terrain?.getTerrain(id)) {
        const metrics = this.bodyMarkerMetrics(frame,id,clientHeight)
        const metresPerPixel = metrics.scale * this.bodies[id].radius * 1000
        this.terrain.draw(pass,id,bodyMatrix(frame,this.bodies[id],1),this.sunlight(id),this.sourceEye(frame,id),metresPerPixel,hdr,pixelRatio)
      }
    }
    flushBodies(ordered.length)
  }

  drawHalo(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.haloPipeline)
    pass.setBindGroup(0,this.haloGroup)
    pass.draw(3)
    if (this.haloScratch[2] > .001) {
      pass.setPipeline(this.lensFlarePipeline)
      pass.setBindGroup(0,this.lensFlareGroup)
      pass.draw(3)
    }
  }

  private atmoFade(frame: SolarFrame,id: 'earth' | 'mars',clientHeight: number) {
    const body = this.bodies[id]
    const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
    const distance = length(subtract(frame.eye,body.center))
    const radiusPx = (radius / Math.max(radius,distance)) * Math.max(1,clientHeight) / (2 * Math.tan(FOV / 2))
    const t = Math.max(0,Math.min(1,(radiusPx - 6) / 12))
    return t * t * (3 - 2 * t)
  }

  private writeEarthAtmo(frame: SolarFrame,strength: number) {
    const pos = relativeToEye(frame,this.bodies.earth.center)
    this.earthAtmoScratch.set([pos[0],pos[1],pos[2],EARTH_EQUATORIAL_KM,strength,0,0,0])
    this.device.queue.writeBuffer(this.earthAtmoBuffer,0,this.earthAtmoScratch)
  }

  drawEarthAtmosphere(pass: GPURenderPassEncoder,frame: SolarFrame,morph: number,clientHeight: number) {
    const strength = morph * this.atmoFade(frame,'earth',clientHeight)
    if (strength <= .001) return
    this.writeEarthAtmo(frame,strength)
    pass.setPipeline(this.earthAtmoPipeline)
    pass.setBindGroup(0,this.earthAtmoGroup)
    pass.draw(3)
  }

  drawLoupeEarthAtmosphere(pass: GPURenderPassEncoder,frame: SolarFrame,morph: number) {
    this.writeEarthAtmo(frame,morph)
    pass.setPipeline(this.earthAtmoPipeline)
    pass.setBindGroup(0,this.loupeEarthAtmoGroup)
    pass.draw(3)
  }

  private writeMarsAtmo(frame: SolarFrame,strength: number) {
    const mars = this.bodies.mars
    const pos = relativeToEye(frame,mars.center)
    this.marsAtmoScratch.set([pos[0],pos[1],pos[2],mars.radius,strength,1,0,0])
    this.device.queue.writeBuffer(this.marsAtmoBuffer,0,this.marsAtmoScratch)
  }

  drawMarsAtmosphere(pass: GPURenderPassEncoder,frame: SolarFrame,height: number,clientHeight: number) {
    if (!this.terrain?.getTerrain('mars') || !this.detailed(frame,'mars',height)) return
    const strength = this.atmoFade(frame,'mars',clientHeight)
    if (strength <= .001) return
    this.writeMarsAtmo(frame,strength)
    pass.setPipeline(this.earthAtmoPipeline)
    pass.setBindGroup(0,this.marsAtmoGroup)
    pass.draw(3)
  }

  drawLoupeMarsAtmosphere(pass: GPURenderPassEncoder,frame: SolarFrame) {
    if (!this.terrain?.getTerrain('mars')) return
    this.writeMarsAtmo(frame,1)
    pass.setPipeline(this.earthAtmoPipeline)
    pass.setBindGroup(0,this.loupeMarsAtmoGroup)
    pass.draw(3)
  }

  drawAnnotations(pass: GPURenderPassEncoder,frame: SolarFrame,height: number) {
    for (const body of sources) {
      if (this.terrain?.getTerrain(body) && this.detailed(frame,body,height)) {
        this.terrain.drawAnnotation(pass,body,bodyMatrix(frame,this.bodies[body],1))
      }
    }
  }

  bodyMarkerMetrics(frame: SolarFrame,id: SourceBody | 'earth',clientHeight: number) {
    const body = this.bodies[id]
    const radius = id === 'earth' ? EARTH_EQUATORIAL_KM : body.radius
    const distance = length(subtract(frame.eye,body.center))
    const tanHalf = Math.tan(FOV / 2)
    const h = Math.max(1,clientHeight)
    const scale = 2 * tanHalf * Math.max(.001,distance / radius - 1) / h
    const radiusPx = (radius / Math.max(radius,distance)) * h / (2 * tanHalf)
    const t = Math.max(0,Math.min(1,(radiusPx - 12) / 22))
    const alpha = t * t * (3 - 2 * t)
    return { scale,alpha }
  }

  sourceMatrix(frame: SolarFrame,body: SourceBody) { return bodyMatrix(frame,this.bodies[body],1) }

  sourcePosition(body: SourceBody,point: SourcePoint): Vec3 | null {
    const terrain = this.terrain?.getTerrain(body)
    if (!terrain) return null
    const rad = Math.PI / 180
    return terrain.positionAt(point.longitude * rad,point.latitude * rad)
  }

  sourceEye(frame: SolarFrame,body: SourceBody): Vec3 {
    const state = this.bodies[body]
    return worldToGeographic(state,subtract(frame.eye,state.center))
  }

  drawLoupeSky(pass: GPURenderPassEncoder,frame: SolarFrame,nx: number,ny: number,invSx: number,invSy: number) {
    const tangent = Math.tan(FOV / 2)
    this.loupeViewScratch.set(frame.right,0); this.loupeViewScratch[3] = frame.aspect
    this.loupeViewScratch.set(frame.up,4); this.loupeViewScratch[7] = tangent
    this.loupeViewScratch.set(frame.back,8); this.loupeViewScratch[11] = frame.near
    this.loupeViewScratch.set(relativeToEye(frame,this.bodies.sun.center),12); this.loupeViewScratch[15] = tangent * 2 / LOUPE_TEXTURE_SIZE
    this.loupeViewScratch.set([nx,ny,invSx,invSy],16)
    this.device.queue.writeBuffer(this.loupeViewBuffer,0,this.loupeViewScratch)
    pass.setPipeline(this.skyPipeline)
    pass.setBindGroup(0,this.loupeSkyGroup)
    pass.draw(3)
  }

  drawLoupe(pass: GPURenderPassEncoder,body: SourceBody,frame: SolarFrame,matrix: Float32Array,metresPerPixel: number,hdr: boolean,pixelRatio: number) {
    if (this.terrain?.getTerrain(body)) {
      this.terrain.drawLoupe(pass,body,matrix,this.sunlight(body),this.sourceEye(frame,body),metresPerPixel,hdr,pixelRatio)
    }
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
    if (this.skyOwned) this.skyTexture.destroy()
    this.viewBuffer.destroy()
    this.loupeViewBuffer.destroy()
    this.earthAtmoBuffer.destroy()
    this.marsAtmoBuffer.destroy()
    this.haloBuffer.destroy()
    this.bodyBuffer.destroy()
    this.orbitBuffer.destroy()
  }
}

function intersectsBlock(block: { min: Vec3; max: Vec3 },eye: Vec3,ray: Vec3,nearest: number) {
  let near = 0, far = nearest
  for (let axis = 0; axis < 3; axis++) {
    const min = block.min[axis] - 1e-6, max = block.max[axis] + 1e-6
    if (Math.abs(ray[axis]) < 1e-12) {
      if (eye[axis] < min || eye[axis] > max) return false
    } else {
      const a = (min - eye[axis]) / ray[axis], b = (max - eye[axis]) / ray[axis]
      near = Math.max(near,Math.min(a,b)); far = Math.min(far,Math.max(a,b))
      if (near > far) return false
    }
  }
  return true
}

// Ray/triangle intersection on the exact indexed vertex positions in the pass,
// accelerated by 8x8-cell bounding blocks for interactive 120 Hz hover picking.
function pickTerrain({ columns,indices,positions,blocks }: PlanetTerrain,eye: Vec3,ray: Vec3) {
  const [ox,oy,oz] = eye, [dx,dy,dz] = ray
  let nearest = Infinity
  for (const block of blocks) {
    if (!intersectsBlock(block,eye,ray,nearest)) continue
    for (let row = block.y; row < block.endY; row++) for (let col = block.x; col < block.endX; col++) {
      const base = (row * columns + col) * 6
      for (let tri = 0; tri < 6; tri += 3) {
        const i = base + tri
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
    }
  }
  return nearest
}
