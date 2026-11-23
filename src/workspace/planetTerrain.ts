import { colorShader } from '../shader-test/color.ts'
import { contourDetail, contourShader } from '../shader-test/contours.ts'
import { createMesh, type Vec3 } from '../shader-test/geometry.ts'
import { copyExternalImageChunked, generateMipmaps, raceAbort, shaderModule, uploadBuffer, writeTextureChunkedAsync } from '../shader-test/gpu.ts'
import { bodyRadius } from './solarSystem.ts'
import { elevationAt, elevationPyramid, packRg8Levels, poleElevations, type Terrain } from '../shader-test/terrain.ts'
import { loadPreparedTerrain, type PreparedTerrain } from '../shader-test/terrain-loader.ts'
import type { TerrainLoader } from './earthWorkspaceResources.ts'
import type { WorkspaceAssets } from './workspaceAssets.ts'
import { PLANET_TILE_PAIRS, PlanetTileCache, type PlanetTilePackLoader } from './planetTiles.ts'

export type TerrainBody = 'moon' | 'mars'

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

export function buildPlanetTerrain(body: TerrainBody, terrain: Terrain, retainElevation = true) {
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

  return { radius, columns, rows, positions, indices, blocks, elevationAt: retainElevation ? readElevation : (() => 0), positionAt }
}

const surfaceShader = /* wgsl */ `
struct View {
  camera: mat4x4f,
  sun: vec4f, // body-fixed sun direction, HDR flag
  terrain: vec4f, // reference radius metres, vertical exaggeration, major contours, DPR
  low: vec4f, // sRGB low albedo, minor contour detail
  high: vec4f, // sRGB high albedo, fine contour detail
  tiles: vec4f, // x: virtual tile pool active (0/1)
  eye: vec4f, // body-local camera position (km), w: body kind (0 = Moon, 1 = Mars)
}
@group(0) @binding(0) var<uniform> u: View;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var gridSampler: sampler;
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var albedoSampler: sampler;
@group(0) @binding(5) var tilePage: texture_2d<u32>;
@group(0) @binding(6) var tilePool: texture_2d_array<f32>;
@group(0) @binding(7) var tileSampler: sampler;
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
fn wrappedGrad(g: vec2f) -> vec2f {
  return vec2f(g.x - round(g.x), g.y);
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
  let pDx = dpdx(p);
  let pDy = dpdy(p);
  let pixelAngle = max(length(pDx), length(pDy));
  // Minor singular value of dp/d(screen) gives the un-foreshortened angular scale everywhere.
  let pxx = dot(pDx, pDx);
  let pyy = dot(pDy, pDy);
  let pxy = dot(pDx, pDy);
  let pDet = max(0., pxx * pyy - pxy * pxy);
  let pMax = max(1e-12, .5 * (pxx + pyy + length(vec2f(pxx - pyy, 2. * pxy))));
  let minorAngle = sqrt(pDet / pMax);
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
  var hillNormal = normalize(p - tangentGrad);
  let uv = dirToUv(p);
  let uvDx = wrappedGrad(dpdx(uv));
  let uvDy = wrappedGrad(dpdy(uv));
  var tileAlbedo = vec3f(0.);
  var tileWeight = 0.;
  if (u.tiles.x > .5) {
    let tileCoord = vec2f(fract(uv.x), clamp(uv.y, 0., .999999));
    let cell = vec2i(clamp(floor(tileCoord * vec2f(48., 24.)), vec2f(0.), vec2f(47., 23.)));
    let entry = textureLoad(tilePage, cell, 0);
    if (entry.r < 255u) {
      let innerScale = 508. / 512.;
      var slot = entry.r;
      var grid = vec2f(24., 12.) * exp2(f32(entry.g));
      var gradScale = grid * innerScale;
      var dx = uvDx * gradScale;
      var dy = uvDy * gradScale;
      var tileMip = log2(max(1., max(length(dx), length(dy)) * 512.));
      if (tileMip > 1.75 && entry.b < 255u) {
        slot = entry.b;
        grid = vec2f(24., 12.) * exp2(f32(entry.a));
        gradScale = grid * innerScale;
        dx = uvDx * gradScale;
        dy = uvDy * gradScale;
        tileMip = log2(max(1., max(length(dx), length(dy)) * 512.));
      }
      if (tileMip < 2.8) {
        let localUv = fract(tileCoord * grid);
        let tileUv = localUv * innerScale + (2. / 512.);
        tileWeight = 1. - smoothstep(2.0, 2.8, tileMip);
        tileAlbedo = textureSampleGrad(tilePool, tileSampler, tileUv, i32(slot), dx, dy).rgb;
        let polarSafe = smoothstep(.04, .14, length(p.xz));
        if (polarSafe > .001) {
          let nSample = textureSampleGrad(tilePool, tileSampler, tileUv, i32(slot + ${PLANET_TILE_PAIRS}u), dx, dy).rgb;
          let tn = normalize(vec3f(nSample.x * 2. - 1., nSample.y * 2. - 1., max(.05, nSample.z)));
          let east = normalize(vec3f(p.z, 0., -p.x));
          let north = cross(p, east);
          let tileNormal = normalize(east * tn.x + north * tn.y + p * tn.z);
          hillNormal = normalize(mix(hillNormal, tileNormal, tileWeight * polarSafe));
        }
      }
    }
  }
  let sun = normalize(u.sun.xyz);
  let viewDir = normalize(u.eye.xyz - input.local);
  let pDotL = dot(p, sun);
  let daylight = smoothstep(-.035, .045, pDotL);
  let ndotl = max(0., dot(hillNormal, sun));
  let ndotv = max(.04, dot(hillNormal, viewDir));
  let isMars = u.eye.w > .5;
  var diffuse = ndotl;
  if (!isMars) {
    // Lommel-Seeliger lunar regolith BRDF: flat dusty opposition brightness + punchy crater relief near the terminator.
    let ls = (2. * ndotl) / max(.06, ndotl + ndotv);
    diffuse = mix(ndotl, ls, .72);
  } else {
    // Oren-Nayar rough desert dust BRDF (sigma = 0.72 rad) for authentic Martian matte-regolith falloff.
    let s2 = .5184;
    let A = 1. - .5 * s2 / (s2 + .33);
    let B = .45 * s2 / (s2 + .09);
    let lProj = sun - hillNormal * ndotl;
    let vProj = viewDir - hillNormal * ndotv;
    let lpLen = length(lProj);
    let vpLen = length(vProj);
    let cosPhi = select(0., clamp(dot(lProj, vProj) / max(1e-5, lpLen * vpLen), -1., 1.), lpLen > 1e-4 && vpLen > 1e-4);
    let sinA = sqrt(max(0., 1. - min(ndotl, ndotv) * min(ndotl, ndotv)));
    let tanB = sqrt(max(0., 1. - max(ndotl, ndotv) * max(ndotl, ndotv))) / max(.04, max(ndotl, ndotv));
    diffuse = ndotl * (A + B * max(0., cosPhi) * sinA * tanB);
  }
  // Distance-aware fill: keep dramatic phases when a planet is tiny in the background
  // (e.g. Moon seen from Earth), while lifting smoothly by the default focused orbital view
  // (~200 CSS px radius) so night-side craters & basins stay readable without brightness pops on zoom.
  let inspect = 1. - smoothstep(.0040, .022, minorAngle * u.terrain.w);
  let fillSide = normalize(cross(viewDir, vec3f(0., 1., 0.)) + vec3f(1e-4, 0., 0.));
  let fillDir = normalize(p + .45 * viewDir + .22 * fillSide + .18 * vec3f(0., 1., 0.));
  let fill = max(0., dot(hillNormal, fillDir));
  let ambient = select(
    mix(.14, .25, inspect) + mix(.06, .13, inspect) * fill,
    mix(.16, .27, inspect) + mix(.07, .14, inspect) * fill,
    isMars,
  );
  let sunTerm = select(
    mix(.80, .72, inspect),
    mix(.76, .70, inspect),
    isMars,
  ) * daylight * diffuse;
  let ambientTint = select(vec3f(.86, .93, 1.06), vec3f(1., .96, .93), isMars);
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
  let albedoSample = textureSampleLevel(albedo, albedoSampler, uv, albedoLod);
  let hasAlbedo = albedoSample.a > .5 || tileWeight > .001;
  if (albedoSample.a > .5) {
    // Convert linear rgba8unorm-srgb sample into display space so contours and toLinear match Earth.
    surface = pow(max(albedoSample.rgb, vec3f(0.)), vec3f(1. / 2.2));
  }
  if (tileWeight > .001) {
    surface = mix(surface, tileAlbedo, tileWeight);
  }
  let lineColor = mix(surface * .52, surface + vec3f(.12, .125, .135), .64);
  surface = mix(surface, lineColor, select(.45, .32, hasAlbedo) * contours);
  var color = surface * (ambientTint * ambient + vec3f(sunTerm));
  // Thin dusty Martian atmospheric limb scatter (warm peach day limb + faint blue twilight).
  var marsAtmo = vec3f(0.);
  if (isMars) {
    let oneMinusV = clamp(1. - dot(p, viewDir), 1e-4, 1.);
    let sunDay = smoothstep(-.12, .22, pDotL);
    let fresnel = pow(oneMinusV, 4.0);
    let horizon = pow(oneMinusV, 7.5);
    let twilight = exp(-pow((pDotL + .01) / .11, 2.)) * pow(oneMinusV, 4.2);
    marsAtmo = vec3f(.88, .54, .34) * (fresnel * .15 + horizon * .28) * sunDay
      + vec3f(.38, .56, .78) * twilight * .07
      + vec3f(.10, .06, .04) * horizon * .08 * (1. - sunDay);
  }
  // Analytic subpixel limb antialiasing only for small apparent radii (< ~50 CSS px),
  // where 4x MSAA stair-steps the silhouette and limb relief is sub-pixel. Larger views
  // rely on 4x MSAA so tall 3D limbs like Olympus Mons are never clipped by mipmapped height.
  let toFrag = input.local - u.eye.xyz;
  let impact = length(cross(u.eye.xyz, input.local)) / max(length(toFrag), 1e-4);
  let kmPerPx = max(length(vec2f(dpdx(impact), dpdy(impact))), 1e-4);
  let smallPlanetAa = smoothstep(.020, .040, minorAngle * u.terrain.w);
  let rSurf = u.terrain.x * .001;
  let limbAlpha = mix(1., smoothstep(-.55, .85, (rSurf - impact) / kmPerPx), smallPlanetAa);
  if (limbAlpha <= .002) { discard; }
  // Subtle linear night-side floor when focused so dark Martian basalt and lunar maria
  // clear the ACES shadow toe while preserving strong day/night phase contrast.
  let nightFillMask = (1. - daylight) * inspect;
  let nightFloor = select(
    vec3f(.0040, .0048, .0062),
    vec3f(.0068, .0044, .0032),
    isMars,
  ) * (.65 + .35 * fill) * nightFillMask;
  if (u.sun.w > .5) { color = toLinear(color) + nightFloor + marsAtmo; }
  else { color = color + pow(max(nightFloor + marsAtmo, vec3f(0.)), vec3f(1. / 2.2)); }
  return vec4f(color, limbAlpha);
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
}

type DevicePlanetCache = {
  controller: AbortController
  pipelines?: Promise<PlanetPipelines>
  indices?: GPUBuffer
  albedoSampler?: GPUSampler
  fallbackAlbedo?: GPUTexture
  planetTiles?: PlanetTileCache
  bodies: WeakMap<Terrain, Promise<SharedBodyTerrain>>
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
      entry!.planetTiles?.destroy()
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
          fragment: {
            module,
            entryPoint: 'fragmentMain',
            targets: [{
              format: 'rgba16float',
              blend: {
                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              },
            }],
          },
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
    ]).then(([surface, annotation]) => ({ surface, annotation }))
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

async function buildBodyAlbedo(device: GPUDevice, cache: DevicePlanetCache, body: TerrainBody, url: string): Promise<GPUTexture> {
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
    await copyExternalImageChunked(device.queue, bitmap, texture, signal)
    generateMipmaps(device, texture, { addressModeU: 'repeat', label: `Build ${body} albedo mipmaps` })
    cache.createdAlbedos.add(texture)
    return texture
  } catch (error) {
    texture?.destroy()
    throw error
  } finally {
    bitmap.close()
  }
}

function loadSharedBodyAlbedo(device: GPUDevice, body: TerrainBody, url: string, signal: AbortSignal): Promise<GPUTexture> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  let pending = cache.albedos.get(url)
  if (!pending) {
    pending = buildBodyAlbedo(device, cache, body, url)
    cache.albedos.set(url, pending)
    void pending.catch(() => { if (cache.albedos.get(url) === pending) cache.albedos.delete(url) })
  }
  return raceAbort(pending, signal)
}

async function getSharedBodyTerrain(device: GPUDevice, body: TerrainBody, prepared: PreparedTerrain<'rg8unorm'>, signal: AbortSignal): Promise<SharedBodyTerrain> {
  signal.throwIfAborted()
  const cache = getPlanetCache(device)
  let pending = cache.bodies.get(prepared.terrain)
  if (!pending) {
    pending = (async () => {
      const { terrain } = prepared
      const levels = prepared.levels ?? packRg8Levels(elevationPyramid(terrain, true))
      const mesh = buildPlanetTerrain(body, terrain, false)
      const heights = device.createTexture({
        label: `${body} numeric terrain grid`,
        size: [terrain.width, terrain.height], mipLevelCount: levels.length,
        format: 'rg8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      try {
        for (const [mipLevel, { data, width, height }] of levels.entries()) {
          await writeTextureChunkedAsync(device.queue, heights, mipLevel, data, width, height, 2, cache.controller.signal)
        }
      } catch (error) {
        heights.destroy()
        throw error
      }
      const vertices = uploadBuffer(device, `${body} displaced terrain vertices`, mesh.positions, GPUBufferUsage.VERTEX)
      const indices = (cache.indices ??= uploadBuffer(device, 'Planetary displaced terrain indices', mesh.indices, GPUBufferUsage.INDEX))
      const sampler = device.createSampler({
        addressModeU: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      })
      const shared: SharedBodyTerrain = { mesh, heights, vertices, indices, sampler }
      cache.created.push(shared)
      // Mip levels and raw CPU elevation bytes are no longer needed once resident on the GPU.
      // Keeping the terrain object itself as the WeakMap key avoids re-uploading on the same device.
      prepared.levels = null
      if (prepared.terrain.pixels.byteLength) prepared.terrain.pixels = new Uint8ClampedArray(0)
      return shared
    })()
    cache.bodies.set(prepared.terrain, pending)
    void pending.catch(() => { if (cache.bodies.get(prepared.terrain) === pending) cache.bodies.delete(prepared.terrain) })
  }
  return raceAbort(pending, signal)
}

export class PlanetTerrainRenderer {
  private readonly bodies = new Map<TerrainBody, GpuTerrain>()
  private readonly device: GPUDevice
  private readonly pipeline: GPURenderPipeline
  private readonly annotationPipeline: GPURenderPipeline
  private readonly tiles: PlanetTileCache
  private readonly unsubscribeTiles: () => void
  private readonly annotationScratch = new Float32Array(20)

  private constructor(device: GPUDevice, pipelines: PlanetPipelines, tiles: PlanetTileCache, onTileUpdate?: () => void) {
    this.device = device
    this.pipeline = pipelines.surface
    this.annotationPipeline = pipelines.annotation
    this.tiles = tiles
    this.unsubscribeTiles = tiles.subscribe(
      () => onTileUpdate?.(),
      () => { this.rebindAll(); onTileUpdate?.() },
    )
  }

  static async create(
    device: GPUDevice,
    signal: AbortSignal,
    assetUrl: WorkspaceAssets['url'],
    loadTerrain?: TerrainLoader,
    onTileUpdate?: () => void,
    loadPlanetTilePack?: PlanetTilePackLoader,
  ) {
    const bodies: TerrainBody[] = ['moon', 'mars']
    const pipelines = await getPlanetPipelines(device)
    signal.throwIfAborted()
    const cache = getPlanetCache(device)
    const tiles = (cache.planetTiles ??= new PlanetTileCache(device))
    if (loadPlanetTilePack) tiles.attachPacks(loadPlanetTilePack)
    const renderer = new PlanetTerrainRenderer(device, pipelines, tiles, onTileUpdate)
    try {
      // Load Moon then Mars sequentially so we never overlap two 8K terrain workers
      // or two 8K RGBA ImageBitmaps (~128 MiB each) in RAM at the same instant.
      for (const body of bodies) {
        const terrainPath = `textures/${body}-terrain.bin`
        let prepared = loadTerrain
          ? await loadTerrain(terrainPath, 'rg8unorm', s => assetUrl(terrainPath, s), signal)
          : await loadPreparedTerrain(await assetUrl(terrainPath, signal), 'rg8unorm', signal)
        if (!prepared.terrain.pixels.byteLength && !cache.bodies.has(prepared.terrain)) {
          // Device recovered after an earlier upload freed CPU pixels; re-decode once.
          prepared = await loadPreparedTerrain(await assetUrl(terrainPath, signal), 'rg8unorm', signal)
        }
        signal.throwIfAborted()
        let albedo: GPUTexture
        try {
          const url = await assetUrl(`textures/${body}-albedo.webp`, signal)
          albedo = await loadSharedBodyAlbedo(device, body, url, signal)
        } catch {
          if (signal.aborted) throw signal.reason
          albedo = getFallbackAlbedo(device)
        }
        signal.throwIfAborted()
        const shared = await getSharedBodyTerrain(device, body, prepared, signal)
        signal.throwIfAborted()
        renderer.attach(body, shared, albedo)
      }
      return renderer
    } catch (error) {
      renderer.destroy()
      throw error
    }
  }

  private createBodyBindGroup(body: TerrainBody, uniform: GPUBuffer, shared: SharedBodyTerrain, albedo: GPUTexture, albedoSampler: GPUSampler) {
    return this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: shared.heights.createView() },
        { binding: 2, resource: shared.sampler },
        { binding: 3, resource: albedo.createView() },
        { binding: 4, resource: albedoSampler },
        { binding: 5, resource: this.tiles.pageTexture(body).createView() },
        { binding: 6, resource: this.tiles.poolTexture.createView({ dimension: '2d-array' }) },
        { binding: 7, resource: this.tiles.sampler },
      ],
    })
  }

  private rebindAll() {
    for (const [body, data] of this.bodies) {
      data.bindGroup = this.createBodyBindGroup(body, data.uniform, data.shared, data.albedo, data.albedoSampler)
      data.loupeBindGroup = this.createBodyBindGroup(body, data.loupeUniform, data.shared, data.albedo, data.albedoSampler)
    }
  }

  private attach(body: TerrainBody, shared: SharedBodyTerrain, albedo: GPUTexture) {
    const device = this.device
    const cache = getPlanetCache(device)
    const albedoSampler = (cache.albedoSampler ??= device.createSampler({
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    }))
    const parameters = new Float32Array(40)
    const { relief, interval, low, high } = settings[body]
    parameters.set([shared.mesh.radius * 1000, relief, interval, 1], 20)
    parameters.set(low, 24)
    parameters.set(high, 28)
    parameters[39] = body === 'mars' ? 1 : 0
    const uniform = device.createBuffer({
      label: `${body} terrain parameters`,
      size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bindGroup = this.createBodyBindGroup(body, uniform, shared, albedo, albedoSampler)
    const loupeParameters = new Float32Array(parameters)
    const loupeUniform = device.createBuffer({
      label: `${body} loupe terrain parameters`,
      size: loupeParameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const loupeBindGroup = this.createBodyBindGroup(body, loupeUniform, shared, albedo, albedoSampler)
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

  updateTiles(
    body: TerrainBody,
    eyeUnit: readonly [number, number, number],
    right: readonly [number, number, number],
    up: readonly [number, number, number],
    back: readonly [number, number, number],
    aspect: number,
    physicalMetresPerPixel: number,
    loupe?: { lon: number; lat: number; zoom: number } | null,
  ) {
    const data = this.bodies.get(body)
    if (!data) return
    this.tiles.updateView(body, eyeUnit, right, up, back, aspect, data.shared.mesh.radius * 1000, physicalMetresPerPixel, loupe)
  }

  tileStatus() {
    const parts: string[] = []
    for (const body of ['moon', 'mars'] as const) {
      if (this.tiles.hasIndex(body)) {
        parts.push(`${body}:${this.tiles.residentCount(body)}/${this.tiles.indexedCount(body)}`)
      }
    }
    return parts.join(' ')
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
    eye: readonly [number, number, number], metresPerPixel: number, hdr: boolean, pixelRatio: number,
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
    parameters[32] = Number(this.tiles.ready(body))
    parameters.set(eye, 36)
    parameters[39] = body === 'mars' ? 1 : 0
    this.device.queue.writeBuffer(uniform, 0, parameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  drawLoupe(
    pass: GPURenderPassEncoder, body: TerrainBody, matrix: Float32Array, sunlight: readonly [number, number, number],
    eye: readonly [number, number, number], metresPerPixel: number, hdr: boolean, pixelRatio: number,
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
    loupeParameters[32] = Number(this.tiles.ready(body))
    loupeParameters.set(eye, 36)
    loupeParameters[39] = body === 'mars' ? 1 : 0
    this.device.queue.writeBuffer(loupeUniform, 0, loupeParameters)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, loupeBindGroup)
    pass.setVertexBuffer(0, vertices)
    pass.setIndexBuffer(indices, 'uint32')
    pass.drawIndexed(mesh.indices.length)
  }

  destroy() {
    this.unsubscribeTiles()
    for (const { uniform, loupeUniform, annotationUniform } of this.bodies.values()) {
      uniform.destroy()
      loupeUniform.destroy()
      annotationUniform.destroy()
    }
    this.bodies.clear()
  }
}
