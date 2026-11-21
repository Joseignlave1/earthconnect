struct Scene {
  eye: vec4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  viewport: vec4f,
  dem: vec4f,
  extent: vec4f,
  detail: vec4f,
  region: vec4f,
  regionExtent: vec4f,
  earth: vec4f,
  station: vec4f,
  sun: vec4f,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var ortho: texture_2d<f32>;
@group(0) @binding(2) var terrainSampler: sampler;
@group(0) @binding(3) var regionalElevation: texture_2d<f32>;
@group(0) @binding(4) var ctxImage: texture_2d<f32>;
@group(0) @binding(5) var elevation: texture_2d<f32>;
@group(0) @binding(6) var orthoDetail: texture_2d<f32>;
@group(0) @binding(7) var terrainLight: texture_2d<f32>;
@group(0) @binding(8) var slabTexture: texture_2d<f32>;
@group(0) @binding(9) var screeTexture: texture_2d<f32>;
@group(0) @binding(10) var rockSampler: sampler;
@group(0) @binding(11) var earthImage: texture_2d<f32>;
@group(0) @binding(12) var earthSampler: sampler;

fn project(world: vec3f) -> vec4f {
  let p = world - scene.eye.xyz;
  let z = dot(p, scene.forward.xyz);
  return vec4f(dot(p, scene.right.xyz) / scene.viewport.x,
    dot(p, scene.up.xyz) / scene.viewport.y, .004, z);
}

fn film(color: vec3f) -> vec3f {
  let v = max(color, vec3f(0.));
  return pow(v / (1. + v * .78), vec3f(.4545));
}

fn hash(p: vec2f) -> f32 {
  let a = fract(vec3f(p.xyx) * .1031);
  let b = a + dot(a, a.yzx + 33.33);
  return fract((b.x + b.y) * b.z);
}

fn noise(p: vec2f) -> f32 {
  let cell = floor(p);
  let f = fract(p);
  let t = f * f * (3. - 2. * f);
  return mix(mix(hash(cell), hash(cell + vec2f(1., 0.)), t.x),
    mix(hash(cell + vec2f(0., 1.)), hash(cell + 1.), t.x), t.y);
}

fn sunlight(ray: vec3f) -> f32 {
  return max(0., dot(ray, normalize(scene.sun.xyz)));
}

fn horizonColor(ray: vec3f) -> vec3f {
  let towardSun = pow(sunlight(ray), 5.);
  return mix(vec3f(.14, .071, .084), vec3f(.81, .262, .119), towardSun);
}

fn atmosphere(ray: vec3f) -> vec3f {
  let horizonDip = sqrt(2. * max(0., scene.eye.y) / 3393.83);
  let aboveHorizon = ray.y + horizonDip;
  let high = max(aboveHorizon, 0.);
  let upper = vec3f(.0029, .0026, .0060);
  let warmBand = exp(-abs(aboveHorizon) * 29.);
  let afterglow = exp(-high * 5.9) * (1. - warmBand);
  // The blue sunset glow is confined to the solar direction.
  let blue = vec3f(.028, .058, .095) * pow(sunlight(ray), 52.);
  let veil = noise(vec2f(atan2(ray.x, -ray.z) * 6., ray.y * 44.));
  let glow = afterglow * vec3f(.071, .025, .043) * (.94 + veil * .12);
  let sunset = pow(sunlight(ray), 8.) * warmBand * vec3f(1.48, .47, .135);
  return mix(upper + glow, horizonColor(ray), warmBand) + blue + sunset;
}

struct Screen {
  @builtin(position) position: vec4f,
  @location(0) ndc: vec2f,
}

@vertex fn skyVertex(@builtin(vertex_index) vertex: u32) -> Screen {
  let corners = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  let p = corners[vertex];
  return Screen(vec4f(p, 0., 1.), p);
}

@fragment fn skyFragment(v: Screen) -> @location(0) vec4f {
  let ray = normalize(scene.forward.xyz + scene.right.xyz * v.ndc.x * scene.viewport.x
    + scene.up.xyz * v.ndc.y * scene.viewport.y);
  var color = atmosphere(ray);
  let starPos = vec2f(atan2(ray.z, ray.x), asin(ray.y)) * 570.;
  let cell = floor(starPos);
  let grain = hash(cell);
  let point = vec2f(hash(cell + 14.31), hash(cell - 8.91)) * .78 + .11;
  let offset = fract(starPos) - point;
  let stars = smoothstep(.997, .9995, grain) * exp(-dot(offset, offset) * 45.);
  color += stars * smoothstep(.19, .55, ray.y) * vec3f(.09, .084, .102);

  let direction = normalize(scene.earth.xyz);
  let east = normalize(cross(direction, scene.up.xyz));
  let north = cross(east, direction);
  let disk = vec2f(dot(ray, east), dot(ray, north)) / scene.earth.w;
  let radius = length(disk);
  color += vec3f(.012, .026, .075) * exp(-max(0., radius - .8) * 4.8);
  if (radius < 1.02 && dot(ray, direction) > 0.) {
    let nz = sqrt(max(0., 1. - dot(disk, disk)));
    let normal = vec3f(disk, nz);
    let uv = vec2f(.26 + atan2(disk.x, nz) / 6.283185, .5 - asin(clamp(disk.y, -1., 1.)) / 3.141593);
    let albedo = pow(textureSampleLevel(earthImage, earthSampler, uv, 0.).rgb, vec3f(2.2));
    let light = max(0., dot(normal, normalize(vec3f(-.87, .28, .45))));
    let rim = pow(1. - nz, 3.) * (.12 + light * 1.4);
    let surface = albedo * (.029 + light * 1.6) + rim * vec3f(.073, .19, .45);
    color = mix(color, surface, 1. - smoothstep(.96, 1.02, radius));
  }
  let dither = (hash(floor(v.position.xy)) - .5) / 255.;
  return vec4f(film(color) + dither, 1.);
}

fn uvAt(p: vec2f) -> vec2f {
  return (p - scene.dem.xy) / scene.extent.xy;
}

fn hiriseWeight(p: vec2f) -> f32 {
  let distances = min(p - scene.dem.xy, scene.dem.xy + scene.extent.xy - p);
  return smoothstep(0., .62, min(distances.x, distances.y));
}

fn sampleElevation(image: texture_2d<f32>, uv: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(image));
  return textureSampleLevel(image, terrainSampler, (uv * (size - 1.) + .5) / size, 0.).rgb;
}

fn regionalHeight(p: vec2f) -> f32 {
  let uv = (p - scene.region.xy) / scene.regionExtent.xy;
  let packed = sampleElevation(regionalElevation, uv);
  return scene.region.w + dot(packed.rg, vec2f(.255, 65.28));
}

fn regionalDisplacement(p: vec2f) -> f32 {
  let strike = vec2f(.51, -.86);
  let fallLine = vec2f(.86, .51);
  let slope = abs(regionalHeight(p + fallLine * .4) - regionalHeight(p - fallLine * .4));
  let along = dot(p, strike);
  let across = dot(p, fallLine);
  let bend = noise(vec2f(across * .29, along * .43));
  let ribs = noise(vec2f(across * .73, along * 1.28 + bend * .37));
  let slopeWeight = smoothstep(.075, .36, slope);
  return (ribs - .5) * .13 * slopeWeight;
}

fn heightAt(p: vec2f) -> f32 {
  let grains = noise(p * vec2f(7.9, 3.3)) - .5;
  let fissures = noise(p * vec2f(1.9, 5.3) + grains * .4) - .5;
  let rough = regionalHeight(p) + regionalDisplacement(p) + grains * .009 + fissures * .031;
  let strength = hiriseWeight(p);
  if (strength == 0.) { return rough; }
  let packed = sampleElevation(elevation, uvAt(p));
  let fine = scene.dem.w + dot(packed.rg, vec2f(.031875, 8.16));
  return mix(rough, fine, strength);
}

fn curvature(p: vec2f) -> f32 {
  return dot(p, p) / (2. * 3393.83);
}

struct Ground {
  @builtin(position) position: vec4f,
  @location(0) point: vec3f,
}

@vertex fn groundVertex(@location(0) xz: vec2f) -> Ground {
  let p = vec3f(xz.x, heightAt(xz) - curvature(xz), xz.y);
  return Ground(project(p), p);
}

fn haze(p: vec3f) -> f32 {
  // Integrate an exponential dust layer between the camera and the surface.
  let d = distance(p, scene.eye.xyz);
  let lo = min(p.y, scene.eye.y);
  let heightDifference = abs(p.y - scene.eye.y) * .56;
  let mean = exp(-max(0., lo + 1.5) * .56)
    * (1. - exp(-max(heightDifference, .001))) / max(heightDifference, .001);
  let wind = vec2f(scene.viewport.z * .020, scene.viewport.z * .006);
  let hazePosition = mix(scene.eye.xz, p.xz, .48);
  let drift = noise(hazePosition * vec2f(.19, .82) - wind);
  let density = mean * (.037 + drift * .047) + .00185;
  return 1. - exp(-d * density);
}

fn litThroughDust(color: vec3f, p: vec3f) -> vec3f {
  let ray = normalize(p - scene.eye.xyz);
  let hazed = mix(color, horizonColor(ray) * .73, haze(p));
  let boundary = min(p.xz - scene.region.xy, scene.region.xy + scene.regionExtent.xy - p.xz);
  let farEdge = 1. - smoothstep(0., 48., min(boundary.x, boundary.y));
  return mix(hazed, atmosphere(ray), farEdge);
}

fn mineralNormal(n: vec3f, packed: vec2f, influence: f32) -> vec3f {
  let tangent = normalize(vec3f(n.y, -n.x, 0.));
  let across = cross(tangent, n);
  return normalize(n + influence * (packed.x * tangent + packed.y * across));
}

// Fine grooves follow the slope of the escarpment at several scales.
fn erosion(p: vec2f) -> f32 {
  let downSlope = dot(p, vec2f(.86, .51));
  let alongCliff = dot(p, vec2f(.51, -.86));
  let warp = noise(vec2f(downSlope * .53, alongCliff * 1.24));
  let acrossGully = alongCliff + (warp - .5) * .36;
  let channels = noise(vec2f(downSlope * 1.05, acrossGully * 4.8));
  let cuts = pow(1. - abs(2. * channels - 1.), 2.8);
  let ledges = noise(vec2f(downSlope * 3.8, acrossGully * 3.4));
  return cuts * .70 + ledges * .30;
}

@fragment fn groundFragment(v: Ground) -> @location(0) vec4f {
  let p = v.point;
  let xz = p.xz;
  let regionDx = vec2f(.37, 0.);
  let regionDz = regionDx.yx;
  let gradient = vec2f(regionalHeight(xz + regionDx) - regionalHeight(xz - regionDx),
    regionalHeight(xz + regionDz) - regionalHeight(xz - regionDz));
  let regionalNormal = normalize(vec3f(-gradient.x, .74, -gradient.y));
  let uv = uvAt(xz);
  let baked = textureSample(terrainLight, terrainSampler, uv);
  let nxz = baked.rg * 2. - 1.;
  let localNormal = normalize(vec3f(nxz.x, sqrt(max(.006, 1. - dot(nxz, nxz))), nxz.y));
  let local = hiriseWeight(xz);
  let normal = normalize(mix(regionalNormal, localNormal, local));
  let footprint = max(length(dpdx(xz)), length(dpdy(xz)));
  let cliff = smoothstep(.075, .44, length(normal.xz)) * (1. - smoothstep(.03, .21, footprint));
  let erosionStep = vec2f(max(.042, footprint * .85), 0.);
  let cuts = vec2f(erosion(xz + erosionStep) - erosion(xz - erosionStep),
    erosion(xz + erosionStep.yx) - erosion(xz - erosionStep.yx));
  let carved = normalize(normal + cliff * vec3f(cuts.x, 0., cuts.y) * .91);

  let overview = textureSample(ortho, terrainSampler, uv).r;
  let detailUV = (xz - scene.detail.xy) / scene.detail.zw;
  let edge = min(min(detailUV.x, 1. - detailUV.x), min(detailUV.y, 1. - detailUV.y));
  let closeTone = textureSample(orthoDetail, terrainSampler, detailUV).r;
  let ctxUV = (xz - vec2f(-65., -66.)) / vec2f(120., 105.);
  let ctxEdge = min(min(ctxUV.x, 1. - ctxUV.x), min(ctxUV.y, 1. - ctxUV.y));
  let ctxTone = textureSample(ctxImage, terrainSampler, ctxUV).r;
  let regionalTone = mix(.52, ctxTone, smoothstep(0., .13, ctxEdge));
  let tone = mix(regionalTone, mix(overview, closeTone, smoothstep(0., .11, edge)), local);

  // Extensive dust deposits cover flatter terrain, leaving darker exposed strata.
  let flow = noise(xz * vec2f(.26, .72)) * .65 + noise(xz * 1.45) * .35;
  let strataUV = vec2f(xz.y * 1.62 + xz.x * .55, (p.y + curvature(xz)) * 7.);
  let strata = noise(strataUV + flow * .92);
  let seams = noise(strataUV * vec2f(.53, 3.9) + flow * 1.47);
  let dust = smoothstep(.40, .93, normal.y * .49 + flow * .42 + strata * .17 + seams * .13);
  let slab = textureSample(slabTexture, rockSampler, xz * vec2f(2.9, 5.4) + flow * .50);
  let scree = textureSample(screeTexture, rockSampler, xz * 71.);
  let micro = mix(slab, scree, dust);
  let relief = (1. - smoothstep(.025, .116, footprint)) * .31;
  let n = mineralNormal(carved, micro.gb * 2. - 1., relief);

  let basalt = vec3f(.105, .083, .099);
  let sediment = vec3f(.39, .238, .152);
  let rock = mix(basalt, sediment, dust) * (.23 + tone * .97 + flow * .30 + strata * .08 + seams * .09)
    * mix(1., .33 + micro.r * 1.23, relief);
  let sun = normalize(scene.sun.xyz);
  let regionUV = (xz - scene.region.xy) / scene.regionExtent.xy;
  let visibility = sampleElevation(regionalElevation, regionUV).b;
  let direct = max(0., dot(n, sun)) * mix(visibility, baked.b, local);
  let warm = vec3f(3.3, 1.47, .74) * direct;
  let occlusion = 1. - erosion(xz) * cliff * .42;
  let bounce = vec3f(.123, .103, .165) * (.45 + .55 * n.y) * mix(1., baked.a, local) * occlusion;
  let color = rock * (warm + bounce);
  return vec4f(film(litThroughDust(color, p)), 1.);
}

struct Station {
  @builtin(position) position: vec4f,
  @location(0) point: vec3f,
  @location(1) @interpolate(flat) normal: vec3f,
  @location(2) @interpolate(flat) part: u32,
}

@vertex fn stationVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) part: u32) -> Station {
  // A single original field station. Dimensions are in kilometres.
  let dimensions = array<vec3f, 5>(vec3f(.015, .0021, .005), vec3f(.004, .0016, .003),
    vec3f(.00034, .006, .00034), vec3f(.0022, .00023, .001), vec3f(.0045, .00038, .0003));
  let centers = array<vec3f, 5>(vec3f(0., .0009, 0.), vec3f(.018, .0006, .003),
    vec3f(-.017, .0047, -.005), vec3f(-.017, .0104, -.005), vec3f(.004, .00135, .0051));
  let normals = array<vec3f, 6>(vec3f(1., 0., 0.), vec3f(-1., 0., 0.), vec3f(0., 1., 0.),
    vec3f(0., -1., 0.), vec3f(0., 0., 1.), vec3f(0., 0., -1.));
  let corners = array<vec2f, 6>(vec2f(-1., -1.), vec2f(1., -1.), vec2f(1., 1.),
    vec2f(-1., -1.), vec2f(1., 1.), vec2f(-1., 1.));
  let normal = normals[vertex / 6u];
  let corner = corners[vertex % 6u];
  let xz = scene.station.xz;
  let center = vec3f(xz.x, heightAt(xz) - curvature(xz), xz.y) + centers[part];
  let p = center + (abs(normal) * normal + corner.x * normal.zxy * normal.zxy
    + corner.y * normal.yzx * normal.yzx) * dimensions[part];
  return Station(project(p), p, normal, part);
}

@fragment fn stationFragment(v: Station) -> @location(0) vec4f {
  let paint = vec3f(.64, .57, .47);
  let light = paint * (.17 + .84 * max(0., dot(v.normal, normalize(scene.sun.xyz))));
  let color = select(light, vec3f(.21, .15, .077), v.part == 4u);
  return vec4f(film(litThroughDust(color, v.point)), 1.);
}

struct Signal {
  @builtin(position) position: vec4f,
  @location(0) ndc: vec2f,
  @location(1) @interpolate(flat) start: vec2f,
  @location(2) @interpolate(flat) end: vec2f,
}

@vertex fn signalVertex(@builtin(vertex_index) vertex: u32) -> Signal {
  let corners = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  let p = corners[vertex];
  let xz = scene.station.xz;
  let station = project(vec3f(xz.x, heightAt(xz) - curvature(xz) + .012, xz.y));
  let earth = scene.earth.xyz;
  let finish = vec2f(dot(earth, scene.right.xyz) / scene.viewport.x, dot(earth, scene.up.xyz) / scene.viewport.y)
    / dot(earth, scene.forward.xyz);
  return Signal(vec4f(p, 0., 1.), p, station.xy / station.w, finish);
}

@fragment fn signalFragment(v: Signal) -> @location(0) vec4f {
  let aspect = scene.viewport.x / scene.viewport.y;
  let dx = max(.01, v.end.x - v.start.x);
  let t = clamp((v.ndc.x - v.start.x) / dx, 0., 1.);
  let rise = 1. - pow(1. - t, 2.85);
  let point = vec2f(mix(v.start.x, v.end.x, t), mix(v.start.y, v.end.y, rise));
  let slope = 2.85 * pow(1. - t, 1.85) * (v.end.y - v.start.y) / (dx * aspect);
  let distance = length((v.ndc - point) * vec2f(aspect, 1.)) * scene.viewport.w * .5
    / sqrt(1. + slope * slope);
  let visible = step(v.start.x, v.ndc.x) * step(v.ndc.x, v.end.x)
    * smoothstep(.022, .061, length((v.ndc - v.end) * vec2f(aspect, 1.)));
  let line = exp(-distance * distance * 2.8) * visible * mix(.44, .16, t);
  let pulseTime = fract(t * 2. + .68 - scene.viewport.z * .16);
  let pulse = exp(-pow((pulseTime - .5) * 60., 2.));
  let highlight = exp(-distance * distance * .57) * pulse * visible * .92;
  let beacon = length((v.ndc - v.start) * vec2f(aspect, 1.)) * scene.viewport.w * .5;
  let flare = exp(-beacon * beacon * .17) * .76 + exp(-beacon * beacon * .014) * .26;
  let alpha = clamp(line + highlight + flare, 0., 1.);
  let tint = mix(vec3f(1., .63, .34), vec3f(1., .92, .74), clamp(highlight + flare, 0., 1.));
  return vec4f(tint * alpha, alpha);
}
