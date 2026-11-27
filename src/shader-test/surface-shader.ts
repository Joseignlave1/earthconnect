import { joinWgsl } from './gpu.ts'
import colorWgsl from './shaders/color.wgsl?raw'
import contoursWgsl from './shaders/contours.wgsl?raw'
import equalEarthWgsl from './shaders/equal-earth.wgsl?raw'
import magmaWgsl from './shaders/magma.wgsl?raw'
import quadWgsl from './shaders/quad.wgsl?raw'
import scoreStyleWgsl from './shaders/score-style.wgsl?raw'
import surfaceTerrainFallbackWgsl from './shaders/surface-terrain-fallback.wgsl?raw'
import surfaceTerrainFilterableWgsl from './shaders/surface-terrain-filterable.wgsl?raw'
import surfaceWgsl from './shaders/surface.wgsl?raw'
import terrainCommonWgsl from './shaders/terrain-common.wgsl?raw'

// Only the score bank texture bindings depend on the adapter's maxTextureArrayLayers limit;
// all scalar constants (TILE_SIZE, BANK_SIZE, HIGHLIGHT_START, HIGHLIGHT_FULL) use WGSL overrides.
function scoreBankBindings(banks: number) {
  const declarations = [
    ...Array.from({ length: banks }, (_, i) => `@group(0) @binding(${6 + i}) var scores${i}: texture_2d_array<u32>;`),
    `@group(0) @binding(${6 + banks}) var earthTilePage: texture_2d<u32>;`,
    `@group(0) @binding(${7 + banks}) var earthTilePool: texture_2d_array<f32>;`,
    `@group(0) @binding(${8 + banks}) var earthTileSampler: sampler;`,
  ].join('\n')

  const singleBranches = Array.from(
    { length: banks },
    (_, i) => `${i ? '  else ' : '  '}if (bank == ${i}u) { return textureLoad(scores${i}, pixel, layer, lod).r; }`,
  ).join('\n')

  const quadBranches = Array.from(
    { length: banks },
    (_, i) =>
      `${i ? '  else ' : '  '}if (bank == ${i}u) {\n` +
      `    return vec4u(\n` +
      `      textureLoad(scores${i}, local, layer, lod).r,\n` +
      `      textureLoad(scores${i}, local + vec2i(1, 0), layer, lod).r,\n` +
      `      textureLoad(scores${i}, local + vec2i(0, 1), layer, lod).r,\n` +
      `      textureLoad(scores${i}, local + vec2i(1, 1), layer, lod).r,\n` +
      `    );\n` +
      `  }`,
  ).join('\n')

  return `${declarations}
fn loadScoreBank(bank: u32, pixel: vec2i, layer: i32, lod: i32) -> u32 {
${singleBranches}
  return 65535u;
}
fn loadScoreQuadBank(bank: u32, local: vec2i, layer: i32, lod: i32) -> vec4u {
${quadBranches}
  return vec4u(65535u);
}`
}

export function surfaceShader(banks: number, filterableTerrain: boolean) {
  return joinWgsl(
    'diagnostic(off, derivative_uniformity);',
    colorWgsl,
    magmaWgsl,
    scoreStyleWgsl,
    equalEarthWgsl,
    contoursWgsl,
    quadWgsl,
    terrainCommonWgsl,
    filterableTerrain ? surfaceTerrainFilterableWgsl : surfaceTerrainFallbackWgsl,
    scoreBankBindings(banks),
    surfaceWgsl,
  )
}
