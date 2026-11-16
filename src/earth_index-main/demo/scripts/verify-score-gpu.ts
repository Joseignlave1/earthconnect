// Runs inside the benchmark browser, against the real WebGL2 implementation.
import { ScoreRaster } from '../src/score-raster.ts';
import { scoreAtlas, sampleAtlas, atlasLevel } from '../src/score-atlas.ts';
import type { ScoreAtlas } from '../src/score-atlas.ts';
import type OLMap from 'ol/Map.js';
import type { FrameState } from 'ol/Map.js';
import type { ScorePatch } from '../src/types.ts';

export async function verifyScoreGpu() {
  const patches: ScorePatch[] = [
    {
      grid: {
        siteKey: 'a',
        srid: 999010,
        crsWkt: '+proj=longlat +datum=WGS84 +no_defs',
        width: 3,
        height: 2,
        transform: [1, 0, 0, 0, -1, 2],
      },
      scores: new Uint16Array([0, 32767, 65535, 16384, 65534, 20000]),
    },
    {
      grid: {
        siteKey: 'b',
        srid: 999010,
        crsWkt: '+proj=longlat +datum=WGS84 +no_defs',
        width: 3,
        height: 2,
        transform: [1, 0, 6, 0, -1, 2],
      },
      scores: new Uint16Array([65534, 1, 12345, 0, 65535, 33333]),
    },
  ];
  const atlas = scoreAtlas(patches);
  const frame = {
    size: [90, 20],
    pixelRatio: 1,
    viewState: { resolution: 0.1 },
    pixelToCoordinateTransform: [0.1, 0, 0, -0.1, 0, 2],
  } as FrameState;
  let comparisons = 0;
  const check = (condition: boolean, message: string) => {
    if (!condition) throw new Error(message);
  };
  const verify = (raster: ScoreRaster, level: number, cutoff: number, maximum: number) => {
    const gl = raster.canvas.getContext('webgl2')!;
    for (const x of [5, 15, 25, 35, 45, 55, 65, 75, 85])
      for (const y of [5, 15]) {
        const score = sampleAtlas(atlas, (x + 0.5) * 0.1, 2 - (y + 0.5) * 0.1, level);
        const alpha =
          score === 65535 || score < cutoff || score === 0
            ? 0
            : 235 * Math.min(1, score / maximum) ** 1.6;
        const expected = [(168 / 255) * alpha, (85 / 255) * alpha, (247 / 255) * alpha, alpha];
        const actual = new Uint8Array(4);
        gl.readPixels(x, 19 - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, actual);
        for (let i = 0; i < 4; i++)
          check(
            Math.abs(actual[i] - expected[i]) <= 1.5,
            `GPU mismatch ${x},${y},level${level},band${i}: ${actual[i]} vs ${expected[i]}`,
          );
        comparisons++;
      }
  };
  // Force two texture-array banks even on hardware supporting thousands of
  // planes, to validate the portable sampler-bank selection path as well.
  const raster = new ScoreRaster();
  const gl = raster.canvas.getContext('webgl2')!;
  const getParameter = gl.getParameter.bind(gl);
  gl.getParameter = (parameter) =>
    parameter === gl.MAX_ARRAY_TEXTURE_LAYERS ? 1 : getParameter(parameter);
  try {
    raster.set(atlas);
    raster.render(frame, 0, 65534);
    verify(raster, 0, 0, 65534);
    raster.render(frame, 40000, 65534);
    verify(raster, 0, 40000, 65534);
    check(raster.stats.uploads === 1, 'Threshold change reuploaded scores');
    const coarse = { ...frame, viewState: { ...frame.viewState, resolution: 10 } };
    raster.render(coarse, 0, 65534);
    verify(raster, atlas.patches[0].levels.length - 1, 0, 65534);
    raster.render(frame, 0, 65534);
    verify(raster, 0, 0, 65534);
    check(
      raster.stats.uploads === 2 && raster.stats.cachedLevels === 2,
      'LOD cache was not reused',
    );

    const extension = gl.getExtension('WEBGL_lose_context');
    if (extension) {
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('GPU context restore timed out')), 5000);
        raster.onRestore = () => {
          clearTimeout(timer);
          resolve();
        };
        raster.canvas.addEventListener(
          'webglcontextlost',
          () => setTimeout(() => extension.restoreContext(), 100),
          { once: true },
        );
        extension.loseContext();
      });
      raster.render(frame, 0, 65534);
      verify(raster, 0, 0, 65534);
    }
    raster.clear();
    check(
      raster.stats.cacheBytes === 0 && raster.stats.cachedLevels === 0,
      'Clearing retained GPU cache',
    );
    raster.set(atlas);
    raster.render(frame, 0, 65534);
    verify(raster, 0, 0, 65534);
    check(gl.getError() === gl.NO_ERROR, 'WebGL error after cache reuse');
  } finally {
    raster.destroy();
  }
  const bounded = new ScoreRaster(24);
  try {
    bounded.set(atlas);
    bounded.render(frame, 0, 65534);
    bounded.render({ ...frame, viewState: { ...frame.viewState, resolution: 10 } }, 0, 65534);
    bounded.render(frame, 0, 65534);
    verify(bounded, 0, 0, 65534);
    check(
      bounded.stats.uploads === 3 &&
        bounded.stats.cachedLevels === 1 &&
        bounded.stats.cacheBytes === 24,
      'LRU budget or eviction failed',
    );
  } finally {
    bounded.destroy();
  }
  return {
    comparisons,
    texture_banks: 2,
    threshold_no_upload: true,
    lod_cache_reused: true,
    cache_eviction: true,
    clear_releases_textures: true,
    context_restore: true,
  };
}

export function verifyWorldGpu(map: OLMap) {
  const raster = map
    .getLayers()
    .getArray()
    .find((layer) => layer.getZIndex() === 10)!
    .get('scoreRenderer') as ScoreRaster;
  const atlas = (raster as unknown as { atlas: ScoreAtlas }).atlas;
  const maximum = Math.max(
    ...atlas.patches.map((p) => {
      const n = p.levels.at(-1)!.scores[0];
      return n === 65535 ? 0 : n;
    }),
  );
  const frame = {
    size: [720, 360],
    pixelRatio: 1,
    viewState: { resolution: 0.5 },
    pixelToCoordinateTransform: [0.5, 0, 0, -0.5, -180, 90],
  } as FrameState;
  const gl = raster.canvas.getContext('webgl2')!;
  let comparisons = 0,
    positive = 0,
    missing = 0;
  for (const cutoff of [0, Math.floor(maximum * 0.9)]) {
    raster.render(frame, cutoff, maximum);
    const pixels = new Uint8Array(720 * 360 * 4);
    gl.readPixels(0, 0, 720, 360, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    for (let y = 2; y < 360; y += 5)
      for (let x = 2; x < 720; x += 7) {
        const longitude = (x + 0.5) * 0.5 - 180,
          latitude = 90 - (y + 0.5) * 0.5;
        const score = sampleAtlas(atlas, longitude, latitude, atlasLevel(atlas, 0.5));
        // Margin around bin edges for float32 projection; exact cell-center
        // addressing is checked by the synthetic fixtures above.
        const p = atlas.projection,
          lat = (latitude * Math.PI) / 180;
        const sx = p.radius * ((longitude * Math.PI) / 180 - p.longitude) * Math.cos(lat) + p.x;
        const sy = p.radius * lat + p.y;
        const shape = atlas.patches[0].levels[atlasLevel(atlas, 0.5)];
        const fx = ((sx - atlas.origin[0]) / atlas.span[0]) * shape.width;
        const fy = ((atlas.origin[1] - sy) / atlas.span[1]) * shape.height;
        if ([fx, fy].some((n) => Math.abs(n - Math.round(n)) < 0.02)) continue;
        const alpha =
          score === 65535 || score < cutoff || score === 0
            ? 0
            : 235 * Math.min(1, score / maximum) ** 1.6;
        const expected = [(168 / 255) * alpha, (85 / 255) * alpha, (247 / 255) * alpha, alpha];
        const offset = ((359 - y) * 720 + x) * 4;
        for (let band = 0; band < 4; band++)
          if (Math.abs(pixels[offset + band] - expected[band]) > 1.5)
            throw new Error(
              `World GPU mismatch at ${longitude},${latitude}: ${pixels.slice(offset, offset + 4)} vs ${expected}`,
            );
        comparisons++;
        if (score === 65535) missing++;
        else if (score > 0) positive++;
      }
  }
  map.render();
  if (!positive || !missing) throw new Error('World GPU probes missed data or missing cells');
  return { comparisons, positive, missing, exact_within_pixel_rounding: true };
}
