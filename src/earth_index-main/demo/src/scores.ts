import type OLMap from 'ol/Map.js';
import type { Coordinate } from 'ol/coordinate.js';
import type { ScoreGrid, ScorePatch, Area, DecodedValue, GridMetadata } from './types.ts';
import Layer from 'ol/layer/Layer.js';
import { transform, transformExtent } from 'ol/proj.js';
import { scoreAtlas } from './score-atlas.ts';
import type { AtlasPatch } from './score-atlas.ts';
import { ScoreRaster } from './score-raster.ts';
export { gridExtent } from './score-atlas.ts';
import VectorLayer from 'ol/layer/Vector.js';
import VectorSource from 'ol/source/Vector.js';
import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import { fromExtent } from 'ol/geom/Polygon.js';
import { Circle, Stroke, Style, Icon } from 'ol/style.js';

import { SCORE_PURPLE, scoreIntensity, strongestAreas, glowOpacity } from './score-visuals.ts';

export const MAX_SCORE = 65534;
export const MISSING_SCORE = 65535;

export function decodeScores(
  bytes: unknown,
  grid: Pick<ScoreGrid, 'siteKey' | 'width' | 'height'>,
) {
  if (!(bytes instanceof Uint8Array))
    throw new Error(`${grid.siteKey}: scores must be a packed uint16 BLOB`);
  const length = grid.width * grid.height;
  if (
    !Number.isSafeInteger(length) ||
    length <= 0 ||
    length > 5_000_000 ||
    bytes.byteLength !== length * 2
  )
    throw new Error(
      `${grid.siteKey}: expected ${length * 2} score bytes, received ${bytes.byteLength}. Return every cell in ORDER BY y,x.`,
    );
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    scores = new Uint16Array(length);
  for (let i = 0; i < length; i++) scores[i] = view.getUint16(i * 2, true);
  return scores;
}
export function scorePixels(scores: Uint16Array, cutoff = 0, maximum = MAX_SCORE) {
  const pixels = new Uint8ClampedArray(scores.length * 4);
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i];
    if (score === MISSING_SCORE || score < cutoff) continue;
    const alpha = Math.round(235 * scoreIntensity(score, maximum));
    pixels.set([...SCORE_PURPLE, alpha], i * 4);
  }
  return pixels;
}
export function scoreSummary(patches: ScorePatch[]) {
  let valid = 0,
    missing = 0,
    rejected = 0,
    best: { siteKey: string; index: number; score: number } | null = null;
  for (const patch of patches)
    for (let i = 0; i < patch.scores.length; i++) {
      const score = patch.scores[i];
      if (score === MISSING_SCORE) {
        missing++;
        continue;
      }
      valid++;
      if (!score) rejected++;
      if (score > 0 && (!best || score > best.score))
        best = { siteKey: patch.grid.siteKey, index: i, score };
    }
  return { valid, missing, rejected, best };
}
export function prepareScores(
  rows: Record<string, DecodedValue>[],
  datasource: GridMetadata,
): ScorePatch[] {
  const grids = new Map(datasource.grids.map((grid) => [grid.siteKey, grid]));
  const seen = new Set();
  return rows.map((row) => {
    const grid = grids.get(typeof row.site_key === 'string' ? row.site_key : '');
    if (!grid) throw new Error(`Unknown result grid: ${String(row.site_key)}`);
    if (seen.has(grid.siteKey)) throw new Error(`Duplicate result grid: ${grid.siteKey}`);
    seen.add(grid.siteKey);
    return { grid, scores: decodeScores(row.scores, grid) };
  });
}

export function earthScores(map: OLMap) {
  let patches: AtlasPatch[] = [];
  const raster = new ScoreRaster();
  const scoreLayer = new Layer({
    zIndex: 10,
    visible: false,
    render: (frame) => raster.render(frame, cutoff, maximum),
    properties: { scoreRenderer: raster },
  });
  raster.onRestore = () => scoreLayer.changed();
  map.addLayer(scoreLayer);
  let maximum = 0;
  const glows = new VectorSource<Feature<Point>>({ wrapX: false });
  const sprite = document.createElement('canvas');
  sprite.width = sprite.height = 64;
  const context = sprite.getContext('2d')!;
  const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, 'rgba(192,132,252,0.95)');
  gradient.addColorStop(0.3, 'rgba(192,132,252,0.9)');
  gradient.addColorStop(0.45, 'rgba(168,85,247,0.25)');
  gradient.addColorStop(1, 'rgba(168,85,247,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  const glowStyles = Array.from(
    { length: 33 },
    (_, i) => new Style({ image: new Icon({ img: sprite, scale: 0.125, opacity: i / 32 }) }),
  );
  const glowLayer = new VectorLayer({
    source: glows,
    zIndex: 25,
    style: (feature, resolution) => {
      if (feature.get('peak') < cutoff) return;
      const opacity = glowOpacity(resolution, feature.get('areaSpan'), feature.get('relative'));
      return opacity > 0 ? glowStyles[Math.round(opacity * 32)] : undefined;
    },
  });
  let cutoff = 0;
  map.addLayer(glowLayer);
  const selection = new VectorSource();
  const selectionLayer = new VectorLayer({
    source: selection,
    zIndex: 30,
    style: new Style({
      stroke: new Stroke({ color: 'rgba(255,255,255,0.8)', width: 1 }),
      image: new Circle({
        radius: 3,
        stroke: new Stroke({ color: 'rgba(255,255,255,0.8)', width: 1 }),
      }),
    }),
  });
  map.addLayer(selectionLayer);
  const clear = () => {
    scoreLayer.setVisible(false);
    raster.clear();
    patches = [];
    selection.clear();
    glows.clear();
    maximum = 0;
    map.getTargetElement()!.dataset.scorePatches = '0';
  };
  return {
    clear,
    destroy() {
      clear();
      raster.destroy();
      map.removeLayer(scoreLayer);
      map.removeLayer(glowLayer);
      map.removeLayer(selectionLayer);
    },
    show(next: ScorePatch[], nextCutoff = 0) {
      const summary = scoreSummary(next);
      const ceiling = summary.best?.score ?? 0;
      if (!next.length) {
        clear();
        return summary;
      }
      // Validate and pool before replacing the active GPU scene.
      const atlas = scoreAtlas(next);
      clear();
      raster.set(atlas);
      patches = atlas.patches;
      maximum = ceiling;
      cutoff = nextCutoff;
      scoreLayer.setVisible(ceiling > 0);
      scoreLayer.changed();
      map.getTargetElement()!.dataset.scorePatches = String(patches.length);
      const bySite = new Map(patches.map((patch) => [patch.grid.siteKey, patch]));
      for (const area of strongestAreas(next, maximum)) {
        const patch = bySite.get(area.siteKey)!;
        const [a, , c, , e, f] = patch.grid.transform;
        const x = c + a * ((area.peakIndex % patch.grid.width) + 0.5);
        const y = f + e * (Math.floor(area.peakIndex / patch.grid.width) + 0.5);
        const center = transform([x, y], patch.code, 'EPSG:4326');
        const bounds = transformExtent(
          [c + a * area.minX, f + e * (area.maxY + 1), c + a * (area.maxX + 1), f + e * area.minY],
          patch.code,
          'EPSG:4326',
          16,
        );
        const areaSpan = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
        const feature = new Feature(new Point(center));
        feature.setProperties({ peak: area.peak, relative: area.peak / maximum, areaSpan });
        glows.addFeature(feature);
      }
      return summary;
    },
    threshold(nextCutoff: number) {
      cutoff = nextCutoff;
      glowLayer.changed();
      selection.clear();
      scoreLayer.changed();
    },
    focusArea(area: Area) {
      selection.clear();
      const patch = patches.find((p) => p.grid.siteKey === area.siteKey);
      if (!patch) return;
      const [a, , c, , , f] = patch.grid.transform,
        e = patch.grid.transform[4];
      const bounds = [
        c + a * area.minX,
        f + e * (area.maxY + 1),
        c + a * (area.maxX + 1),
        f + e * area.minY,
      ];
      selection.addFeature(new Feature(fromExtent(bounds).transform(patch.code, 'EPSG:4326')));
      const x = area.peakIndex % patch.grid.width,
        y = Math.floor(area.peakIndex / patch.grid.width);
      const point = [c + a * (x + 0.5), f + e * (y + 0.5)];
      selection.addFeature(new Feature(new Point(transform(point, patch.code, 'EPSG:4326'))));
      // Leave context around a one-cell pick rather than filling the map with it.
      const extent = [
        Math.min(bounds[0], point[0] - 2500),
        Math.min(bounds[1], point[1] - 2500),
        Math.max(bounds[2], point[0] + 2500),
        Math.max(bounds[3], point[1] + 2500),
      ];
      map.getView().fit(transformExtent(extent, patch.code, 'EPSG:4326', 16), {
        padding: [25, 25, 25, 25],
        maxZoom: 15,
      });
    },
    focus(siteKey: string) {
      const patch = patches.find((p) => p.grid.siteKey === siteKey);
      if (patch)
        map.getView().fit(transformExtent(patch.extent, patch.code, 'EPSG:4326', 16), {
          padding: [20, 20, 20, 20],
          maxZoom: 15,
        });
    },
    inspect(coordinate: Coordinate) {
      for (const patch of patches) {
        const point = transform(coordinate, 'EPSG:4326', patch.code);
        const [a, , c, , , f] = patch.grid.transform,
          e = patch.grid.transform[4];
        const x = Math.floor((point[0] - c) / a),
          y = Math.floor((point[1] - f) / e);
        if (x >= 0 && y >= 0 && x < patch.grid.width && y < patch.grid.height) {
          const score = patch.scores[y * patch.grid.width + x];
          return {
            siteKey: patch.grid.siteKey,
            x: x + 1,
            y: y + 1,
            score: score === MISSING_SCORE ? null : score / MAX_SCORE,
          };
        }
      }
      return null;
    },
  };
}
