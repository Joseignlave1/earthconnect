import { afterEach, expect, it, vi } from 'vitest';
import type OLMap from 'ol/Map.js';
import type BaseLayer from 'ol/layer/Base.js';
import VectorLayer from 'ol/layer/Vector.js';
import View from 'ol/View.js';
import type Style from 'ol/style/Style.js';
const rasterMocks = vi.hoisted(() => ({
  set: vi.fn(),
  clear: vi.fn(),
  destroy: vi.fn(),
  render: vi.fn(),
}));
vi.mock('../../src/score-raster.ts', () => ({
  ScoreRaster: class {
    onRestore = () => {};
    set = rasterMocks.set;
    clear = rasterMocks.clear;
    destroy = rasterMocks.destroy;
    render = rasterMocks.render;
  },
}));
import { earthScores } from '../../src/scores.ts';

// Exercise real OpenLayers layers/projections; only the canvas bitmap sink and
// map attachment are replaced, since jsdom does not provide a canvas renderer.
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
it('preserves peaks across zoom levels, uses small highlights, and clears old results', () => {
  vi.clearAllMocks();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
    () =>
      ({
        createRadialGradient: () => ({ addColorStop: vi.fn() }),
        fillRect: vi.fn(),
      }) as unknown as CanvasRenderingContext2D,
  );
  const png = vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockImplementation(() => {
    throw new Error('PNG encoding must not run');
  });
  vi.stubGlobal(
    'ImageData',
    class {
      constructor(public data: Uint8ClampedArray) {}
    },
  );
  const target = document.createElement('div');
  const layers: BaseLayer[] = [];
  const view = new View({ projection: 'EPSG:4326', resolution: 1, center: [0, 0] });
  const map = {
    addLayer: (layer: BaseLayer) => layers.push(layer),
    removeLayer: (layer: BaseLayer) => layers.splice(layers.indexOf(layer), 1),
    getTargetElement: () => target,
    getView: () => view,
  } as unknown as OLMap;
  const ui = earthScores(map);
  ui.show([
    {
      grid: {
        siteKey: 'peak',
        width: 3,
        height: 1,
        srid: 999001,
        crsWkt: '+proj=longlat +datum=WGS84 +no_defs',
        transform: [0.01, 0, 0, 0, -0.01, 1],
      },
      scores: new Uint16Array([0, 20000, 19000]),
    },
  ]);
  const layer = layers.find((l) => l.getZIndex() === 25) as VectorLayer;
  const source = layer.getSource()!;
  const feature = source.getFeatures()[0];
  expect(source.getFeatures()).toHaveLength(1);
  const extent = feature.getGeometry()!.getExtent();
  [0.015, 0.995, 0.015, 0.995].forEach((value, i) => expect(extent[i]).toBeCloseTo(value, 10));
  const style = layer.getStyleFunction()!;
  expect(style(feature, 1)).toBeTruthy();
  expect(style(feature, 0.001)).toBeUndefined();
  const distant = style(feature, 1) as Style;
  const icon = distant.getImage()!;
  expect(icon.getSize()![0] * (icon.getScale() as number)).toBeLessThanOrEqual(8);
  const scoreLayer = layers.find((l) => l.getZIndex() === 10)!;
  expect(layers).toHaveLength(3); // One score layer, highlights and selection.
  const atlas = rasterMocks.set.mock.calls[0][0];
  expect(atlas.patches[0].levels.at(-1).scores[0]).toBe(20000);
  expect(atlas.patches[0].levels[0].scores).toEqual(new Uint16Array([0, 20000, 19000]));
  view.setResolution(0.001);
  expect(rasterMocks.set).toHaveBeenCalledTimes(1);
  ui.threshold(20001);
  expect(style(feature, 1)).toBeUndefined();
  ui.threshold(19000);
  expect(style(feature, 1)).toBeTruthy();
  expect(rasterMocks.set).toHaveBeenCalledTimes(1); // Threshold/zoom reuse the same data.
  expect(layers.find((l) => l.getZIndex() === 10)).toBe(scoreLayer);
  expect(png).not.toHaveBeenCalled();
  expect(ui.inspect([0.015, 0.995])?.score).toBe(20000 / 65534);
  ui.clear();
  expect(source.getFeatures()).toHaveLength(0);
  expect(target.dataset.scorePatches).toBe('0');
  ui.destroy();
  view.setResolution(1);
  expect(rasterMocks.destroy).toHaveBeenCalledOnce();
  expect(layers).toHaveLength(0);
});
