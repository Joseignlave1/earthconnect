import Map from 'ol/Map.js';
import View from 'ol/View.js';
import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import Projection from 'ol/proj/Projection.js';
import { get as getProjection } from 'ol/proj.js';
import TileLayer from 'ol/layer/Tile.js';
import VectorLayer from 'ol/layer/Vector.js';
import XYZ from 'ol/source/XYZ.js';
import VectorSource from 'ol/source/Vector.js';
import TileGrid from 'ol/tilegrid/TileGrid.js';
import { defaults as controls } from 'ol/control/defaults.js';
import Attribution from 'ol/control/Attribution.js';
import { Circle, Fill, Stroke, Style } from 'ol/style.js';
import { earthScores } from './scores.ts';
import { targetAt, tileUrl } from './target.ts';
import type { MapConfig, Target } from './types.ts';

const extent = [-180, -90, 180, 90];
export type MapController = ReturnType<typeof createMap>;
export type InspectedCell = ReturnType<ReturnType<typeof earthScores>['inspect']>;
const markerStyle = [
  new Style({
    image: new Circle({
      radius: 9,
      fill: new Fill({ color: '#fff' }),
      stroke: new Stroke({ color: '#111', width: 2 }),
    }),
  }),
  new Style({
    image: new Circle({ radius: 3, fill: new Fill({ color: '#e00000' }) }),
  }),
];

export function createMap(
  element: HTMLDivElement,
  config: MapConfig,
  onChoose: (target: Target) => void,
  onInspect: (cell: InspectedCell) => void,
) {
  let cleanupKeyboard = () => {};
  const projection =
    config.id === 'earth'
      ? getProjection('EPSG:4326')!
      : new Projection({
          code: `${config.id.toUpperCase()}:LONLAT`,
          units: 'degrees',
          extent,
          worldExtent: extent,
          global: true,
          metersPerUnit: (2 * Math.PI * config.radius) / 360,
        });
  const source = new XYZ({
    projection,
    crossOrigin: 'anonymous',
    wrapX: false,
    transition: 0,
    tileGrid: new TileGrid({
      extent,
      origin: [-180, 90],
      tileSize: config.tileSize,
      resolutions: config.resolutions,
      sizes: config.sizes,
    }),
    tileUrlFunction: (coordinate) => tileUrl(config, coordinate, import.meta.env.BASE_URL),
    attributions: `<a href="${config.source}" target="_blank" rel="noopener noreferrer">${config.attribution}</a>`,
  });
  const baseLayer = new TileLayer({
    source,
    extent,
  });
  const marker = new VectorSource({ wrapX: false });
  const map = new Map({
    target: element,
    controls: controls({ rotate: false, attribution: false }).extend([
      new Attribution({ collapsible: false }),
    ]),
    layers: [baseLayer, new VectorLayer({ source: marker, style: markerStyle })],
    view: new View({
      projection,
      center: [0, 0],
      extent,
      showFullExtent: true,
      maxResolution: 360 / 256,
      minResolution: config.resolutions.at(-1)! / 2,
      constrainRotation: true,
      enableRotation: false,
    }),
  });
  const reset = () => map.getView().fit(extent, { size: map.getSize(), padding: [8, 8, 8, 8] });
  const scores = config.id === 'earth' ? earthScores(map) : undefined;
  if (scores) map.on('singleclick', (event) => onInspect(scores.inspect(event.coordinate)));
  reset();

  if (config.id !== 'earth') {
    element.classList.add('selectable');
    const choose = (coordinate: number[] | undefined) => {
      // Ignore clicks in the margin outside the projected planet extent.
      if (!coordinate || coordinate[0] < -180 || coordinate[0] > 180) return;
      const target = targetAt(config.id, coordinate);
      if (target) onChoose(target);
    };
    map.on('singleclick', (event) => choose(event.coordinate));
    element.addEventListener('keydown', onKeyDown);
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Enter') {
        event.preventDefault();
        choose(map.getView().getCenter());
      }
    }
    cleanupKeyboard = () => element.removeEventListener('keydown', onKeyDown);
  }

  element.dataset.loadedTiles = '0';
  element.dataset.tileErrors = '0';
  source.on('tileloadend', () => {
    element.dataset.loadedTiles = String(Number(element.dataset.loadedTiles) + 1);
  });
  source.on('tileloaderror', () => {
    element.dataset.tileErrors = String(Number(element.dataset.tileErrors) + 1);
  });
  const observer = new ResizeObserver(() => map.updateSize());
  observer.observe(element);
  return {
    reset,
    scores,
    setTarget(target: Target | null) {
      marker.clear();
      if (target?.body === config.id)
        marker.addFeature(new Feature(new Point([target.longitude, target.latitude])));
    },
    destroy() {
      observer.disconnect();
      cleanupKeyboard();
      scores?.destroy();
      map.setTarget(undefined);
      map.dispose();
    },
  };
}
