import type { Target, MapConfig } from './types.ts';
// Map coordinates are planet-local longitude/latitude, positive east/north.
export function targetAt(body: string, coordinate: number[] | undefined): Target | null {
  if ((body !== 'moon' && body !== 'mars') || !Array.isArray(coordinate)) return null;
  const [longitude, latitude] = coordinate;
  if (!Number.isFinite(longitude) || !Number.isFinite(latitude) || latitude < -90 || latitude > 90)
    return null;
  const wrapped = ((((longitude + 180) % 360) + 360) % 360) - 180;
  return { body, latitude: round(latitude), longitude: round(wrapped) };
}

function round(value: number) {
  const rounded = Number(value.toFixed(6));
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function tileUrl(config: MapConfig, [z, x, y]: number[], base = '/demo/') {
  const size = config.sizes[z];
  if (!size || x < 0 || y < 0 || x >= size[0] || y >= size[1]) return undefined;
  if (z === 0) return `${base}overview/${config.id}-${x}.jpg`;
  return config.url.replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y));
}
