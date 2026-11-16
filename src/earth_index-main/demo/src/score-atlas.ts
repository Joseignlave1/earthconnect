import proj4 from 'proj4';
import { register } from 'ol/proj/proj4.js';
import { scorePyramid, overviewLevel } from './score-overviews.ts';
import type { ScorePatch, ScoreGrid } from './types.ts';

export function gridExtent(grid: Pick<ScoreGrid, 'siteKey' | 'width' | 'height' | 'transform'>) {
  const [a, b, c, d, e, f] = grid.transform;
  if (
    grid.transform.length !== 6 ||
    !grid.transform.every(Number.isFinite) ||
    b !== 0 ||
    d !== 0 ||
    a <= 0 ||
    e >= 0
  )
    throw new Error(`${grid.siteKey}: unsupported grid transform`);
  return [c, f + e * grid.height, c + a * grid.width, f];
}

const registered = new Map<string, string>();
export function registerScoreProjection(srid: number, wkt: string) {
  const code = `EARTH-INDEX:${srid}`;
  if (registered.get(code) !== wkt) {
    if (registered.has(code)) throw new Error(`Conflicting projection definition: ${code}`);
    proj4.defs(code, wkt);
    register(proj4);
    registered.set(code, wkt);
  }
  return code;
}

export type AtlasPatch = ScorePatch & {
  code: string;
  extent: number[];
  levels: ReturnType<typeof scorePyramid>;
};
export interface ScoreAtlas {
  patches: AtlasPatch[];
  lookup: Uint16Array;
  columns: number;
  rows: number;
  origin: number[];
  span: number[];
  projection: { sinusoidal: boolean; radius: number; longitude: number; x: number; y: number };
}

// A tiny lattice maps native geographic chunks to texture-array slots. The
// shader projects each screen pixel once, then performs two texture lookups;
// neither per-region image layers nor viewport-dependent reprojection is needed.
export function scoreAtlas(next: ScorePatch[]): ScoreAtlas {
  if (!next.length || next.length >= 65535) throw new Error('Invalid score region count');
  const first = next[0].grid;
  const code = registerScoreProjection(first.srid, first.crsWkt);
  const definition = proj4.defs(code);
  const name = definition.projName?.toLowerCase();
  const sinusoidal = name === 'sinu' || name === 'sinusoidal';
  if (!sinusoidal && name !== 'longlat') throw new Error('Unsupported score display projection');
  const radius = Number(definition.a);
  if (sinusoidal && (!(radius > 0) || Number(definition.b ?? radius) !== radius))
    throw new Error('Score display requires a spherical sinusoidal projection');
  const patches = next.map((patch) => {
    const g = patch.grid;
    if (
      g.srid !== first.srid ||
      g.crsWkt !== first.crsWkt ||
      g.width !== first.width ||
      g.height !== first.height ||
      g.transform[0] !== first.transform[0] ||
      g.transform[4] !== first.transform[4]
    )
      throw new Error('Score regions must share one regular grid and projection');
    return {
      ...patch,
      code,
      extent: gridExtent(g),
      levels: scorePyramid(patch.scores, g.width, g.height),
    };
  });
  const origin = [
    Math.min(...patches.map((p) => p.extent[0])),
    Math.max(...patches.map((p) => p.extent[3])),
  ];
  const span = [first.width * first.transform[0], -first.height * first.transform[4]];
  const integer = (n: number) => {
    if (Math.abs(n - Math.round(n)) > 1e-6) throw new Error('Misaligned score region');
    return Math.round(n);
  };
  const locations = patches.map((p) => [
    integer((p.extent[0] - origin[0]) / span[0]),
    integer((origin[1] - p.extent[3]) / span[1]),
  ]);
  const columns = Math.max(...locations.map((p) => p[0])) + 1;
  const rows = Math.max(...locations.map((p) => p[1])) + 1;
  if (columns * rows > 1_000_000) throw new Error('Score grid is too sparse to display');
  const lookup = new Uint16Array(columns * rows).fill(65535);
  locations.forEach(([x, y], index) => {
    const address = y * columns + x;
    if (lookup[address] !== 65535) throw new Error('Overlapping score regions');
    lookup[address] = index;
  });
  return {
    patches,
    lookup,
    columns,
    rows,
    origin,
    span,
    projection: {
      sinusoidal,
      radius: sinusoidal ? radius : 1,
      longitude: Number(definition.long0 ?? 0),
      x: Number(definition.x0 ?? 0),
      y: Number(definition.y0 ?? 0),
    },
  };
}

export function atlasLevel(atlas: ScoreAtlas, resolution: number) {
  const degrees = atlas.projection.sinusoidal ? 180 / (Math.PI * atlas.projection.radius) : 1;
  return overviewLevel(
    atlas.patches[0].levels,
    atlas.span.map((n) => n * degrees),
    resolution,
  );
}

// Reference sampler shared by numeric/browser tests. The GPU follows these
// exact missing/zero, lookup and max-pooled footprint rules.
export function sampleAtlas(atlas: ScoreAtlas, longitude: number, latitude: number, level = 0) {
  if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) return 65535;
  const p = atlas.projection;
  const lat = (latitude * Math.PI) / 180;
  const x = p.sinusoidal
    ? p.radius * ((longitude * Math.PI) / 180 - p.longitude) * Math.cos(lat) + p.x
    : longitude;
  const y = p.sinusoidal ? p.radius * lat + p.y : latitude;
  const u = (x - atlas.origin[0]) / atlas.span[0];
  const v = (atlas.origin[1] - y) / atlas.span[1];
  if (u < 0 || v < 0 || u >= atlas.columns || v >= atlas.rows) return 65535;
  const slot = atlas.lookup[Math.floor(v) * atlas.columns + Math.floor(u)];
  if (slot === 65535) return 65535;
  const image = atlas.patches[slot].levels[level];
  return image.scores[
    Math.floor((v % 1) * image.height) * image.width + Math.floor((u % 1) * image.width)
  ];
}
