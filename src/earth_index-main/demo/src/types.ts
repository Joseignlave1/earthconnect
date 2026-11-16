import type { Datasource, Grid, PointMeasurement } from './gen/earth_index_pb.ts';

export type Body = 'moon' | 'earth' | 'mars';
export type Target = { body: Exclude<Body, 'earth'>; longitude: number; latitude: number };
export type DecodedValue =
  null | boolean | number | string | Uint8Array | DecodedValue[] | { [key: string]: DecodedValue };
export type Measurement = PointMeasurement & { decoded: DecodedValue };
export interface Sample {
  measurements: Measurement[];
  failed?: boolean;
}
export type FieldsMetadata = {
  relations: { name: string; columns: { name: string; unit: string; description?: string }[] }[];
};
export interface Field {
  key: string;
  label: string;
  unit: string;
  description?: string;
  categorical: boolean;
  tolerance: number;
  group: string;
  qualityKey: string | null;
}
export interface Criterion {
  key: string;
  target: number | null;
  enabled: boolean;
  tolerance: number;
  weight: number;
  min?: number | null;
  max?: number | null;
  selectionTouched?: boolean;
  selectedByUser?: boolean;
  toleranceTouched?: boolean;
  weightTouched?: boolean;
  targetTouched?: boolean;
  kind?: 'measured' | 'approximate' | 'preference' | 'manual';
  origin?: string;
}
export type ScoreGrid = Pick<
  Grid,
  'siteKey' | 'width' | 'height' | 'srid' | 'crsWkt' | 'transform'
>;
export interface ScorePatch {
  grid: ScoreGrid;
  scores: Uint16Array;
}
export type AreaPatch = {
  grid: Pick<ScoreGrid, 'siteKey' | 'width' | 'height'>;
  scores: Uint16Array;
};
export interface Area {
  siteKey: string;
  count: number;
  peak: number;
  peakIndex: number;
  mean: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}
export type MapConfig = {
  id: string;
  label: string;
  radius: number;
  tileSize: number;
  resolutions: number[];
  sizes: number[][];
  url: string;
  source: string;
  attribution: string;
};
export type CoverageMetadata = Pick<Datasource, 'coverage'>;
export type GridMetadata = { grids: ScoreGrid[] };
export const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export const percent = (score: number) => `${(100 * score).toFixed(1)}%`;
