import type { SamplePointResponse } from './gen/earth_index_pb.ts';
import type { Measurement, Target, Sample } from './types.ts';

import { api } from './api.ts';
import { PlanetaryBody, MeasurementStatus } from './gen/earth_index_pb.ts';
import { decodeValue } from './values.ts';

export function pointMeasurements(response: SamplePointResponse): Measurement[] {
  return response.measurements.map((measurement) => ({
    ...measurement,
    decoded: decodeValue(measurement.value),
  }));
}
export function numericMeasurement(
  measurements: Measurement[],
  id: string,
  unit: string,
): number | null {
  const measurement = measurements.find(
    (m) =>
      m.variableId === id &&
      m.status === MeasurementStatus.AVAILABLE &&
      m.unit === unit &&
      !m.qualityFlags.includes('semantic_review_required') &&
      Number.isFinite(m.decoded),
  );
  return typeof measurement?.decoded === 'number' ? measurement.decoded : null;
}
export async function loadCharacteristics(target: Target, signal: AbortSignal, client = api) {
  const response = await client.samplePoint(
    {
      body: target.body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS,
      longitude: target.longitude,
      latitude: target.latitude,
    },
    { signal, timeoutMs: 25000 },
  );
  return {
    ...target,
    response,
    measurements: pointMeasurements(response),
    failed: response.partial,
  };
}

// An aborted/obsolete completion can never replace a newer target, including clear.
export type LoadState = 'empty' | 'loading' | 'partial' | 'ready' | 'error';
export function targetLoader(
  onResult: (result: Sample | { error: unknown } | null, state: LoadState) => void,
  reader: (target: Target, signal: AbortSignal) => Promise<Sample> = loadCharacteristics,
) {
  let controller: AbortController | undefined;
  let revision = 0;
  return async (target: Target | null) => {
    controller?.abort();
    const current = ++revision;
    if (!target) {
      onResult(null, 'empty');
      return;
    }
    controller = new AbortController();
    onResult(null, 'loading');
    try {
      const result = await reader(target, controller.signal);
      if (current === revision) onResult(result, result.failed ? 'partial' : 'ready');
    } catch (error) {
      if (current === revision) onResult({ error }, 'error');
    }
  };
}
