import { createClient } from '@connectrpc/connect'
import { createGrpcWebTransport } from '@connectrpc/connect-web'
import type { BodyId } from './explorerData'
import { EarthIndex, PlanetaryBody, type PointMeasurement, type Value } from '../vendor/earth-index/earth_index_pb.ts'

const API_BASE_URL = import.meta.env.VITE_EARTH_INDEX_API_URL ?? 'https://earth-index-crisrpia.fly.dev'

export const earthIndexClient = createClient(EarthIndex, createGrpcWebTransport({ baseUrl: API_BASE_URL }))

type DecodedValue = null | boolean | number | string | Uint8Array | DecodedValue[] | { [key: string]: DecodedValue }
export type SampleMeasurement = PointMeasurement & { decoded: DecodedValue }
export type SampleResult = { measurements: SampleMeasurement[]; partial: boolean; cached: boolean }

export { applySample, criteriaFor, loadDatasource } from './earthIndexCriteria'

function errorText(error: unknown) {
  if (error instanceof Error) return error.message
  return String(error)
}

export function friendlyApiError(error: unknown) {
  const message = errorText(error)
  if (/deadline|timeout/i.test(message)) return 'Earth Index timed out. Try again.'
  if (/fetch|network|unavailable/i.test(message)) return 'Earth Index is offline. Try again shortly.'
  return message
}

export async function samplePoint(
  body: BodyId,
  latitude: number,
  longitude: number,
  signal: AbortSignal,
): Promise<SampleResult> {
  const response = await earthIndexClient.samplePoint(
    {
      body: body === 'moon' ? PlanetaryBody.MOON : PlanetaryBody.MARS,
      latitude,
      longitude,
    },
    { signal, timeoutMs: 60000 },
  )
  return {
    measurements: response.measurements.map(measurement => ({
      ...measurement,
      decoded: decodeValue(measurement.value),
    })),
    partial: response.partial,
    cached: response.cached,
  }
}

function decodeValue(value: Value | undefined): DecodedValue {
  const kind = value?.kind
  if (!kind) return null
  switch (kind.case) {
    case 'nullValue':
      return null
    case 'integerValue':
      return kind.value >= BigInt(Number.MIN_SAFE_INTEGER) && kind.value <= BigInt(Number.MAX_SAFE_INTEGER)
        ? Number(kind.value)
        : kind.value.toString()
    case 'numberValue':
    case 'boolValue':
    case 'stringValue':
    case 'bytesValue':
      return kind.value
    case 'listValue':
      return kind.value.values.map(decodeValue)
    case 'objectValue':
      return Object.fromEntries(Object.entries(kind.value.fields).map(([key, item]) => [key, decodeValue(item)]))
    default:
      return null
  }
}
