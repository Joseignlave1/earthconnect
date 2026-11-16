import type { MessageInitShape } from '@bufbuild/protobuf';
import type { Value, ValueSchema, QueryResponse } from './gen/earth_index_pb.ts';
import type { DecodedValue } from './types.ts';

// The console uses ordinary JSON; Protobuf carries exact int64 and bytes natively.
export function encodeValue(value: unknown): MessageInitShape<typeof ValueSchema> {
  let kind: MessageInitShape<typeof ValueSchema>['kind'];
  if (value === null) kind = { case: 'nullValue', value: 0 };
  else if (typeof value === 'boolean') kind = { case: 'boolValue', value };
  else if (typeof value === 'string') kind = { case: 'stringValue', value };
  else if (typeof value === 'number') {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)))
      throw new Error(
        'Params contain an unsafe number; use a string and an SQL cast for exact large values',
      );
    kind = Number.isInteger(value)
      ? { case: 'integerValue', value: BigInt(value) }
      : { case: 'numberValue', value };
  } else if (Array.isArray(value))
    kind = { case: 'listValue', value: { values: value.map(encodeValue) } };
  else if (value && typeof value === 'object')
    kind = { case: 'objectValue', value: { fields: encodeParams(value) } };
  else throw new Error('Unsupported parameter value');
  return { kind };
}
export function encodeParams(
  params: unknown,
): Record<string, MessageInitShape<typeof ValueSchema>> {
  if (!params || Array.isArray(params) || typeof params !== 'object')
    throw new Error('Params must be a JSON object');
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => [key, encodeValue(value)]),
  );
}
export function decodeValue(value: Value | undefined): DecodedValue {
  const kind = value?.kind;
  if (!kind) return null;
  const { case: type, value: raw } = kind;
  switch (type) {
    case 'nullValue':
      return null;
    case 'integerValue':
      return raw >= BigInt(Number.MIN_SAFE_INTEGER) && raw <= BigInt(Number.MAX_SAFE_INTEGER)
        ? Number(raw)
        : raw.toString();
    case 'numberValue':
    case 'boolValue':
    case 'stringValue':
    case 'bytesValue':
      return raw;
    case 'listValue':
      return raw.values.map(decodeValue);
    case 'objectValue':
      return Object.fromEntries(
        Object.entries(raw.fields).map(([key, item]) => [key, decodeValue(item)]),
      );
    default:
      return null;
  }
}
export function resultRows(response: QueryResponse): Record<string, DecodedValue>[] {
  const names = response.columns.map((column) => column.name);
  if (new Set(names).size !== names.length)
    throw new Error('Map result column names must be unique');
  return response.rows.map((row) =>
    Object.fromEntries(names.map((name, i) => [name, decodeValue(row.values[i])])),
  );
}
