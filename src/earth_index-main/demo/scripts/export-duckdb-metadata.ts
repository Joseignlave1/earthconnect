// Run once against the existing API before replacing PostgreSQL. Keep this
// binary snapshot with the imported dataset so rebuilding needs no live PG.
import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { toBinary, toJson } from '@bufbuild/protobuf';
import { createClient } from '@connectrpc/connect';
import { createGrpcWebTransport } from '@connectrpc/connect-web';
import { EarthIndex, MetadataResponseSchema } from '../src/gen/earth_index_pb.ts';

const api = createClient(EarthIndex, createGrpcWebTransport({
  baseUrl: process.env.MATCHING_API ?? 'http://localhost:8080',
}));
const metadata = await api.metadata({}, { timeoutMs: 60000 });
const id = process.argv[2] ?? 'earth-ed5bc9904a5f';
metadata.datasources = metadata.datasources.filter((d) => d.id === id);
if (metadata.datasources.length !== 1) throw new Error(`Missing datasource ${id}`);
if (metadata.datasources[0].relations.some((r) => r.name === 'measurements'))
  throw new Error('Export the original PostgreSQL metadata before migration; keep the existing source snapshot');
const out = resolve(import.meta.dirname, '../../db/runtime');
await mkdir(out, { recursive: true });
await writeFile(resolve(out, 'source-metadata.pb'), toBinary(MetadataResponseSchema, metadata));
console.log(`Exported ${id}: ${metadata.datasources[0].grids.length} grids`);
await writeFile(resolve(out, 'source-metadata.json'), JSON.stringify(toJson(MetadataResponseSchema, metadata)));
