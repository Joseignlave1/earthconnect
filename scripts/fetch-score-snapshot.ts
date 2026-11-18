// Explicit one-shot capture: no metadata request, retry, sampling or live-page API call.
import { createClient } from '@connectrpc/connect'
import { createGrpcWebTransport } from '@connectrpc/connect-web'
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { gzipSync, gunzipSync } from 'node:zlib'
import { resolve } from 'node:path'
import { EarthIndex } from '../vendor/earth-index/earth_index_pb.ts'
import { packedScoreQuery, scoreQueries } from './score-queries.ts'
import { CELL, SIZE, worldTiles } from '../src/shader-test/world-grid.ts'
import { decodeScoreSnapshot, type SnapshotMetadata } from '../src/shader-test/score-snapshot.ts'

const key = process.argv[2] ?? 'basins'
if (key !== 'flat' && key !== 'basins') throw new Error('Choose flat or basins.')
const spec = scoreQueries[key]
const directory = resolve(import.meta.dirname,'../public/data')
const destination = resolve(directory,`${spec.file}.bin`)
try { await readFile(destination); throw new Error('Snapshot already exists; refusing to issue another API request.') }
catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
const endpoint = 'https://earth-index-crisrpia.fly.dev'
const datasourceId = 'earth-ed5bc9904a5f'
const query = packedScoreQuery(spec.expression)
let requests = 0
const api = createClient(EarthIndex,createGrpcWebTransport({ baseUrl: endpoint, fetch: (input,init) => {
  if (++requests !== 1) throw new Error('Only one API request is allowed for this capture.')
  return fetch(input,{ ...init,redirect: 'error' })
} }))
console.log(`Issuing one global score query: ${spec.name}…`)
const response = await api.query({ datasourceId,query,params: Object.fromEntries(Object.entries(spec.params).map(([name,value]) => [name,{ kind: { case: 'numberValue' as const,value } }])) },{ timeoutMs: 180000 })
const columns = response.columns.map(c => c.name)
const addresses = worldTiles(), cells = SIZE * SIZE
const rows = new Map<string,Uint8Array>()
for (const row of response.rows) {
  const field = (key: string) => row.values[columns.indexOf(key)]?.kind
  const site = field('site_key'), scores = field('scores'), width = field('width'), height = field('height')
  if (site?.case !== 'stringValue' || scores?.case !== 'bytesValue' || width?.case !== 'integerValue' || height?.case !== 'integerValue' || Number(width.value) !== SIZE || Number(height.value) !== SIZE || scores.value.byteLength !== cells * 2) throw new Error('Unexpected score query result shape.')
  if (rows.has(site.value)) throw new Error('Duplicate score tile.')
  rows.set(site.value,scores.value)
}
if (rows.size !== addresses.length) throw new Error(`Expected ${addresses.length} global tiles, received ${rows.size}.`)
const native = Buffer.alloc(addresses.length * cells * 2)
let valid = 0, best = 0
addresses.forEach(({ h,v },index) => {
  const key = `h${String(h).padStart(2,'0')}v${String(v).padStart(2,'0')}`, scores = rows.get(key)
  if (!scores) throw new Error(`Missing tile ${key}.`)
  native.set(scores,index * cells * 2)
  const view = new DataView(scores.buffer,scores.byteOffset,scores.byteLength)
  for (let i = 0; i < cells; i++) { const code = view.getUint16(i * 2,true); if (code !== 65535) { valid++; best = Math.max(best,code) } }
})
// Keep the received native bytes locally for independent round-trip verification.
await writeFile(`/tmp/earth-${key}-score-response.u16`,native)
const shuffled = Buffer.alloc(native.length)
for (let tile = 0; tile < addresses.length; tile++) {
  const begin = tile * cells * 2
  let previous = 0
  for (let i = 0; i < cells; i++) {
    if (i % SIZE === 0) previous = 0
    const code = native.readUInt16LE(begin + i * 2), delta = (code - previous) & 65535
    shuffled[begin + i] = delta & 255; shuffled[begin + cells + i] = delta >> 8
    previous = code
  }
}
const base = {
  name: spec.name, description: spec.description,
  fetchedAt: new Date().toISOString(), endpoint,datasourceId,query,params: spec.params,
  tileSize: SIZE,cellMetres: CELL,tiles: addresses.map(({ h,v }): [number,number] => [h,v]),
  sha256: createHash('sha256').update(native).digest('hex'),valid,best,
}
const candidates = (['u16le','row-delta-byteplanes'] as const).map(encoding => {
  const metadata: SnapshotMetadata = { ...base,encoding }, json = Buffer.from(JSON.stringify(metadata)), offset = (8 + json.length + 3) & ~3
  const header = Buffer.alloc(offset); header.write('ECS1'); header.writeUInt32LE(json.length,4); header.set(json,8)
  return { metadata,compressed: gzipSync(Buffer.concat([header,encoding === 'u16le' ? native : shuffled]),{ level: 9 }) }
}).sort((a,b) => a.compressed.length - b.compressed.length)
const chosen = candidates[0], decoded = gunzipSync(chosen.compressed)
const dataset = decodeScoreSnapshot(decoded.buffer.slice(decoded.byteOffset,decoded.byteOffset + decoded.byteLength))
const hash = createHash('sha256')
for (const tile of dataset.tiles) hash.update(Buffer.from(tile.levels[0].scores.buffer))
if (hash.digest('hex') !== base.sha256) throw new Error('Snapshot round trip changed score bytes.')
await mkdir(directory,{ recursive: true })
await writeFile(destination,chosen.compressed)
await writeFile(resolve(directory,`${spec.file}.json`),JSON.stringify({ ...chosen.metadata,requests,rawBytes: native.length,downloadBytes: chosen.compressed.length,alternatives: candidates.map(c => ({ encoding: c.metadata.encoding,bytes: c.compressed.length })) },null,2) + '\n')
console.log(JSON.stringify({ requests,tiles: rows.size,valid,best: best / 65534,rawBytes: native.length,downloadBytes: chosen.compressed.length,encoding: chosen.metadata.encoding,sha256: base.sha256,destination },null,2))
