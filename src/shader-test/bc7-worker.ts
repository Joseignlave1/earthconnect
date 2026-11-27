import { decompressZstd } from '../earth-score/decode.ts'

export type Bc7WorkerRequest = { url: string }
export type Bc7WorkerResponse =
  | { kind: 'ok'; width: number; height: number; mipCount: number; data: ArrayBuffer }
  | { kind: 'error'; message: string }

const send = globalThis.postMessage as (message: Bc7WorkerResponse, transfer?: Transferable[]) => void

function totalBc7Bytes(width: number, height: number, mipCount: number): number {
  let total = 0
  for (let mip = 0; mip < mipCount; mip++) {
    const mw = Math.max(1, width >> mip)
    const mh = Math.max(1, height >> mip)
    total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * 16
  }
  return total
}

globalThis.onmessage = async ({ data }: MessageEvent<Bc7WorkerRequest>) => {
  try {
    const response = await fetch(data.url)
    if (!response.ok) throw new Error(`BC7 texture HTTP ${response.status}`)
    const raw = await response.arrayBuffer()
    if (raw.byteLength < 16) throw new Error('Invalid BC7 texture header.')
    const view = new DataView(raw)
    if (view.getUint32(0) !== 0x45434243) throw new Error('Invalid BC7 magic.')
    const width = view.getUint32(4, true)
    const height = view.getUint32(8, true)
    const mipCount = view.getUint32(12, true)
    const expectedBytes = totalBc7Bytes(width, height, mipCount)
    const decoded = await decompressZstd(raw.slice(16), expectedBytes)
    send({ kind: 'ok', width, height, mipCount, data: decoded.buffer }, [decoded.buffer])
  } catch (error) {
    send({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
