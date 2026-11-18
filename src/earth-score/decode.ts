import wasmUrl from './zstd.wasm?url&no-inline'
import { BLOCK_SIZE } from './manifest.ts'

export type DecodeRequest = {
  frame: ArrayBuffer
  packedBytes: number
  rows: number
  fields: number
}

type Decoder = {
  memory: WebAssembly.Memory
  allocate: (bytes: number) => number
  release: (ptr: number, bytes: number) => void
  decode: (frame: number, frameBytes: number, output: number, outputBytes: number) => number
}

async function loadDecoder(signal?: AbortSignal) {
  const response = await fetch(wasmUrl, { signal })
  if (!response.ok) throw new Error(`Zstd decoder: HTTP ${response.status}`)
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer())
  return instance.exports as Decoder
}

function unpack(zstd: Decoder, { frame, packedBytes, rows, fields }: DecodeRequest) {
  const input = zstd.allocate(frame.byteLength)
  const output = zstd.allocate(packedBytes)
  try {
    if (!input || !output) throw new Error('Unable to allocate Zstd decode buffers')
    // Allocations can grow WASM memory. Read its buffer only after both allocations.
    new Uint8Array(zstd.memory.buffer, input, frame.byteLength).set(new Uint8Array(frame))
    const status = zstd.decode(input, frame.byteLength, output, packedBytes)
    if (status === 1) throw new Error('Incorrect Zstd frame content size')
    if (status !== 0) throw new Error('Zstd decompression or checksum failed')
    const header = new DataView(zstd.memory.buffer, output, 16)
    if (header.getUint32(0, true) !== 0x31425045 || header.getUint32(4, true) !== BLOCK_SIZE ||
        header.getUint32(8, true) !== rows || header.getUint32(12, true) !== fields) {
      throw new Error('Incorrect GPU block header')
    }
    // WASM memory cannot be transferred. Copy the compact chunk once into a
    // transferable buffer so postMessage adds no further payload copy.
    return new Uint8Array(zstd.memory.buffer, output, packedBytes).slice().buffer
  } finally {
    zstd.release(output, packedBytes)
    zstd.release(input, frame.byteLength)
  }
}

export function createDecoder(signal?: AbortSignal) {
  let decoder: Promise<Decoder> | undefined
  return async (request: DecodeRequest) => {
    decoder ??= loadDecoder(signal)
    const zstd = await decoder
    signal?.throwIfAborted()
    return unpack(zstd, request)
  }
}
