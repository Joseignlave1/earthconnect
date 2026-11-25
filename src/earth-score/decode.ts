import wasmUrl from './zstd.wasm?url&no-inline'
import { BLOCK_SIZE } from './manifest.ts'

export type DecodeRequest = {
  frame: ArrayBuffer | Blob
  packedBytes: number
  rows: number
  fields: number
  module?: WebAssembly.Module
}

type Decoder = {
  memory: WebAssembly.Memory
  allocate: (bytes: number) => number
  release: (ptr: number, bytes: number) => void
  decode: (frame: number, frameBytes: number, output: number, outputBytes: number) => number
}

export async function compileZstdModule(signal?: AbortSignal): Promise<WebAssembly.Module> {
  const response = await fetch(wasmUrl, { signal })
  if (!response.ok) throw new Error(`Zstd decoder: HTTP ${response.status}`)
  if (typeof WebAssembly.compileStreaming === 'function') {
    try {
      return await WebAssembly.compileStreaming(response.clone())
    } catch {
      signal?.throwIfAborted()
    }
  }
  return WebAssembly.compile(await response.arrayBuffer())
}

async function loadDecoder(module?: WebAssembly.Module, signal?: AbortSignal) {
  const compiled = module ?? (await compileZstdModule(signal))
  const instance = await WebAssembly.instantiate(compiled)
  return instance.exports as Decoder
}

type Scratch = {
  inputPtr: number
  inputCap: number
  outputPtr: number
  outputCap: number
}

function reserve(zstd: Decoder, ptr: number, cap: number, needed: number): [number, number] {
  if (needed <= cap && ptr) return [ptr, cap]
  if (ptr) zstd.release(ptr, cap)
  const next = zstd.allocate(needed)
  if (!next) throw new Error('Unable to allocate Zstd decode buffers')
  return [next, needed]
}

function unpack(zstd: Decoder, scratch: Scratch, frame: ArrayBuffer, { packedBytes, rows, fields }: DecodeRequest) {
  ;[scratch.inputPtr, scratch.inputCap] = reserve(zstd, scratch.inputPtr, scratch.inputCap, frame.byteLength)
  ;[scratch.outputPtr, scratch.outputCap] = reserve(zstd, scratch.outputPtr, scratch.outputCap, packedBytes)
  // Allocations can grow WASM memory. Read its buffer only after both reservations.
  new Uint8Array(zstd.memory.buffer, scratch.inputPtr, frame.byteLength).set(new Uint8Array(frame))
  const status = zstd.decode(scratch.inputPtr, frame.byteLength, scratch.outputPtr, packedBytes)
  if (status === 1) throw new Error('Incorrect Zstd frame content size')
  if (status !== 0) throw new Error('Zstd decompression or checksum failed')
  const header = new DataView(zstd.memory.buffer, scratch.outputPtr, 16)
  if (
    header.getUint32(0, true) !== 0x31425045 ||
    header.getUint32(4, true) !== BLOCK_SIZE ||
    header.getUint32(8, true) !== rows ||
    header.getUint32(12, true) !== fields
  ) {
    throw new Error('Incorrect GPU block header')
  }
  // WASM memory cannot be transferred. Copy the compact chunk once into a
  // transferable buffer so postMessage adds no further payload copy.
  return new Uint8Array(zstd.memory.buffer, scratch.outputPtr, packedBytes).slice().buffer
}

let sharedZstdModule: Promise<WebAssembly.Module> | undefined

export async function decompressZstd(
  frame: ArrayBuffer,
  outputBytes: number,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  sharedZstdModule ??= compileZstdModule().catch(error => {
    sharedZstdModule = undefined
    throw error
  })
  const zstd = await loadDecoder(await sharedZstdModule, signal)
  signal?.throwIfAborted()
  const inputPtr = zstd.allocate(frame.byteLength)
  if (!inputPtr) throw new Error('Unable to allocate Zstd input buffer')
  const outputPtr = zstd.allocate(outputBytes)
  if (!outputPtr) {
    zstd.release(inputPtr, frame.byteLength)
    throw new Error('Unable to allocate Zstd output buffer')
  }
  try {
    new Uint8Array(zstd.memory.buffer, inputPtr, frame.byteLength).set(new Uint8Array(frame))
    const status = zstd.decode(inputPtr, frame.byteLength, outputPtr, outputBytes)
    if (status === 1) throw new Error('Incorrect Zstd frame content size')
    if (status !== 0) throw new Error('Zstd decompression or checksum failed')
    return new Uint8Array(zstd.memory.buffer, outputPtr, outputBytes).slice()
  } finally {
    zstd.release(outputPtr, outputBytes)
    zstd.release(inputPtr, frame.byteLength)
  }
}

export function createDecoder(signal?: AbortSignal) {
  let decoder: Promise<Decoder> | undefined
  const scratch: Scratch = { inputPtr: 0, inputCap: 0, outputPtr: 0, outputCap: 0 }
  return async (request: DecodeRequest) => {
    decoder ??= loadDecoder(request.module, signal)
    const [zstd, frame] = await Promise.all([
      decoder,
      request.frame instanceof Blob ? request.frame.arrayBuffer() : request.frame,
    ])
    signal?.throwIfAborted()
    return unpack(zstd, scratch, frame, request)
  }
}
