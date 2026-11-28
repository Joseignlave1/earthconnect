import wasmUrl from './zstd.wasm?url&no-inline'
import { EPB1_MAGIC, EPB2_HEADER_BYTES, EPB2_MAGIC } from './epb2-gpu.ts'
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
  frame_content_size: (frame: number, frameBytes: number) => number
  count_epb2_escapes: (lo: number, rows: number, fields: number, blockSize: number, outOffsets: number) => number
  decode: (frame: number, frameBytes: number, output: number, outputBytes: number) => number
  decode_ect2: (
    frame: number,
    frameBytes: number,
    width: number,
    height: number,
    flags: number,
    output: number,
  ) => number
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

type ScratchBuffer = {
  ptr: number
  capacity: number
}

type Scratch = {
  input: ScratchBuffer
  output: ScratchBuffer
}

function reserve(zstd: Decoder, slice: ScratchBuffer, needed: number) {
  if (needed <= slice.capacity && slice.ptr) return
  if (slice.ptr) {
    zstd.release(slice.ptr, slice.capacity)
    slice.ptr = 0
    slice.capacity = 0
  }
  const next = zstd.allocate(needed)
  if (!next) throw new Error('Unable to allocate Zstd decode buffers')
  slice.ptr = next
  slice.capacity = needed
}

function unpack(zstd: Decoder, scratch: Scratch, frame: ArrayBuffer, { packedBytes, rows, fields }: DecodeRequest) {
  reserve(zstd, scratch.input, frame.byteLength)
  new Uint8Array(zstd.memory.buffer, scratch.input.ptr, frame.byteLength).set(new Uint8Array(frame))
  const contentSize = zstd.frame_content_size(scratch.input.ptr, frame.byteLength)
  if (!contentSize) throw new Error('Incorrect Zstd frame content size')

  const blocks = Math.ceil(rows / BLOCK_SIZE)
  const totalBlocks = fields * blocks
  // Reserve enough room for either EPB1 or EPB2 + trailing blockEscapeOffsets table.
  const epb2TailBytes = totalBlocks * 4
  const maxOutput = Math.max(contentSize, contentSize + epb2TailBytes)
  reserve(zstd, scratch.output, maxOutput)

  const status = zstd.decode(scratch.input.ptr, frame.byteLength, scratch.output.ptr, contentSize)
  if (status === 1) throw new Error('Incorrect Zstd frame content size')
  if (status !== 0) throw new Error('Zstd decompression or checksum failed')

  const header = new DataView(zstd.memory.buffer, scratch.output.ptr, Math.min(EPB2_HEADER_BYTES, contentSize))
  const magic = header.getUint32(0, true)

  if (magic === EPB1_MAGIC) {
    if (
      contentSize !== packedBytes ||
      header.getUint32(4, true) !== BLOCK_SIZE ||
      header.getUint32(8, true) !== rows ||
      header.getUint32(12, true) !== fields
    ) {
      throw new Error('Incorrect GPU block header')
    }
    return new Uint8Array(zstd.memory.buffer, scratch.output.ptr, packedBytes).slice().buffer
  }

  if (magic === EPB2_MAGIC) {
    if (
      contentSize < EPB2_HEADER_BYTES ||
      header.getUint32(4, true) !== BLOCK_SIZE ||
      header.getUint32(8, true) !== rows ||
      header.getUint32(12, true) !== fields ||
      header.getUint32(16, true) !== packedBytes
    ) {
      throw new Error('Incorrect GPU block header')
    }
    const loBytes = header.getUint32(24, true)
    const escapeCount = header.getUint32(28, true)
    const loAligned = (loBytes + 3) & ~3
    const hiAligned = (escapeCount * 2 + 3) & ~3
    const expectedRaw = EPB2_HEADER_BYTES + loAligned + hiAligned
    if (loBytes !== rows * fields || contentSize !== expectedRaw) {
      throw new Error('Incorrect EPB2 payload size')
    }
    const escOffsetsPtr = scratch.output.ptr + expectedRaw
    const counted = zstd.count_epb2_escapes(
      scratch.output.ptr + EPB2_HEADER_BYTES,
      rows,
      fields,
      BLOCK_SIZE,
      escOffsetsPtr,
    )
    if (counted !== escapeCount) {
      throw new Error('EPB2 escape count mismatch')
    }
    const totalOutBytes = expectedRaw + epb2TailBytes
    return new Uint8Array(zstd.memory.buffer, scratch.output.ptr, totalOutBytes).slice().buffer
  }

  throw new Error('Incorrect GPU block header')
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

export async function decompressTerrainEct2Wasm(
  frame: ArrayBuffer,
  width: number,
  height: number,
  flags: number,
  signal?: AbortSignal,
): Promise<Uint8ClampedArray<ArrayBuffer>> {
  sharedZstdModule ??= compileZstdModule().catch(error => {
    sharedZstdModule = undefined
    throw error
  })
  const zstd = await loadDecoder(await sharedZstdModule, signal)
  signal?.throwIfAborted()
  const outStride = flags & 4 ? 4 : 2
  const outputBytes = width * height * outStride
  const inputPtr = zstd.allocate(frame.byteLength)
  if (!inputPtr) throw new Error('Unable to allocate Zstd input buffer')
  const outputPtr = zstd.allocate(outputBytes)
  if (!outputPtr) {
    zstd.release(inputPtr, frame.byteLength)
    throw new Error('Unable to allocate ECT2 output buffer')
  }
  try {
    new Uint8Array(zstd.memory.buffer, inputPtr, frame.byteLength).set(new Uint8Array(frame))
    const status = zstd.decode_ect2(inputPtr, frame.byteLength, width, height, flags, outputPtr)
    if (status === 1) throw new Error('Incorrect Zstd frame content size')
    if (status === 3) throw new Error('Invalid ECT2 terrain dimensions.')
    if (status !== 0) throw new Error('Zstd decompression or checksum failed')
    const out = new Uint8ClampedArray(outputBytes)
    out.set(new Uint8Array(zstd.memory.buffer, outputPtr, outputBytes))
    return out
  } finally {
    zstd.release(outputPtr, outputBytes)
    zstd.release(inputPtr, frame.byteLength)
  }
}

export function createDecoder(signal?: AbortSignal) {
  let decoder: Promise<Decoder> | undefined
  const scratch: Scratch = { input: { ptr: 0, capacity: 0 }, output: { ptr: 0, capacity: 0 } }
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
