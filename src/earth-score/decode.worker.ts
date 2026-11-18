import { createDecoder, type DecodeRequest } from './decode.ts'

export type DecodeResult = { ready: true } | { packed: ArrayBuffer } | { error: string }

const decode = createDecoder()
export const send = globalThis.postMessage as (result: DecodeResult, transfer?: Transferable[]) => void

globalThis.onmessage = async ({ data }: MessageEvent<DecodeRequest>) => {
  try {
    const packed = await decode(data)
    send({ packed }, [packed])
  } catch (error) {
    send({ error: error instanceof Error ? error.message : String(error) })
  }
}
