import DecoderWorker from '../earth-score/decoder-entry.worker.ts?worker&inline'

export function createDecoderWorker() {
  return new DecoderWorker()
}
