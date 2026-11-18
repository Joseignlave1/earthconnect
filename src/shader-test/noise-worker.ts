import { prepareDataset, type DataRequest, type DatasetMessage } from './prepare-dataset.ts'

const send = globalThis.postMessage as (message: DatasetMessage, transfer?: Transferable[]) => void

globalThis.onmessage = async ({ data }: MessageEvent<DataRequest>) => {
  try {
    const dataset = await prepareDataset(data,(done,total) => send({ kind: 'progress',done,total }))
    const transfer = dataset.tiles.flatMap(tile => tile.levels.map(level => level.scores.buffer))
    send({ kind: 'complete',dataset },[dataset.lookup.buffer,dataset.histogram.buffer,...transfer])
  } catch (error) {
    send({ kind: 'error',message: error instanceof Error ? error.message : String(error) })
  }
}
