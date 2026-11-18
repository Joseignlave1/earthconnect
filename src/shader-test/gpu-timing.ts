export const GPU_STAGES = ['surface','annotations','bloom','composition','presentation'] as const
export type GpuStage = typeof GPU_STAGES[number]
export type GpuTimings = Record<GpuStage | 'total',number>
export type TimingResult = { activity: 'orbit' | 'hover' | 'exposure' | 'bloom'; median: number; p95: number; stages: GpuTimings; samples: number }

// One in-flight sample. Timestamp differences exclude CPU, queue waiting and
// readback; absent (cached) stages report zero, never an older frame's query.
export class GpuTimer {
  private queries: GPUQuerySet
  private resolve: GPUBuffer
  private readback: GPUBuffer
  private active = new Set<GpuStage>()
  constructor(device: GPUDevice) {
    const size = GPU_STAGES.length * 16
    this.queries = device.createQuerySet({ type: 'timestamp',count: GPU_STAGES.length * 2 })
    this.resolve = device.createBuffer({ size,usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    this.readback = device.createBuffer({ size,usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  }
  beginFrame() { this.active.clear() }
  writes(stage: GpuStage,part: 'begin' | 'end' | 'both' = 'both'): GPURenderPassTimestampWrites {
    this.active.add(stage)
    const index = GPU_STAGES.indexOf(stage) * 2
    return { querySet: this.queries,beginningOfPassWriteIndex: part === 'end' ? undefined : index,endOfPassWriteIndex: part === 'begin' ? undefined : index + 1 }
  }
  encode(encoder: GPUCommandEncoder) {
    encoder.resolveQuerySet(this.queries,0,GPU_STAGES.length * 2,this.resolve,0)
    encoder.copyBufferToBuffer(this.resolve,0,this.readback,0,GPU_STAGES.length * 16)
  }
  async read(): Promise<GpuTimings> {
    const active = new Set(this.active)
    await this.readback.mapAsync(GPUMapMode.READ)
    const stamps = new BigUint64Array(this.readback.getMappedRange())
    const result = {} as GpuTimings
    let start: bigint | undefined,end = 0n
    GPU_STAGES.forEach((stage,i) => {
      result[stage] = active.has(stage) ? Number(stamps[i * 2 + 1] - stamps[i * 2]) / 1e6 : 0
      if (active.has(stage)) { start ??= stamps[i * 2]; end = stamps[i * 2 + 1] }
    })
    result.total = start == null ? 0 : Number(end - start) / 1e6
    this.readback.unmap()
    return result
  }
  destroy() { this.queries.destroy(); this.resolve.destroy(); this.readback.destroy() }
}
