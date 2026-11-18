import { WORKGROUP_SIZE, type ScoreManifest } from './manifest.ts'
import { lookupBytes, PARAMETER_BYTES } from './scoring.ts'

const filters = ['internal', 'out-of-memory', 'validation'] as const

// Pop every scope in the same turn it was pushed, including synchronous errors.
// Other users of the caller's device can interleave work after the operation.
export async function checkedGpu<T>(device: GPUDevice, operation: () => T, label: string) {
  for (const filter of filters) device.pushErrorScope(filter)
  let value: T
  let errors: Promise<(GPUError | null)[]>
  try {
    value = operation()
  } finally {
    errors = Promise.all(filters.map(() => device.popErrorScope()))
    // A thrown operation skips the await below.
    void errors.catch(() => {})
  }
  const error = (await errors).find(Boolean)
  if (error) throw new Error(`${label}: ${error.message}`)
  return value
}

export function checkDeviceLimits(device: GPUDevice, manifest: ScoreManifest) {
  const limits = device.limits
  const storageBytes = Math.max(lookupBytes(manifest),
    ...manifest.chunks.map(({ rows, packedBytes }) => Math.max(rows * 4, packedBytes)))
  if (storageBytes > limits.maxStorageBufferBindingSize ||
      Math.max(storageBytes, PARAMETER_BYTES) > limits.maxBufferSize ||
      PARAMETER_BYTES > limits.maxUniformBufferBindingSize ||
      limits.maxStorageBuffersPerShaderStage < 3 || limits.maxUniformBuffersPerShaderStage < 1 ||
      limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE || limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE ||
      manifest.chunks.some(({ rows }) => Math.ceil(rows / WORKGROUP_SIZE) > limits.maxComputeWorkgroupsPerDimension)) {
    throw new Error(`WebGPU limits do not support Earth Score (requires ${storageBytes} bytes per storage binding and ${WORKGROUP_SIZE} invocations per workgroup)`)
  }
}

export function deviceLifetime(device: GPUDevice) {
  const loading = new AbortController()
  const state: { loading?: AbortController; lost?: Error } = { loading }
  void device.lost.then(info => {
    state.lost = new Error(`Earth Score GPU device lost: ${info.message || info.reason}`)
    state.loading?.abort(state.lost)
  })
  return { controller: loading, state }
}
