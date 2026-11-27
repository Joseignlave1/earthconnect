export function createBc7Worker() {
  return new Worker(new URL('./bc7-worker.ts', import.meta.url), { type: 'module' })
}
