// The score loader resolves chunk filenames relative to this logical URL.
// Actual asset URLs, including their content hashes, come from the CDN manifest.
export const scoreManifestUrl = 'https://workspace.invalid/compute-data/world/manifest.json'

export type WorkspaceAssets = {
  blob(path: string, signal: AbortSignal): Promise<Blob>
  url(path: string, signal: AbortSignal): Promise<string>
  readAsset(url: string | URL, signal: AbortSignal): Promise<Response>
  dispose(): void
}

export function workspaceAssets(get: (path: string, signal: AbortSignal) => Blob | Promise<Blob>): WorkspaceAssets {
  const urls = new Map<string, string>()
  async function blob(path: string, signal: AbortSignal) {
    signal.throwIfAborted()
    const result = await get(path, signal)
    signal.throwIfAborted()
    return result
  }
  return {
    blob,
    async url(path, signal) {
      const cached = urls.get(path)
      if (cached) {
        signal.throwIfAborted()
        return cached
      }
      const data = await blob(path, signal)
      let url = urls.get(path)
      if (!url) {
        url = URL.createObjectURL(data)
        urls.set(path, url)
      }
      return url
    },
    async readAsset(url, signal) {
      return new Response(await blob(new URL(url).pathname.slice(1), signal))
    },
    dispose() {
      for (const url of urls.values()) URL.revokeObjectURL(url)
      urls.clear()
    },
  }
}
