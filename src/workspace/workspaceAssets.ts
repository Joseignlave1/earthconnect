// The score loader resolves chunk filenames relative to this logical URL.
// Actual asset URLs, including their content hashes, come from the CDN manifest.
export const scoreManifestUrl = 'https://workspace.invalid/compute-data/world/manifest.json'

export type WorkspaceAssets = {
  blob(path: string, signal: AbortSignal): Promise<Blob>
  url(path: string, signal: AbortSignal): Promise<string>
  readAsset(url: string | URL, signal: AbortSignal): Promise<Response>
  dispose(): void
}

export function workspaceAssets(
  get: (path: string, signal: AbortSignal) => Blob | Promise<Blob>,
  resolveDirectUrl?: (path: string, signal: AbortSignal) => Promise<string | undefined>,
): WorkspaceAssets {
  const objectUrls = new Map<string, string>()
  const directUrls = new Map<string, string>()
  async function blob(path: string, signal: AbortSignal) {
    signal.throwIfAborted()
    const result = await get(path, signal)
    signal.throwIfAborted()
    return result
  }
  return {
    blob,
    async url(path, signal) {
      const cached = directUrls.get(path) ?? objectUrls.get(path)
      if (cached) {
        signal.throwIfAborted()
        return cached
      }
      if (resolveDirectUrl) {
        const direct = await resolveDirectUrl(path, signal)
        signal.throwIfAborted()
        if (direct) {
          directUrls.set(path, direct)
          return direct
        }
      }
      const data = await blob(path, signal)
      let url = objectUrls.get(path)
      if (!url) {
        url = URL.createObjectURL(data)
        objectUrls.set(path, url)
      }
      return url
    },
    async readAsset(url, signal) {
      return new Response(await blob(new URL(url).pathname.slice(1), signal))
    },
    dispose() {
      for (const url of objectUrls.values()) URL.revokeObjectURL(url)
      objectUrls.clear()
      directUrls.clear()
    },
  }
}

/** Shared tile-pack loader used by both the landing atlas and the full workspace. */
export async function loadWorkspaceTilePack(
  assets: WorkspaceAssets,
  path: string,
  signal: AbortSignal,
): Promise<Blob | string | null> {
  try {
    if (import.meta.env.DEV) {
      const url = await assets.url(path, signal)
      if (!url.startsWith('blob:')) return url
    }
    return await assets.blob(path, signal)
  } catch {
    return null
  }
}
