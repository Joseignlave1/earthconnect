import { workspaceAssets } from './workspaceAssets.ts'

// Index the archive once. Decode each bounded script part only when its file is needed.
export function embeddedWorkspaceAssets() {
  const typedArrays = Uint8Array as Uint8ArrayConstructor & {
    fromBase64?(input: string): Uint8Array<ArrayBuffer>
  }
  const parts = new Map<string, HTMLScriptElement[]>()
  for (const script of document.querySelectorAll<HTMLScriptElement>('script[data-workspace-asset]')) {
    const path = script.dataset.workspaceAsset!
    if (!parts.has(path)) parts.set(path, [])
    parts.get(path)!.push(script)
  }
  return workspaceAssets(path => {
    const scripts = parts.get(path)
    if (!scripts) throw new Error(`Workspace archive is missing ${path}.`)
    const bytes = scripts.map(
      script =>
        typedArrays.fromBase64?.(script.text) ??
        Uint8Array.from(atob(script.text), character => character.charCodeAt(0)),
    )
    const type = path.endsWith('.jpg') ? 'image/jpeg' : path.endsWith('.webp') ? 'image/webp' : ''
    return new Blob(bytes, { type })
  })
}
