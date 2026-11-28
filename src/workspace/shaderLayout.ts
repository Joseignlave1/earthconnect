export const EDITOR_LAYOUT_KEY = 'earth-connect-wgsl-editor-layout'

export type LayoutTab = string | { previewFileId: string }

export type SavedGroupLayout = {
  orientation: 0 | 1
  groups: unknown[]
}

export type SavedEditorLayout = {
  layout: SavedGroupLayout
  groups: { tabs: LayoutTab[]; active?: LayoutTab }[]
  activeGroup: number
}

function isLayoutTab(tab: unknown): tab is LayoutTab {
  return (
    typeof tab === 'string' ||
    (typeof tab === 'object' && tab !== null && 'previewFileId' in tab && typeof tab.previewFileId === 'string')
  )
}

export function loadSavedEditorLayout(
  activeFileId: string,
): { snapshot: SavedEditorLayout; addPreview: boolean } | undefined {
  try {
    const saved = JSON.parse(localStorage.getItem(EDITOR_LAYOUT_KEY) ?? 'null') as
      | (Omit<SavedEditorLayout, 'groups'> & {
          groups: { tabs?: (LayoutTab | null)[]; active?: LayoutTab | null; files?: string[] }[]
        })
      | null
    if (
      saved &&
      Array.isArray(saved.groups) &&
      saved.groups.length > 0 &&
      Array.isArray(saved.layout?.groups) &&
      Number.isInteger(saved.activeGroup) &&
      (saved.layout.orientation === 0 || saved.layout.orientation === 1)
    ) {
      if (
        saved.groups.every(
          group => Array.isArray(group?.tabs) && group.tabs.every(tab => tab === null || isLayoutTab(tab)),
        )
      ) {
        const migrate = (tab: LayoutTab | null | undefined) => (tab === null ? { previewFileId: activeFileId } : tab)
        return {
          snapshot: {
            ...saved,
            groups: saved.groups.map(group => ({
              tabs: group.tabs!.map(tab => migrate(tab)!),
              active: migrate(group.active),
            })),
          },
          addPreview: false,
        }
      }
      if (
        saved.groups.every(group => Array.isArray(group?.files) && group.files.every(file => typeof file === 'string'))
      ) {
        return {
          snapshot: {
            ...saved,
            groups: saved.groups.map(group => ({ tabs: group.files!, active: group.active ?? undefined })),
          },
          addPreview: true,
        }
      }
    }
  } catch {
    /* Storage may be unavailable. */
  }
  return undefined
}

export function initialPreviewFileId(files: readonly { id: string }[], activeFileId: string): string | undefined {
  const valid = new Set(files.map(file => file.id))
  const saved = loadSavedEditorLayout(activeFileId)
  if (!saved) return valid.has(activeFileId) ? activeFileId : undefined
  if (saved.addPreview) return valid.has(activeFileId) ? activeFileId : undefined
  for (const group of saved.snapshot.groups) {
    if (group.active && typeof group.active === 'object' && valid.has(group.active.previewFileId)) {
      return group.active.previewFileId
    }
  }
  // Saved layout explicitly exists with no active preview tab; don't prewarm one.
  return undefined
}
