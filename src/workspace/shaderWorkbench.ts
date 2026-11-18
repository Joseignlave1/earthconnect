import {
  getService, GroupDirection, GroupOrientation, GroupsOrder, ICodeEditorService, IEditorGroupsService,
  IEditorService, IExtensionService, initialize, type EditorGroupLayout, type IEditorGroup,
} from '@codingame/monaco-vscode-api'
import { isCodeEditor } from '@codingame/monaco-vscode-api/vscode/vs/editor/browser/editorBrowser'
import { isIMenuItem, MenuId, MenuRegistry } from '@codingame/monaco-vscode-api/vscode/vs/platform/actions/common/actions'
import { ContextKeyExpr } from '@codingame/monaco-vscode-api/vscode/vs/platform/contextkey/common/contextkey'
import { ConfigureEditorAction } from '@codingame/monaco-vscode-api/vscode/vs/workbench/browser/actions/layoutActions'
import {
  COPY_EDITOR_INTO_NEW_WINDOW_COMMAND_ID, MOVE_EDITOR_INTO_NEW_WINDOW_COMMAND_ID, REOPEN_WITH_COMMAND_ID,
} from '@codingame/monaco-vscode-api/vscode/vs/workbench/browser/parts/editor/editorCommands'
import { EditorCloseContext } from '@codingame/monaco-vscode-api/vscode/vs/workbench/common/editor'
import getConfigurationServiceOverride from '@codingame/monaco-vscode-configuration-service-override'
import getFilesServiceOverride, {
  RegisteredFileSystemProvider, RegisteredMemoryFile, registerFileSystemOverlay,
} from '@codingame/monaco-vscode-files-service-override'
import getLanguagesServiceOverride from '@codingame/monaco-vscode-languages-service-override'
import getModelServiceOverride from '@codingame/monaco-vscode-model-service-override'
import getTextMateServiceOverride from '@codingame/monaco-vscode-textmate-service-override'
import TextMateWorker from '@codingame/monaco-vscode-textmate-service-override/worker?worker'
import getThemeServiceOverride from '@codingame/monaco-vscode-theme-service-override'
import getViewsServiceOverride, { renderEditorPart } from '@codingame/monaco-vscode-views-service-override'
import { editor, Uri } from 'monaco-editor'
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker.js?worker'
import { movePreviewHost, registerShaderPreviewPane, ShaderPreviewInput, ShaderPreviewPane } from './shaderPreviewEditor'
import type { ShaderFile } from './shaderProject'
import { registerWgslTheme, wgslLanguageId, wgslThemeName } from './wgslTheme'
import './shaderWorkbench.css'

type WorkbenchCallbacks = {
  onSelect: (fileId: string) => void
  onClose: (fileId: string) => boolean
  onReopen: (fileId: string) => void
  onEditor: (editor: editor.ICodeEditor | undefined) => void
  onPreviewHost: (fileId: string, host: HTMLElement | undefined) => void
  onPreviewOpenChange: (fileIds: readonly string[]) => void
  onVisiblePreviews: (fileIds: readonly string[]) => void
}

type FileEntry = {
  id: string
  name: string
  resource: Uri
  registration: { dispose: () => void }
  reference: Awaited<ReturnType<typeof editor.createModelReference>>
  model: editor.ITextModel
}

type LayoutTab = string | { previewFileId: string }

type EditorLayout = {
  layout: EditorGroupLayout
  groups: { tabs: LayoutTab[]; active?: LayoutTab }[]
  activeGroup: number
}

const layoutKey = 'earth-connect-wgsl-editor-layout'
const provider = new RegisteredFileSystemProvider(false)
const filesById = new Map<string, FileEntry>()
const filesByResource = new Map<string, FileEntry>()
let initialization: Promise<void> | undefined
let currentMount: (() => void) | undefined
let syncQueue = Promise.resolve()
let layoutRestored = false

function configureEditorMenus() {
  // The embedded workbench has one text editor type and no Settings or auxiliary window UI.
  const unsupported = new Set([
    ConfigureEditorAction.ID, REOPEN_WITH_COMMAND_ID,
    MOVE_EDITOR_INTO_NEW_WINDOW_COMMAND_ID, COPY_EDITOR_INTO_NEW_WINDOW_COMMAND_ID,
  ])
  for (const menu of [MenuId.EditorTitle, MenuId.EditorTitleContext]) {
    for (const item of MenuRegistry.getMenuItems(menu)) {
      if (isIMenuItem(item) && unsupported.has(item.command.id)) item.when = ContextKeyExpr.false()
    }
  }
}

function initializeWorkbench() {
  if (initialization) return initialization
  registerFileSystemOverlay(1, provider)
  configureEditorMenus()
  registerShaderPreviewPane()
  const themeReady = registerWgslTheme()
  window.MonacoEnvironment = {
    getWorker: (_moduleId, label) => label === 'TextMateWorker' ? new TextMateWorker() : new EditorWorker(),
  }
  initialization = initialize({
    ...getFilesServiceOverride(),
    ...getModelServiceOverride(),
    ...getConfigurationServiceOverride(),
    ...getLanguagesServiceOverride(),
    ...getThemeServiceOverride(),
    ...getTextMateServiceOverride(),
    ...getViewsServiceOverride(undefined, undefined, false),
  }, document.body, {
    configurationDefaults: {
      'workbench.colorTheme': wgslThemeName,
      'workbench.startupEditor': 'none',
      'workbench.editor.enablePreview': false,
      'workbench.editor.closeEmptyGroups': true,
      'workbench.editor.restoreViewState': true,
      'workbench.editor.tabSizing': 'shrink',
      'workbench.editor.alwaysShowEditorActions': true,
      'workbench.editor.dragToOpenWindow': false,
      'workbench.editor.showIcons': false,
      'workbench.tips.enabled': false,
      'breadcrumbs.enabled': false,
      'files.autoSave': 'onFocusChange',
      'editor.minimap.enabled': false,
      'editor.fontSize': 13,
      'editor.wordWrap': 'on',
      'editor.scrollBeyondLastLine': false,
      'editor.lineNumbersMinChars': 3,
      'editor.renderValidationDecorations': 'on',
      'editor.tabSize': 2,
      'editor.insertSpaces': true,
    },
  }).then(async () => {
    await themeReady
    const extensionService = await getService(IExtensionService)
    await extensionService.whenInstalledExtensionsRegistered()
  })
  return initialization
}

function isLayoutTab(tab: unknown): tab is LayoutTab {
  return typeof tab === 'string' || (typeof tab === 'object' && tab !== null
    && 'previewFileId' in tab && typeof tab.previewFileId === 'string')
}

function loadLayout(activeFileId: string): { snapshot: EditorLayout; addPreview: boolean } | undefined {
  try {
    const saved = JSON.parse(localStorage.getItem(layoutKey) ?? 'null') as (Omit<EditorLayout, 'groups'> & {
      groups: { tabs?: (LayoutTab | null)[]; active?: LayoutTab | null; files?: string[] }[]
    }) | null
    if (saved && Array.isArray(saved.groups) && saved.groups.length > 0 && Array.isArray(saved.layout?.groups)
      && Number.isInteger(saved.activeGroup)
      && (saved.layout.orientation === GroupOrientation.HORIZONTAL || saved.layout.orientation === GroupOrientation.VERTICAL)) {
      if (saved.groups.every(group => Array.isArray(group?.tabs)
        && group.tabs.every(tab => tab === null || isLayoutTab(tab)))) {
        const migrate = (tab: LayoutTab | null | undefined) => tab === null ? { previewFileId: activeFileId } : tab
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
      if (saved.groups.every(group => Array.isArray(group?.files)
        && group.files.every(file => typeof file === 'string'))) {
        return {
          snapshot: { ...saved, groups: saved.groups.map(group => ({ tabs: group.files!, active: group.active ?? undefined })) },
          addPreview: true,
        }
      }
    }
  } catch { /* Storage may be unavailable. */ }
}

export function getFileId(resource: Uri | undefined) {
  return resource && filesByResource.get(resource.toString())?.id
}

async function addFile(file: ShaderFile) {
  const resource = Uri.file(`/shader-project/${encodeURIComponent(file.id)}/${file.name}`)
  const registration = provider.registerFile(new RegisteredMemoryFile(resource, file.source))
  try {
    const reference = await editor.createModelReference(resource)
    if (!reference.object.isResolved()) {
      reference.dispose()
      throw new Error(`Could not resolve ${file.name}`)
    }
    const model = reference.object.textEditorModel
    editor.setModelLanguage(model, wgslLanguageId)
    const entry = { id: file.id, name: file.name, resource, registration, reference, model }
    filesById.set(file.id, entry)
    filesByResource.set(resource.toString(), entry)
    return entry
  } catch (error) {
    registration.dispose()
    throw error
  }
}

export async function mountShaderWorkbench(host: HTMLElement, callbacks: WorkbenchCallbacks) {
  await initializeWorkbench()
  const [groups, editorService, codeEditorService] = await Promise.all([
    getService(IEditorGroupsService), getService(IEditorService), getService(ICodeEditorService),
  ])
  await groups.whenRestored
  currentMount?.()
  host.classList.add('shader-workbench')
  const attachment = renderEditorPart(host)
  const parking = host.ownerDocument.createElement('div')
  parking.hidden = true
  host.append(parking)
  const subscriptions = [attachment]
  const groupSubscriptions = new Map<number, { dispose: () => void }>()
  let disposed = false
  let syncing = false
  let updateScheduled = false
  let selectedId: string | undefined
  let visiblePreviews: string[] = []
  const previewHosts = new Map<string, HTMLElement>()
  const previewGroupIds = new Map<string, number>()
  const sourceGroupIds = new Map<string, number>()

  function findPreview(group: IEditorGroup, fileId: string) {
    return group.editors.find(input => input instanceof ShaderPreviewInput && input.fileId === fileId)
  }

  function updatePreviews() {
    const open = new Map<string, { group: IEditorGroup; input: ShaderPreviewInput }>()
    for (const group of groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE)) {
      for (const input of group.editors) {
        if (!(input instanceof ShaderPreviewInput)) continue
        const existing = open.get(input.fileId)
        // Native whole-group copies can clone a singleton tab.
        if (existing && group.id === previewGroupIds.get(input.fileId)) {
          void existing.group.closeEditor(existing.input, { preserveFocus: true })
          open.set(input.fileId, { group, input })
        } else if (existing) {
          void group.closeEditor(input, { preserveFocus: true })
        } else {
          open.set(input.fileId, { group, input })
        }
      }
    }
    let changed = false
    for (const [fileId, previewHost] of previewHosts) {
      const entry = open.get(fileId)
      const pane = entry?.group.activeEditorPane
      if (!(pane instanceof ShaderPreviewPane) || pane.input !== entry?.input) movePreviewHost(previewHost, parking)
      if (entry) continue
      callbacks.onPreviewHost(fileId, undefined)
      previewHost.remove()
      previewHosts.delete(fileId)
      changed = true
    }
    const visible: string[] = []
    for (const [fileId, { group, input }] of open) {
      previewGroupIds.set(fileId, group.id)
      let previewHost = previewHosts.get(fileId)
      if (!previewHost) {
        previewHost = host.ownerDocument.createElement('div')
        previewHost.className = 'sky-preview-host'
        parking.append(previewHost)
        previewHosts.set(fileId, previewHost)
        callbacks.onPreviewHost(fileId, previewHost)
        changed = true
      }
      const pane = group.activeEditorPane
      if (pane instanceof ShaderPreviewPane && pane.input === input) {
        pane.showHost(previewHost)
        visible.push(fileId)
      }
    }
    if (changed) callbacks.onPreviewOpenChange([...previewHosts.keys()])
    if (visible.length !== visiblePreviews.length || visible.some((id, index) => id !== visiblePreviews[index])) {
      visiblePreviews = visible
      callbacks.onVisiblePreviews(visible)
    }
  }

  function editors() {
    return codeEditorService.listCodeEditors().filter(codeEditor => host.contains(codeEditor.getContainerDomNode()))
  }

  function activeEditor() {
    const control = editorService.activeTextEditorControl
    return isCodeEditor(control) && host.contains(control.getContainerDomNode()) ? control : undefined
  }

  function saveLayout() {
    if (disposed || syncing || !layoutRestored) return
    const part = groups.mainPart
    const ordered = part.getGroups(GroupsOrder.GRID_APPEARANCE)
    function layoutTab(input: IEditorGroup['activeEditor']): LayoutTab | undefined {
      return input instanceof ShaderPreviewInput ? { previewFileId: input.fileId } : getFileId(input?.resource)
    }
    const snapshot: EditorLayout = {
      layout: part.getLayout(),
      groups: ordered.map(group => ({
        tabs: group.editors.flatMap(input => {
          const tab = layoutTab(input)
          return tab === undefined ? [] : [tab]
        }),
        active: layoutTab(group.activeEditor),
      })),
      activeGroup: ordered.indexOf(part.activeGroup),
    }
    try { localStorage.setItem(layoutKey, JSON.stringify(snapshot)) } catch { /* Storage may be unavailable. */ }
  }

  function scheduleUpdate() {
    if (updateScheduled) return
    updateScheduled = true
    queueMicrotask(() => {
      updateScheduled = false
      if (disposed || syncing) return
      updatePreviews()
      const fileId = getFileId(groups.activeGroup.activeEditor?.resource)
      if (fileId && fileId !== selectedId) {
        selectedId = fileId
        callbacks.onSelect(fileId)
      }
      callbacks.onEditor(activeEditor())
      saveLayout()
    })
  }

  function trackGroup(group: IEditorGroup) {
    const model = group.onDidModelChange(scheduleUpdate)
    const active = group.onDidActiveEditorChange(scheduleUpdate)
    groupSubscriptions.set(group.id, {
      dispose() {
        model.dispose()
        active.dispose()
      },
    })
  }

  function open(fileId: string, group: IEditorGroup, inactive = false) {
    const entry = filesById.get(fileId)
    if (!entry) return
    return editorService.openEditor({
      resource: entry.resource,
      options: { pinned: true, inactive, preserveFocus: true, index: inactive ? group.count : undefined },
    }, group)
  }

  function openPreview(fileId: string, group: IEditorGroup, inactive = false) {
    const entry = filesById.get(fileId)
    if (!entry) return
    return group.openEditor(findPreview(group, fileId) ?? new ShaderPreviewInput(fileId, entry.name), {
      pinned: true, inactive, preserveFocus: true, index: inactive ? group.count : undefined,
    })
  }

  function addPreviewGroup(source: IEditorGroup) {
    return groups.mainPart.addGroup(source, host.clientWidth >= 760 ? GroupDirection.RIGHT : GroupDirection.UP)
  }

  async function activateShader(fileId: string, fallback: IEditorGroup) {
    const entry = filesById.get(fileId)
    if (!entry) return
    const active = groups.activeGroup
    const previous = sourceGroupIds.get(fileId)
    const group = (active.findEditors(entry.resource).length > 0 ? active : undefined)
      ?? groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE).find(candidate => candidate.findEditors(entry.resource).length > 0)
      ?? (previous === undefined ? undefined : groups.mainPart.getGroup(previous))
      ?? fallback
    await open(fileId, group)
    groups.activateGroup(group)
  }

  subscriptions.push(
    groups.onDidChangeActiveGroup(scheduleUpdate),
    editorService.onDidActiveEditorChange(scheduleUpdate),
    editorService.onDidVisibleEditorsChange(scheduleUpdate),
    codeEditorService.onCodeEditorAdd(scheduleUpdate),
    groups.onDidAddGroup(group => {
      trackGroup(group)
      scheduleUpdate()
    }),
    groups.onDidMoveGroup(scheduleUpdate),
    groups.onDidRemoveGroup(group => {
      groupSubscriptions.get(group.id)?.dispose()
      groupSubscriptions.delete(group.id)
      scheduleUpdate()
    }),
    editorService.onDidCloseEditor(event => {
      if (disposed || syncing || event.context === EditorCloseContext.MOVE) return
      if (event.editor instanceof ShaderPreviewInput) {
        const { fileId } = event.editor
        queueMicrotask(() => {
          if (!disposed && groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE).every(group => group.isEmpty)) {
            callbacks.onReopen(fileId)
          }
        })
        return
      }
      const id = getFileId(event.editor.resource)
      const entry = id && filesById.get(id)
      if (!entry || groups.groups.some(group => group.findEditors(entry.resource).length > 0)) return
      sourceGroupIds.set(entry.id, event.groupId)
      if (!callbacks.onClose(entry.id)) {
        queueMicrotask(() => {
          if (!disposed) void open(entry.id, groups.getGroup(event.groupId) ?? groups.activeGroup)
        })
      }
      scheduleUpdate()
    }),
  )
  groups.groups.forEach(trackGroup)
  window.addEventListener('pagehide', saveLayout)

  async function sync(files: readonly ShaderFile[], openFileIds: readonly string[], activeFileId: string) {
    const apply = async () => {
      if (disposed) return
      syncing = true
      const selectionChanged = selectedId !== activeFileId
      selectedId = activeFileId
      try {
        const present = new Map(files.map(file => [file.id, file]))
        const openIds = new Set(openFileIds)
        for (const group of groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE)) {
          for (const input of group.editors) {
            if (!(input instanceof ShaderPreviewInput)) continue
            const file = present.get(input.fileId)
            if (file) input.setFileName(file.name)
            else {
              await group.closeEditor(input, { preserveFocus: true })
              previewGroupIds.delete(input.fileId)
            }
          }
        }
        for (const entry of [...filesById.values()]) {
          const file = present.get(entry.id)
          const obsolete = !file || file.name !== entry.name
          if (file && openIds.has(entry.id)) {
            if (!obsolete) continue
            const replacement = await addFile(file)
            for (const group of groups.groups) {
              const inputs = group.findEditors(entry.resource)
              if (inputs.length > 0) await editorService.replaceEditors(inputs.map(input => ({
                editor: input, replacement: { resource: replacement.resource }, forceReplaceDirty: true,
              })), group)
            }
          } else {
            const toClose = groups.groups.flatMap(group =>
              group.findEditors(entry.resource).map(input => ({ groupId: group.id, editor: input })))
            if (toClose.length > 0) await editorService.closeEditors(toClose)
          }
          if (!obsolete) continue
          entry.reference.dispose()
          entry.registration.dispose()
          if (filesById.get(entry.id) === entry) filesById.delete(entry.id)
          filesByResource.delete(entry.resource.toString())
        }
        for (const file of files) {
          if (!filesById.has(file.id)) await addFile(file)
        }
        if (disposed) return
        const firstLayout = !layoutRestored
        let addPreview = false
        let freshLayout = false
        if (firstLayout) {
          const saved = loadLayout(activeFileId)
          freshLayout = !saved
          addPreview = saved?.addPreview ?? true
          if (saved) {
            const { snapshot } = saved
            groups.mainPart.applyLayout(snapshot.layout)
            const ordered = groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE)
            const restoredPreviews = new Set<string>()
            for (const [index, group] of ordered.entries()) {
              const previous = snapshot.groups[index]
              if (!previous) continue
              for (const tab of previous.tabs) {
                if (typeof tab === 'string') {
                  if (openIds.has(tab)) await open(tab, group, true)
                } else if (!restoredPreviews.has(tab.previewFileId)) {
                  restoredPreviews.add(tab.previewFileId)
                  await openPreview(tab.previewFileId, group, true)
                }
              }
              const active = previous.active
              if (typeof active === 'string') {
                if (openIds.has(active)) await open(active, group)
              } else if (active && findPreview(group, active.previewFileId)) {
                await openPreview(active.previewFileId, group)
              }
            }
            const group = ordered[snapshot.activeGroup]
            if (group) groups.activateGroup(group)
          }
        }
        const activeGroup = groups.activeGroup
        const previewActive = activeGroup.activeEditor instanceof ShaderPreviewInput
        const target = previewActive
          ? groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE).find(group => group.editors.some(input => getFileId(input.resource))) ?? activeGroup
          : activeGroup
        let openedActiveFile = false
        for (const id of openFileIds) {
          const entry = filesById.get(id)
          if (entry && !groups.groups.some(group => group.findEditors(entry.resource).length > 0)) {
            const affinity = sourceGroupIds.get(id)
            const destination = (affinity === undefined ? undefined : groups.mainPart.getGroup(affinity)) ?? target
            await open(id, destination, true)
            if (id === activeFileId) openedActiveFile = true
          }
        }
        if (openIds.has(activeFileId) && (!previewActive || (!firstLayout && (selectionChanged || openedActiveFile)))) {
          if (getFileId(activeGroup.activeEditor?.resource) !== activeFileId) await activateShader(activeFileId, target)
        }
        if (addPreview || groups.mainPart.getGroups(GroupsOrder.GRID_APPEARANCE).every(group => group.isEmpty)) {
          const priorGroup = groups.activeGroup
          const previewGroup = priorGroup.isEmpty ? priorGroup : addPreviewGroup(priorGroup)
          const width = host.clientWidth
          if (freshLayout && width >= 760 && previewGroup !== priorGroup) {
            const shaderShare = Math.min(0.4, 680 / width)
            groups.mainPart.applyLayout({
              orientation: GroupOrientation.HORIZONTAL,
              groups: [{ size: shaderShare }, { size: 1 - shaderShare }],
            })
          }
          await openPreview(activeFileId, previewGroup)
          groups.activateGroup(priorGroup)
        }
        layoutRestored = true
      } finally {
        syncing = false
        if (!disposed) {
          updatePreviews()
          callbacks.onEditor(activeEditor())
          saveLayout()
        }
      }
    }
    const pending = syncQueue.then(apply)
    syncQueue = pending.catch(() => {})
    return pending
  }

  function dispose() {
    if (disposed) return
    saveLayout()
    disposed = true
    for (const [fileId, previewHost] of previewHosts) {
      callbacks.onPreviewHost(fileId, undefined)
      previewHost.remove()
    }
    callbacks.onPreviewOpenChange([])
    callbacks.onVisiblePreviews([])
    parking.remove()
    window.removeEventListener('pagehide', saveLayout)
    subscriptions.forEach(subscription => subscription.dispose())
    groupSubscriptions.forEach(subscription => subscription.dispose())
    if (currentMount === dispose) currentMount = undefined
  }
  currentMount = dispose

  return {
    sync, editors, dispose,
    renderVisibleEditors() {
      for (const editor of editors()) {
        const container = editor.getContainerDomNode()
        if (container.clientWidth && container.clientHeight) editor.render(true)
      }
    },
    model: (fileId: string) => filesById.get(fileId)?.model,
    focus: () => activeEditor()?.focus(),
    async focusShader(fileId: string) {
      await syncQueue
      if (disposed) return
      await activateShader(fileId, groups.activeGroup)
      activeEditor()?.focus()
    },
    async showPreview(fileId: string) {
      await syncQueue
      const entry = filesById.get(fileId)
      if (disposed || !entry) return
      const main = groups.mainPart
      const ordered = main.getGroups(GroupsOrder.GRID_APPEARANCE)
      const source = (main.activeGroup.findEditors(entry.resource).length > 0 ? main.activeGroup : undefined)
        ?? ordered.find(group => group.findEditors(entry.resource).length > 0)
        ?? main.activeGroup
      const direction = host.clientWidth >= 760 ? GroupDirection.RIGHT : GroupDirection.UP
      const adjacent = main.findGroup({ direction }, source)
      const previous = previewGroupIds.get(fileId)
      const group = ordered.find(candidate => findPreview(candidate, fileId))
        ?? (previous === undefined ? undefined : main.getGroup(previous))
        ?? (source.isEmpty ? source : adjacent ?? addPreviewGroup(source))
      await openPreview(fileId, group)
      group.focus()
    },
  }
}

export type ShaderWorkbench = Awaited<ReturnType<typeof mountShaderWorkbench>>
