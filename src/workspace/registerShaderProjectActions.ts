import { getService, IContextKeyService, IEditorGroupsService, IQuickInputService } from '@codingame/monaco-vscode-api'
import { URI } from '@codingame/monaco-vscode-api/vscode/vs/base/common/uri'
import { DisposableStore } from '@codingame/monaco-vscode-api/vscode/vs/base/common/lifecycle'
import { MenuId, MenuRegistry } from '@codingame/monaco-vscode-api/vscode/vs/platform/actions/common/actions'
import {
  CommandsRegistry,
  type ICommandHandler,
} from '@codingame/monaco-vscode-api/vscode/vs/platform/commands/common/commands'
import { ContextKeyExpr } from '@codingame/monaco-vscode-api/vscode/vs/platform/contextkey/common/contextkey'
import type { ShaderFile, ShaderFileKind } from './shaderProject'
import { getPreviewFileId, shaderPreviewScheme } from './shaderPreviewEditor'
import { getFileId } from './shaderWorkbench'
import { wgslLanguageId } from './wgslTheme'

type ProjectActions = {
  getFiles: () => readonly ShaderFile[]
  getActiveFileId: () => string
  onNewFile: (kind: ShaderFileKind) => void
  onRenameFile: (id: string, name: string) => void
  onReopenFile: (id: string) => void
  onToggleVim: () => void
  onShowPreview: (id: string) => Promise<void>
  onRevealSource: (id: string) => Promise<void>
}

const editorIsWgsl = ContextKeyExpr.equals('resourceLangId', wgslLanguageId)
const editorIsPreview = ContextKeyExpr.equals('resourceScheme', shaderPreviewScheme)
const editorIsProject = ContextKeyExpr.or(editorIsWgsl, editorIsPreview)
const vimKey = 'earthWgsl.vimEnabled'
const closedFilesKey = 'earthWgsl.hasClosedFiles'

function wgslFileName(value: string) {
  const name = value.trim()
  return /\.wgsl$/i.test(name) ? name : `${name}.wgsl`
}

function resourceForAction(groups: IEditorGroupsService, resource: unknown, commandContext: unknown) {
  const context = commandContext ?? resource
  const { groupId, editorIndex } = (context ?? {}) as { groupId?: number; editorIndex?: number }
  const group = groupId === undefined ? groups.activeGroup : groups.getGroup(groupId)
  // The toolbar can pass the resource of a previously focused tab.
  if (groupId !== undefined && editorIndex === undefined) return group?.activeEditor?.resource
  if (URI.isUri(resource)) return resource
  return editorIndex === undefined ? group?.activeEditor?.resource : group?.getEditorByIndex(editorIndex)?.resource
}

export async function registerShaderProjectActions(project: ProjectActions) {
  const [context, quickInput] = await Promise.all([getService(IContextKeyService), getService(IQuickInputService)])
  const vim = context.createKey<boolean>(vimKey, false)
  const closedFiles = context.createKey<boolean>(closedFilesKey, false)
  const store = new DisposableStore()

  function register(
    id: string,
    title: string,
    order: number,
    run: ICommandHandler,
    when = editorIsWgsl,
    toggled?: ReturnType<typeof ContextKeyExpr.has>,
  ) {
    const command = { id: `earthWgsl.${id}`, title, toggled }
    store.add(CommandsRegistry.registerCommand(command.id, run))
    store.add(MenuRegistry.appendMenuItem(MenuId.EditorTitle, { command, when, group: '2_wgsl', order }))
    return command
  }

  register('newFile', 'New WGSL Shader', 1, () => project.onNewFile('shader'), editorIsProject)
  register('newHelper', 'New WGSL Helper', 2, () => project.onNewFile('helper'), editorIsProject)
  register('renameFile', 'Rename WGSL File…', 3, async (accessor, resource, commandContext) => {
    const fileId =
      getFileId(resourceForAction(accessor.get(IEditorGroupsService), resource, commandContext)) ??
      project.getActiveFileId()
    const file = project.getFiles().find(file => file.id === fileId)
    if (!file) return
    const value = await quickInput.input({
      title: 'Rename WGSL File',
      value: file.name,
      valueSelection: [0, file.name.replace(/\.wgsl$/i, '').length],
      validateInput: async value => {
        const name = wgslFileName(value)
        if (name.toLowerCase() === '.wgsl' || /[\\/]/.test(name)) return 'Enter a file name without a path.'
        if (name.length > 80) return 'Enter a file name with at most 80 characters.'
        if (project.getFiles().some(other => other.id !== file.id && other.name.toLowerCase() === name.toLowerCase())) {
          return 'A file with that name already exists.'
        }
      },
    })
    if (value !== undefined) project.onRenameFile(file.id, wgslFileName(value))
  })
  register(
    'reopenFile',
    'Reopen WGSL File…',
    4,
    async () => {
      const files = project
        .getFiles()
        .filter(file => file.closed)
        .map(file => ({ label: file.name, id: file.id }))
      const selected = await quickInput.pick(files, { title: 'Reopen WGSL File', placeHolder: 'Select a closed file' })
      if (selected) project.onReopenFile(selected.id)
    },
    ContextKeyExpr.and(editorIsProject, ContextKeyExpr.has(closedFilesKey)),
  )
  register('toggleVim', 'Vim Keybindings', 5, () => project.onToggleVim(), editorIsWgsl, ContextKeyExpr.has(vimKey))
  const showPreview = register('showPreview', 'Open Preview to the Side', 6, (accessor, resource, commandContext) => {
    const id = getFileId(resourceForAction(accessor.get(IEditorGroupsService), resource, commandContext))
    if (id) return project.onShowPreview(id)
  })
  const revealSource = register(
    'revealSource',
    'Reveal Shader Source',
    6,
    (accessor, resource, commandContext) => {
      const id = getPreviewFileId(resourceForAction(accessor.get(IEditorGroupsService), resource, commandContext))
      if (id) return project.onRevealSource(id)
    },
    editorIsPreview,
  )
  for (const menu of [MenuId.EditorTitleContext, MenuId.CommandPalette]) {
    store.add(MenuRegistry.appendMenuItem(menu, { command: showPreview, when: editorIsWgsl, group: '2_wgsl' }))
    store.add(MenuRegistry.appendMenuItem(menu, { command: revealSource, when: editorIsPreview, group: '2_wgsl' }))
  }
  store.add(
    MenuRegistry.appendMenuItem(MenuId.EditorContext, {
      command: showPreview,
      when: editorIsWgsl,
      group: 'earthQuery',
      order: 4,
    }),
  )

  return {
    update(vimEnabled: boolean, hasClosedFiles: boolean) {
      vim.set(vimEnabled)
      closedFiles.set(hasClosedFiles)
    },
    dispose() {
      store.dispose()
      vim.reset()
      closedFiles.reset()
    },
  }
}
