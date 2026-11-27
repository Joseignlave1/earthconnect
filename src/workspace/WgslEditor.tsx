import { useEffect, useLayoutEffect, useRef } from 'react'
import * as monaco from 'monaco-editor'
import type { CodeEditorWidget } from '@codingame/monaco-vscode-api/vscode/vs/editor/browser/widget/codeEditor/codeEditorWidget'
import { OverlayWidgetPositionPreference } from '@codingame/monaco-vscode-api/vscode/vs/editor/browser/editorBrowser'
import { MenuId, MenuRegistry } from '@codingame/monaco-vscode-api/vscode/vs/platform/actions/common/actions'
import { CommandsRegistry } from '@codingame/monaco-vscode-api/vscode/vs/platform/commands/common/commands'
import { ContextKeyExpr } from '@codingame/monaco-vscode-api/vscode/vs/platform/contextkey/common/contextkey'
import {
  KeybindingWeight,
  KeybindingsRegistry,
} from '@codingame/monaco-vscode-api/vscode/vs/platform/keybinding/common/keybindingsRegistry'
import { registerShaderProjectActions } from './registerShaderProjectActions'
import { mountShaderWorkbench, type ShaderWorkbench } from './shaderWorkbench'
import type { CompletionItem, Hover, Position, Range } from 'vscode-languageserver-types'
import {
  assembleShaderProject,
  findShaderLocation,
  selectShaderModule,
  type ShaderFile,
  type ShaderFileKind,
} from './shaderProject'
import { EARTH_CELL_FIELDS } from '../earth-score/custom-shader.ts'
import { wgslKeywords as keywords, wgslLanguageId as language, wgslTypes as types } from './wgslTheme'
import { startWgslLanguageServer } from './wgslLanguageServer'
import { registerSharedWgslAnalyzer } from './planetaryShaderAnalysis'
import './WgslEditor.css'

export type WgslDiagnostic = {
  message: string
  severity: 'error' | 'warning' | 'info'
  line: number
  column: number
  length?: number
  fileId?: string
}

type Props = {
  onPreviewHost: (id: string, host: HTMLElement | undefined) => void
  onPreviewOpenChange: (ids: readonly string[]) => void
  onVisiblePreviews: (ids: readonly string[]) => void
  takePrewarmedHost?: (id: string) => HTMLElement | undefined
  onLayoutReady: () => void
  analyzerEnabled?: boolean
  onAnalyzerReady?: () => void
  onStartError?: () => void
  files: readonly ShaderFile[]
  activeFileId: string
  onActiveFileChange: (id: string) => void
  onCloseFile: (id: string) => boolean
  onNewFile: (kind: ShaderFileKind) => void
  onRenameFile: (id: string, name: string) => void
  onReopenFile: (id: string) => void
  onToggleVim: () => void
  onChange: (fileId: string, source: string) => void
  onRun: (fileId: string, source: string) => void
  vim: boolean
  insertion?: { id: number; fileId: string; text: string }
  onInserted?: (id: number) => void
  diagnostics?: readonly WgslDiagnostic[]
  prelude?: string
  label?: string
}

const analyzerOwner = 'wgsl-analyzer'
const gpuOwner = 'webgpu'
const builtins: Readonly<Record<string, string>> = {
  abs: 'Returns the absolute value.',
  clamp: 'Clamps a value between a lower and upper bound.',
  distance: 'Returns the distance between two points.',
  dot: 'Returns the dot product of two vectors.',
  exp: 'Raises e to the power of its argument.',
  length: 'Returns the magnitude of a vector.',
  max: 'Returns the larger of two values.',
  min: 'Returns the smaller of two values.',
  mix: 'Interpolates linearly between two values.',
  normalize: 'Returns a unit vector in the same direction.',
  pow: 'Raises the first argument to the power of the second.',
  select: 'Returns the first value when the condition is false, or the second when true.',
  smoothstep: 'Interpolates smoothly between two edge values.',
  sqrt: 'Returns the square root of a nonnegative value.',
  has_data: 'Returns true when this field contains a usable measurement.',
  moon_point: 'Return typed Moon measurements at a constant latitude and longitude in degrees.',
  mars_point: 'Return typed Mars measurements at a constant latitude and longitude in degrees.',
  NO_SCORE: 'Return this value to exclude the current cell from the map.',
}

function vimEditor(editor: monaco.editor.ICodeEditor) {
  const context = (editor as CodeEditorWidget).contextKeyService
  const createContextKey = context.createKey.bind(context)
  // Monaco Vim only needs one standalone-only helper from the editor.
  return new Proxy(editor, {
    get(target, property) {
      if (property === 'createContextKey') return createContextKey
      const value = Reflect.get(target, property)
      return typeof value === 'function' ? value.bind(target) : value
    },
  }) as Parameters<typeof import('monaco-vim').initVimMode>[0]
}

function virtualSource(prelude: string, code: string) {
  return prelude ? code + '\n' + prelude : code
}

function editorRange(range: Range, firstLine = 1) {
  return new monaco.Range(
    range.start.line + 2 - firstLine,
    range.start.character + 1,
    range.end.line + 2 - firstLine,
    range.end.character + 1,
  )
}

function hoverContents(hover: Hover): monaco.IMarkdownString[] {
  const entries = Array.isArray(hover.contents) ? hover.contents : [hover.contents]
  return entries.map(entry => {
    if (typeof entry === 'string') return { value: entry }
    if ('language' in entry) return { value: '\x60\x60\x60' + entry.language + '\n' + entry.value + '\n\x60\x60\x60' }
    if (entry.kind === 'plaintext') return { value: entry.value.replace(/[\\`*_[\]{}<>]/g, '\\$&') }
    return { value: entry.value }
  })
}

function editorSuggestion(
  item: CompletionItem,
  defaultRange: monaco.IRange,
  firstLine: number,
  lineCount: number,
): monaco.languages.CompletionItem {
  const edit = item.textEdit
  const localRange = (range: Range) => {
    const local = editorRange(range, firstLine)
    return local.startLineNumber >= 1 && local.endLineNumber <= lineCount ? local : undefined
  }
  const insert = edit && 'insert' in edit ? localRange(edit.insert) : undefined
  const replace = edit && 'replace' in edit ? localRange(edit.replace) : undefined
  const range = edit && ('range' in edit ? localRange(edit.range) : insert && replace && { insert, replace })
  const kinds = monaco.languages.CompletionItemKind
  const kindsByProtocol: Partial<Record<number, monaco.languages.CompletionItemKind>> = {
    2: kinds.Method,
    3: kinds.Function,
    5: kinds.Field,
    6: kinds.Variable,
    7: kinds.Class,
    14: kinds.Keyword,
    21: kinds.Constant,
    22: kinds.Struct,
    25: kinds.TypeParameter,
  }
  return {
    label: item.label,
    kind: kindsByProtocol[item.kind ?? 0] ?? kinds.Text,
    insertText: edit?.newText ?? item.insertText ?? item.label,
    range: range ?? defaultRange,
    detail: item.detail,
    documentation:
      typeof item.documentation === 'string'
        ? item.documentation
        : item.documentation && { value: item.documentation.value },
    filterText: item.filterText,
    sortText: item.sortText,
  }
}

const earthCellFieldLookup = new Map(
  EARTH_CELL_FIELDS.map(field => [field.name, { ...field, doc: `${field.description} (${field.unit})` }]),
)

function fallbackSuggestions(model: monaco.editor.ITextModel, position: monaco.Position, range: monaco.IRange) {
  const kind = monaco.languages.CompletionItemKind
  const before = model.getLineContent(position.lineNumber).slice(0, position.column - 1)
  if (/\.\s*\w*$/.test(before)) {
    return Array.from(earthCellFieldLookup.values(), ({ name, type, doc }) => ({
      label: name,
      insertText: name,
      kind: kind.Field,
      range,
      detail: type,
      documentation: doc,
    }))
  }
  return [
    ...keywords.map(name => ({ label: name, insertText: name, kind: kind.Keyword, range })),
    ...types.map(name => ({ label: name, insertText: name, kind: kind.Class, range })),
    ...Object.entries(builtins).map(([name, doc]) => ({
      label: name,
      insertText: name,
      kind: name === 'NO_SCORE' ? kind.Constant : kind.Function,
      range,
      documentation: doc,
    })),
  ]
}

// The upstream server can return no hover information. Keep common shader
// functions, declarations and EarthCell field documentation available locally.
function localHover(model: monaco.editor.ITextModel, position: monaco.Position, program: string) {
  const token = model.getWordAtPosition(position)
  if (!token) return null
  const name = token.word
  const field = earthCellFieldLookup.get(name)
  if (field) {
    return {
      contents: [{ value: '`' + name + ': ' + field.type + '`' }, { value: field.doc }],
    }
  }
  if (Object.hasOwn(builtins, name)) return { contents: [{ value: builtins[name] }] }
  if (!/^[a-zA-Z_]\w*$/.test(name)) return null
  const declaration = program.match(new RegExp('\\b(?:let|const|var|fn|struct)\\s+' + name + '\\b[^;{\\n]*'))
  return declaration ? { contents: [{ value: '\x60\x60\x60wgsl\n' + declaration[0].trim() + '\n\x60\x60\x60' }] } : null
}

const severity = {
  error: monaco.MarkerSeverity.Error,
  warning: monaco.MarkerSeverity.Warning,
  info: monaco.MarkerSeverity.Info,
}

function setGpuMarkers(model: monaco.editor.ITextModel, diagnostics: readonly WgslDiagnostic[] = []) {
  monaco.editor.setModelMarkers(
    model,
    gpuOwner,
    diagnostics.map(item => ({
      ...model.validateRange(
        item.line > 0
          ? new monaco.Range(item.line, item.column, item.line, item.column + Math.max(1, item.length ?? 1))
          : new monaco.Range(1, 1, 1, model.getLineMaxColumn(1)),
      ),
      message: item.message,
      severity: severity[item.severity],
      source: 'WebGPU',
    })),
  )
}

type FileModel = {
  model: monaco.editor.ITextModel
  changes: monaco.IDisposable
  source: string
  pendingEchoes: string[]
}

export default function WgslEditor(props: Props) {
  const {
    files,
    activeFileId,
    diagnostics,
    insertion,
    prelude = '',
    label = 'WGSL score function',
    vim,
    analyzerEnabled = true,
  } = props
  const current = { ...props, prelude, label, analyzerEnabled }
  const panel = useRef<HTMLDivElement>(null)
  const container = useRef<HTMLDivElement>(null)
  const insertedId = useRef<number | undefined>(undefined)
  const instance = useRef<{
    reconcile: () => void
    updateMarkers: () => void
    updateEditors: () => void
    maybeStartAnalyzer: () => void
  }>(null)
  const latest = useRef(current)
  useLayoutEffect(() => {
    latest.current = current
  })

  useEffect(() => {
    const host = container.current
    const editorPanel = panel.current
    if (!host || !editorPanel) return
    const controller = new AbortController()
    const analyzerController = new AbortController()
    const models = new Map<string, FileModel>()
    const modelIds = new Map<monaco.editor.ITextModel, string>()
    const previewHosts = new Map<string, HTMLElement>()
    const editorBindings = new Map<
      monaco.editor.ICodeEditor,
      {
        dispose: () => void
      }
    >()
    const editorLanguage = ContextKeyExpr.equals('editorLangId', language)
    const runCommand = 'earthQuery.run'
    const resources: monaco.IDisposable[] = []
    let workbench: ShaderWorkbench | undefined
    let projectActions: Awaited<ReturnType<typeof registerShaderProjectActions>> | undefined
    let activeEditor: monaco.editor.ICodeEditor | undefined
    let focusingForInsertion: { id: number; fileId: string } | undefined
    let activeVimEditor: monaco.editor.ICodeEditor | undefined
    let activeVimModel: monaco.editor.ITextModel | null | undefined
    let stopVim: (() => void) | undefined
    let replacing = false
    let reconciling = false
    let needsReconcile = false
    let syncedProject: { files: readonly ShaderFile[]; activeFileId: string } | undefined
    let server: Awaited<ReturnType<typeof startWgslLanguageServer>> | undefined
    let unregisterAnalyzer: (() => void) | undefined
    let startupTimer: ReturnType<typeof window.setTimeout> | undefined
    let program = assembleShaderProject([])
    let virtualProgram = ''
    let sentSource = ''
    let sentProgram = program

    function assemble(fileId: string) {
      return assembleShaderProject(
        selectShaderModule(latest.current.files, fileId).map(file => ({
          ...file,
          source: models.get(file.id)?.model.getValue() ?? file.source,
        })),
      )
    }

    function runFile(fileId: string) {
      latest.current.onRun(fileId, assemble(fileId).source)
    }

    function runInEditor(editor = activeEditor) {
      const model = editor?.getModel()
      const fileId = (model && modelIds.get(model)) || latest.current.activeFileId
      runFile(fileId)
      if (!previewHosts.get(fileId)?.parentElement?.classList.contains('shader-preview-pane')) {
        void workbench?.showPreview(fileId).catch(error => {
          if (!controller.signal.aborted) console.error('Could not open the shader preview', error)
        })
      }
    }

    function runFromPreview(event: KeyboardEvent) {
      if (
        event.defaultPrevented ||
        !(event.ctrlKey || event.metaKey) ||
        event.key !== 'Enter' ||
        !(event.target instanceof Node)
      )
        return
      const target = event.target
      const fileId = [...previewHosts].find(
        ([, preview]) =>
          preview.contains(target) ||
          (preview.parentElement === target && preview.parentElement.classList.contains('shader-preview-pane')),
      )?.[0]
      if (!fileId) return
      event.preventDefault()
      event.stopPropagation()
      runFile(fileId)
    }

    host.addEventListener('keydown', runFromPreview)

    function registerCommands() {
      resources.push(
        CommandsRegistry.registerCommand(runCommand, () => runInEditor()),
        MenuRegistry.appendMenuItem(MenuId.EditorContext, {
          command: { id: runCommand, title: 'Run Earth query' },
          when: editorLanguage,
          group: 'earthQuery',
          order: 1,
        }),
      )
      resources.push(
        KeybindingsRegistry.registerKeybindingRule({
          id: runCommand,
          primary: monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,
          weight: KeybindingWeight.EditorContrib,
          when: ContextKeyExpr.and(editorLanguage, ContextKeyExpr.has('editorTextFocus')),
        }),
      )
    }

    function applyInsertion(focusShader = true) {
      const { insertion, onInserted, files } = latest.current
      if (!insertion || insertedId.current === insertion.id) return
      const { fileId } = insertion
      const model = activeEditor?.getModel()
      if (!model || modelIds.get(model) !== fileId) {
        const file = files.find(file => file.id === fileId)
        if (!focusShader || !workbench || !file || focusingForInsertion?.id === insertion.id) return
        const request = { id: insertion.id, fileId }
        focusingForInsertion = request
        if (file.closed) latest.current.onReopenFile(fileId)
        void workbench
          .focusShader(fileId)
          .then(() => {
            if (!controller.signal.aborted && focusingForInsertion === request) applyInsertion(false)
          })
          .catch(error => {
            if (!controller.signal.aborted) console.error('Could not focus the shader for point insertion', error)
          })
          .finally(() => {
            if (focusingForInsertion === request) focusingForInsertion = undefined
          })
        return
      }
      const selection = activeEditor?.getSelection()
      if (!activeEditor || !selection) return
      activeEditor.pushUndoStop()
      if (
        activeEditor.executeEdits('earth-insert-point', [
          { range: selection, text: insertion.text, forceMoveMarkers: true },
        ])
      ) {
        activeEditor.pushUndoStop()
        insertedId.current = insertion.id
        activeEditor.focus()
        onInserted?.(insertion.id)
      }
    }

    function updateEditors() {
      projectActions?.update(
        latest.current.vim,
        latest.current.files.some(file => file.closed),
      )
      const editors = new Set(workbench?.editors() ?? [])
      if (activeEditor) editors.add(activeEditor)
      for (const [editor, binding] of editorBindings) {
        if (editors.has(editor)) continue
        binding.dispose()
        editorBindings.delete(editor)
      }
      for (const editor of editors) {
        editor.updateOptions({ ariaLabel: latest.current.label })
        let binding = editorBindings.get(editor)
        if (!binding) {
          const keydown = editor.onKeyDown(event => {
            // Handle the shortcut in the editor before Vim or workbench keybindings consume it.
            if (
              event.browserEvent.defaultPrevented ||
              !(event.ctrlKey || event.metaKey) ||
              event.keyCode !== monaco.KeyCode.Enter
            )
              return
            event.preventDefault()
            event.stopPropagation()
            runInEditor(editor)
          })
          const modelChanged = editor.onDidChangeModel(() => {
            if (editor === activeEditor) updateEditors()
          })
          binding = {
            dispose() {
              keydown.dispose()
              modelChanged.dispose()
            },
          }
          editorBindings.set(editor, binding)
        }
      }

      const editor = latest.current.vim ? activeEditor : undefined
      const editorModel = editor?.getModel()
      if (editor !== activeVimEditor || editorModel !== activeVimModel) {
        stopVim?.()
        activeVimEditor = editor
        activeVimModel = editorModel
        if (editor && editorModel && editorPanel) {
          let canceled = false
          let adapter: import('monaco-vim').VimAdapterInstance | undefined
          const status = document.createElement('div')
          status.className = 'wgsl-vim-status'
          status.setAttribute('aria-label', 'Vim command line')
          const statusWidget: monaco.editor.IOverlayWidget = {
            getId: () => 'earth-wgsl-vim',
            getDomNode: () => status,
            getPosition: () => ({ preference: OverlayWidgetPositionPreference.BOTTOM_RIGHT_CORNER }),
          }
          editor.addOverlayWidget(statusWidget)
          editorPanel.dataset.vimMode = 'loading'
          void import('monaco-vim')
            .then(({ initVimMode }) => {
              if (canceled) return
              adapter = initVimMode(vimEditor(editor), status)
              editorPanel.dataset.vimMode = 'active'
              editor.layoutOverlayWidget(statusWidget)
            })
            .catch(() => {
              if (canceled) return
              editorPanel.dataset.vimMode = 'unavailable'
              status.textContent = 'Vim unavailable'
              editor.layoutOverlayWidget(statusWidget)
            })
          stopVim = () => {
            if (canceled) return
            canceled = true
            const focus = document.activeElement
            adapter?.dispose()
            if (focus instanceof HTMLElement && !host?.contains(focus) && !status.contains(focus)) {
              focus.focus({ preventScroll: true })
            }
            editor.removeOverlayWidget(statusWidget)
            delete editorPanel.dataset.vimMode
          }
        }
      }
      applyInsertion()
    }

    function clearAnalyzerMarkers() {
      for (const { model } of models.values()) monaco.editor.setModelMarkers(model, analyzerOwner, [])
    }

    function sync() {
      const next = assemble(latest.current.activeFileId)
      if (
        next.source !== program.source ||
        next.segments.length !== program.segments.length ||
        next.segments.some(
          (segment, index) =>
            segment.fileId !== program.segments[index].fileId ||
            segment.lineCount !== program.segments[index].lineCount,
        )
      ) {
        program = next
      }
      virtualProgram = virtualSource(latest.current.prelude, program.source)
      if (!server || (sentSource === virtualProgram && sentProgram === program)) return
      sentSource = virtualProgram
      sentProgram = program
      clearAnalyzerMarkers()
      server.change(virtualProgram)
    }

    function updateMarkers() {
      const entries = latest.current.diagnostics ?? []
      const defaultFile = latest.current.activeFileId
      for (const [id, { model }] of models) {
        setGpuMarkers(
          model,
          entries.filter(item => (item.fileId ?? defaultFile) === id),
        )
      }
    }

    function reconcileModels() {
      replacing = true
      const wanted = new Set<string>()
      for (const file of latest.current.files) {
        const model = workbench?.model(file.id)
        if (!model) continue
        wanted.add(file.id)
        const source = file.source.replace(/\r\n?/g, '\n')
        const previous = models.get(file.id)
        if (previous?.model === model) {
          // React can acknowledge an earlier keystroke after Monaco has already
          // applied the next one. Never reset the text or selection for that echo.
          if (previous.source !== source) {
            previous.source = source
            const echo = previous.pendingEchoes.lastIndexOf(source)
            if (echo >= 0) previous.pendingEchoes.splice(0, echo + 1)
            else {
              previous.pendingEchoes.length = 0
              if (previous.model.getValue() !== source) previous.model.setValue(source)
            }
          }
          continue
        }
        previous?.changes.dispose()
        if (previous) modelIds.delete(previous.model)
        model.updateOptions({ tabSize: 2, insertSpaces: true })
        const pendingEchoes: string[] = []
        models.set(file.id, {
          model,
          source,
          pendingEchoes,
          changes: model.onDidChangeContent(() => {
            if (replacing) return
            const value = model.getValue()
            pendingEchoes.push(value)
            if (pendingEchoes.length > 64) pendingEchoes.shift()
            latest.current.onChange(file.id, value)
            sync()
          }),
        })
        modelIds.set(model, file.id)
      }
      replacing = false

      for (const [id, item] of models) {
        if (wanted.has(id)) continue
        item.changes.dispose()
        models.delete(id)
        modelIds.delete(item.model)
      }
      updateMarkers()
      sync()
    }

    let canReconcile = false
    async function reconcile(initial = false) {
      reconcileModels()
      const bench = workbench
      if (!bench || !canReconcile) return
      if (reconciling) {
        needsReconcile = true
        return
      }
      reconciling = true
      try {
        do {
          needsReconcile = false
          const { files, activeFileId } = latest.current
          if (
            syncedProject?.activeFileId === activeFileId &&
            files.length === syncedProject.files.length &&
            files.every((file, index) => {
              const synced = syncedProject?.files[index]
              return file.id === synced?.id && file.name === synced.name && file.closed === synced.closed
            })
          ) {
            updateEditors()
            continue
          }
          await bench.sync(
            files,
            files.filter(file => !file.closed).map(file => file.id),
            activeFileId,
          )
          if (controller.signal.aborted) return
          syncedProject = { files, activeFileId }
          reconcileModels()
          updateEditors()
        } while (needsReconcile)
      } catch (error) {
        if (initial) throw error
        if (!controller.signal.aborted) console.error('Could not synchronize shader editors', error)
      } finally {
        reconciling = false
      }
    }

    function lspPosition(position: monaco.Position, firstLine: number): Position {
      return { line: position.lineNumber + firstLine - 2, character: position.column - 1 }
    }

    const fallback = () => {
      if (analyzerController.signal.aborted) return
      unregisterAnalyzer?.()
      unregisterAnalyzer = undefined
      server?.dispose()
      server = undefined
      editorPanel.dataset.languageService = 'local'
      clearAnalyzerMarkers()
    }

    function configureLanguage() {
      resources.push(
        monaco.languages.registerCompletionItemProvider(language, {
          triggerCharacters: ['.'],
          async provideCompletionItems(model, position, _context, token) {
            const snapshot = program
            const fileId = modelIds.get(model)
            if (!fileId) return null
            const segment = snapshot.segments.find(part => part.fileId === fileId)
            const word = model.getWordUntilPosition(position)
            const range = new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn)
            if (server && segment) {
              const text = virtualProgram
              const result = await server.complete(lspPosition(position, segment.startLine)).catch(() => null)
              if (token.isCancellationRequested || snapshot !== program || text !== virtualProgram) return null
              const entries = Array.isArray(result) ? result : result?.items
              if (entries?.length)
                return {
                  suggestions: entries.map(item => editorSuggestion(item, range, segment.startLine, segment.lineCount)),
                  incomplete: !Array.isArray(result) && result?.isIncomplete,
                }
            }
            return { suggestions: fallbackSuggestions(model, position, range) }
          },
        }),
      )
      resources.push(
        monaco.languages.registerHoverProvider(language, {
          async provideHover(model, position, token) {
            const snapshot = program
            const fileId = modelIds.get(model)
            if (!fileId) return null
            const segment = snapshot.segments.find(part => part.fileId === fileId)
            if (server && segment) {
              const text = virtualProgram
              const result = await server.hover(lspPosition(position, segment.startLine)).catch(() => null)
              if (token.isCancellationRequested || snapshot !== program || text !== virtualProgram) return null
              if (result) {
                const range = result.range && model.validateRange(editorRange(result.range, segment.startLine))
                return { contents: hoverContents(result), range }
              }
            }
            return localHover(
              model,
              position,
              segment ? virtualProgram : virtualSource(latest.current.prelude, assemble(fileId).source),
            )
          },
        }),
      )
    }

    function startAnalyzer(editorPanel: HTMLDivElement) {
      if (globalThis.crossOriginIsolated) {
        editorPanel.dataset.languageService = 'starting'
        startupTimer = window.setTimeout(() => {
          fallback()
          analyzerController.abort()
        }, 15_000)
        void startWgslLanguageServer(
          () => {
            sentProgram = program
            sentSource = virtualProgram
            return virtualProgram
          },
          ({ diagnostics: entries }) => {
            if (analyzerController.signal.aborted) return
            const markers = new Map<string, monaco.editor.IMarkerData[]>()
            for (const item of entries) {
              const location = findShaderLocation(sentProgram.segments, item.range.start.line + 1)
              const entry = location && models.get(location.fileId)
              if (!entry) continue
              const firstLine = item.range.start.line + 2 - location.line
              const list = markers.get(location.fileId) ?? []
              list.push({
                ...entry.model.validateRange(editorRange(item.range, firstLine)),
                message: typeof item.message === 'string' ? item.message : item.message.value,
                severity:
                  item.severity === 2
                    ? monaco.MarkerSeverity.Warning
                    : item.severity === 3
                      ? monaco.MarkerSeverity.Info
                      : item.severity === 4
                        ? monaco.MarkerSeverity.Hint
                        : monaco.MarkerSeverity.Error,
                source: item.source ?? 'wgsl-analyzer',
              })
              markers.set(location.fileId, list)
            }
            for (const [id, { model }] of models)
              monaco.editor.setModelMarkers(model, analyzerOwner, markers.get(id) ?? [])
          },
          fallback,
          analyzerController.signal,
        )
          .then(ready => {
            if (analyzerController.signal.aborted) return
            server = ready
            unregisterAnalyzer = registerSharedWgslAnalyzer(ready.syntaxTreeFor)
            editorPanel.dataset.languageService = 'wgsl-analyzer'
            sync()
            latest.current.onAnalyzerReady?.()
          })
          .catch(fallback)
          .finally(() => window.clearTimeout(startupTimer))
      } else {
        editorPanel.dataset.languageService = 'local'
      }
    }

    let analyzerStarted = false
    let workbenchReady = false
    function maybeStartAnalyzer() {
      if (analyzerStarted || !workbenchReady || !latest.current.analyzerEnabled || controller.signal.aborted) return
      analyzerStarted = true
      startAnalyzer(editorPanel!)
    }

    instance.current = {
      reconcile: () => {
        void reconcile()
      },
      updateMarkers,
      updateEditors,
      maybeStartAnalyzer,
    }
    void mountShaderWorkbench(
      host,
      {
        onSelect: id => {
          if (!controller.signal.aborted) latest.current.onActiveFileChange(id)
        },
        onClose: id => !controller.signal.aborted && latest.current.onCloseFile(id),
        onReopen: id => {
          if (!controller.signal.aborted) latest.current.onReopenFile(id)
        },
        onEditor: editor => {
          if (controller.signal.aborted) return
          activeEditor = editor
          if (workbench) updateEditors()
        },
        onPreviewHost: (id, preview) => {
          if (controller.signal.aborted) return
          if (preview) previewHosts.set(id, preview)
          else previewHosts.delete(id)
          latest.current.onPreviewHost(id, preview)
        },
        onPreviewOpenChange: ids => {
          if (!controller.signal.aborted) latest.current.onPreviewOpenChange(ids)
        },
        onVisiblePreviews: ids => {
          if (!controller.signal.aborted) latest.current.onVisiblePreviews(ids)
        },
        takePrewarmedHost: id => (controller.signal.aborted ? undefined : latest.current.takePrewarmedHost?.(id)),
      },
      controller.signal,
    )
      .then(async bench => {
        if (controller.signal.aborted) {
          bench.dispose()
          return
        }
        workbench = bench
        configureLanguage()
        registerCommands()
        const actions = await registerShaderProjectActions({
          getFiles: () => latest.current.files,
          getActiveFileId: () => latest.current.activeFileId,
          onNewFile: kind => latest.current.onNewFile(kind),
          onRenameFile: (id, name) => latest.current.onRenameFile(id, name),
          onReopenFile: id => latest.current.onReopenFile(id),
          onToggleVim: () => latest.current.onToggleVim(),
          onShowPreview: id => bench.showPreview(id),
          onRevealSource: async id => {
            if (latest.current.files.find(file => file.id === id)?.closed) latest.current.onReopenFile(id)
            await bench.focusShader(id)
          },
        })
        if (controller.signal.aborted) {
          actions.dispose()
          return
        }
        projectActions = actions
        resources.push(actions)
        canReconcile = true
        await reconcile(true)
        if (!controller.signal.aborted) {
          bench.renderVisibleEditors()
          workbenchReady = true
          latest.current.onLayoutReady()
          maybeStartAnalyzer()
        }
      })
      .catch(error => {
        if (controller.signal.aborted) return
        editorPanel.dataset.languageService = 'unavailable'
        host.textContent = 'The shader editor could not start.'
        console.error('Could not start the shader editor', error)
        latest.current.onStartError?.()
      })

    return () => {
      controller.abort()
      unregisterAnalyzer?.()
      unregisterAnalyzer = undefined
      analyzerController.abort()
      host.removeEventListener('keydown', runFromPreview)
      window.clearTimeout(startupTimer)
      for (const resource of resources) resource.dispose()
      stopVim?.()
      for (const binding of editorBindings.values()) binding.dispose()
      for (const { model, changes } of models.values()) {
        changes.dispose()
        monaco.editor.setModelMarkers(model, analyzerOwner, [])
        monaco.editor.setModelMarkers(model, gpuOwner, [])
      }
      workbench?.dispose()
      instance.current = null
    }
  }, [])

  useLayoutEffect(() => {
    instance.current?.reconcile()
  }, [files, activeFileId, prelude])
  useEffect(() => {
    instance.current?.updateMarkers()
  }, [diagnostics])
  useEffect(() => {
    instance.current?.updateEditors()
  }, [vim, label, insertion])
  useEffect(() => {
    if (analyzerEnabled) instance.current?.maybeStartAnalyzer()
  }, [analyzerEnabled])

  return (
    <div ref={panel} className="wgsl-editor">
      <div ref={container} className="wgsl-monaco" />
    </div>
  )
}
