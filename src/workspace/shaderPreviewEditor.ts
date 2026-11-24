import { URI } from '@codingame/monaco-vscode-api/vscode/vs/base/common/uri'
import { SyncDescriptor } from '@codingame/monaco-vscode-api/vscode/vs/platform/instantiation/common/descriptors'
import { Registry } from '@codingame/monaco-vscode-api/vscode/vs/platform/registry/common/platform'
import { IStorageService } from '@codingame/monaco-vscode-api/vscode/vs/platform/storage/common/storage.service'
import { ITelemetryService } from '@codingame/monaco-vscode-api/vscode/vs/platform/telemetry/common/telemetry.service'
import { IThemeService } from '@codingame/monaco-vscode-api/vscode/vs/platform/theme/common/themeService.service'
import {
  EditorPaneDescriptor,
  type IEditorPaneRegistry,
} from '@codingame/monaco-vscode-api/vscode/vs/workbench/browser/editor'
import { EditorPane } from '@codingame/monaco-vscode-api/vscode/vs/workbench/browser/parts/editor/editorPane'
import {
  EditorExtensions,
  EditorInputCapabilities,
  type IUntypedEditorInput,
} from '@codingame/monaco-vscode-api/vscode/vs/workbench/common/editor'
import { EditorInput } from '@codingame/monaco-vscode-api/vscode/vs/workbench/common/editor/editorInput'
import type { IEditorGroup } from '@codingame/monaco-vscode-api/vscode/vs/workbench/services/editor/common/editorGroupsService'

const previewId = 'earth-connect.shaderPreview'
export const shaderPreviewScheme = 'earth-connect'

export function getPreviewFileId(resource: URI | undefined) {
  const prefix = '/preview/'
  if (resource?.scheme !== shaderPreviewScheme || !resource.path.startsWith(prefix)) return
  try {
    return decodeURIComponent(resource.path.slice(prefix.length)) || undefined
  } catch {
    return undefined
  }
}

export class ShaderPreviewInput extends EditorInput {
  override readonly resource: URI

  constructor(
    readonly fileId: string,
    private fileName: string,
  ) {
    super()
    this.resource = URI.from({ scheme: shaderPreviewScheme, path: `/preview/${encodeURIComponent(fileId)}` })
  }

  get typeId() {
    return previewId
  }
  get editorId() {
    return previewId
  }
  get capabilities() {
    return EditorInputCapabilities.Readonly | EditorInputCapabilities.Singleton | EditorInputCapabilities.ForceReveal
  }

  getName() {
    return `Preview of ${this.fileName}`
  }

  setFileName(name: string) {
    if (name === this.fileName) return
    this.fileName = name
    this._onDidChangeLabel.fire()
  }

  override matches(other: EditorInput | IUntypedEditorInput) {
    return super.matches(other) || (other instanceof ShaderPreviewInput && other.fileId === this.fileId)
  }
}

export class ShaderPreviewPane extends EditorPane {
  private surface!: HTMLElement

  constructor(group: IEditorGroup, telemetry: ITelemetryService, theme: IThemeService, storage: IStorageService) {
    super(previewId, group, telemetry, theme, storage)
  }

  protected createEditor(parent: HTMLElement) {
    this.surface = parent.ownerDocument.createElement('div')
    this.surface.className = 'shader-preview-pane'
    this.surface.tabIndex = -1
    parent.append(this.surface)
  }

  showHost(host: HTMLElement) {
    if (host.parentElement !== this.surface) movePreviewHost(host, this.surface)
  }

  override focus() {
    this.surface.focus()
  }
  layout() {}
}

ITelemetryService(ShaderPreviewPane, undefined, 1)
IThemeService(ShaderPreviewPane, undefined, 2)
IStorageService(ShaderPreviewPane, undefined, 3)

export function registerShaderPreviewPane() {
  Registry.as<IEditorPaneRegistry>(EditorExtensions.EditorPane).registerEditorPane(
    EditorPaneDescriptor.create(ShaderPreviewPane, previewId, 'Preview'),
    [new SyncDescriptor(ShaderPreviewInput)],
  )
}

export function movePreviewHost(host: HTMLElement, destination: HTMLElement) {
  if (host.parentElement === destination) return
  // Preserve the state of embedded content when both parents are in the document.
  if (typeof destination.moveBefore === 'function' && host.isConnected && destination.isConnected) {
    destination.moveBefore(host, null)
  } else {
    destination.append(host)
  }
}
