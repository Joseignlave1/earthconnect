import type {
  CompletionItem, CompletionList, Diagnostic, Hover, Position,
} from 'vscode-languageserver-types'
import type { WgslAnalyzerServer } from 'wgsl-analyzer-web'

type PublishDiagnosticsParams = { uri: string; version?: number; diagnostics: Diagnostic[] }

type Reply = {
  id?: number
  method?: string
  params?: PublishDiagnosticsParams
  result?: unknown
  error?: { message: string; code: number }
}

class WgslRequestError extends Error {
  readonly code: number

  constructor(message: string, code: number) {
    super(message)
    this.code = code
  }
}

const file = 'shaders/score.wgsl'

export async function startWgslLanguageServer(
  source: () => string,
  publishDiagnostics: ((value: PublishDiagnosticsParams) => void) | null,
  onFailure: () => void,
  signal: AbortSignal,
) {
  const { WgslAnalyzerServer } = await import('wgsl-analyzer-web')
  signal.throwIfAborted()
  const server: WgslAnalyzerServer = await WgslAnalyzerServer.start({
    files: { 'wesl.toml': 'edition = "2026_pre"\n', [file]: source() },
    onExit: onFailure,
  })
  if (signal.aborted) {
    server.dispose()
    signal.throwIfAborted()
  }

  const uri = server.uriOf(file)
  const analysisUri = server.uriOf('shaders/analyze.wgsl')
  let analysisOpened = false
  let analysisVersion = 0
  let analysisText: string | undefined
  let analysisQueue = Promise.resolve()
  let id = 0
  let version = 1
  let disposed = false
  let diagnosticTimer: ReturnType<typeof setTimeout> | undefined
  const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()
  const subscription = server.onMessage((message) => {
    const reply = message as Reply
    if (reply.method === 'textDocument/publishDiagnostics' && reply.params?.uri === uri &&
        (reply.params.version === undefined || reply.params.version === version)) {
      publishDiagnostics?.(reply.params)
    } else if (reply.method && reply.id != null) {
      // A server request needs a reply even when the client has no handler.
      const result = reply.method === 'workspace/configuration' ? [] : null
      server.sendMessage({ jsonrpc: '2.0', id: reply.id, result })
      if (reply.method === 'workspace/diagnostic/refresh') scheduleDiagnostics()
    } else if (reply.id != null) {
      const request = pending.get(reply.id)
      pending.delete(reply.id)
      if (reply.error) request?.reject(new WgslRequestError(reply.error.message, reply.error.code))
      else request?.resolve(reply.result)
    }
  })

  function notify(method: string, params: object) {
    server.sendMessage({ jsonrpc: '2.0', method, params })
  }

  function request<T>(method: string, params: object): Promise<T> {
    return new Promise((resolve, reject) => {
      if (disposed) {
        reject(new Error('WGSL analysis has stopped.'))
        return
      }
      const requestId = ++id
      pending.set(requestId, { resolve: value => resolve(value as T), reject })
      server.sendMessage({ jsonrpc: '2.0', id: requestId, method, params })
    })
  }

  function scheduleDiagnostics() {
    if (!publishDiagnostics) return
    clearTimeout(diagnosticTimer)
    const requestedVersion = version
    diagnosticTimer = setTimeout(() => {
      void request<{ kind: string; items?: Diagnostic[] }>('textDocument/diagnostic', {
        textDocument: { uri },
      }).then(result => {
        if (result?.kind === 'full' && requestedVersion === version) {
          publishDiagnostics({ uri, version, diagnostics: result.items ?? [] })
        }
      }).catch(() => { /* Diagnostics can be superseded or canceled by editor changes. */ })
    }, 180)
  }

  function dispose() {
    if (disposed) return
    disposed = true
    clearTimeout(diagnosticTimer)
    signal.removeEventListener('abort', dispose)
    subscription.dispose()
    server.dispose()
    for (const request of pending.values()) request.reject(new Error('WGSL analysis has stopped.'))
    pending.clear()
  }

  signal.addEventListener('abort', dispose, { once: true })
  try {
    await request('initialize', {
      processId: null,
      rootUri: server.rootUri,
      capabilities: {
        general: { positionEncodings: ['utf-16'] },
        textDocument: {
          completion: { completionItem: { documentationFormat: ['markdown', 'plaintext'] } },
          hover: { contentFormat: ['markdown', 'plaintext'] },
          publishDiagnostics: { versionSupport: true },
          diagnostic: {},
        },
        workspace: { diagnostics: { refreshSupport: true } },
      },
      workspaceFolders: [{ uri: server.rootUri, name: 'Earth Connect' }],
    })
    notify('initialized', {})
    notify('textDocument/didOpen', {
      textDocument: { uri, languageId: 'wesl', version, text: source() },
    })
    scheduleDiagnostics()
  } catch (error) {
    dispose()
    throw error
  }

  return {
    change(text: string) {
      if (disposed) return
      notify('textDocument/didChange', {
        textDocument: { uri, version: ++version },
        contentChanges: [{ text }],
      })
      scheduleDiagnostics()
    },
    complete(position: Position) {
      return request<CompletionList | CompletionItem[] | null>('textDocument/completion', {
        textDocument: { uri }, position,
      })
    },
    hover(position: Position) {
      return request<Hover | null>('textDocument/hover', { textDocument: { uri }, position })
    },
    async syntaxTree(signal?: AbortSignal) {
      for (let attempt = 0; attempt < 8; attempt++) {
        signal?.throwIfAborted()
        try {
          const tree = await request<string>('wgsl-analyzer/viewSyntaxTree', { textDocument: { uri } })
          if (tree) return tree
        } catch (error) {
          // Initial indexing and edits can invalidate an analyzer snapshot.
          if (!(error instanceof WgslRequestError) || error.code !== -32801) throw error
        }
        await new Promise(resolve => setTimeout(resolve, 25 * (attempt + 1)))
      }
      throw new Error('The WGSL analyzer could not produce a syntax tree.')
    },
    syntaxTreeFor(text: string, signal?: AbortSignal) {
      const task = analysisQueue.then(async () => {
        signal?.throwIfAborted()
        if (disposed) throw new Error('WGSL analysis has stopped.')
        if (!analysisOpened) {
          analysisOpened = true
          analysisVersion = 1
          analysisText = text
          notify('textDocument/didOpen', {
            textDocument: { uri: analysisUri, languageId: 'wesl', version: analysisVersion, text },
          })
        } else if (analysisText !== text) {
          analysisText = text
          notify('textDocument/didChange', {
            textDocument: { uri: analysisUri, version: ++analysisVersion },
            contentChanges: [{ text }],
          })
        }
        for (let attempt = 0; attempt < 8; attempt++) {
          signal?.throwIfAborted()
          try {
            const tree = await request<string>('wgsl-analyzer/viewSyntaxTree', { textDocument: { uri: analysisUri } })
            if (tree) return tree
          } catch (error) {
            if (!(error instanceof WgslRequestError) || error.code !== -32801) throw error
          }
          await new Promise(resolve => setTimeout(resolve, 25 * (attempt + 1)))
        }
        throw new Error('The WGSL analyzer could not produce a syntax tree.')
      })
      analysisQueue = task.then(() => undefined, () => undefined)
      return task
    },
    dispose,
  }
}
