// The generated protobuf contains enums, which Node's type stripping cannot run.
import { registerHooks } from 'node:module'
import ts from 'typescript'
registerHooks({
  load(url,context,next) {
    const loaded = next(url,context)
    if (!url.endsWith('/earth_index_pb.ts')) return loaded
    return { ...loaded, format: 'module', source: ts.transpileModule(String(loaded.source),{ compilerOptions: { target: ts.ScriptTarget.ES2023,module: ts.ModuleKind.ESNext,verbatimModuleSyntax: true } }).outputText }
  },
})
await import('./fetch-score-snapshot.ts')
