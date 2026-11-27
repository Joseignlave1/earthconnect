import { PLANET_SHADER_RESERVED_NAMES } from '../earth-score/planetShaderSchema.ts'
import { canonicalPoint } from './planetaryPointCache.ts'
import { startWgslLanguageServer } from './wgslLanguageServer.ts'

const functions = { moon_point: 'moon', mars_point: 'mars' } as const
type Range = { line: number; column: number; length: number }
export type PlanetaryCall = Range & {
  body: 'moon' | 'mars'
  latitude: number
  longitude: number
  /** UTF-16 offsets spanning the full call expression, end exclusive. */
  offset: number
  endOffset: number
}
type Diagnostic = Range & { severity: 'error'; message: string }

export class PlanetaryAnalysisError extends Error {
  readonly diagnostics: readonly Diagnostic[]

  constructor(diagnostics: readonly Diagnostic[]) {
    super(diagnostics[0]?.message ?? 'Cannot analyze planetary samples.')
    this.name = 'PlanetaryAnalysisError'
    this.diagnostics = diagnostics
  }
}

type Point = readonly [offset: number, line: number, column: number]
type Ast = { kind: string; type: 'Node' | 'Token'; start: Point; end: Point; children?: Ast[] }
type NumberValue = { value: number; type: 'abstract' | 'f32' | 'other' }
const runtimeBindings = new Set(['ConstantDeclaration', 'LetDeclaration', 'VariableDeclaration', 'OverrideDeclaration'])
const startIdentifier = /[_\p{XID_Start}]/u
const continueIdentifier = /[_\p{XID_Continue}]/u
const lineBreak = /[\n\v\f\r\u0085\u2028\u2029]/

// A lexical gate only: the AST determines whether a name is a call or binding.
export function mentionsPlanetaryName(source: string) {
  if (!source.includes('moon_point') && !source.includes('mars_point')) return -1
  let blockDepth = 0
  let identifier = ''
  let first = -1
  function finish(offset: number) {
    if (Object.hasOwn(functions, identifier)) first = offset - identifier.length
    identifier = ''
  }
  for (let index = 0; index < source.length; ) {
    const pair = source.slice(index, index + 2)
    if (pair === '/*') {
      finish(index)
      if (first >= 0) break
      blockDepth++
      index += 2
    } else if (blockDepth > 0) {
      if (pair === '*/') {
        blockDepth--
        index += 2
      } else index++
    } else if (pair === '//') {
      finish(index)
      if (first >= 0) break
      while (index < source.length && !lineBreak.test(source[index])) index++
    } else {
      const char = String.fromCodePoint(source.codePointAt(index)!)
      if ((identifier ? continueIdentifier : startIdentifier).test(char)) {
        identifier += char
      } else {
        finish(index)
        if (first >= 0) break
      }
      index += char.length
    }
  }
  if (first < 0) finish(source.length)
  return first
}

function children(node: Ast | undefined) {
  return node?.children ?? []
}

function branch(node: Ast | undefined, kind: string) {
  return children(node).find(child => child.kind === kind)
}

function expressionAfterEqual(node: Ast) {
  const parts = children(node)
  const equal = parts.findIndex(child => child.kind === 'Equal')
  return equal >= 0 ? parts.slice(equal + 1).find(child => child.type === 'Node') : undefined
}

function numericLiteral(text: string, floating: boolean): NumberValue {
  if (floating && /^0x/i.test(text)) {
    const [mantissa, rawExp] = text.slice(2).toLowerCase().split('p')
    let type: NumberValue['type'] = 'abstract'
    let exponentText = rawExp
    if (exponentText !== undefined) {
      const expSuffix = exponentText.at(-1)
      if (expSuffix === 'f' || expSuffix === 'h') {
        type = expSuffix === 'f' ? 'f32' : 'other'
        exponentText = exponentText.slice(0, -1)
      }
    }
    const [whole, fraction = ''] = mantissa.split('.')
    const hexFraction = [...fraction].reduceRight((rest, digit) => (rest + parseInt(digit, 16)) / 16, 0)
    const exponent = exponentText === undefined ? 0 : Number(exponentText)
    const value = (Number.parseInt(whole || '0', 16) + hexFraction) * 2 ** exponent
    return { type, value: type === 'f32' ? Math.fround(value) : value }
  }
  const suffix = text.at(-1)?.toLowerCase()
  const explicit = floating ? suffix === 'f' || suffix === 'h' : suffix === 'i' || suffix === 'u'
  const type = !explicit ? 'abstract' : suffix === 'f' ? 'f32' : 'other'
  const digits = explicit ? text.slice(0, -1) : text
  const value = Number(digits)
  return { type, value: type === 'f32' ? Math.fround(value) : value }
}

export function readPlanetaryCalls(source: string, syntaxTree: string): readonly PlanetaryCall[] {
  const tree = JSON.parse(syntaxTree) as Ast
  const bytes = new TextEncoder().encode(source)
  const decoder = new TextDecoder(undefined, { ignoreBOM: true })
  if (tree.kind !== 'SourceFile' || tree.end[0] !== bytes.length) {
    throw new Error('The WGSL syntax tree does not match this shader revision.')
  }
  const offsets = [0]
  for (let index = 0; index < bytes.length; index++) {
    if (bytes[index] === 10) offsets.push(index + 1)
  }
  const lineStarts = source.split('\n').reduce<number[]>(
    (starts, line) => {
      starts.push(starts[starts.length - 1] + line.length + 1)
      return starts
    },
    [0],
  )
  const sourceOffset = ([offset, line]: Point) =>
    lineStarts[line] + decoder.decode(bytes.subarray(offsets[line], offset)).length
  const text = (node: Ast) => decoder.decode(bytes.subarray(node.start[0], node.end[0]))
  function range(node: Ast): Range {
    const [start, line] = node.start
    const end = node.end[1] === line ? node.end[0] : offsets[line + 1] - 1
    return {
      line: line + 1,
      column: decoder.decode(bytes.subarray(offsets[line], start)).length + 1,
      length: Math.max(1, decoder.decode(bytes.subarray(start, end)).length),
    }
  }
  const name = (node: Ast) => branch(branch(node, 'Name'), 'Identifier')
  const pathParts = (node: Ast | undefined) =>
    children(branch(node, 'Path')).filter(child => child.kind === 'Identifier')
  function simplePath(node: Ast | undefined) {
    const parts = pathParts(node)
    if (parts.length === 1 && !branch(branch(node, 'Path'), 'TemplateList') && !branch(node, 'TemplateList'))
      return text(parts[0])
  }

  const globals = new Map<string, Ast>()
  for (const node of children(tree)) {
    const identifier = name(node)
    if (identifier) globals.set(text(identifier), node)
  }

  function isFloatType(identifier: string | undefined, seen = new Set<string>()): boolean {
    if (identifier === 'f32') return true
    if (!identifier || seen.has(identifier)) return false
    seen.add(identifier)
    const alias = globals.get(identifier)
    return alias?.kind === 'TypeAliasDeclaration' && isFloatType(simplePath(branch(alias, 'TypeSpecifier')), seen)
  }

  function shadows(identifier: string, position: number, ancestors: readonly Ast[]) {
    for (const scope of ancestors) {
      const definitions =
        scope.kind === 'FunctionDeclaration'
          ? children(branch(scope, 'FunctionParameters'))
          : scope.kind === 'ForStatement'
            ? children(branch(scope, 'ForInitializer'))
            : scope.kind === 'CompoundStatement'
              ? children(scope).filter(child => runtimeBindings.has(child.kind))
              : []
      if (
        definitions.some(definition => {
          const declaration = name(definition)
          return declaration && definition.end[0] <= position && text(declaration) === identifier
        })
      )
        return true
    }
    return false
  }

  function numeric(node: Ast | undefined, ancestors: readonly Ast[], seen: Set<string>): NumberValue | undefined {
    if (!node) return
    switch (node.kind) {
      case 'Literal': {
        const token = children(node).find(child => child.kind === 'FloatLiteral' || child.kind === 'IntLiteral')
        if (token) return numericLiteral(text(token), token.kind === 'FloatLiteral')
        return
      }
      case 'PrefixExpression': {
        if (!branch(node, 'Minus')) return
        const number = numeric(
          children(node).find(child => child.type === 'Node'),
          ancestors,
          seen,
        )
        if (number) return { ...number, value: -number.value }
        return
      }
      case 'ParenthesisExpression': {
        if (!branch(node, 'ParenthesisRight')) return
        return numeric(
          children(node).find(child => child.type === 'Node'),
          ancestors,
          seen,
        )
      }
      case 'IdentExpression': {
        const identifier = simplePath(node)
        if (!identifier || seen.has(identifier) || shadows(identifier, node.start[0], ancestors)) return
        const constant = globals.get(identifier)
        if (constant?.kind !== 'ConstantDeclaration') return
        seen.add(identifier)
        // A module initializer resolves names in module scope, regardless of
        // which local scope uses the constant.
        const value = numeric(expressionAfterEqual(constant), [], seen)
        const annotation = branch(constant, 'TypeSpecifier')
        if (!value || !annotation) return value
        if (value.type !== 'other' && isFloatType(simplePath(annotation))) {
          return { type: 'f32', value: Math.fround(value.value) }
        }
        return
      }
      default:
        return
    }
  }

  const calls: PlanetaryCall[] = []
  const diagnostics: Diagnostic[] = []
  function fail(node: Ast, message: string) {
    diagnostics.push({ ...range(node), severity: 'error', message })
  }

  function visit(node: Ast, ancestors: Ast[]) {
    const declaration = name(node)
    if (declaration && node.kind !== 'StructMember' && PLANET_SHADER_RESERVED_NAMES.has(text(declaration))) {
      fail(declaration, `${text(declaration)} is reserved for planetary sampling.`)
    }
    if (node.kind === 'FunctionCall') {
      const callee = branch(node, 'IdentExpression')
      const calleePath = pathParts(callee)
      const token = calleePath.at(-1)
      const label = token && text(token)
      if (label && Object.hasOwn(functions, label)) {
        const argumentsNode = branch(node, 'Arguments')
        const argumentsList = children(argumentsNode).filter(child => child.type === 'Node')
        const close = branch(argumentsNode, 'ParenthesisRight')
        if (simplePath(callee) !== label) {
          fail(token, `Call ${label} directly, without a namespace or type arguments.`)
        } else if (argumentsList.length !== 2 || !close) {
          fail(token, `${label} requires exactly two arguments, latitude and longitude.`)
        } else {
          const coordinates = argumentsList.map((argument, index) => {
            const number = numeric(argument, ancestors, new Set())
            const axis = index === 0 ? 'Latitude' : 'Longitude'
            const limit = index === 0 ? 90 : 180
            if (!number || number.type === 'other' || !Number.isFinite(number.value)) {
              fail(
                argument,
                `${axis} must be a finite numeric literal or a module-scope numeric const that converts to f32.`,
              )
              return
            }
            if (Math.abs(number.value) > limit) {
              fail(argument, `${axis} must be between ${-limit}° and ${limit}°.`)
              return
            }
            return number.value
          })
          const [latitude, longitude] = coordinates
          if (latitude !== undefined && longitude !== undefined) {
            calls.push({
              ...range(token),
              offset: sourceOffset(token.start),
              endOffset: sourceOffset(close.end),
              ...canonicalPoint({ body: functions[label as keyof typeof functions], latitude, longitude }),
            })
          }
        }
      }
    }
    ancestors.push(node)
    for (const child of children(node)) {
      if (child.type === 'Node') visit(child, ancestors)
    }
    ancestors.pop()
  }

  visit(tree, [])
  if (diagnostics.length) throw new PlanetaryAnalysisError(diagnostics)
  return calls
}

type Server = Awaited<ReturnType<typeof startWgslLanguageServer>>
type SharedSyntaxTreeProvider = (source: string, signal?: AbortSignal) => Promise<string>
let sharedProvider: SharedSyntaxTreeProvider | undefined
let session: Promise<Server> | undefined
let openedSource: string | undefined
let shutdown: AbortController | undefined
let idle: ReturnType<typeof setTimeout> | undefined
let queue = Promise.resolve()

/** Reuse an already-running editor wgsl-analyzer instance for planetary AST inspection. */
export function registerSharedWgslAnalyzer(provider: SharedSyntaxTreeProvider) {
  sharedProvider = provider
  return () => {
    if (sharedProvider === provider) sharedProvider = undefined
  }
}

function stopSession() {
  shutdown?.abort()
  shutdown = undefined
  session = undefined
  openedSource = undefined
}

async function inspect(source: string, signal: AbortSignal) {
  signal.throwIfAborted()
  const shared = sharedProvider
  if (shared) {
    try {
      const tree = await shared(source, signal)
      signal.throwIfAborted()
      return readPlanetaryCalls(source, tree)
    } catch (error) {
      if (signal.aborted || error instanceof PlanetaryAnalysisError) throw error
      // Fall back to a dedicated analyzer session if the editor's session stopped.
    }
  }
  clearTimeout(idle)
  let timeout: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      stopSession()
      reject(new Error('The local WGSL analyzer did not respond.'))
    }, 15_000)
  })
  try {
    if (!session) {
      const controller = new AbortController()
      shutdown = controller
      const onExit = () => {
        if (shutdown === controller) stopSession()
      }
      session = startWgslLanguageServer(() => source, null, onExit, controller.signal)
      openedSource = source
    }
    const running = await Promise.race([session, deadline])
    signal.throwIfAborted()
    if (openedSource !== source) {
      running.change(source)
      openedSource = source
    }
    const tree = await Promise.race([running.syntaxTree(signal), deadline])
    signal.throwIfAborted()
    return readPlanetaryCalls(source, tree)
  } catch (error) {
    if (!signal.aborted && !(error instanceof PlanetaryAnalysisError)) stopSession()
    throw error
  } finally {
    clearTimeout(timeout)
    idle = setTimeout(stopSession, 30_000)
  }
}

export async function analyzePlanetaryShader(source: string, signal: AbortSignal): Promise<readonly PlanetaryCall[]> {
  signal.throwIfAborted()
  const offset = mentionsPlanetaryName(source)
  if (offset < 0) return []
  const name = Object.keys(functions).find(name => source.startsWith(name, offset))
  const unavailable = () =>
    new PlanetaryAnalysisError([
      {
        line: source.slice(0, offset).split('\n').length,
        column: offset - source.lastIndexOf('\n', offset - 1),
        length: name?.length ?? 1,
        severity: 'error',
        message: 'Planetary calls require the local WGSL analyzer. Reload on a cross-origin-isolated page to use them.',
      },
    ])
  if (!globalThis.crossOriginIsolated) throw unavailable()
  const job = queue.then(() => inspect(source, signal))
  queue = job.then(
    () => undefined,
    () => undefined,
  )
  try {
    return await job
  } catch (error) {
    signal.throwIfAborted()
    if (error instanceof PlanetaryAnalysisError) throw error
    throw unavailable()
  }
}
