export type StartupPhase = 'bootstrap' | 'prepare' | 'settle'

const PHASE_ORDER: readonly StartupPhase[] = ['bootstrap', 'prepare', 'settle']

type Span = {
  phase: StartupPhase
  start: number
  end?: number
  detail?: string
}

const spans = new Map<string, Span>()
let reported = false

export function startStartupSpan(name: string, phase: StartupPhase = 'prepare', detail?: string) {
  if (reported || spans.has(name)) return
  spans.set(name, { phase, start: performance.now(), detail })
}

export function endStartupSpan(name: string, detail?: string) {
  if (reported) return
  const span = spans.get(name)
  if (!span || span.end !== undefined) return
  span.end = performance.now()
  if (detail !== undefined) span.detail = detail
}

export function reportStartupTiming() {
  if (reported) return
  reported = true
  const total = performance.now()
  const fmt = (ms: number) => `${Math.round(ms)} ms`

  type Row = { name: string; phase: StartupPhase; start: number; end: number; open: boolean; detail?: string }
  const byPhase = new Map<StartupPhase, Row[]>()
  for (const [name, span] of spans) {
    const list = byPhase.get(span.phase) ?? []
    list.push({
      name,
      phase: span.phase,
      start: span.start,
      end: span.end ?? total,
      open: span.end === undefined,
      detail: span.detail,
    })
    byPhase.set(span.phase, list)
  }

  let nameWidth = 10
  for (const list of byPhase.values()) {
    for (const row of list) nameWidth = Math.max(nameWidth, row.name.length)
  }

  const lines: string[] = []
  for (const phase of PHASE_ORDER) {
    const rows = byPhase.get(phase)
    if (!rows?.length) continue
    rows.sort((a, b) => a.start - b.start)
    const phaseStart = Math.min(...rows.map(r => r.start))
    const phaseEnd = Math.max(...rows.map(r => r.end))
    const phaseDur = fmt(phaseEnd - phaseStart).padStart(7)
    lines.push(`  ${phase.padEnd(nameWidth + 2)}  ${phaseDur}  (${fmt(phaseStart)} → ${fmt(phaseEnd)})`)
    for (const row of rows) {
      const label = row.name.padEnd(nameWidth)
      const timing = row.open
        ? `${fmt(row.end - row.start).padStart(7)}  (open since ${fmt(row.start)}, elapsed so far)`
        : `${fmt(row.end - row.start).padStart(7)}  (${fmt(row.start)} → ${fmt(row.end)})`
      lines.push(`    ${label}  ${timing}${row.detail ? `  ${row.detail}` : ''}`)
    }
  }
  console.log(`[Earth Connect] ready in ${fmt(total)}\n${lines.join('\n')}`)
}
