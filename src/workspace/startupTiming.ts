type Span = { start: number; end?: number; detail?: string }

const marks = new Map<string, number>()
const spans = new Map<string, Span>()
let reported = false

export function markStartup(name: string) {
  if (reported || marks.has(name)) return
  marks.set(name, performance.now())
}

export function startStartupSpan(name: string, detail?: string) {
  if (reported || spans.has(name)) return
  spans.set(name, { start: performance.now(), detail })
}

export function endStartupSpan(name: string, detail?: string) {
  if (reported) return
  const span = spans.get(name)
  if (!span || span.end !== undefined) return
  span.end = performance.now()
  if (detail !== undefined) span.detail = detail
}

export function reportStartupTiming(note?: string) {
  if (reported) return
  reported = true
  const total = performance.now()
  const fmt = (ms: number) => `${Math.round(ms)} ms`
  const rows: { phase: string; start: number; end: number; detail?: string }[] = []
  for (const [phase, span] of spans) {
    rows.push({ phase, start: span.start, end: span.end ?? total, detail: span.detail })
  }
  for (const [phase, at] of marks) {
    rows.push({ phase, start: at, end: at })
  }
  rows.sort((a, b) => a.start - b.start)
  const width = Math.max(...rows.map(row => row.phase.length), 8)
  const lines = rows.map(row => {
    const label = row.phase.padEnd(width)
    const timing =
      row.end > row.start
        ? `${fmt(row.end - row.start).padStart(7)}  (${fmt(row.start)} → ${fmt(row.end)})`
        : `      —  (at ${fmt(row.start)})`
    return `  ${label}  ${timing}${row.detail ? `  ${row.detail}` : ''}`
  })
  console.log(`[Earth Connect] ready in ${fmt(total)}${note ? ` (${note})` : ''}\n${lines.join('\n')}`)
}
