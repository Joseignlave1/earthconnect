import type { ReactNode } from 'react'

const WGSL_TOKEN_RE =
  /(\/\/[^\n]*|\b(?:fn|let|var|if|else|return)\b|\b(?:EarthCell|f32|vec2f|vec3f|vec4f|u32|i32|bool)\b|\b(?:mars_point|moon_point|has_data|exp|dot|smoothstep|clamp|mix|sqrt|abs|max|min)\b|\bNO_SCORE\b|(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_]))/g

export function highlightWgslLine(line: string, lineIdx: number): ReactNode[] {
  const parts: ReactNode[] = []
  let last = 0
  WGSL_TOKEN_RE.lastIndex = 0
  let match = WGSL_TOKEN_RE.exec(line)
  let tokIdx = 0
  while (match !== null) {
    if (match.index > last) parts.push(line.slice(last, match.index))
    const token = match[0]
    let cls: string
    if (token.startsWith('//')) cls = 'wgsl-tok-comment'
    else if (/^(fn|let|var|if|else|return)$/.test(token)) cls = 'wgsl-tok-kw'
    else if (/^(EarthCell|f32|vec2f|vec3f|vec4f|u32|i32|bool)$/.test(token)) cls = 'wgsl-tok-type'
    else if (/^(mars_point|moon_point|has_data|exp|dot|smoothstep|clamp|mix|sqrt|abs|max|min)$/.test(token))
      cls = 'wgsl-tok-fn'
    else if (token === 'NO_SCORE') cls = 'wgsl-tok-const'
    else cls = 'wgsl-tok-num'
    parts.push(
      <span key={`${lineIdx}-${tokIdx++}`} className={cls}>
        {token}
      </span>,
    )
    last = match.index + token.length
    match = WGSL_TOKEN_RE.exec(line)
  }
  if (last < line.length) parts.push(line.slice(last))
  return parts
}
