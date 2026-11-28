import { useState } from 'react'
import { COPILOT_EXAMPLES } from './editorialData.ts'
import { highlightWgslLine } from './wgslHighlight.tsx'

export function CopilotSection() {
  const [copilotId, setCopilotId] = useState<string>(COPILOT_EXAMPLES[0].id)
  const activeCopilot = COPILOT_EXAMPLES.find(item => item.id === copilotId) ?? COPILOT_EXAMPLES[0]

  return (
    <section className="copilot-section" aria-label="Workspace AI assistant">
      <div className="copilot-grid">
        <div className="copilot-copy">
          <p className="copilot-status">Workspace assistant · In development</p>
          <h2 className="copilot-heading">Ask in plain language. Edit the shader when you want.</h2>
          <p className="copilot-body">
            You don&apos;t have to write WGSL by hand to explore another world. Describe the terrain or mission
            constraints you care about—an integrated assistant translates your question into a live GPU scoring rule,
            inspects matching regions with tools, and leaves the shader open to refine.
          </p>
          <div className="copilot-switches" role="tablist" aria-label="Assistant example prompts">
            {COPILOT_EXAMPLES.map(example => {
              const selected = example.id === activeCopilot.id
              return (
                <button
                  key={example.id}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  className="copilot-switch-btn"
                  onClick={() => setCopilotId(example.id)}
                >
                  {example.label}
                </button>
              )
            })}
          </div>
        </div>

        <div className="copilot-trace" role="tabpanel" aria-label={activeCopilot.label}>
          <p className="copilot-prompt">
            <span className="copilot-chevron" aria-hidden="true">
              &gt;
            </span>
            <span>{activeCopilot.prompt}</span>
          </p>

          <ul className="copilot-steps">
            {activeCopilot.steps.map(step => (
              <li key={step.call} className="copilot-step">
                <span className="copilot-step-call">{step.call}</span>
                <span className="copilot-step-detail">{step.detail}</span>
              </li>
            ))}
          </ul>

          <pre className="copilot-snippet">
            <code>
              {activeCopilot.snippet.map((line, idx) => (
                <span
                  key={`${activeCopilot.id}-${idx}`}
                  className="copilot-snippet-line"
                  data-added={line.added ? 'true' : undefined}
                >
                  <span className="copilot-snippet-mark" aria-hidden="true">
                    {line.added ? '+' : ' '}
                  </span>
                  <span className="atlas-code-src">{highlightWgslLine(line.code, idx)}</span>
                </span>
              ))}
            </code>
          </pre>
        </div>
      </div>
    </section>
  )
}
