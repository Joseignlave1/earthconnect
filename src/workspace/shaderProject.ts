import { comparisonExamples } from './comparisonExamples.ts'
import { helloWorldExample } from './helloWorldExample.ts'

const examples = [...comparisonExamples, helloWorldExample]

export type ShaderFileKind = 'shader' | 'helper'
export type ShaderFile = { id: string; name: string; source: string; closed?: boolean; kind?: ShaderFileKind }
type Segment = { fileId: string; startLine: number; lineCount: number }

export function assembleShaderProject(files: readonly ShaderFile[]) {
  const segments: Segment[] = []
  let startLine = 1
  const source = files
    .map(file => {
      const source = file.source.replace(/\r\n?/g, '\n')
      const lineCount = source.split('\n').length
      segments.push({ fileId: file.id, startLine, lineCount })
      startLine += lineCount
      return source
    })
    .join('\n')
  return { source, segments }
}

export function findShaderLocation(segments: readonly Segment[], line: number) {
  const segment = segments.find(part => line >= part.startLine && line < part.startLine + part.lineCount)
  return segment && { fileId: segment.fileId, line: line - segment.startLine + 1 }
}

export type ProjectFile = Omit<ShaderFile, 'source'> & { source: string | null }
export type ShaderProject = { files: ProjectFile[]; activeFileId: string }
const storageKey = 'earth-connect-wgsl-project'

function addExamples(files: ProjectFile[], newProject = false): ProjectFile[] {
  const ids = new Set(files.map(file => file.id))
  const names = new Set(files.map(file => file.name.toLowerCase()))
  const additions = examples
    .filter(file => !ids.has(file.id))
    .map(file => {
      const stem = file.name.replace(/\.wgsl$/i, '')
      let name: string = file.name
      let suffix = 2
      while (names.has(name.toLowerCase())) name = `${stem}-${suffix++}.wgsl`
      names.add(name.toLowerCase())
      return {
        id: file.id,
        name,
        source: null,
        ...(!newProject && file.id === helloWorldExample.id && { closed: true }),
      }
    })
  return [...files, ...additions]
}

export function readShaderProject(): ShaderProject {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(storageKey) ?? 'null')
    if (
      saved &&
      typeof saved === 'object' &&
      'files' in saved &&
      'activeFileId' in saved &&
      typeof saved.activeFileId === 'string' &&
      Array.isArray(saved.files) &&
      saved.files.every(
        file =>
          file &&
          typeof file.id === 'string' &&
          typeof file.name === 'string' &&
          (file.source === null || typeof file.source === 'string') &&
          (file.closed === undefined || typeof file.closed === 'boolean') &&
          (file.kind === undefined || file.kind === 'shader' || file.kind === 'helper'),
      ) &&
      saved.files.some(file => file.id === saved.activeFileId) &&
      saved.files.some(file => file.id === 'score') &&
      new Set(saved.files.map(file => file.id)).size === saved.files.length
    ) {
      return { files: addExamples(saved.files), activeFileId: saved.activeFileId }
    }
  } catch {
    /* Storage may be unavailable. */
  }
  return {
    files: addExamples([{ id: 'score', name: 'score.wgsl', source: null }], true),
    activeFileId: helloWorldExample.id,
  }
}

export function saveShaderProject(project: ShaderProject) {
  try {
    localStorage.setItem(storageKey, JSON.stringify(project))
  } catch {
    /* Storage may be unavailable. */
  }
}

export function resolveShaderFiles(project: ShaderProject, baseline: string): ShaderFile[] {
  return project.files.map(file => ({ ...file, source: file.source ?? defaultShaderSource(file.id, baseline) }))
}

export function defaultShaderSource(fileId: string, baseline: string): string {
  return examples.find(file => file.id === fileId)?.source ?? baseline
}

function isComparisonFile(file: Pick<ShaderFile, 'id' | 'kind'>) {
  return file.kind === 'shader' || examples.some(example => example.id === file.id)
}

export function shaderModuleId(fileId: string, files: readonly Pick<ShaderFile, 'id' | 'kind'>[]) {
  return files.some(file => file.id === fileId && isComparisonFile(file)) ? fileId : 'score'
}

export function selectShaderModule(files: readonly ShaderFile[], activeFileId: string): ShaderFile[] {
  const moduleId = shaderModuleId(activeFileId, files)
  if (moduleId !== 'score') return files.filter(file => file.id === moduleId)
  return files.filter(file => file.id === 'score' || !isComparisonFile(file))
}
