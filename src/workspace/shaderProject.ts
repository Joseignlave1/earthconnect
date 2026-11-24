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

export type ShaderProjectStorage = {
  load(): Promise<unknown>
  save(project: ShaderProject): Promise<void>
}

const legacyStorageKey = 'earth-connect-wgsl-project'
const dbName = 'earth-connect-wgsl-project'
const filesStore = 'files'
const metaStore = 'meta'
const metaKey = 'project'

type ProjectMeta = { activeFileId: string; fileOrder: string[] }

let openDatabasePromise: Promise<IDBDatabase | null> | null = null
let writeQueue = Promise.resolve()

function openProjectDatabase(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null)
  if (!openDatabasePromise) {
    openDatabasePromise = new Promise(resolve => {
      let settled = false
      const finish = (db: IDBDatabase | null) => {
        if (settled) return
        settled = true
        if (!db) openDatabasePromise = null
        resolve(db)
      }
      try {
        const request = indexedDB.open(dbName, 1)
        request.onupgradeneeded = () => {
          const db = request.result
          if (!db.objectStoreNames.contains(filesStore)) db.createObjectStore(filesStore, { keyPath: 'id' })
          if (!db.objectStoreNames.contains(metaStore)) db.createObjectStore(metaStore)
        }
        request.onsuccess = () => {
          const db = request.result
          db.onversionchange = () => {
            db.close()
            openDatabasePromise = null
          }
          db.onclose = () => {
            openDatabasePromise = null
          }
          finish(db)
        }
        request.onerror = () => finish(null)
        request.onblocked = () => finish(null)
      } catch {
        finish(null)
      }
    })
  }
  return openDatabasePromise
}

function createIndexedDbStorage(): ShaderProjectStorage {
  return {
    async load() {
      const db = await openProjectDatabase()
      if (!db) return undefined
      return new Promise<unknown>((resolve, reject) => {
        const tx = db.transaction([filesStore, metaStore], 'readonly')
        const metaRequest = tx.objectStore(metaStore).get(metaKey)
        const filesRequest = tx.objectStore(filesStore).getAll()
        let meta: unknown
        let records: unknown[] = []
        metaRequest.onsuccess = () => {
          meta = metaRequest.result
        }
        filesRequest.onsuccess = () => {
          records = Array.isArray(filesRequest.result) ? filesRequest.result : []
        }
        tx.oncomplete = () => {
          if (!meta || typeof meta !== 'object') {
            resolve(null)
            return
          }
          const { activeFileId, fileOrder } = meta as Partial<ProjectMeta>
          const byId = new Map<string, unknown>()
          for (const record of records) {
            if (record && typeof record === 'object' && 'id' in record && typeof record.id === 'string') {
              byId.set(record.id, record)
            }
          }
          const ordered: unknown[] = []
          const seen = new Set<string>()
          if (Array.isArray(fileOrder)) {
            for (const id of fileOrder) {
              if (typeof id !== 'string' || seen.has(id) || !byId.has(id)) continue
              seen.add(id)
              ordered.push(byId.get(id))
            }
          }
          for (const [id, record] of byId) {
            if (!seen.has(id)) ordered.push(record)
          }
          resolve({ activeFileId, files: ordered })
        }
        tx.onerror = () => reject(tx.error ?? new Error('Could not read shader project from IndexedDB.'))
        tx.onabort = () => reject(tx.error ?? new Error('Shader project read transaction was aborted.'))
      })
    },
    async save(project) {
      const db = await openProjectDatabase()
      if (!db) throw new Error('IndexedDB is unavailable.')
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction([filesStore, metaStore], 'readwrite')
        const store = tx.objectStore(filesStore)
        const nextIds = new Set(project.files.map(file => file.id))
        const keysRequest = store.getAllKeys()
        keysRequest.onsuccess = () => {
          for (const key of keysRequest.result) {
            if (typeof key === 'string' && !nextIds.has(key)) store.delete(key)
          }
          for (const file of project.files) {
            store.put(file)
          }
          const meta: ProjectMeta = {
            activeFileId: project.activeFileId,
            fileOrder: project.files.map(file => file.id),
          }
          tx.objectStore(metaStore).put(meta, metaKey)
        }
        tx.oncomplete = () => resolve()
        tx.onerror = () => reject(tx.error ?? new Error('Could not save shader project to IndexedDB.'))
        tx.onabort = () => reject(tx.error ?? new Error('Shader project write transaction was aborted.'))
      })
    },
  }
}

const defaultStorage = createIndexedDbStorage()

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

function parseShaderProject(saved: unknown): ShaderProject | null {
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
        typeof file === 'object' &&
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
    return {
      files: addExamples(saved.files as ProjectFile[]),
      activeFileId: saved.activeFileId,
    }
  }
  return null
}

function readLegacyProject(): ShaderProject | null {
  try {
    const raw = localStorage.getItem(legacyStorageKey)
    if (!raw) return null
    return parseShaderProject(JSON.parse(raw))
  } catch {
    return null
  }
}

function writeLegacyProject(project: ShaderProject) {
  try {
    localStorage.setItem(legacyStorageKey, JSON.stringify(project))
  } catch {
    /* Storage may be unavailable. */
  }
}

function clearLegacyProject() {
  try {
    localStorage.removeItem(legacyStorageKey)
  } catch {
    /* Storage may be unavailable. */
  }
}

export function defaultShaderProject(): ShaderProject {
  return {
    files: addExamples([{ id: 'score', name: 'score.wgsl', source: null }], true),
    activeFileId: helloWorldExample.id,
  }
}

function enqueueWrite<T>(task: () => Promise<T>): Promise<T> {
  const pending = writeQueue.then(task, task)
  writeQueue = pending.then(
    () => undefined,
    () => undefined,
  )
  return pending
}

function cloneProject(project: ShaderProject): ShaderProject {
  return {
    activeFileId: project.activeFileId,
    files: project.files.map(file => ({ ...file })),
  }
}

export async function readShaderProject(storage: ShaderProjectStorage | null = defaultStorage): Promise<ShaderProject> {
  if (storage) {
    try {
      const parsed = parseShaderProject(await storage.load())
      if (parsed) return parsed
    } catch {
      /* Fall back to legacy storage or defaults when IndexedDB is unavailable. */
    }
  }
  const legacy = readLegacyProject()
  if (legacy) {
    if (storage) {
      try {
        await enqueueWrite(() => storage.save(cloneProject(legacy)))
        clearLegacyProject()
      } catch {
        /* Keep the legacy copy if IndexedDB migration failed. */
      }
    }
    return legacy
  }
  return defaultShaderProject()
}

export function saveShaderProject(
  project: ShaderProject,
  storage: ShaderProjectStorage | null = defaultStorage,
): Promise<void> {
  const snapshot = cloneProject(project)
  return enqueueWrite(async () => {
    if (storage) {
      try {
        await storage.save(snapshot)
        return
      } catch {
        /* Fall back to localStorage when IndexedDB is unavailable. */
      }
    }
    writeLegacyProject(snapshot)
  })
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
