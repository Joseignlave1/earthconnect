import { strict as assert } from 'node:assert'
import { readFileSync } from 'node:fs'
import { registerHooks } from 'node:module'
import test, { type TestContext } from 'node:test'
import { comparisonExamples } from '../src/workspace/comparisonExamples.ts'
const wgsl = registerHooks({
  load(url, context, nextLoad) {
    if (!url.endsWith('.wgsl?raw')) return nextLoad(url, context)
    return { format: 'module', source: `export default ${JSON.stringify(readFileSync(new URL(url), 'utf8'))}`, shortCircuit: true }
  },
})
const { assembleShaderProject, findShaderLocation, readShaderProject, resolveShaderFiles, saveShaderProject, selectShaderModule } = await import('../src/workspace/shaderProject.ts')
const { helloWorldExample } = await import('../src/workspace/helloWorldExample.ts')
wgsl.deregister()

function mockStorage(t: TestContext, saved: string | null) {
  const storage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: () => saved, setItem: (_key: string, value: string) => { saved = value } },
  })
  t.after(() => {
    if (storage) Object.defineProperty(globalThis, 'localStorage', storage)
    else Reflect.deleteProperty(globalThis, 'localStorage')
  })
}

test('project assembly keeps diagnostic locations across CRLF, trailing lines and empty files', () => {
  const { source, segments } = assembleShaderProject([
    { id: 'score', name: 'score.wgsl', source: '// score\r\n\r\n' },
    { id: 'empty', name: 'empty.wgsl', source: '' },
    { id: 'helper', name: 'helper.wgsl', source: '// helper\nfn helper() {}' },
  ])
  assert.equal(source, '// score\n\n\n\n// helper\nfn helper() {}')
  assert.deepEqual(findShaderLocation(segments, 3), { fileId: 'score', line: 3 })
  assert.deepEqual(findShaderLocation(segments, 4), { fileId: 'empty', line: 1 })
  assert.deepEqual(findShaderLocation(segments, 6), { fileId: 'helper', line: 2 })
  assert.equal(findShaderLocation(segments, 7), undefined)
})

test('closed files remain in the module and an untouched score follows the sampled baseline', () => {
  const project = {
    activeFileId: 'score',
    files: [
      { id: 'score', name: 'main.wgsl', source: null },
      { id: 'helper', name: 'helper.wgsl', source: 'fn helper() {}', closed: true },
    ],
  }
  const files = resolveShaderFiles(project, '// basin')
  assert.equal(files.find(file => file.id === 'helper')?.closed, true)
  assert.equal(assembleShaderProject(files).source, '// basin\nfn helper() {}')
  assert.equal(assembleShaderProject(resolveShaderFiles(project, '// sample')).source, '// sample\nfn helper() {}')
})

test('default examples run separately from each other and the shared score files', t => {
  mockStorage(t, null)
  const project = readShaderProject()
  const helper = { id: 'helper', name: 'helper.wgsl', source: 'fn helper() {}', closed: true }
  const files = resolveShaderFiles({ ...project, files: [...project.files, helper] }, '// basin')
  assert.equal(project.activeFileId, helloWorldExample.id)
  assert.equal(project.files.find(file => file.id === helloWorldExample.id)?.closed, undefined)
  assert.equal(comparisonExamples.length, 3)
  for (const example of [...comparisonExamples, helloWorldExample]) {
    assert.equal(project.files.find(file => file.id === example.id)?.source, null)
    const { source, segments } = assembleShaderProject(selectShaderModule(files, example.id))
    assert.equal(source, example.source.replace(/\r\n?/g, '\n'))
    assert.deepEqual(findShaderLocation(segments, 1), { fileId: example.id, line: 1 })
  }
  for (const id of ['score', helper.id]) {
    assert.equal(assembleShaderProject(selectShaderModule(files, id)).source, '// basin\nfn helper() {}')
  }
})

test('saved user shaders compile independently while helpers still belong to the score module', t => {
  mockStorage(t, null)
  const initial = readShaderProject()
  const terrain = { id: 'terrain', name: 'terrain.wgsl', kind: 'shader', source: 'fn score(cell: EarthCell) -> f32 { return 0.2; }' } as const
  const vegetation = { id: 'vegetation', name: 'vegetation.wgsl', kind: 'shader', source: 'fn score(cell: EarthCell) -> f32 { return 0.9; }', closed: true } as const
  const legacyHelper = { id: 'legacy', name: 'legacy.wgsl', source: 'fn legacy() {}' }
  const helper = { id: 'shared', name: 'shared.wgsl', kind: 'helper', source: 'fn shared() {}' } as const
  const project = { ...initial, files: [...initial.files, terrain, vegetation, legacyHelper, helper] }
  saveShaderProject(project)
  const restored = readShaderProject()
  assert.deepEqual(restored, project)
  const files = resolveShaderFiles(restored, '// baseline score')
  for (const shader of [terrain, vegetation]) {
    const module = assembleShaderProject(selectShaderModule(files, shader.id))
    assert.equal(module.source, shader.source)
    assert.deepEqual(findShaderLocation(module.segments, 1), { fileId: shader.id, line: 1 })
  }
  for (const id of ['score', legacyHelper.id, helper.id]) {
    assert.deepEqual(selectShaderModule(files, id).map(file => file.id), ['score', legacyHelper.id, helper.id])
  }
})

test('adding the default tabs preserves existing files, tab selection and saved example changes', t => {
  const [first, second] = comparisonExamples
  const oldProject = {
    activeFileId: 'helper',
    files: [
      { id: 'score', name: 'renamed.wgsl', source: 'fn score() {}', closed: true },
      { id: 'helper', name: first.name, source: 'fn helper() {}' },
    ],
  }
  mockStorage(t, JSON.stringify(oldProject))
  const merged = readShaderProject()
  assert.deepEqual(merged.files.slice(0, 2), oldProject.files)
  assert.equal(merged.activeFileId, oldProject.activeFileId)
  assert.equal(new Set(merged.files.map(file => file.name.toLowerCase())).size, merged.files.length)
  assert.deepEqual(merged.files.slice(2).map(file => file.id), [...comparisonExamples.map(file => file.id), helloWorldExample.id])
  assert.equal(merged.files.find(file => file.id === helloWorldExample.id)?.closed, true)
  const edited = {
    activeFileId: second.id,
    files: merged.files.map(file => file.id === first.id
      ? { ...file, name: 'edited-example.wgsl', source: '// custom example', closed: true }
      : file),
  }
  saveShaderProject(edited)
  const restored = readShaderProject()
  assert.deepEqual(restored, edited)
  const files = resolveShaderFiles(restored, '// basin')
  assert.equal(assembleShaderProject(selectShaderModule(files, first.id)).source, '// custom example')
  assert.equal(assembleShaderProject(selectShaderModule(files, second.id)).source, second.source)
})

test('a project restores its active shader when every WGSL tab is closed', t => {
  mockStorage(t, null)
  const initial = readShaderProject()
  const activeFileId = comparisonExamples[0].id
  const project = {
    activeFileId,
    files: initial.files.map(file => ({
      ...file,
      source: file.id === activeFileId ? '// retained shader' : file.source,
      closed: true,
    })),
  }
  saveShaderProject(project)
  const restored = readShaderProject()
  assert.deepEqual(restored, project)
  const files = resolveShaderFiles(restored, '// basin')
  assert.equal(assembleShaderProject(selectShaderModule(files, restored.activeFileId)).source, '// retained shader')
})
