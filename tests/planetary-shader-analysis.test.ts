import { strict as assert } from 'node:assert'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import {
  analyzePlanetaryShader,
  mentionsPlanetaryName,
  PlanetaryAnalysisError,
  readPlanetaryCalls,
} from '../src/workspace/planetaryShaderAnalysis.ts'

// Captured from wgsl-analyzer-web@0.11.539's viewSyntaxTree request.
// Insignificant comment and whitespace tokens were removed from the fixtures.
const fixtures: { source: string; tree: unknown }[] = JSON.parse(
  readFileSync(new URL('./fixtures/planetary-syntax-trees.json', import.meta.url), 'utf8'),
)
const read = ({ source, tree }: (typeof fixtures)[number]) => readPlanetaryCalls(source, JSON.stringify(tree))

test('the syntax tree resolves signed numeric consts and aliases and maps the whole call', () => {
  const file = fixtures[0]
  const calls = read(file)
  assert.deepEqual(
    calls.map(({ body, latitude, longitude }) => [body, latitude, longitude]),
    [
      ['moon', -12, 12],
      ['mars', 90, 0],
      ['moon', 0, -180],
    ],
  )
  const [first, second] = calls
  assert.equal(first.line, 5)
  assert.equal(first.column, file.source.split('\n')[4].indexOf('moon_point') + 1)
  assert.equal(first.length, 'moon_point'.length)
  assert.equal(file.source.slice(first.offset, first.endOffset), 'moon_point(lat, -(x))')
  assert.equal(file.source.slice(second.offset, second.endOffset), 'mars_point(\n    90.0,\n    47.0,\n  )')
})

test('the syntax tree reports runtime expressions, bounds, arity and reserved functions', () => {
  assert.throws(
    () => read(fixtures[1]),
    (error: unknown) => {
      assert.ok(error instanceof PlanetaryAnalysisError)
      const diagnostics = error.diagnostics
      assert.deepEqual(
        diagnostics.map(d => d.line),
        [4, 7, 9, 9, 11, 15, 16, 19],
      )
      assert.match(diagnostics[0].message, /module-scope/)
      assert.match(diagnostics[2].message, /-90° and 90°/)
      assert.match(diagnostics[3].message, /-180° and 180°/)
      assert.deepEqual([diagnostics[4].column, diagnostics[4].length], [5, 6])
      assert.match(diagnostics[5].message, /exactly two arguments/)
      assert.match(diagnostics[6].message, /without a namespace/)
      assert.match(diagnostics[7].message, /reserved/)
      return true
    },
  )
})

test('the syntax tree honors declaration order and rejects shadowed, cyclic and incompatible values', () => {
  assert.throws(
    () => read(fixtures[2]),
    (error: unknown) => {
      assert.ok(error instanceof PlanetaryAnalysisError)
      assert.deepEqual(
        error.diagnostics.map(d => d.line),
        [6, 7, 8],
      )
      return true
    },
  )
})

test('ordinary and frozen shaders work without the isolated browser analyzer', async () => {
  const { signal } = new AbortController()
  assert.deepEqual(
    await analyzePlanetaryShader(
      `
    /* /* mars_point */ moon_point */
    // moon_point(2.0, 3.0)
    fn ordinary() { let lunar = MoonPoint(); }
  `,
      signal,
    ),
    [],
  )
  await assert.rejects(
    analyzePlanetaryShader('fn sample() { let point = moon_point(0.0, 0.0); }', signal),
    /require the local WGSL analyzer/,
  )
})

test('the lexical gate sees complete identifiers following WGSL line breaks', () => {
  assert.equal(mentionsPlanetaryName('ἀmoon_point mars_point_extended'), -1)
  const source = '// ignore moon_point\u2028mars_point'
  assert.equal(mentionsPlanetaryName(source), source.indexOf('mars_point'))
})
