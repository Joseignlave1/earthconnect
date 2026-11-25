import {
  buildEarthQuery,
  loadEarthScore,
  freezePlanetaryShaderPoint,
  SCORE_MAX,
  SCORE_MISSING,
  SCORE_SHADER_EXAMPLES,
  ScoreShaderError,
  sourceCodeWGSL,
} from '/earth-score.mjs'

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

async function rejects(promise, pattern) {
  try {
    await promise
  } catch (error) {
    assert(pattern.test(String(error)), `Unexpected rejection: ${error}`)
    return
  }
  throw new Error(`Expected rejection matching ${pattern}`)
}

async function sha256(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('')
}

function trackResources(device) {
  const buffers = []
  const workers = []
  const create = device.createBuffer.bind(device)
  const WorkerImpl = globalThis.Worker
  device.createBuffer = config => {
    const buffer = create(config)
    const item = { label: config.label, destroyed: false }
    const destroy = buffer.destroy.bind(buffer)
    buffer.destroy = () => {
      item.destroyed = true
      destroy()
    }
    buffers.push(item)
    tracker.onBuffer?.(item)
    return buffer
  }
  globalThis.Worker = class TrackedWorker extends WorkerImpl {
    constructor(...args) {
      super(...args)
      const item = { terminated: false }
      const terminate = this.terminate.bind(this)
      this.terminate = () => {
        item.terminated = true
        terminate()
      }
      workers.push(item)
    }
  }
  const tracker = {
    buffers,
    workers,
    onBuffer: undefined,
    restore() {
      device.createBuffer = create
      globalThis.Worker = WorkerImpl
    },
  }
  return tracker
}

function compareScores(actual, expected, stats, i) {
  for (let row = 0; row < actual.length; row++) {
    const a = actual[row],
      b = expected[row]
    if (a === b) {
      stats.exact++
      if (a === SCORE_MISSING) stats.missing++
    } else {
      assert(
        a !== SCORE_MISSING && b !== SCORE_MISSING && Math.abs(a - b) === 1,
        `Score mismatch at chunk ${i} row ${row}: got ${a}, expected ${b}`,
      )
      stats.withinOne++
    }
  }
}

function codePipeline(device) {
  const shader = device.createShaderModule({
    code: `
    @group(0) @binding(0) var<storage, read> packedInput: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    ${sourceCodeWGSL}
    @compute @workgroup_size(256)
    fn reconstruct(@builtin(global_invocation_id) id: vec3u) {
      let rows = packedInput[2];
      let count = rows * packedInput[3];
      let i = id.x * 2u;
      if (i >= count) { return; }
      let low = sourceCode(rows, i / rows, i % rows);
      var high = 0u;
      if (i + 1u < count) { high = sourceCode(rows, (i + 1u) / rows, (i + 1u) % rows); }
      output[id.x] = low | (high << 16u);
    }
  `,
  })
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shader, entryPoint: 'reconstruct' },
  })
}

function verifyQueryScores(earth, criteria, codes, actual) {
  const rows = actual.length
  const families = new Map()
  const terms = criteria.map(criterion => {
    const field = earth.fields.findIndex(({ name }) => name === criterion.key)
    const coverage = earth.fields.findIndex(({ name }) => name === criterion.qualityKey)
    if (criterion.enabled) {
      const family = families.get(criterion.group) ?? { sum: 0, max: 0 }
      family.sum += criterion.weight
      family.max = Math.max(family.max, criterion.weight)
      families.set(criterion.group, family)
    }
    return { ...criterion, field, coverage }
  })
  const total = [...families.values()].reduce((sum, family) => sum + family.max, 0)
  const physical = (field, row) => {
    const code = codes[field * rows + row]
    const { origin, denominator } = earth.fields[field].encoding
    return code ? (code - origin) / denominator : null
  }
  const stats = { missing: 0, missingAndRejected: 0, rejected: 0, positive: 0, best: 0, withinOne: 0 }
  for (let row = 0; row < rows; row++) {
    let missing = false
    let rejected = false
    let loss = 0
    for (const term of terms) {
      if (term.coverage >= 0) {
        const coverage = physical(term.coverage, row)
        if (coverage === null || coverage < 0.9) missing = true
      }
      const value = physical(term.field, row)
      if (value === null) {
        missing = true
        continue
      }
      if ((term.min !== null && value < term.min) || (term.max !== null && value > term.max)) rejected = true
      if (term.enabled) {
        const family = families.get(term.group)
        const distance = term.categorical
          ? value === term.target
            ? 0
            : 25
          : ((value - term.target) / term.tolerance) ** 2
        loss += ((term.weight * distance) / family.sum) * (family.max / total)
      }
    }
    const code = actual[row]
    if (missing) {
      assert(code === SCORE_MISSING, `Expected missing criteria score at row ${row}, got ${code}`)
      stats.missing++
      if (rejected) stats.missingAndRejected++
    } else if (rejected) {
      assert(code === 0, `Expected hard rejection at row ${row}, got ${code}`)
      stats.rejected++
    } else {
      const score = SCORE_MAX * Math.exp(-0.5 * Math.min(1000, loss))
      const whole = Math.floor(score)
      const fraction = score - whole
      const expected = whole + Number(fraction > 0.5 || (fraction === 0.5 && whole % 2 === 1))
      assert(
        code !== SCORE_MISSING && Math.abs(code - expected) <= 1,
        `Criteria score differs at row ${row}: got ${code}, expected ${expected}`,
      )
      if (code !== expected) stats.withinOne++
      if (code > 0) stats.positive++
      stats.best = Math.max(stats.best, code)
    }
  }
  return stats
}

function baselineQuerySource(earth, overrides = {}) {
  const query = buildEarthQuery(
    earth.baseline.map((feature, index) => ({
      key: feature.field,
      label: feature.field,
      group: String(feature.group),
      unit: '',
      enabled: true,
      target: feature.target,
      tolerance: feature.tolerance,
      weight: feature.weight,
      min: null,
      max: null,
      qualityKey: index === 0 ? 'terrain_valid_fraction' : index === 1 ? 'landcover_valid_fraction' : undefined,
      ...overrides[feature.field],
    })),
  )
  return earth.queryShaderSource(query)
}

async function verifyFullWorld(device, earth, ref) {
  const makeStats = () => ({ exact: 0, withinOne: 0, missing: 0 })
  const baseline = makeStats()
  const moved = makeStats()
  const hashes = []
  const rawBytes = Math.max(...earth.chunks.map(c => c.rows)) * earth.fields.length * 2
  const staging = device.createBuffer({ size: rawBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const codes = device.createBuffer({ size: rawBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
  const decoder = codePipeline(device)

  async function read(buffer, bytes, consume, encoder = device.createCommandEncoder()) {
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes)
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ, 0, bytes)
    try {
      return await consume(staging.getMappedRange(0, bytes))
    } finally {
      staging.unmap()
    }
  }

  async function checkScores(stats, movedQuery) {
    for (const chunk of earth.chunks) {
      const { index, rows, scores } = chunk
      const response = await fetch(`/reference/${ref.chunks[index].file}`)
      assert(response.ok, `Missing score reference for chunk ${index}`)
      const bytes = await response.arrayBuffer()
      assert(bytes.byteLength === rows * 4, `Incorrect reference size for chunk ${index}`)
      const expected = new Uint16Array(bytes, movedQuery ? rows * 2 : 0, rows)
      await read(scores, rows * 4, async data => {
        compareScores(new Uint32Array(data), expected, stats, index)
        if (!movedQuery) hashes[index] = await sha256(data)
      })
    }
  }

  async function readCodes({ rows, packedInput }, consume) {
    const size = rows * earth.fields.length * 2
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(decoder)
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: decoder.getBindGroupLayout(0),
        entries: [packedInput, codes].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    )
    pass.dispatchWorkgroups(Math.ceil(size / 4 / 256))
    pass.end()
    return read(codes, size, consume, encoder)
  }

  async function checkCriteriaQueries() {
    const address = earth.layout.addressAt(85, 33)
    const chunk = earth.chunks[address.chunk]
    const { rows, scores, index } = chunk
    const storedCodes = await readCodes(chunk, data => new Uint16Array(data.slice(0)))
    const criterion = (key, settings = {}) => ({
      key,
      label: key,
      group: key,
      enabled: true,
      target: 0,
      tolerance: 1,
      weight: 1,
      min: null,
      max: null,
      ...settings,
    })
    const terrain = { qualityKey: 'terrain_valid_fraction' }
    const criteria = [
      criterion('slope_mean_deg', { ...terrain, target: 4, tolerance: 4, min: 0.9 }),
      criterion('relief_5km_m', { ...terrain, target: 200, tolerance: 200, weight: 2, group: 'relief' }),
      criterion('relief_21km_m', { ...terrain, target: 800, tolerance: 450, weight: 0.5, group: 'relief' }),
      criterion('surface_range_c', { target: 56, tolerance: 30, weight: 0.5 }),
      criterion('dominant_lithology_code', { target: 5, categorical: true, weight: 0.125 }),
      criterion('bare_fraction', {
        enabled: false,
        target: null,
        weight: 0,
        min: 0.25,
        max: 0.95,
        qualityKey: 'landcover_valid_fraction',
      }),
    ]
    const noCandidates = [criterion('slope_mean_deg', { ...terrain, target: 4, min: 1e10 })]
    const plan = buildEarthQuery(criteria)
    await earth.dispatchShader(earth.queryShaderSource(plan))
    const criteriaScores = await read(scores, rows * 4, data =>
      verifyQueryScores(earth, criteria, storedCodes, new Uint32Array(data)),
    )
    await earth.dispatchShader(earth.queryShaderSource(buildEarthQuery(noCandidates)))
    const emptyScores = await read(scores, rows * 4, data =>
      verifyQueryScores(earth, noCandidates, storedCodes, new Uint32Array(data)),
    )
    assert(
      criteriaScores.missing > 0 &&
        criteriaScores.missingAndRejected > 0 &&
        criteriaScores.rejected > 0 &&
        criteriaScores.positive > 0 &&
        criteriaScores.best > SCORE_MAX / 2,
      'Expected missing, rejected, and strong criteria matches',
    )
    assert(
      emptyScores.missing > 0 && emptyScores.rejected > 0 && emptyScores.positive === 0,
      'Impossible filter should yield no candidates',
    )
    assert(emptyScores.missing < criteriaScores.missing, 'A sampled subset should require fewer values')

    const elevation = earth.fields.findIndex(({ name }) => name === 'elevation_median_m')
    const { origin, denominator } = earth.fields[elevation].encoding
    const elevationCode = storedCodes[elevation * rows + address.row]
    assert(elevationCode !== 0, 'Expected elevation data at the sampled test address')
    const exactElevation = (elevationCode - origin) / denominator
    const tolerance = 1e-6
    const narrow = [criterion('elevation_median_m', { target: exactElevation + tolerance, tolerance })]
    const narrowPlan = buildEarthQuery(narrow)
    await earth.dispatchShader(earth.queryShaderSource(narrowPlan))
    const narrowScore = await read(scores, rows * 4, data => new Uint32Array(data)[address.row])
    assert(narrowScore === Math.round(SCORE_MAX * Math.exp(-0.5)), `Unexpected narrow-tolerance score: ${narrowScore}`)

    await earth.dispatchShader(baselineQuerySource(earth))
    await read(scores, rows * 4, async data => {
      assert((await sha256(data)) === hashes[index], 'Criteria query changed the baseline')
    })
    return { criteriaScores, emptyScores, narrowScore }
  }

  async function checkShaderCoordinates() {
    const boundary = earth.chunks[1].firstAddress
    const checks = [
      { address: earth.layout.locate(0), x: 0, y: 1946 },
      { address: earth.layout.locate(boundary - 1), x: 833, y: 2223 },
      { address: earth.layout.locate(boundary), x: 556, y: 2224 },
      { address: earth.layout.addressOf('h02v06', 71 * 278 + 6), x: 562, y: 1739 },
      { address: earth.layout.addressOf('h02v06', 71 * 278 + 7), x: 563, y: 1739 },
      { address: earth.layout.addressOf('h20v09', 71 * 278 + 6), x: 5566, y: 2573 },
      { address: earth.layout.addressOf('h33v11', 71 * 278 + 6), x: 9180, y: 3129 },
    ]
    const assertCoordinates = checks
      .map(({ address, x, y }) => {
        const comparison = address.center
          ? `abs(cell.latitude_deg - ${address.center.latitude}) < 0.00003 &&
           abs(cell.longitude_deg - ${address.center.longitude}) < 0.00003`
          : 'cell.latitude_deg == NO_DATA && cell.longitude_deg == NO_DATA'
        return `if (cell.grid_x == ${x}u && cell.grid_y == ${y}u) {
        return select(NO_SCORE, 1.0, ${comparison});
      }`
      })
      .join('\n')
    await earth.dispatchShader(`fn score(cell: EarthCell) -> f32 {
      ${assertCoordinates}
      return NO_SCORE;
    }`)
    for (const { index, scores, rows } of earth.chunks) {
      const selected = checks.filter(({ address }) => address.chunk === index)
      if (!selected.length) continue
      await read(scores, rows * 4, data => {
        const codes = new Uint32Array(data)
        for (const { address, x, y } of selected) {
          assert(
            codes[address.row] === SCORE_MAX,
            `Wrong coordinates at chunk ${index}, row ${address.row} (grid ${x}, ${y})`,
          )
        }
      })
    }
  }

  async function checkShaderScores() {
    const address = earth.layout.addressAt(85, 33)
    const chunk = earth.chunks[address.chunk]
    const { rows, scores } = chunk
    const storedCodes = await readCodes(chunk, data => new Uint16Array(data.slice(0)))
    const physical = (name, row) => {
      const field = earth.fields.findIndex(value => value.name === name)
      const { origin, denominator } = earth.fields[field].encoding
      const code = storedCodes[field * rows + row]
      return code ? (code - origin) / denominator : null
    }

    await earth.dispatchShader(`diagnostic(off, derivative_uniformity);
      const scale = 8.0;
      fn flat(pixel: EarthCell) -> f32 {
        let terrain = array<f32, 2>(pixel.bare_fraction, pixel.slope_mean_deg);
        for (var i = 0u; i < 2u; i++) {
          if (!has_data(terrain[i])) { return NO_SCORE; }
        }
        return terrain[0] * exp(-terrain[1] / scale);
      }
      fn score(other: EarthCell) -> f32 {
        let copy = other;
        return flat(copy);
      }
    `)
    const reference = await read(scores, rows * 4, async data => {
      const scoreCodes = new Uint32Array(data)
      for (let row = 0; row < rows; row++) {
        const bare = physical('bare_fraction', row),
          slope = physical('slope_mean_deg', row)
        const expected =
          bare === null || slope === null ? SCORE_MISSING : Math.round(bare * Math.exp(-slope / 8) * SCORE_MAX)
        const actual = scoreCodes[row]
        assert(
          expected === SCORE_MISSING
            ? actual === SCORE_MISSING
            : actual !== SCORE_MISSING && Math.abs(expected - actual) <= 1,
          `Incorrect WGSL score at row ${row}: ${actual} against ${expected}`,
        )
      }
      return sha256(data)
    })
    try {
      await earth.dispatchShader('fn score(cell: EarthCell) -> f32 {\n  return cell.unknown_field;\n}')
      throw new Error('Invalid WGSL compiled')
    } catch (error) {
      assert(
        error instanceof ScoreShaderError &&
          error.diagnostics.some(
            diagnostic =>
              diagnostic.line === 2 && diagnostic.column >= 10 && diagnostic.message.includes('unknown_field'),
          ),
        `Unexpected WGSL diagnostics: ${JSON.stringify(error.diagnostics)}`,
      )
    }
    await read(scores, rows * 4, async data => {
      assert((await sha256(data)) === reference, 'Invalid WGSL changed the score buffers')
    })
    const superseded = earth.dispatchShader(SCORE_SHADER_EXAMPLES[0].source)
    const constant = earth.dispatchShader('fn score(cell: EarthCell) -> f32 { return 1.5; }')
    await Promise.all([rejects(superseded, /superseded/), constant])
    await read(scores, rows * 4, data => {
      assert(
        new Uint32Array(data).every(code => code === SCORE_MAX),
        'Custom score failed to clamp or lost the latest edit',
      )
    })
    await rejects(earth.dispatchShader(SCORE_SHADER_EXAMPLES[1].source, AbortSignal.abort()), /abort/i)
    await earth.dispatchShader(`fn score(cell: EarthCell) -> f32 {
      return cell.bare_fraction / (cell.slope_mean_deg - cell.slope_mean_deg);
    }`)
    await read(scores, rows * 4, data => {
      assert(
        new Uint32Array(data).every(code => code === SCORE_MISSING),
        'Nonfinite shader outputs should remain missing',
      )
    })
    await earth.dispatchShader(SCORE_SHADER_EXAMPLES[0].source)
    const basin = await read(scores, rows * 4, data => new Uint32Array(data)[address.row])
    assert(basin > 0 && basin <= SCORE_MAX, 'Starter WGSL must score the Himalayan sample')
    const measurement = (variableId, unit, decoded, status = 1) => ({
      variableId,
      unit,
      decoded,
      status,
      qualityFlags: [],
    })
    const moon = {
      body: 'moon',
      latitude: 90,
      longitude: -27,
      partial: true,
      measurements: [
        measurement('moon.relief_5km', 'm', 300),
        measurement('moon.surface_noon', 'K', 150),
        measurement('moon.surface_min', 'K', 110, 2),
      ],
    }
    const mars = {
      body: 'mars',
      latitude: 18.65,
      longitude: -133.8,
      measurements: [
        measurement('mars.relief_5km', 'm', 800),
        measurement('mars.modeled_surface_temperature_mean', 'degree_C', -50),
      ],
    }
    const sourceWithPoints = (moonCode, marsCode) => `fn score(cell: EarthCell) -> f32 {
      let lunar = ${moonCode};
      let martian = ${marsCode};
      if (!lunar.sampled || !lunar.partial || !martian.sampled ||
          has_data(lunar.temperature.min_k)) { return NO_SCORE; }
      return (lunar.terrain.relief_5km_m + martian.terrain.relief_5km_m +
        lunar.temperature.noon_k + martian.temperature.mean_c) / 4000.;
    }`
    const lunar = 'moon_point(90., 180.)'
    const martian = 'mars_point(18.65, -133.8)'
    const liveSource = sourceWithPoints(lunar, martian)
    const samples = [moon, ...Array.from({ length: 39 }, (_, longitude) => ({ ...moon, latitude: 0, longitude })), mars]
    const calls = [
      { body: 'moon', offset: liveSource.indexOf(lunar), length: lunar.length, slot: 0 },
      { body: 'mars', offset: liveSource.indexOf(martian), length: martian.length, slot: samples.length - 1 },
    ]
    const readOne = () => read(scores, rows * 4, data => new Uint32Array(data)[address.row])
    await earth.dispatchShader(liveSource, undefined, { calls, samples })
    const live = await readOne()
    assert(live === Math.round(0.3 * SCORE_MAX), `Planetary intrinsic returned ${live}`)
    await earth.dispatchShader(sourceWithPoints(freezePlanetaryShaderPoint(moon), freezePlanetaryShaderPoint(mars)))
    assert((await readOne()) === live, 'Frozen planetary values differ from the live binding')

    const line = '  /* 🌖 café */ return moon_point(90., 180.).bad_field;'
    const badSource = `fn score(cell: EarthCell) -> f32 {\r\n${line}\r\n}`
    try {
      await earth.dispatchShader(badSource, undefined, {
        samples: [moon],
        calls: [{ ...calls[0], offset: badSource.indexOf(lunar) }],
      })
      throw new Error('Invalid planetary point field compiled')
    } catch (error) {
      assert(
        error instanceof ScoreShaderError &&
          error.diagnostics.some(
            diagnostic =>
              diagnostic.line === 2 &&
              diagnostic.column === line.indexOf(lunar) + 1 &&
              diagnostic.length === `${lunar}.bad_field`.length &&
              diagnostic.message.includes('bad_field'),
          ),
        `Incorrect Unicode position for planetary point field: ${JSON.stringify(error.diagnostics)}`,
      )
    }

    const compile = device.createComputePipelineAsync
    device.createComputePipelineAsync = () => {
      throw new Error('Expected shader pipeline to be cached')
    }
    try {
      const warmer = [
        ...samples.slice(0, -1),
        {
          ...mars,
          longitude: 45,
          measurements: [measurement('mars.relief_5km', 'm', 1600), ...mars.measurements.slice(1)],
        },
      ]
      await earth.dispatchShader(liveSource, undefined, { calls, samples: warmer })
      assert(
        (await readOne()) === Math.round(0.5 * SCORE_MAX),
        'Existing pipeline did not see changed planetary values',
      )
      await earth.dispatchShader(liveSource, undefined, {
        calls,
        samples: [...samples.slice(0, -1), { ...mars, measurements: [] }],
      })
      assert((await readOne()) === SCORE_MISSING, 'A fresh planetary binding can still read stale samples')
    } finally {
      device.createComputePipelineAsync = compile
    }
    return { basin, flat: reference, live }
  }

  let reconstructedBytes = 0
  try {
    await checkScores(baseline, false)
    const warmStart = performance.now()
    await earth.dispatchShader(baselineQuerySource(earth, ref.changed))
    const changedQueryMs = Math.round(performance.now() - warmStart)
    await checkScores(moved, true)
    await earth.dispatchShader(baselineQuerySource(earth))
    for (const { index, rows, scores, packedInput } of earth.chunks) {
      await read(scores, rows * 4, async data => {
        assert((await sha256(data)) === hashes[index], `Restored baseline differs at chunk ${index}`)
      })
      await readCodes({ rows, packedInput }, async data => {
        assert(
          (await sha256(data)) === ref.chunks[index].sourceCodesSha256,
          `Reconstructed codes differ at chunk ${index}`,
        )
      })
      reconstructedBytes += rows * earth.fields.length * 2
    }
    const criteriaQueries = await checkCriteriaQueries()
    await checkShaderCoordinates()
    const shaderScores = await checkShaderScores()
    await earth.dispatchShader(baselineQuerySource(earth))
    return { baseline, moved, reconstructedBytes, changedQueryMs, hashes, shaderScores, ...criteriaQueries }
  } finally {
    staging.destroy()
    codes.destroy()
  }
}

async function verifyFailurePaths(device, tracker) {
  const nativeFetch = globalThis.fetch
  const manifest = await (await nativeFetch('/world/manifest.json')).json()
  const small = {
    ...manifest,
    sites: manifest.sites.slice(0, 39),
    chunks: manifest.chunks.slice(0, 3),
    rows: 39 * 278 ** 2,
  }
  const manifestUrl = new URL('/test-world/manifest.json', location.href)

  async function runFailure(reason, configure, { signal, selected = small } = {}) {
    const bufferMark = tracker.buffers.length
    const workerMark = tracker.workers.length
    globalThis.fetch = async (input, options) => {
      const file = String(input).split('/').at(-1)
      if (file === 'manifest.json') return new Response(JSON.stringify(selected))
      return configure(file, options)
    }
    try {
      await rejects(loadEarthScore(device, { manifestUrl, signal }), reason)
    } finally {
      globalThis.fetch = nativeFetch
    }
    const buffers = tracker.buffers.slice(bufferMark)
    const workers = tracker.workers.slice(workerMark)
    assert(buffers.length > 0 && buffers.every(item => item.destroyed), 'Failed load leaked GPU buffers')
    assert(workers.length > 0 && workers.every(item => item.terminated), 'Failed load leaked decode workers')
    return buffers.length
  }

  await runFailure(/HTTP 404/, () => new Response(null, { status: 404 }))
  await runFailure(/compressed length/, () => new Response(new ArrayBuffer(32)))
  await runFailure(
    /Zstd frame/,
    file => new Response(new ArrayBuffer(small.chunks.find(c => c.file === file).compressedBytes)),
  )
  await runFailure(/GPU block header/, (file, options) => nativeFetch(`/world/${file}`, options), {
    selected: { ...small, columns: small.columns.slice(0, -1), encodings: small.encodings.slice(0, -1) },
  })

  const createBindGroup = device.createBindGroup.bind(device)
  device.createBindGroup = descriptor => createBindGroup({ ...descriptor, entries: [] })
  try {
    await runFailure(/Earth Score upload/, (file, options) => nativeFetch(`/world/${file}`, options))
  } finally {
    device.createBindGroup = createBindGroup
  }

  const controller = new AbortController()
  tracker.onBuffer = item => {
    if (item.label?.startsWith('Earth Score input')) controller.abort(new Error('Cancelled after a partial upload'))
  }
  try {
    const released = await runFailure(
      /Cancelled after a partial upload/,
      (file, options) => {
        if (file === small.chunks[0].file) return nativeFetch(`/world/${file}`, options)
        return new Promise((_, reject) => {
          options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true })
        })
      },
      { signal: controller.signal },
    )
    assert(released > 1, 'Partial load did not release resident chunk resources')
  } finally {
    tracker.onBuffer = undefined
  }
}

export async function verify() {
  assert(!crossOriginIsolated, 'The worker build must work without cross-origin isolation')
  const adapter = await navigator.gpu?.requestAdapter()
  assert(adapter && !adapter.info.isFallbackAdapter, 'A hardware WebGPU adapter is required')
  console.log(`WebGPU hardware: ${adapter.info.vendor} ${adapter.info.architecture}`)
  const device = await adapter.requestDevice()
  const uncaptured = []
  device.addEventListener('uncapturederror', event => uncaptured.push(event.error.message))
  const tracker = trackResources(device)
  let earth
  try {
    const reference = await (await fetch('/reference/manifest.json')).json()
    const start = performance.now()
    earth = await loadEarthScore(device, { manifestUrl: '/world/manifest.json' })
    const loadedMs = Math.round(performance.now() - start)
    await earth.dispatchShader(baselineQuerySource(earth))
    const firstQueryMs = Math.round(performance.now() - start)
    console.log(
      `Uploaded ${earth.chunks.length} chunks and scored ${earth.layout.rows} addresses in ${firstQueryMs} ms`,
    )
    assert(
      earth.layout.rows === 35550640 && earth.layout.sites.length === 460 && earth.fields.length === 25,
      'Unexpected world layout',
    )
    assert(
      tracker.workers.length >= 4 && tracker.workers.length <= 8 && tracker.workers.every(w => w.terminated),
      'Decode worker pool leaked',
    )
    const { hashes, ...report } = await verifyFullWorld(device, earth, reference)
    const baselineHash = await sha256(new TextEncoder().encode(hashes.join('')))
    await rejects(
      Promise.resolve().then(() => earth.queryShaderSource({})),
      /buildEarthQuery/,
    )
    earth.destroy()
    earth.destroy()
    assert(
      tracker.buffers.every(b => b.destroyed),
      'Successful load or validation leaked GPU buffers',
    )
    await rejects(earth.dispatchShader('fn score(cell: EarthCell) -> f32 { return 0.; }'), /destroyed/)
    device.pushErrorScope('validation')
    await verifyFailurePaths(device, tracker)
    assert(
      (await device.popErrorScope()) === null && uncaptured.length === 0,
      `Unexpected WebGPU errors: ${uncaptured}`,
    )
    const probe = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(probe, 0, new Uint32Array([1]))
    await device.queue.onSubmittedWorkDone()
    probe.destroy()
    return {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      loadedMs,
      firstQueryMs,
      ...report,
      baselineHash,
      cleanupAndFailurePaths: 'passed',
    }
  } finally {
    earth?.destroy()
    tracker.restore()
    device.destroy()
  }
}
