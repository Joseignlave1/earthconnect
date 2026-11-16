// @ts-nocheck
// Optional Playwright driver is installed outside the application dependencies.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright-core');
const output = process.env.BROWSER_PROFILE_DIR ?? '/tmp/earth-browser-profile';
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  headless: process.env.PROFILE_HEADLESS === '1',
  args: ['--use-gl=angle', '--use-angle=gl'],
});
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  deviceScaleFactor: 1.2,
});
const errors = [];
const page = await context.newPage();
page.on('pageerror', (e) => errors.push(String(e)));
const save = async (report) => {
  await mkdir(output, { recursive: true });
  await writeFile(resolve(output, 'browser.json'), JSON.stringify(report, null, 2) + '\n');
};
const report = {
  method:
    'Warm local Chromium with hardware ANGLE/Mesa, 1280x800, 1.2 DPR. Production minification, build-only source/prototype instrumentation; no drawImage hook. One warmup and 3 measured queries, 3 deterministic 12-update pan/zoom runs per state. Overview interactions start at 4x world-fit zoom to permit real movement. Frame gaps are animation opportunities, synchronous renderer timings exclude GPU/compositor. All world responses use the identical Valles query. No renderer changes.',
  errors,
};
try {
  await page.goto('http://localhost:5174/demo/');
  await page.waitForFunction(() => globalThis.__earthProfile?.app?.ready());
  report.environment = await page.evaluate(() => __earthProfile.environment());
  console.log(JSON.stringify({ environment: report.environment }));
  report.gpu_validation = await page.evaluate(() => __earthProfile.verifyGpu());
  if (process.argv.includes('--controls-only')) {
    report.method = report.method.replace(
      'One warmup and 3 measured queries, 3 deterministic 12-update pan/zoom runs per state.',
      'Controls: one setup query, then three 12-update pan runs with scores but no coverage clipping, followed by three base-only pan runs.',
    );
    await page.evaluate(() => __earthProfile.query('control-query-warmup'));
    await page.evaluate(() => {
      const map = __earthProfile.maps.get('earth').map;
      const coverage = map
        .getLayers()
        .getArray()
        .find((layer) => layer.getClassName() === 'earth-covered');
      coverage?.setVisible(false);
    });
    for (const state of ['no-clip', 'base-only']) {
      if (state === 'base-only') await page.evaluate(() => __earthProfile.app.clear());
      for (let i = 0; i < 3; i++) {
        const s = await page.evaluate(
          (name) => __earthProfile.interact(name, 'pan'),
          `world-${state}-pan-${i}`,
        );
        console.log(JSON.stringify({ scenario: s.name, frames: s.frame_gaps, stages: s.stages }));
      }
    }
    report.samples = await page.evaluate(() => __earthProfile.samples);
    await writeFile(
      resolve(output, 'browser-controls.json'),
      JSON.stringify(report, null, 2) + '\n',
    );
  } else {
    report.raw_rpc = [];
    for (let i = 0; i < 4; i++) {
      const s = await page.evaluate(() => __earthProfile.rawRpc());
      report.raw_rpc.push({ warmup: i === 0, ...s });
      console.log(JSON.stringify({ raw_rpc: report.raw_rpc.at(-1) }));
    }
    for (let i = 0; i < 3; i++) {
      for (const kind of ['pan', 'zoom']) {
        const s = await page.evaluate(({ name, kind }) => __earthProfile.interact(name, kind), {
          name: `empty-${kind}-${i}`,
          kind,
        });
        console.log(JSON.stringify({ scenario: s.name, frames: s.frame_gaps, stages: s.stages }));
      }
    }
    for (let i = 0; i < 4; i++) {
      // Start every query at the same world extent. The existing result-controls
      // effect subsequently focuses its best pick, exactly as in the real app.
      await page.evaluate(async () => {
        await __earthProfile.app.clear();
        __earthProfile.maps.get('earth').reset();
      });
      const s = await page.evaluate((name) => __earthProfile.query(name), `query-${i}`);
      console.log(
        JSON.stringify({
          scenario: s.name,
          action_ms: s.action_ms,
          result: s.result,
          stages: s.stages,
          frames: s.frame_gaps,
        }),
      );
    }
    const hashes = await page.evaluate(() => __earthProfile.app.hashes());
    const reference = JSON.parse(
      await readFile(
        resolve(import.meta.dirname, '../../research/world-4km/scoring-benchmark.json'),
        'utf8',
      ),
    ).paths.production_sql.all_regions_sha256;
    assert.deepEqual(hashes, reference);
    report.world_validation = {
      regions: Object.keys(hashes).length,
      exact_all_region_hashes: true,
    };
    report.world_gpu_validation = await page.evaluate(() => __earthProfile.verifyWorldGpu());
    await page.evaluate(() => __earthProfile.maps.get('earth').reset());
    for (let i = 0; i < 3; i++) {
      for (const kind of ['pan', 'zoom']) {
        const s = await page.evaluate(({ name, kind }) => __earthProfile.interact(name, kind), {
          name: `world-${kind}-${i}`,
          kind,
        });
        console.log(JSON.stringify({ scenario: s.name, frames: s.frame_gaps, stages: s.stages }));
      }
    }
    await page.evaluate(() => document.querySelector('#best-result').click());
    for (let i = 0; i < 3; i++) {
      const s = await page.evaluate(
        (name) => __earthProfile.interact(name, 'pan'),
        `detail-pan-${i}`,
      );
      console.log(JSON.stringify({ scenario: s.name, frames: s.frame_gaps, stages: s.stages }));
    }
    for (let i = 0; i < 3; i++) {
      // Ensure each input changes the threshold, triggering the real control.
      await page.evaluate(() => {
        const input = document.querySelector('#score-threshold');
        input.value = '25';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await page.waitForTimeout(500);
      const s = await page.evaluate((name) => __earthProfile.threshold(name), `threshold-${i}`);
      console.log(
        JSON.stringify({
          scenario: s.name,
          action_ms: s.action_ms,
          frames: s.frame_gaps,
          stages: s.stages,
        }),
      );
    }
    report.samples = await page.evaluate(() => __earthProfile.samples);
    await save(report);

    // Control the deployed, uninstrumented production app through its real form.
    // This catches profiling-build distortions in the end-to-end status timer.
    await page.goto('http://localhost:8080/demo/');
    await page.waitForFunction(() => document.querySelector('#earth-datasource')?.options.length);
    const request = JSON.parse(await readFile(resolve(output, 'request.json'), 'utf8'));
    await page.locator('#console-tab').click();
    await page.locator('#query-text').fill(request.query);
    await page.locator('#query-params').fill(JSON.stringify(request.params));
    await page.evaluate(() => scrollTo(0, 0));
    report.production_controls = [];
    for (let i = 0; i < 4; i++) {
      await page.locator('[aria-label="Reset Earth map"]').click();
      await page.evaluate(() => document.querySelector('#run-query').click());
      await page.waitForFunction(
        () => /cells.*patches/.test(document.querySelector('#query-status')?.textContent ?? ''),
        { timeout: 30000 },
      );
      const status = await page.locator('#query-status').textContent();
      report.production_controls.push({ warmup: i === 0, status });
      console.log(JSON.stringify({ production: report.production_controls.at(-1) }));
      await page.waitForTimeout(1000);
    }
    await save(report);
  }
} finally {
  await browser.close();
}
