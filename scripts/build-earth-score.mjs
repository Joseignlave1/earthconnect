import { copyFile, mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { build } from 'vite'
import { hydrateAssets } from './hydrate-assets.mjs'

const fromRoot = path => fileURLToPath(new URL(path, new URL('../', import.meta.url)))
const outDir = fromRoot('exports/earth-score')
await hydrateAssets('world')
await build({
  configFile: false,
  root: fromRoot('.'),
  base: './',
  publicDir: fromRoot('compute-data'),
  worker: { format: 'es' },
  build: {
    outDir,
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        assetFileNames: 'assets/[name]-[hash][extname]',
        chunkFileNames: 'assets/[name]-[hash].js',
      },
    },
    lib: {
      entry: fromRoot('src/earth-score/index.ts'),
      formats: ['es'],
      fileName: () => 'earth-score.mjs',
    },
  },
})
await mkdir(`${outDir}/licenses`, { recursive: true })
for (const file of ['ZSTD-LICENSE.txt', 'ZSTD-SYS-LICENSE.txt']) {
  await copyFile(fromRoot(`tools/earth-score-decoder/${file}`), `${outDir}/licenses/${file}`)
}
await writeFile(`${outDir}/earth-score.d.mts`, "export * from './types/index.js'\n")
