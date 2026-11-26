import { cp } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

// The WASM language server requires SharedArrayBuffer. Credentialless lets
// the existing public fonts and images load without cookies.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

const pagesOutput = fileURLToPath(new URL('./dist-pages', import.meta.url))
const pagesHeaders = fileURLToPath(new URL('./public/_headers', import.meta.url))
const copyPagesHeaders: Plugin = {
  name: 'copy-pages-headers',
  apply: 'build',
  async writeBundle({ dir }) {
    if (dir) await cp(pagesHeaders, resolve(dir, '_headers'))
  },
}
const devOptimizerCache: Plugin = {
  name: 'dev-optimizer-cache',
  apply: (_config, { command, isPreview }) => command === 'serve' && !isPreview,
  config({ cacheDir, server }) {
    if (cacheDir) return
    const port = server?.port ?? 5173
    // Keep explicitly selected QA and test ports separate from the primary dev cache.
    return { cacheDir: `node_modules/.vite/serve-${port}` }
  },
}
// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  publicDir: mode === 'pages' ? false : undefined,
  plugins: [react(), mode === 'pages' && copyPagesHeaders, devOptimizerCache],
  worker: { format: 'es' },
  resolve: {
    // Keep Vim and the workbench on the same Monaco services and editor types.
    dedupe: ['monaco-editor'],
  },
  optimizeDeps: { exclude: ['wgsl-analyzer-web'] },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: mode === 'pages' ? { outDir: pagesOutput, emptyOutDir: true } : undefined,
}))
