import { createRequire } from 'node:module'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const require = createRequire(import.meta.url)
const cesiumPlugin = require('vite-plugin-cesium') as { default: () => Plugin }

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), cesiumPlugin.default()],
})
