import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
export default defineConfig({
  plugins: [svelte()],
  base: '/demo/',
  build: { target: 'es2022' },
  server: { proxy: { '/earth_index.v1.EarthIndex/': 'http://localhost:8080' } },
  test: {
    environment: 'jsdom',
    include: ['tests/components/**/*.test.ts'],
    // Svelte's client entry is needed when mounting components in jsdom.
    server: { deps: { inline: ['svelte'] } },
  },
  resolve: { conditions: ['browser'] },
});
