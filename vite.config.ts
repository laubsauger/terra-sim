import { defineConfig } from 'vitest/config';

// TERRA_NO_HMR=1 (GPU tests): no file watching, so edits during a run don't reload test pages.
const noHmr = process.env.TERRA_NO_HMR === '1';

export default defineConfig({
  // GitHub Pages serves the project under /terra-sim/ (the deploy workflow sets PAGES_BASE)
  base: process.env.PAGES_BASE ?? '/',
  build: { target: 'es2022' },
  server: noHmr ? { hmr: false, watch: null } : {},
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
});
