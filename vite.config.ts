import { defineConfig } from 'vitest/config';

export default defineConfig({
  build: { target: 'es2022' },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
});
