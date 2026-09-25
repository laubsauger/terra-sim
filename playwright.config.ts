import { defineConfig } from '@playwright/test';

// GPU tests need real WebGPU: Chromium new headless + unsafe WebGPU flag.
export default defineConfig({
  testDir: 'tests/gpu',
  timeout: process.env.SOAK ? 0 : 60_000,
  use: {
    baseURL: 'http://localhost:5199',
    browserName: 'chromium',
    channel: 'chromium',
    launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal'] },
  },
  webServer: {
    command: 'npx vite --port 5199 --strictPort',
    url: 'http://localhost:5199',
    reuseExistingServer: true,
  },
});
