import { test, expect } from '@playwright/test';

// GPU tests are only meaningful if the test browser exposes a real WebGPU adapter.
test('test browser exposes a WebGPU adapter', async ({ page }) => {
  await page.goto('/');
  const info = await page.evaluate(async () => {
    if (!('gpu' in navigator)) return { ok: false, reason: 'no navigator.gpu' };
    const adapter = await navigator.gpu.requestAdapter();
    return { ok: !!adapter, reason: adapter ? adapter.info.vendor : 'no adapter' };
  });
  expect(info.ok, info.reason).toBe(true);
});
