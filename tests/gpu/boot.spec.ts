import { test, expect } from '@playwright/test';

test('app boots on WebGPU and renders a canvas without page errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 20_000 });
  await expect(page.locator('canvas')).toHaveCount(1);
  expect(errors).toEqual([]);
});

// V18: a browser without WebGPU must get an explanation, never a blank page.
test('missing WebGPU shows error screen with reason', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
  });
  await page.goto('/');
  await expect(page.locator('.terra-error h1')).toHaveText('WebGPU unavailable');
  await expect(page.locator('.terra-error .reason')).toContainText('navigator.gpu');
  await expect(page.locator('canvas')).toHaveCount(0);
});
