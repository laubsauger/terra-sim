import { test, expect } from '@playwright/test';

// §I.overlays: every overlay key either renders cleanly or says it is not available — never a GPU error.
test('overlay keys 1-9 toggle without GPU errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { const t = m.text(); if (/status of 404/.test(t)) return; if (m.type() === 'error' || /WebGPU|WGSL|validation/i.test(t)) errors.push(t); });
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const label = page.locator('.terra-overlay-label');
  for (let k = 1; k <= 9; k++) {
    await page.keyboard.press(String(k));
    await expect(label).toBeVisible();
    await page.waitForTimeout(150);
  }
  await expect(label).toContainText('precip');
  await page.keyboard.press('9'); // same key again → off
  await expect(label).toBeHidden();
  await page.keyboard.press('2');
  await expect(label).toContainText('crust age');
  await page.keyboard.press('0');
  await expect(label).toBeHidden();
  expect(errors).toEqual([]);
});
