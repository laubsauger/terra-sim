import { test, expect } from '@playwright/test';

type W = { terra: { audio: { started: boolean; isMuted: boolean } }; terraReady: boolean };
const ready = (page: import('@playwright/test').Page) =>
  page.waitForFunction(() => (window as unknown as W).terraReady === true, null, { timeout: 30_000 });

// V14: browsers punish sites that make sound before a gesture; mute choice must survive reloads.
test('audio waits for a gesture and mute persists', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.waitForTimeout(500);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.started)).toBe(false);
  await page.mouse.click(10, 400);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.started)).toBe(true);
  await page.keyboard.press('m');
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted)).toBe(true);
  await page.reload();
  await ready(page);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted)).toBe(true);
});
