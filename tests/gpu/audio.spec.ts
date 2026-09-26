import { test, expect } from '@playwright/test';

type W = { terra: { audio: { started: boolean; isMuted: boolean } }; terraReady: boolean };
const ready = (page: import('@playwright/test').Page) =>
  page.waitForFunction(() => (window as unknown as W).terraReady === true, null, { timeout: 30_000 });

// V14: browsers punish sites that make sound before a gesture; mute choice must survive reloads.
// Sound starts muted (user: a constant hum was the first thing new visitors heard); M turns it on.
test('audio waits for a gesture, starts muted and the choice persists', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.waitForTimeout(500);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.started)).toBe(false);
  await page.mouse.click(10, 400);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.started)).toBe(true);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted), 'muted by default').toBe(true);
  await page.keyboard.press('m');
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted)).toBe(false);
  await page.reload();
  await ready(page);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted), 'unmute persists').toBe(false);
  await page.keyboard.press('m');
  await page.reload();
  await ready(page);
  expect(await page.evaluate(() => (window as unknown as W).terra.audio.isMuted), 'mute persists').toBe(true);
});
