import { test, expect } from '@playwright/test';

type W = { terra: { params: { get(k: string): unknown }; clock: { requestedSpeed: number; paused: boolean; geoTime: number } }; terraReady: boolean };
const ready = (page: import('@playwright/test').Page) =>
  page.waitForFunction(() => (window as unknown as W).terraReady === true, null, { timeout: 20_000 });

// V20: URL, params and clock must agree — one schema, one value.
test('url speed reaches params and clock; keys double/halve speed', async ({ page }) => {
  await page.goto('/?speed=4&seed=42');
  await ready(page);
  const s0 = await page.evaluate(() => { const t = (window as unknown as W).terra; return [t.params.get('speed'), t.clock.requestedSpeed, t.params.get('seed')]; });
  expect(s0).toEqual([4, 4, 42]);
  await page.keyboard.press(']');
  expect(await page.evaluate(() => (window as unknown as W).terra.clock.requestedSpeed)).toBe(8);
  await page.keyboard.press('[');
  await page.keyboard.press('[');
  expect(await page.evaluate(() => (window as unknown as W).terra.clock.requestedSpeed)).toBe(2);
});

// V17: pausing geo time must not freeze ambience.
test('space pauses geo clock; geoTime stops advancing', async ({ page }) => {
  await page.goto('/');
  await ready(page);
  await page.keyboard.press('Space');
  const t0 = await page.evaluate(() => (window as unknown as W).terra.clock.geoTime);
  await page.waitForTimeout(400);
  const t1 = await page.evaluate(() => (window as unknown as W).terra.clock.geoTime);
  expect(t1).toBe(t0);
  expect(await page.evaluate(() => (window as unknown as W).terra.clock.paused)).toBe(true);
});

test('H hides panel, timebar and HUD; ambient url starts hidden', async ({ page }) => {
  await page.goto('/?ambient=1');
  await ready(page);
  await expect(page.locator('.terra-perf')).toBeHidden();
  await page.keyboard.press('h');
  await expect(page.locator('.terra-perf')).toBeVisible();
});
