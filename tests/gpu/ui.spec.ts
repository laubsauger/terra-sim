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

// Game UI time of day: the Day pill sets the sun directly, runs / stops a day cycle, and hiding the UI (H)
// must not undo the player's cycle choice.
test('Day pill drags the sun and toggles the day cycle; H keeps the choice', async ({ page }) => {
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const pill = page.locator('.terra-day');
  await expect(pill).toBeVisible();
  await pill.locator('input').fill('500'); // noon
  const look = () => page.evaluate(() => { const l = (window as any).terra.look; return { tod: l.timeOfDay as number, day: l.dayLength as number }; });
  expect((await look()).tod).toBeCloseTo(0.5, 2);
  await expect(pill.locator('.td-clock')).toHaveText('12:00');
  await pill.locator('button').click();
  expect((await look()).day).toBeGreaterThan(0);
  const t0 = (await look()).tod;
  expect(Math.abs(t0 - 0.5), 'starting the cycle keeps the sun where it is').toBeLessThan(0.01);
  await page.keyboard.press('h');
  await page.keyboard.press('h');
  expect((await look()).day, 'H does not stop the cycle').toBeGreaterThan(0);
  await pill.locator('button').click();
  expect((await look()).day).toBe(0);
});

// Close ground shots (trees in front, mountains behind): over the block the camera is only kept above the actual
// ground, not above the grid ceiling, may look up at a peak, and is never let inside the terrain.
test('camera can sit just above the ground over the block, never inside it', async ({ page }) => {
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  await page.waitForTimeout(1500); // ground cache refresh
  const r = await page.evaluate(async () => {
    const t = (window as any).terra;
    const s = new Float32Array(await t.fields.read(t.stage.renderer, 'surfY')), w = new Float32Array(await t.fields.read(t.stage.renderer, 'water'));
    const { voxelToWorldY } = await import('/src/render/space.ts');
    const c = 128 + 128 * 256; const g = voxelToWorldY(s[c]! + w[c]!);
    const place = async (y: number) => {
      t.stage.camera.position.set(0.004, y, 0.004); t.stage.controls.target.set(0.6, g + 0.3, 0.6); t.stage.controls.update();
      await new Promise((res) => setTimeout(res, 300));
      return t.stage.camera.position.y as number;
    };
    return { g, low: await place(g + 0.06), inside: await place(g - 0.2) };
  });
  expect(r.low, 'stays low (was pushed to the grid ceiling)').toBeLessThan(r.g + 0.1);
  expect(r.inside, 'pushed out of the terrain').toBeGreaterThan(r.g);
});
