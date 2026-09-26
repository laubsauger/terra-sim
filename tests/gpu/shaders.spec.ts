import { test, expect } from '@playwright/test';
import { checkWgslScopes } from './support/wgscope';

// B19 / B25: TSL assigns a node where it is first built; built inside a branch, every other read sees 0.
// It broke the tectonics kernel (continents decayed), every plume particle kind (spawned on one point) and the
// flora base height. Every shader the running app compiles (sim, render, life, atmo, FX, overlays, slice) must be
// free of such reads.
test('no app shader reads a TSL temp assigned only in another branch', async ({ page }) => {
  test.setTimeout(120_000);
  await page.addInitScript(() => {
    const w = window as unknown as { __wgsl: string[] };
    w.__wgsl = [];
    const orig = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (d: GPUShaderModuleDescriptor) { w.__wgsl.push(String(d.code)); return orig.call(this, d); };
  });
  await page.goto('/?seed=3&speed=5');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 60_000 });
  await page.waitForTimeout(6000);
  // compile the optional paths too: plate layer, slice caps, an overlay, a god-tool meteor
  await page.evaluate(() => {
    const t = (window as any).terra;
    t.overlays.tectonics = true; t.overlays.select(2); t.slice.setCut(0.2, 0.3);
  });
  await page.waitForTimeout(3000);
  const codes: string[] = await page.evaluate(() => [...new Set((window as unknown as { __wgsl: string[] }).__wgsl)]);
  expect(codes.length).toBeGreaterThan(50);
  const bad: string[] = [];
  for (const code of codes) {
    const v = checkWgslScopes(code);
    if (v.length) bad.push(`${v.length} reads, e.g. ${v[0]!.v} in fn ${v[0]!.fn}: ${code.split('\n').find((l) => l.includes(v[0]!.v + ' = '))?.trim().slice(0, 120)}`);
  }
  expect(bad).toEqual([]);
});
