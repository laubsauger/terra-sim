import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V16: god tools are fun only if the world stays honest: every edit moves mass through the reservoir.
test('uplift, subsidence and meteor keep crust + reservoir exact and shape the land', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = twoPlateWorld([0, 0], [0, 0]);
    w.mantleReservoir = 50_000_000;
    uploadWorld(f, w);
    const P = new Params(); P.set('erosionRate', 0); P.set('thermalErosion', 0); P.set('eventRate', 0);
    const sim = new Sim(r, f, w, P, 0.05);
    const total = async () => { const ci = new Uint32Array(await f.read(r, 'colInfo')); let m = 0; for (let i = 1; i < ci.length; i += 2) m += ci[i]!; return m + new Int32Array(await f.read(r, 'counters'))[0]!; };
    const surf = async () => new Float32Array(await f.read(r, 'surfY'));
    const frame = () => new Promise((res) => setTimeout(res, 0));
    const run = async (n: number) => { let d = 0; while (d < n) { d += sim.runTicks(8); await frame(); } };
    const m0 = await total();
    const s0 = await surf();
    sim.events.enqueue({ kind: 'uplift', my: sim.geoMy, x: 190, z: 128, radius: 12, magnitude: 20 });
    sim.events.enqueue({ kind: 'uplift', my: sim.geoMy, x: 190, z: 40, radius: 12, magnitude: -10 });
    sim.events.enqueue({ kind: 'meteor', my: sim.geoMy, x: 220, z: 200, radius: 10, magnitude: 1 });
    await run(160); // isostasy responds
    const s1 = await surf();
    const m1 = await total();
    const at = (s: Float32Array, x: number, z: number) => s[L.colIdx(x, z)]!;
    return {
      m0, m1,
      up: at(s1, 190, 128) - at(s0, 190, 128),
      down: at(s1, 190, 40) - at(s0, 190, 40),
      craterCenter: at(s1, 220, 200) - at(s0, 220, 200),
      craterRim: at(s1, 220 + 12, 200) - at(s0, 220 + 12, 200),
      unhandled: sim.unhandledEvents,
    };
  });
  expect(res.m1).toBe(res.m0);
  expect(res.up).toBeGreaterThan(3);
  expect(res.down).toBeLessThan(-1);
  expect(res.craterCenter).toBeLessThan(-2);
  expect(res.craterRim).toBeGreaterThan(0.5);
  expect(res.unhandled).toBe(0);
});
