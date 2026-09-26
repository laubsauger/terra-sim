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
    // M_total (V3): crust + reservoir + chamber magma + pending melt + lava
    const total = async () => {
      const ci = new Uint32Array(await f.read(r, 'colInfo'));
      const mc = new Uint32Array(await f.read(r, 'magCol'));
      const lv = new Uint32Array(await f.read(r, 'lava'));
      let m = 0;
      for (let i = 1; i < ci.length; i += 2) m += ci[i]! + mc[i - 1]! + mc[i]! + lv[i - 1]!;
      return m + new Int32Array(await f.read(r, 'counters'))[0]!;
    };
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

// User report: god tools seemed to do nothing. They must act immediately, even with the geo clock paused.
test('god tools apply immediately in the live app while paused', async ({ page }) => {
  await page.goto('/?seed=3&speed=1');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 60_000 });
  const res = await page.evaluate(async () => {
    const t = (window as any).terra;
    t.clock.paused = true;
    await new Promise((r) => setTimeout(r, 300));
    const read = async () => new Float32Array(await t.fields.read(t.stage.renderer, 'surfY'));
    const s0 = await read();
    // pick a land column
    let c = 0; for (let i = 0; i < s0.length; i++) if (s0[i]! > s0[c]!) c = i;
    const x = c % 256, z = c >> 8;
    t.sim.events.enqueue({ kind: 'uplift', my: t.sim.geoMy, x, z, radius: 8, magnitude: 10 });
    await new Promise((r) => setTimeout(r, 400)); // a few frames
    const s1 = await read();
    return { d: s1[c]! - s0[c]!, ticked: t.clock.paused };
  });
  expect(res.ticked).toBe(true);
  expect(res.d).toBeGreaterThan(8);
});

// An oblique impact must read as directional: more ejecta downrange than uprange, mass still exact.
test('oblique meteor throws ejecta downrange', async ({ page }) => {
  await page.goto('/tests/gpu/support/blank.html');
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { GodTools } = await import('/src/sim/godTools.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = twoPlateWorld([0, 0], [0, 0]); w.mantleReservoir = 50_000_000; uploadWorld(f, w);
    const derive = createDerivePass(f); derive.run(r);
    const s0 = new Float32Array(await f.read(r, 'surfY'));
    new GodTools(f).meteor(r, 200, 128, 10, 1, 0); // travelling +x
    derive.run(r);
    const s1 = new Float32Array(await f.read(r, 'surfY'));
    const d = (x: number) => s1[L.colIdx(x, 128)]! - s0[L.colIdx(x, 128)]!;
    // ejecta rim band (radius 10, bowl shifted ~1.5 cells downrange): 12-18 cells out, clear of the bowl edge
    let down = 0, up = 0; for (let k = 12; k <= 18; k++) { down += d(200 + k); up += d(200 - k); }
    const vox = new Uint32Array(await f.read(r, 'vox'));
    const c = L.colIdx(201, 128), top = Math.ceil(s1[c]!) - 1;
    return { down, up, centre: d(201), floorMat: vox[c + top * L.NCOL]! & 0xff, BASALT: L.Mat.BASALT };
  });
  // user: craters should be deeper and visible, with charring: a radius-10 strike digs well over 5 layers
  // and leaves a dark impact-melt (basalt) floor
  expect(res.centre).toBeLessThan(-5);
  expect(res.floorMat, 'impact-melt floor').toBe(res.BASALT);
  expect(res.down).toBeGreaterThan(res.up * 1.5);
});
