import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// The cut faces should read like real strata: buried sediment becomes rock suited to where it was laid
// down, deep burial metamorphoses it, and rock ages. None of it may change crust mass (V3).
test('burial lithifies by environment, metamorphoses at depth, ages rock, conserves mass', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { emptyWorld } = await import('/src/sim/worldData.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { Diagenesis, DIAG_EVERY } = await import('/src/sim/diagenesis.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = emptyWorld(1);
    // column A (warm shallow sea): sediment 0..70, water 5. column B (deep sea): shale below 30 layers cover.
    for (let z = 0; z < L.NZ; z++) for (let x = 0; x < L.NX; x++) {
      for (let y = 0; y < 70; y++) {
        const mat = y < 10 ? L.Mat.PERIDOTITE : x < 128 ? L.Mat.SEDIMENT : (y < 40 ? L.Mat.SHALE : L.Mat.BASALT);
        w.vox[L.voxIdx(x, y, z)] = L.packVoxel(mat, 255, 0, 0);
      }
      w.water[L.colIdx(x, z)] = x < 128 ? 5 : 20;
    }
    uploadWorld(f, w);
    // warm everywhere: surfTemp 25
    (f.cpuArray('surfTemp') as Float32Array).fill(25); f.markDirty('surfTemp');
    createDerivePass(f).run(r);
    const massOf = async () => { const ci = new Uint32Array(await f.read(r, 'colInfo')); let m = 0; for (let i = 1; i < ci.length; i += 2) m += ci[i]!; return m; };
    const m0 = await massOf();
    const d = new Diagenesis(f);
    for (let k = 1; k <= 200; k++) d.step(r, k * DIAG_EVERY, 0.05); // 200 My
    createDerivePass(f).run(r);
    const vox = new Uint32Array(await f.read(r, 'vox'));
    const at = (x: number, y: number) => vox[L.voxIdx(x, y, 7)]!;
    return {
      m0, m1: await massOf(),
      topA: L.voxMat(at(10, 69)), buriedA: L.voxMat(at(10, 60)),
      deepB: L.voxMat(at(200, 35)),   // 34 layers of cover
      deepestB: L.voxMat(at(200, 15)), // 54 layers of cover
      ageMy: L.codeToAge(L.voxAge(at(200, 50))),
    };
  });
  expect(res.m1).toBe(res.m0);
  expect(res.topA).toBe(5);        // SEDIMENT: uncovered surface stays loose
  expect(res.buriedA).toBe(8);     // LIMESTONE: warm shallow sea
  expect(res.deepB).toBe(9);       // SCHIST: shale under > 26 layers
  expect(res.deepestB).toBe(10);   // GNEISS: schist under > 40 layers
  expect(res.ageMy).toBeGreaterThan(120);
  expect(res.ageMy).toBeLessThan(300);
});
