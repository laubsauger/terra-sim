import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// Terrain render, water, erosion and probe all read surfY; it must match the voxel truth exactly,
// including fractional top fill and wrap-edge columns.
test('derive pass surfY matches CPU reference, before and after vox swap', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { emptyWorld } = await import('/src/sim/worldData.ts');
    const { createDerivePass, surfYCpu } = await import('/src/sim/derive.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = emptyWorld(1);
    // deterministic pseudo-random columns; column (0,0) stays empty
    let s = 12345;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
    for (let z = 0; z < L.NZ; z++) for (let x = 0; x < L.NX; x++) {
      if (x === 0 && z === 0) continue;
      const top = Math.floor(rnd() * L.NY);
      for (let y = 0; y <= top; y++) w.vox[L.voxIdx(x, y, z)] = L.packVoxel(L.Mat.BASALT, y === top ? Math.floor(rnd() * 256) : 255, 0, 0);
    }
    uploadWorld(f, w);
    const pass = createDerivePass(f);
    pass.run(r);
    const gpu = new Float32Array(await f.read(r, 'surfY'));
    const cpu = new Float32Array(L.NCOL);
    surfYCpu(w.vox, cpu);
    let maxErr = 0;
    for (let i = 0; i < L.NCOL; i++) maxErr = Math.max(maxErr, Math.abs(gpu[i]! - cpu[i]!));
    // swap parity: second buffer is empty → surfY must become 0 everywhere (kernel follows current buffer)
    f.swap('vox');
    pass.run(r);
    const after = new Float32Array(await f.read(r, 'surfY'));
    let maxAfter = 0;
    for (let i = 0; i < L.NCOL; i++) maxAfter = Math.max(maxAfter, after[i]!);
    return { maxErr, empty: gpu[0], maxAfter, sample: [gpu[1], cpu[1]] };
  });
  expect(res.maxErr).toBeLessThan(1e-5);
  expect(res.empty).toBe(0);
  expect(res.maxAfter).toBe(0);
});
