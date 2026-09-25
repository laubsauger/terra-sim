import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// Terrain render, water, erosion and probe all read surfY; it must match the voxel truth exactly,
// including fractional top fill and wrap-edge columns.
test('derive pass surfY, colInfo, crustMass match CPU reference; kernel follows vox swap', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { emptyWorld } = await import('/src/sim/worldData.ts');
    const { createDerivePass, deriveCpu } = await import('/src/sim/derive.ts');
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
      const cont = rnd() < 0.5 ? L.FLAG_CONTINENTAL : 0;
      for (let y = 0; y <= top; y++) {
        const mat = y < 10 ? L.Mat.PERIDOTITE : y === 20 ? L.Mat.MAGMA : L.Mat.BASALT;
        w.vox[L.voxIdx(x, y, z)] = L.packVoxel(mat, y === top ? Math.floor(rnd() * 256) : 255, 0, mat === L.Mat.BASALT ? cont : 0);
      }
    }
    uploadWorld(f, w);
    const pass = createDerivePass(f);
    pass.run(r);
    const gpu = new Float32Array(await f.read(r, 'surfY'));
    const gInfo = new Uint32Array(await f.read(r, 'colInfo'));
    const cpu = { surfY: new Float32Array(L.NCOL), colInfo: new Uint32Array(L.NCOL * 2) };
    deriveCpu(w.vox, cpu);
    let maxErr = 0, infoMismatch = 0, massMismatch = 0;
    for (let i = 0; i < L.NCOL; i++) {
      maxErr = Math.max(maxErr, Math.abs(gpu[i]! - cpu.surfY[i]!));
      if (gInfo[i * 2] !== cpu.colInfo[i * 2]) infoMismatch++;
      if (gInfo[i * 2 + 1] !== cpu.colInfo[i * 2 + 1]) massMismatch++;
    }
    // swap parity: second buffer is empty → surfY must become 0 everywhere (kernel follows current buffer)
    f.swap('vox');
    pass.run(r);
    const after = new Float32Array(await f.read(r, 'surfY'));
    let maxAfter = 0;
    for (let i = 0; i < L.NCOL; i++) maxAfter = Math.max(maxAfter, after[i]!);
    return { maxErr, infoMismatch, massMismatch, empty: gpu[0], maxAfter };
  });
  expect(res.maxErr).toBeLessThan(1e-5);
  expect(res.infoMismatch).toBe(0);
  expect(res.massMismatch).toBe(0);
  expect(res.empty).toBe(0);
  expect(res.maxAfter).toBe(0);
});
