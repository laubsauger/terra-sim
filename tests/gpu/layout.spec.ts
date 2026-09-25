import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// GPU and CPU must agree bit-for-bit on voxel packing and torus indexing, or sim kernels
// and CPU worldgen/save code disagree about the same world.
test('TSL pack/unpack and wrap match CPU layout', async ({ page }) => {
  const out = await page.evaluate(async () => {
    const { makeRenderer, TSL } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const T = await import('/src/sim/tslLayout.ts');
    const { Fn, uint, int, instanceIndex } = TSL;
    const r = await makeRenderer();
    const f = new GpuFields();
    f.add('o', 'uint', 8);
    const o = f.cur<'uint'>('o');
    const k = Fn(() => {
      const v = T.tPack(uint(12), uint(200), uint(77), uint(0x81));
      o.element(0).assign(v);
      o.element(1).assign(T.tMat(v));
      o.element(2).assign(T.tFill(v));
      o.element(3).assign(T.tAge(v));
      o.element(4).assign(T.tFlags(v));
      o.element(5).assign(T.tColIdx(int(-1), int(-1)));
      o.element(6).assign(T.tVoxIdx(int(L.NX + 2), int(3), int(-2)));
      o.element(7).assign(instanceIndex.mul(0));
    })().compute(1);
    r.compute(k);
    const a = new Uint32Array(await f.read(r, 'o'));
    return { gpu: [...a], cpu: [L.packVoxel(12, 200, 77, 0x81), 12, 200, 77, 0x81, L.colIdx(-1, -1), L.voxIdx(L.NX + 2, 3, -2), 0] };
  });
  expect(out.gpu).toEqual(out.cpu);
});
