import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V2 + V12: the world after N ticks must not depend on how ticks were split across frames
// or on when async readbacks happened to land.
test('same world + seed → identical hash regardless of ticks-per-frame', async ({ page }) => {
  test.setTimeout(120_000);
  const hashes = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const { hashFields } = await import('/src/sim/hash.ts');
    const r = await makeRenderer();
    const frame = () => new Promise((res) => setTimeout(res, 0));
    async function runWith(chunk: number, total: number) {
      const f = new GpuFields();
      registerSimFields(f);
      f.freeze();
      const w = twoPlateWorld([3, 1.5], [-1, 0.5]);
      uploadWorld(f, w);
      const sim = new Sim(r, f, w, new Params(), 0.05);
      let done = 0, stalls = 0;
      while (done < total) {
        const ran = sim.runTicks(Math.min(chunk, total - done));
        if (ran === 0) stalls++;
        done += ran;
        await frame();
      }
      return { hash: await hashFields(r, f), stalls, vel: sim.tectonics.plates[0]!.vel };
    }
    return [await runWith(1, 400), await runWith(7, 400), await runWith(64, 400)];
  });
  expect(hashes[1]!.hash).toBe(hashes[0]!.hash);
  expect(hashes[2]!.hash).toBe(hashes[0]!.hash);
  expect(hashes[2]!.vel).toEqual(hashes[0]!.vel);
  expect(hashes[2]!.stalls).toBeGreaterThan(0); // big chunks must hit the readback barrier, proving it is exercised
});
