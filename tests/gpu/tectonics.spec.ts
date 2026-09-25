import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

type Run = {
  mass0: number; mass1: number; water0: number; water1: number; cols: number[];
  ridgeAge: number; ridgeTop: number; contAtEdge: number; oceanPlateCols: number; subducted: number; created: number;
};

// Oceanic plate 0 drives east into stationary continental plate 1 (convergent at x=128)
// and pulls away from it at x=0/255 (divergent, across the torus seam).
async function run(page: import('@playwright/test').Page, ticks: number, isoEvery: number): Promise<Run> {
  return page.evaluate(async ({ ticks, isoEvery }) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { Tectonics } = await import('/src/sim/tectonics.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld([4, 0], [0, 0]);
    uploadWorld(f, w);
    const derive = createDerivePass(f);
    derive.run(r);
    const tec = new Tectonics(f, w.plates);
    const sumF = (a: Float32Array) => { let s = 0; for (const v of a) s += v; return s; };
    const massNow = async () => {
      const ci = new Uint32Array(await f.read(r, 'colInfo'));
      let m = 0;
      for (let i = 1; i < ci.length; i += 2) m += ci[i]!;
      return m + new Int32Array(await f.read(r, 'counters'))[0]!;
    };
    const mass0 = await massNow();
    const water0 = sumF(new Float32Array(await f.read(r, 'water')));
    for (let t = 1; t <= ticks; t++) if (tec.tick(r, t, 0.05, { speedMul: 1, isoEvery })) derive.run(r);
    const mass1 = await massNow();
    const water1 = sumF(new Float32Array(await f.read(r, 'water')));
    const pid = new Uint32Array(await f.read(r, 'plateId'));
    const age = new Float32Array(await f.read(r, 'crustAge'));
    const info = new Uint32Array(await f.read(r, 'colInfo'));
    const stats = Tectonics.parseStats(await f.read(r, 'counters'));
    const cols = [0, 0];
    for (const p of pid) cols[p]!++;
    // column just east of the seam at x=0 was vacated by plate 0 moving east → youngest crust
    const c0 = L.colIdx(0, 10);
    const edge = L.colIdx(128, 10); // convergent front: continent must survive
    return {
      mass0, mass1, water0, water1, cols,
      ridgeAge: age[c0]!, ridgeTop: (info[c0 * 2]! >> 8) & 0xff,
      contAtEdge: (info[edge * 2]! >> 16) & 1, oceanPlateCols: cols[0]!,
      subducted: stats.subducted[0]!, created: stats.created[0]!,
    };
  }, { ticks, isoEvery });
}

// V3: every fill unit that leaves the voxel grid must land in the mantle reservoir and vice versa.
test('crust mass + reservoir exactly conserved; water conserved; area conserved', async ({ page }) => {
  const res = await run(page, 100, 4); // 100 ticks × 0.05 My × 4 cells/My = 20 cells of motion
  expect(res.mass1).toBe(res.mass0);
  expect(Math.abs(res.water1 - res.water0) / res.water0).toBeLessThan(1e-5);
  expect(res.cols[0]! + res.cols[1]!).toBe(256 * 256);
});

// Oceanic lithosphere subducts under continents; ridges open behind moving plates, wrapping the seam (V1).
test('convergence: continent survives, ocean subducts; divergence: fresh crust at seam', async ({ page }) => {
  const res = await run(page, 100, 1000);
  expect(res.contAtEdge).toBe(1);
  expect(res.subducted).toBeGreaterThan(0);
  expect(res.created).toBeGreaterThan(0);
  expect(res.ridgeAge).toBeLessThan(5);           // fresh crust, not the 50 My initial ocean
  expect(res.ridgeTop).toBe(58 + 7 - 1);          // ridge profile top voxel
  expect(res.oceanPlateCols).toBe(128 * 256);     // rigid ocean plate: consumes and creates equal area
});

// Isostasy keeps relief alive: thick crust floats up, thin/old ocean sinks, without creating mass.
test('isostasy lifts thick continent toward equilibrium and sinks old ocean', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const T = await import('/src/sim/tectonics.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld([0, 0], [0, 0]);
    // continental crust 40 layers but sitting too low: top at 70 (eq ≈ 84)
    for (let z = 0; z < L.NZ; z++) for (let x = 128; x < L.NX; x++) {
      for (let y = 0; y < L.NY; y++) {
        const mat = y < 30 ? L.Mat.PERIDOTITE : y < 70 ? L.Mat.GRANITE : L.Mat.AIR;
        w.vox[L.voxIdx(x, y, z)] = mat === L.Mat.AIR ? 0 : L.packVoxel(mat, 255, 0, mat === L.Mat.GRANITE ? L.FLAG_CONTINENTAL : 0);
      }
    }
    // ocean: 9 layers, age 200 My → eq ≈ 63.4 + 9*0.091*2.83 - 0.7*sqrt(200) ≈ 55.8, starts at 61
    for (let i = 0; i < L.NCOL; i++) if (w.plateId[i] === 0) w.crustAge[i] = 200;
    uploadWorld(f, w);
    const derive = createDerivePass(f);
    derive.run(r);
    const tec = new T.Tectonics(f, w.plates);
    const massOf = async () => { const ci = new Uint32Array(await f.read(r, 'colInfo')); let m = 0; for (let i = 1; i < ci.length; i += 2) m += ci[i]!; return m; };
    const m0 = await massOf();
    for (let t = 1; t <= 120; t++) if (tec.tick(r, t, 0.001, { speedMul: 1, isoEvery: 1 })) derive.run(r);
    const s = new Float32Array(await f.read(r, 'surfY'));
    return { cont: s[L.colIdx(200, 5)]!, ocean: s[L.colIdx(50, 5)]!, m0, m1: await massOf() };
  });
  expect(res.cont).toBeGreaterThan(82);
  expect(res.cont).toBeLessThan(86.5);
  expect(res.ocean).toBeLessThan(58);
  expect(res.m1).toBe(res.m0);
});

// B2: collisions used to keep 1 layer of the losing continent and delaminate the rest, so continents
// melted away within ~150 My. Continental crust must stack (orogeny), not vanish.
test('continent-continent collision thickens crust and keeps continental mass', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { Tectonics } = await import('/src/sim/tectonics.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld([2, 0], [-2, 0], { bothContinental: true });
    uploadWorld(f, w);
    const derive = createDerivePass(f);
    derive.run(r);
    const tec = new Tectonics(f, w.plates);
    const crust = async () => { const ci = new Uint32Array(await f.read(r, 'colInfo')); let m = 0, maxTop = 0; for (let i = 0; i < ci.length; i += 2) { m += ci[i + 1]!; maxTop = Math.max(maxTop, (ci[i]! >> 8) & 0xff); } return { m, maxTop }; };
    const a = await crust();
    const res0 = new Int32Array(await f.read(r, 'counters'))[0]!;
    for (let t = 1; t <= 80; t++) if (tec.tick(r, t, 0.05, { speedMul: 1, isoEvery: 4 })) derive.run(r);
    const b = await crust();
    const res1 = new Int32Array(await f.read(r, 'counters'))[0]!;
    const ci = new Uint32Array(await f.read(r, "colInfo")); let thick = 0, maxM = 0; for (let i = 0; i < ci.length; i += 2) { if (ci[i + 1]! > 41 * 255) thick++; maxM = Math.max(maxM, ci[i + 1]!); }
    return { m0: a.m, m1: b.m, top0: a.maxTop, top1: b.maxTop, res0, res1, dbg: [thick, maxM / 255] };
  });
  expect(res.m1 + res.res1).toBe(res.m0 + res.res0);   // V3
  // capped orogens (64 layers) delaminate the excess by design; before the fix ~all loser crust vanished
  expect(res.m1 / res.m0).toBeGreaterThan(0.9);
  expect(res.top1).toBeGreaterThan(res.top0 + 3);       // mountains rise
  expect(res.top1).toBeLessThan(126);                   // never hit the grid ceiling
});
