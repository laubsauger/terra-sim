import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V1: the mantle is a torus in X/Z like the crust. A plume straddling the seam must produce the same heat-flow
// footprint as the same plume in mid-grid, shifted; otherwise hotspot chains would bend or die at the edge.
// Rolls are switched off so the field is exactly translation- and mirror-symmetric.
test('mantle + heat flow respect torus wrap: seam plume ≡ shifted mid-grid plume, mirror symmetric', async ({ page }) => {
  test.setTimeout(120_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const only = { tectonics: false, magma: false, lava: false };
    const run = async (px: number) => {
      const rig = await makeMagmaRig(r, twoPlateWorld([0, 0], [0, 0]), { mantle: { plumes: [{ x: px, z: 128, r: 9, age: 20, life: 1e9 }], evolve: false } });
      rig.mantle.uniforms.rollAmp.value = 0;
      rig.step(120, only);
      return rig.readF('heatFlow');
    };
    const A = await run(0), B = await run(128);
    let shiftErr = 0, mirrorErr = 0;
    for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
      shiftErr = Math.max(shiftErr, Math.abs(A[L.colIdx(x, z)]! - B[L.colIdx(x + 128, z)]!));
      mirrorErr = Math.max(mirrorErr, Math.abs(A[L.colIdx(x, z)]! - A[L.colIdx(255 - x, z)]!));
    }
    return { shiftErr, mirrorErr, peakSeam: A[L.colIdx(0, 128)]!, peakWrapped: A[L.colIdx(255, 128)]!, far: A[L.colIdx(128, 0)]! };
  });
  expect(res.shiftErr).toBeLessThan(1e-3);
  expect(res.mirrorErr).toBeLessThan(1e-2);
  expect(res.peakSeam).toBeGreaterThan(150);  // hot enough to melt (qHot = 110)
  expect(res.peakWrapped).toBeGreaterThan(150);
  expect(res.far).toBeGreaterThan(60);
  expect(res.far).toBeLessThan(70);           // background ≈ q0
});

// Subducted slabs are the mantle's cold downwellings: they must cool the mantle under the trench and sink,
// and lower surface heat flow there (cold forearc), while the far field stays at the reference.
test('slab cold sink under the trench: cold top anomaly, sinking, low heat flow', async ({ page }) => {
  test.setTimeout(120_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const MM = await import('/src/sim/mantleModel.ts');
    const r = await makeRenderer();
    const w = twoPlateWorld([3, 0], [0, 0]); // trench at column x = 128 → mantle mx = 32
    w.mantleReservoir = 50_000_000;
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [], evolve: false } });
    rig.mantle.uniforms.rollAmp.value = 0;
    rig.step(400);
    const m = await rig.readF('mantle');
    const anom = (mx0: number, mx1: number, my: number) => {
      let s = 0, n = 0;
      for (let mx = mx0; mx <= mx1; mx++) for (let mz = 0; mz < MM.MZ; mz++) { s += m[(mx + mz * MM.MX + my * MM.MX * MM.MZ) * 4]! - MM.mantleRef(my); n++; }
      return s / n;
    };
    const q = await rig.readF('heatFlow');
    let qTrench = 0; for (let z = 0; z < 256; z++) qTrench += q[128 + z * 256]!; qTrench /= 256;
    return { top: anom(32, 32, MM.MY - 1), deep: anom(31, 33, MM.MY - 6), farTop: anom(4, 12, MM.MY - 1), farDeep: anom(4, 12, MM.MY - 6), qTrench };
  });
  expect(res.top).toBeLessThan(-10);
  expect(res.deep).toBeLessThan(-2);      // cold material has sunk several cells
  expect(Math.abs(res.farTop)).toBeLessThan(1);
  expect(Math.abs(res.farDeep)).toBeLessThan(1);
  expect(res.qTrench).toBeLessThan(62);   // below the 65 background
});

// Crust temperature: a conductive geotherm scaled by heat flow (continental Moho a few hundred °C, thin ocean
// crust cool), air at surface temperature, and a hot halo around magma chambers (contact metamorphism input).
test('crust temp: geotherm rises with depth, Moho plausible, chamber halo, air = surface temp', async ({ page }) => {
  test.setTimeout(120_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const MM = await import('/src/sim/mantleModel.ts');
    const r = await makeRenderer();
    const w = twoPlateWorld([0, 0], [0, 0]);
    // chamber in the continent (surface 84, base 44)
    for (let x = 190; x <= 194; x++) for (let z = 126; z <= 130; z++) for (let y = 68; y <= 73; y++) w.vox[L.voxIdx(x, y, z)] = L.packVoxel(L.Mat.MAGMA, 255, 0, 0);
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [], evolve: false } });
    rig.step(80, { tectonics: false, magma: false, lava: false });
    const t = await rig.readF('crustTemp');
    const T = (x: number, y: number, z: number) => t[(x >> 1) + (z >> 1) * MM.CX + (y >> 1) * MM.CX * MM.CZ]!;
    const prof: number[] = [];
    for (let y = 0; y < 128; y += 2) prof.push(T(220, y, 40));
    let mono = true;
    for (let y = 48; y < 84; y += 2) if (!(T(220, y, 40) > T(220, y + 2, 40))) mono = false;
    let finite = true; for (const v of t) if (!Number.isFinite(v)) { finite = false; break; }
    return {
      mono, finite, air: T(220, 100, 40), moho: T(220, 44, 40), oceanMoho: T(60, 52, 40),
      chamber: T(192, 70, 128), halo: T(192, 76, 128), farSame: T(220, 76, 40), prof,
    };
  });
  expect(res.finite).toBe(true);
  expect(res.mono).toBe(true);
  expect(res.air).toBe(0);                 // rig has no climate pass: surfTemp field = 0
  expect(res.moho).toBeGreaterThan(450);
  expect(res.moho).toBeLessThan(750);
  expect(res.oceanMoho).toBeLessThan(res.moho);
  expect(res.chamber).toBeCloseTo(1150, 0);
  expect(res.halo).toBeGreaterThan(res.farSame + 100);
});
