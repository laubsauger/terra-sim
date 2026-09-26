import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V3: erosion only moves crust between voxels and suspension. Σ crust fill + Σ sedSusp must stay put
// (integer fill-unit exchanges; only f32 rounding of suspended transport may drift).
test('hydraulic + thermal erosion conserve crust + suspended mass', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const g = (x: number, z: number, cx: number, cz: number, s: number) => Math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * s * s));
    const height = (x: number, z: number) => 72 + 40 * g(x, z, 64, 64, 14) + 20 * g(x, z, 170, 60, 6) - 30 * g(x, z, 160, 170, 35) + 2 * Math.sin(x * 0.4);
    const mats = [L.Mat.BASALT, L.Mat.SEDIMENT, L.Mat.GRANITE, L.Mat.SHALE];
    const vox = R.voxFromHeights(height, (x, z) => mats[((x >> 5) + (z >> 5)) & 3]!);
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) water[R.idx(x, z)] = Math.max(0, 74 - height(x, z));
    const rig = await R.makeHydroRig(r, vox, water);
    rig.params.set('erosionRate', 4);
    rig.params.set('thermalErosion', 2);
    rig.hydro.uniforms.rain.value = 0.003;
    const m0 = await rig.mass();
    const v0 = await rig.readU('vox');
    for (let t = 0; t < 150; t++) rig.tick(8);
    const m1 = await rig.mass();
    const v1 = await rig.readU('vox');
    let changed = 0; for (let i = 0; i < v0.length; i++) if (v0[i] !== v1[i]) changed++;
    const susp = R.sum64(await rig.readF('sedSusp'));
    return { m0, m1, changed, susp };
  });
  expect(res.changed).toBeGreaterThan(1000); // erosion/deposition really happened
  expect(res.susp).toBeGreaterThan(0);
  // absolute, not relative to the ~1e9 total: losing even one fill unit per 150 ticks would be a bookkeeping bug
  expect(Math.abs(res.m1 - res.m0)).toBeLessThan(0.5);
});

// Rained-on slope with two shallow troughs (x = 64, x = 192), run at sim.ts cadence (2 substeps/tick,
// erosion every 2nd tick with kGeo × 2) under realistic precipitation (3.5e-4 voxel/tick).
// Returns mean surface drop along each trough and on the flank between them.
async function slopeRun(page: import('@playwright/test').Page, o: { ticks: number; vegRight: number; erosionRate: number }) {
  return page.evaluate(async (o) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    // z 20→180: ramp 110 → 78 (slope 0.2); z 180..255: ocean floor 60; z 0..20: wall back up (torus)
    const profile = (z: number) => z < 20 ? 60 + 50 * (z / 20) : z < 180 ? 110 - 0.2 * (z - 20) : z < 190 ? 78 - 1.8 * (z - 180) : 60;
    const trough = (x: number, cx: number) => 1.5 * Math.max(0, 1 - Math.abs(x - cx) / 20);
    const height = (x: number, z: number) => profile(z) - trough(x, 64) - trough(x, 192);
    const vox = R.voxFromHeights(height, () => L.Mat.SANDSTONE);
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) water[R.idx(x, z)] = Math.max(0, 76 - height(x, z));
    const rig = await R.makeHydroRig(r, vox, water);
    const veg = rig.f.cpuArray('veg') as Float32Array;
    for (let z = 0; z < R.NZ; z++) for (let x = 128; x < R.NX; x++) veg[R.idx(x, z)] = o.vegRight;
    rig.f.markDirty('veg');
    rig.params.set('erosionRate', o.erosionRate);
    rig.params.set('thermalErosion', 0); // isolate hydraulic incision
    rig.hydro.uniforms.rain.value = 3.5e-4 / 2;
    const s0 = await rig.readF('surfY');
    rig.simTicks(o.ticks, 2, 2);
    const s1 = await rig.readF('surfY');
    // mean drop over a band (the incised channel may wander off the trough axis)
    const drop = (x0: number, x1: number) => {
      let d = 0, n = 0;
      for (let z = 60; z < 160; z++) for (let x = x0; x <= x1; x++) { d += s0[R.idx(x, z)]! - s1[R.idx(x, z)]!; n++; }
      return d / n;
    };
    return { left: drop(56, 72), right: drop(184, 200), flank: drop(118, 138) };
  }, o);
}

// §C pass 8: rivers must carve valleys. Flow concentrates in the troughs; saturated sediment flux grows
// like q^1.5, so the troughs must deepen much faster than the sheet-flow flank — at a geologic pace
// (≈0.5-1 voxel per 10 My at max erosionRate), however damped the pipe model is.
test('rain on a slope carves a channel along the flow path', async ({ page }) => {
  const res = await slopeRun(page, { ticks: 200, vegRight: 0, erosionRate: 4 });
  console.log(`channel cut over 200 ticks (10 My): troughs ${res.left.toFixed(3)} / ${res.right.toFixed(3)}, flank ${res.flank.toFixed(3)} voxel`);
  expect(res.left).toBeGreaterThan(0.4);           // ~0.6 voxel per 10 My mean over a 17-cell band
  expect(res.left).toBeLessThan(3);                // not a runaway
  expect(res.left).toBeGreaterThan(3 * res.flank); // incision localised on the flow path
  expect(res.right).toBeGreaterThan(0.5 * res.left); // both troughs incise (channel growth is an instability, not bit-equal)
});

// T31 coupling: vegetation shields soil (erodibility × (1 - 0.7·veg)); a forested catchment must incise
// clearly less than an identical bare one under the same rain. Default erosionRate: detachment-limited
// regime, where erodibility matters (at max rate both saturate to transport-limited incision).
test('vegetated slopes erode less than bare ones', async ({ page }) => {
  const res = await slopeRun(page, { ticks: 400, vegRight: 1, erosionRate: 1 });
  console.log(`veg: bare trough ${res.left.toFixed(3)}, vegetated trough ${res.right.toFixed(3)} voxel`);
  expect(res.left).toBeGreaterThan(0.2);
  expect(res.right).toBeLessThan(0.6 * res.left);
  expect(res.right).toBeGreaterThan(0); // still erodes, just slower
});

// Thermal erosion is talus relaxation: slopes above the talus angle slump until they are at talus;
// slopes below talus must be left exactly alone (otherwise it becomes a diffusion that flattens the world).
test('thermal erosion relaxes cliffs to talus and leaves gentle slopes untouched', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const E = await import('/src/sim/erosion.ts');
    const r = await makeRenderer();
    const step = (z: number) => (z >= 64 && z < 192 ? 86 : 80);        // 6-voxel cliffs at z=64 and z=192
    const gentle = (z: number) => 80 + 0.25 * Math.min(z, 256 - z) * 0.2; // slope 0.05 ≪ talus
    const w = (x: number) => Math.min(1, Math.max(0, (Math.abs(x - 64) - 40) / 16)); // 0 in cliff zone, 1 in gentle zone
    const height = (x: number, z: number) => step(z) * (1 - w(x)) + gentle(z) * w(x);
    const rig = await R.makeHydroRig(r, R.voxFromHeights(height));
    const talus = E.EROSION_DEFAULTS.talus;
    const maxSlope = (s: Float32Array) => {
      let m = 0;
      for (let z = 0; z < R.NZ; z++) for (let x = 40; x <= 88; x++) {
        const h = s[R.idx(x, z)]!;
        m = Math.max(m, Math.abs(h - s[R.idx(x, z + 1)]!), Math.abs(h - s[R.idx(x + 1, z)]!));
      }
      return m;
    };
    const s0 = await rig.readF('surfY');
    const v0 = await rig.readU('vox');
    const m0 = await rig.mass();
    for (let t = 0; t < 400; t++) rig.tick(1);
    const s1 = await rig.readF('surfY');
    const v1 = await rig.readU('vox');
    let gentleChanged = 0;
    for (let y = 0; y < R.NY; y++) for (let z = 0; z < R.NZ; z++) for (let x = 140; x < 245; x++) {
      const i = R.idx(x, z) + y * R.NCOL;
      if (v0[i] !== v1[i]) gentleChanged++;
    }
    return { before: maxSlope(s0), after: maxSlope(s1), talus, gentleChanged, m0, m1: await rig.mass() };
  });
  expect(res.before).toBeGreaterThan(5.9);
  expect(res.after).toBeLessThan(res.talus + 0.05);
  expect(res.gentleChanged).toBe(0);
  expect(res.m1).toBe(res.m0); // thermal alone is pure integer bookkeeping → exact
});

// Convergent fronts rebuild their cliff every tectonics step (~1 cell/My). Talus capped at 0.25 layer per
// direction per step left 50-layer walls standing forever; a tall cliff must slump within a few My.
test('a 40-layer wall slumps to a slope within ~4 My of erosion steps', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const rig = await R.makeHydroRig(r, R.voxFromHeights((_x: number, z: number) => (z >= 64 && z < 192 ? 110 : 70)));
    rig.params.set('erosionRate', 0); // thermal only
    const maxSlope = (s: Float32Array) => {
      let m = 0;
      for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) m = Math.max(m, Math.abs(s[R.idx(x, z)]! - s[R.idx(x, (z + 1) % R.NZ)]!));
      return m;
    };
    const before = maxSlope(await rig.readF('surfY'));
    const m0 = await rig.mass();
    for (let t = 0; t < 40; t++) rig.tick(1); // 40 erosion steps = 4 My at EROSION_EVERY 2 (the old 0.25-layer cap: slope ≈ 30)
    return { before, after: maxSlope(await rig.readF('surfY')), m0, m1: await rig.mass() };
  });
  expect(res.before).toBeGreaterThan(39);
  expect(res.after, `max slope ${res.after.toFixed(1)} layers/cell`).toBeLessThan(12);
  expect(res.m1).toBe(res.m0);
});
