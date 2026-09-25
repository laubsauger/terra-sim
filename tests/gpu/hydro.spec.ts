import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// Shared synthetic terrain: a Gaussian mountain range + a deep ocean basin, water not at rest.
// Built inside the page (functions don't cross the evaluate boundary), so it is a string snippet.
const TERRAIN = `
  const gauss = (x, z, cx, cz, s) => { const dx = Math.min(Math.abs(x - cx), 256 - Math.abs(x - cx)); const dz = Math.min(Math.abs(z - cz), 256 - Math.abs(z - cz)); return Math.exp(-(dx*dx + dz*dz) / (2*s*s)); };
  const height = (x, z) => 72 + 45 * gauss(x, z, 64, 64, 18) + 25 * gauss(x, z, 100, 40, 10) - 38 * gauss(x, z, 180, 180, 40) + 1.5 * Math.sin(x * 0.3) * Math.cos(z * 0.23);
`;

// V4: flow must only move water, never create or destroy it — sea level has to emerge from the total
// volume, so any leak compounds over millions of substeps into a drifting ocean.
test('pipe model conserves water and suspended sediment over 1000 substeps (deep basin + mountains)', async ({ page }) => {
  const res = await page.evaluate(async (TERRAIN) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const height = new Function(`${TERRAIN}; return height;`)() as (x: number, z: number) => number;
    const vox = R.voxFromHeights(height);
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) {
      const H = height(x, z);
      water[R.idx(x, z)] = Math.max(0, 76 + 4 * Math.sin(x * 0.05) - H) + (x < 128 && z < 128 ? 2 : 0);
    }
    const rig = await R.makeHydroRig(r, vox, water);
    const sed0 = rig.f.cpuArray('sedSusp') as Float32Array;
    for (let z = 20; z < 110; z++) for (let x = 20; x < 110; x++) sed0[R.idx(x, z)] = 100;
    rig.f.markDirty('sedSusp');
    const w0 = R.sum64(await rig.readF('water'));
    const s0 = R.sum64(await rig.readF('sedSusp'));
    for (let i = 0; i < 10; i++) rig.hydro.step(r, 100);
    const w1a = await rig.readF('water');
    const s1a = await rig.readF('sedSusp');
    let minW = Infinity, minS = Infinity, finite = true;
    for (let i = 0; i < R.NCOL; i++) { minW = Math.min(minW, w1a[i]!); minS = Math.min(minS, s1a[i]!); finite &&= Number.isFinite(w1a[i]!) && Number.isFinite(s1a[i]!); }
    const vel = await rig.readF('waterVel');
    let maxV = 0; for (let i = 0; i < vel.length; i++) maxV = Math.max(maxV, Math.abs(vel[i]!));
    return { w0, w1: R.sum64(w1a), s0, s1: R.sum64(s1a), minW, minS, finite, maxV };
  }, TERRAIN);
  expect(res.finite).toBe(true);
  expect(Math.abs(res.w1 - res.w0) / res.w0).toBeLessThan(1e-5);
  expect(Math.abs(res.s1 - res.s0) / res.s0).toBeLessThan(1e-5);
  expect(res.minW).toBeGreaterThan(-1e-4); // K factor: a column never ships more than it holds
  expect(res.minS).toBeGreaterThan(-1e-3);
  expect(res.maxV).toBeGreaterThan(0.01); // water actually moved
});

// The GPU kernel is the sim; the CPU reference is the readable spec. Divergence = index/direction bug.
test('hydro GPU substeps match CPU reference', async ({ page }) => {
  const res = await page.evaluate(async (TERRAIN) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const H = await import('/src/sim/hydro.ts');
    const r = await makeRenderer();
    const height = new Function(`${TERRAIN}; return height;`)() as (x: number, z: number) => number;
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) water[R.idx(x, z)] = Math.max(0, 78 - height(x, z)) + ((x * 7 + z * 3) % 5) * 0.2;
    const rig = await R.makeHydroRig(r, R.voxFromHeights(height), water);
    const sed = rig.f.cpuArray('sedSusp') as Float32Array;
    for (let i = 0; i < R.NCOL; i++) sed[i] = (i % 13) * 3;
    rig.f.markDirty('sedSusp');
    const st = {
      surfY: await rig.readF('surfY'), water: water.slice(), flux: new Float32Array(R.NCOL * 4),
      vel: new Float32Array(R.NCOL * 2), sed: sed.slice(), sedRatio: new Float32Array(R.NCOL),
    };
    const N = 30;
    rig.hydro.step(r, N);
    for (let i = 0; i < N; i++) H.hydroStepCpu(st);
    const g = { water: await rig.readF('water'), flux: await rig.readF('flux'), vel: await rig.readF('waterVel'), sed: await rig.readF('sedSusp') };
    const maxErr = (a: Float32Array, b: Float32Array) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!)); return m; };
    return { water: maxErr(g.water, st.water), flux: maxErr(g.flux, st.flux), vel: maxErr(g.vel, st.vel), sed: maxErr(g.sed, st.sed) };
  }, TERRAIN);
  expect(res.water).toBeLessThan(1e-3);
  expect(res.flux).toBeLessThan(1e-3);
  expect(res.vel).toBeLessThan(1e-2);
  expect(res.sed).toBeLessThan(1e-2);
});

// Quasi-steady flow must settle: a closed basin ends as a flat lake (rivers, sea level and shore render
// all assume water at rest is level), and all water stays inside the basin.
test('water poured off-centre in a closed bowl settles to a flat lake', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const height = (x: number, z: number) => {
      const r2 = (x - 128) ** 2 + (z - 128) ** 2;
      return r2 < 2500 ? 100 - 30 * (1 - r2 / 2500) : 100;
    };
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) if ((x - 105) ** 2 + (z - 128) ** 2 < 144) water[R.idx(x, z)] = 20;
    const rig = await R.makeHydroRig(r, R.voxFromHeights(height), water);
    const before = R.sum64(water);
    for (let i = 0; i < 40; i++) rig.hydro.step(r, 100);
    const w = await rig.readF('water');
    const s = await rig.readF('surfY');
    let n = 0, sum = 0, sum2 = 0, outside = 0;
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) {
      const i = R.idx(x, z);
      if ((x - 128) ** 2 + (z - 128) ** 2 >= 2500) outside += Math.abs(w[i]!);
      if (w[i]! > 0.05) { const h = s[i]! + w[i]!; n++; sum += h; sum2 += h * h; }
    }
    const mean = sum / n;
    return { n, std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), mean, outside, rel: Math.abs(R.sum64(w) - before) / before };
  });
  expect(res.n).toBeGreaterThan(500); // it spread into a lake, not a puddle
  expect(res.std).toBeLessThan(0.01);
  expect(res.outside).toBeLessThan(1e-3);
  expect(res.rel).toBeLessThan(1e-5);
});

// V1: the world is a torus. Water at x=0 must reach x=NX-1 exactly as it reaches x=1 (same for z);
// an edge clamp would show up as asymmetry or a wall.
test('flow wraps across X and Z edges symmetrically', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const water = new Float32Array(R.NCOL);
    water[R.idx(0, 128)] = 10;
    water[R.idx(128, 0)] = 10;
    const rig = await R.makeHydroRig(r, R.voxFromHeights(() => 80), water);
    rig.hydro.step(r, 3);
    const w = await rig.readF('water');
    return {
      xm: w[R.idx(R.NX - 1, 128)]!, xp: w[R.idx(1, 128)]!, xz: w[R.idx(0, 129)]!,
      zm: w[R.idx(128, R.NZ - 1)]!, zp: w[R.idx(128, 1)]!, zx: w[R.idx(129, 0)]!,
      far: w[R.idx(64, 64)]!, total: R.sum64(w),
    };
  });
  expect(res.xm).toBeGreaterThan(0.1);
  expect(Math.abs(res.xm - res.xp)).toBeLessThan(1e-5);
  expect(Math.abs(res.xm - res.xz)).toBeLessThan(1e-5);
  expect(res.zm).toBeGreaterThan(0.1);
  expect(Math.abs(res.zm - res.zp)).toBeLessThan(1e-5);
  expect(Math.abs(res.zm - res.zx)).toBeLessThan(1e-5);
  expect(res.far).toBe(0);
  expect(Math.abs(res.total - 20)).toBeLessThan(1e-4);
});

// V7: a god-tool flood or a numeric spike must not blow up into NaN/Inf; the scheme is stable
// independent of depth, and K keeps columns non-negative even with a 1e5-deep column on a peak.
test('no NaN/Inf after stress: huge water column on a mountain peak', async ({ page }) => {
  const res = await page.evaluate(async (TERRAIN) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const height = new Function(`${TERRAIN}; return height;`)() as (x: number, z: number) => number;
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) water[R.idx(x, z)] = Math.max(0, 76 - height(x, z));
    water[R.idx(64, 64)] = 1e5;
    water[R.idx(100, 40)] = 5e4;
    const rig = await R.makeHydroRig(r, R.voxFromHeights(height), water);
    const sed = rig.f.cpuArray('sedSusp') as Float32Array;
    sed[R.idx(64, 64)] = 1e5;
    rig.f.markDirty('sedSusp');
    const w0 = R.sum64(water);
    for (let i = 0; i < 20; i++) rig.hydro.step(r, 100);
    for (let i = 0; i < 20; i++) rig.tick(4);
    let bad = 0;
    for (const n of ['water', 'flux', 'waterVel', 'sedSusp', 'surfY']) for (const v of await rig.readF(n)) if (!Number.isFinite(v)) bad++;
    const w = await rig.readF('water');
    let minW = Infinity; for (const v of w) minW = Math.min(minW, v);
    return { bad, minW, rel: Math.abs(R.sum64(w) - w0) / w0 };
  }, TERRAIN);
  expect(res.bad).toBe(0);
  expect(res.minW).toBeGreaterThan(-1e-2);
  expect(res.rel).toBeLessThan(1e-5);
});

// §C perf budget: hydro substep must be well under 1 ms for 256² columns on a dGPU (V9).
// Wall-clock here includes submission overhead on whatever GPU the runner has; logged for the report.
test('hydro substep timing', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    const water = new Float32Array(R.NCOL).fill(1);
    const rig = await R.makeHydroRig(r, R.voxFromHeights((x, z) => 80 + 10 * Math.sin(x * 0.1) * Math.sin(z * 0.07)), water);
    rig.hydro.step(r, 10);
    await rig.f.read(r, 'water', 0, 4);
    const N = 1000;
    const t0 = performance.now();
    rig.hydro.step(r, N);
    await rig.f.read(r, 'water', 0, 4);
    const hydroMs = (performance.now() - t0) / N;
    const M = 100;
    const t1 = performance.now();
    for (let i = 0; i < M; i++) rig.erosion.step(r);
    await rig.f.read(r, 'water', 0, 4);
    const erosionMs = (performance.now() - t1) / M;
    return { hydroMs, erosionMs };
  });
  console.log(`hydro substep ${res.hydroMs.toFixed(3)} ms, erosion step ${res.erosionMs.toFixed(3)} ms (wall, incl. submit)`);
  expect(res.hydroMs).toBeLessThan(1);
  expect(res.erosionMs).toBeLessThan(2);
});
