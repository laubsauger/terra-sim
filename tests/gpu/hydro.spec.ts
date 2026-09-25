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
  expect(res.sed).toBeLessThan(0.05); // sed up to 36: ~1e-3 relative (sedSpeed amplifies f32/f64 differences)
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
    // damped regime (D = kFlow/friction = 0.3): levelling a 100-cell bowl is diffusive, std ≈ 0.4 @ 2000,
    // 0.02 @ 4000, 0.0015 @ 6000 substeps. 6000 substeps = 3000 ticks at HYDRO_SUBSTEPS = 2.
    for (let i = 0; i < 60; i++) rig.hydro.step(r, 100);
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

// Live-app failure (low friction): the deep ocean rang with neighbour level jumps of 5-6 voxels, re-excited by
// every tectonic column shift (water pools at overlaps, gaps go dry). Quasi-steady water must kill such
// grid-scale level noise within a tick or two of substeps and must not keep oscillating afterwards.
test('deep ocean: random ±3-voxel level perturbations flatten within 40 substeps, no persistent oscillation', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    let seed = 7;
    const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
    const floor = (x: number, z: number) => 53 + 7 * Math.sin(x * 0.07 + 1) * Math.cos(z * 0.05); // depth 16..30
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) water[R.idx(x, z)] = 76 - floor(x, z) + (rnd() * 2 - 1) * 3;
    const rig = await R.makeHydroRig(r, R.voxFromHeights(floor), water);
    const rough = async () => {
      const w = await rig.readF('water'), s = await rig.readF('surfY');
      let t = 0;
      for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) {
        const h = s[R.idx(x, z)]! + w[R.idx(x, z)]!;
        t += Math.abs(h - s[R.idx(x + 1, z)]! - w[R.idx(x + 1, z)]!) + Math.abs(h - s[R.idx(x, z + 1)]! - w[R.idx(x, z + 1)]!);
      }
      return t / (2 * R.NCOL);
    };
    const r0 = await rough();
    const trace: number[] = [];
    for (let i = 0; i < 80; i++) { rig.hydro.step(r, 1); trace.push(await rough()); }
    return { r0, trace, total0: R.sum64(water), total1: R.sum64(await rig.readF('water')) };
  });
  console.log(`ocean roughness ${res.r0.toFixed(3)} → ${[0, 4, 9, 19, 39, 79].map((i) => `${i + 1}:${res.trace[i]!.toExponential(1)}`).join(' ')}`);
  expect(res.r0).toBeGreaterThan(1.5);
  expect(res.trace[39]!).toBeLessThan(0.05); // 40 substeps (20 geo ticks at HYDRO_SUBSTEPS = 2)
  // no ringing: roughness never grows again, substep by substep, and is still falling at 80
  let rises = 0;
  for (let i = 1; i < 80; i++) if (res.trace[i]! > res.trace[i - 1]! * 1.001 + 1e-6) rises++;
  expect(rises).toBe(0);
  expect(res.trace[79]!).toBeLessThan(0.8 * res.trace[39]!);
  expect(Math.abs(res.total1 - res.total0) / res.total0).toBeLessThan(1e-7);
});

// V4 on deep water: realistic inflows (land runoff ~1e-4 voxel/substep) are far below the f32 ulp of a
// 16-30-voxel ocean column (~2e-6 × a few). Unquantised adds round those deltas the same way every substep
// → systematic creation/loss of water (fast-math also folds compensation tricks). Fluxes are floored to
// FLUX_Q so every add is exact: total water must be conserved to float-sum precision, not just ~1e-5.
test('deep ocean fed by thin runoff conserves water exactly (no rounding bias)', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const R = await import('/src/sim/hydroRig.ts');
    const r = await makeRenderer();
    // z < 128: ocean floor 46..60 (depth 16..30); z >= 128: land at 77..79 gently draining into it
    const height = (x: number, z: number) => z < 128 ? 53 + 7 * Math.sin(x * 0.05) * Math.sin(z * Math.PI / 128) : 77 + 2 * Math.sin((z - 128) * Math.PI / 128) + 0.3 * Math.sin(x * 0.2);
    const water = new Float32Array(R.NCOL);
    for (let z = 0; z < R.NZ; z++) for (let x = 0; x < R.NX; x++) {
      const h = height(x, z);
      water[R.idx(x, z)] = z < 128 ? 76.001 + 0.01 * Math.sin(x * 0.03) - h : 0.003 + 0.001 * Math.sin(x * 0.7 + z);
    }
    const rig = await R.makeHydroRig(r, R.voxFromHeights(height), water);
    const w0 = R.sum64(await rig.readF('water'));
    const drift: number[] = [];
    for (let i = 0; i < 4; i++) { rig.hydro.step(r, 1000); drift.push(R.sum64(await rig.readF('water')) - w0); }
    return { w0, drift };
  });
  console.log(`deep-ocean drift after 1k..4k substeps: ${res.drift.map((d) => d.toExponential(2)).join(' ')} (total ${res.w0.toFixed(0)})`);
  // exact adds: only the f64 sum of f32 values remains; allow 1e-10 relative
  for (const d of res.drift) expect(Math.abs(d) / res.w0).toBeLessThan(1e-10);
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
