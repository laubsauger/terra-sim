import { test, expect } from '@playwright/test';

// B19: an invalid WGSL kernel only logs an error and the pass silently stops running; every sim test
// here must fail on GPU validation errors instead of passing on a half-dead sim.
let gpuErrors: string[] = [];
test.beforeEach(async ({ page }) => {
  gpuErrors = [];
  page.on('console', (m) => { if (m.type() === 'error' && /webgpu|wgsl|shader|pipeline|validation/i.test(m.text())) gpuErrors.push(m.text().slice(0, 300)); });
  await page.goto('/tests/gpu/support/blank.html');
});
test.afterEach(() => { expect(gpuErrors, 'GPU validation errors').toEqual([]); });

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

// B13 / V26: open ocean must stay level on the live sim (tectonic shifts re-roughen it every run),
// and leveling must not create or destroy water (V4).
test('full sim keeps open ocean level and water conserved', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const { generateWorld } = await import('/src/sim/worldgen.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = generateWorld(3, { plates: 7 }); uploadWorld(f, w);
    const sim = new Sim(r, f, w, new Params(), 0.05);
    const frame = () => new Promise((res) => setTimeout(res, 0));
    const total = async () => { let t = 0; for (const n of ['water', 'vapor', 'ice']) for (const v of new Float32Array(await f.read(r, n))) t += v; return t; };
    const W0 = await total();
    while (sim.geoMy < 60) { sim.runTicks(32); await frame(); }
    const s = new Float32Array(await f.read(r, 'surfY')), wa = new Float32Array(await f.read(r, 'water'));
    const sea = sim.stats!.seaLevel;
    let n = 0, sum = 0, sum2 = 0;
    for (let i = 0; i < s.length; i++) if (s[i]! < sea - 3) { const L = s[i]! + wa[i]!; n++; sum += L; sum2 += L * L; }
    const mean = sum / n;
    return { std: Math.sqrt(sum2 / n - mean * mean), drift: Math.abs((await total()) - W0) / W0 };
  });
  expect(res.std).toBeLessThan(0.3);
  expect(res.drift).toBeLessThan(1e-4);
});

// Earthquake sites must come from real plate-boundary activity, so FX quakes strike where plates collide.
test('quake sites sit on the active subduction front', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = twoPlateWorld([3, 0], [0, 0]); uploadWorld(f, w);
    const sim = new Sim(r, f, w, new Params(), 0.05);
    const frame = () => new Promise((res) => setTimeout(res, 0));
    while (sim.tick < 200) { sim.runTicks(16); await frame(); }
    const q = sim.quakeSites;
    // distance (cells) from the site to the nearest plate boundary
    const pid = new Uint32Array(await f.read(r, 'plateId'));
    let near = 99;
    if (q.subduct) {
      const me = pid[q.subduct.x + q.subduct.z * 256];
      for (let dz = -14; dz <= 14; dz++) for (let dx = -14; dx <= 14; dx++) {
        const x = (q.subduct.x + dx + 256) % 256, z = (q.subduct.z + dz + 256) % 256;
        if (pid[x + z * 256] !== me) near = Math.min(near, Math.max(Math.abs(dx), Math.abs(dz)));
      }
    }
    // a split/merge applied at the same window end can move or erase the boundary the site came from
    const lastOp = sim.lifecycle.log[sim.lifecycle.log.length - 1];
    const recentOp = !!lastOp && sim.geoMy - lastOp.my <= 4;
    return { q, near, recentOp };
  });
  expect(res.q.subduct).not.toBeNull();
  expect(res.q.subduct!.events).toBeGreaterThan(10);
  // sites come from the previous stats window (2-4 My old); a boundary moving ~2 cells/My has moved on since,
  // so 'at the boundary' means within ~12 cells (plates here are > 100 cells across)
  expect(res.near <= 12 || res.recentOp, `site ${JSON.stringify(res)}`).toBe(true);
});
