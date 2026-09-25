import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V6: plates must split (rifts), be absorbed when tiny and merge after sustained collision,
// so the count stays in [3,12] forever and the world keeps reorganising.
test('split relabels exactly the CPU-predicted side; absorb removes a tiny plate', async ({ page }) => {
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { Lifecycle, splitSideCpu, ABSORB_AREA } = await import('/src/sim/lifecycle.ts');
    const { PCG32 } = await import('/src/core/rng.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld([1, 0], [0, 1]);
    // third plate: small 20×20 island of plate 2 inside plate 0
    w.plates[2]!.alive = true;
    for (let z = 10; z < 30; z++) for (let x = 10; x < 30; x++) w.plateId[L.colIdx(x, z)] = 2;
    uploadWorld(f, w);
    createDerivePass(f).run(r);
    const life = new Lifecycle(f);
    const stats = async () => {
      life.gatherStats(r);
      const s = Lifecycle.parseStats(await f.read(r, 'counters'));
      return s;
    };
    const s0 = await stats();
    const rng = new PCG32(1);
    // force a split of plate 1 (continental half)
    const op = life.decide(w.plates, s0, 100, rng, 1);
    if (!op || op.kind !== 'split') return { err: 'no split: ' + JSON.stringify(op) };
    life.apply(r, w.plates, op, 100);
    const pid = new Uint32Array(await f.read(r, 'plateId'));
    let mismatch = 0, newCount = 0;
    for (let z = 0; z < L.NZ; z++) for (let x = 128; x < L.NX; x++) {
      const want = splitSideCpu(x, z, op) ? op.into : 1;
      if (pid[L.colIdx(x, z)] !== want) mismatch++;
      if (pid[L.colIdx(x, z)] === op.into) newCount++;
    }
    const alive = w.plates.filter((p) => p.alive).length;
    // absorb tiny plate 2 over a few windows
    const tinyArea = s0.area[2]!;
    for (let k = 0; k < 30; k++) {
      // counters accumulate across gathers here (no window clear); use per-gather deltas
      const before = Lifecycle.parseStats(await f.read(r, 'counters'));
      life.gatherStats(r);
      const after = Lifecycle.parseStats(await f.read(r, 'counters'));
      const s = { ...after, area: after.area.map((v, i) => v - before.area[i]!) };
      const o = life.decide(w.plates, s, 200 + k, rng);
      if (!o) break;
      life.apply(r, w.plates, o, 200 + k);
      if (o.kind === 'kill') break;
    }
    const pid2 = new Uint32Array(await f.read(r, 'plateId'));
    let left2 = 0;
    for (const p of pid2) if (p === 2) left2++;
    return { mismatch, newCount, alive, tinyArea, absorbArea: ABSORB_AREA, left2, alive2: w.plates[2]!.alive, log: life.log.map((l) => l.op.kind) };
  });
  expect(res.err).toBeUndefined();
  // GPU sin/mod vs CPU double differ only on cells lying exactly on the wobbly split line
  expect(res.mismatch).toBeLessThan(50);
  expect(res.newCount).toBeGreaterThan(1000);
  expect(res.alive).toBe(4);
  expect(res.tinyArea!).toBeLessThan(res.absorbArea!);
  expect(res.left2).toBe(0);
  expect(res.alive2).toBe(false);
});

// User: plates fragment into stray specks. Rigid plates should stay contiguous: very few cells may have
// ≤ 1 neighbour of their own plate after a long full-sim run.
test('plates stay contiguous (few isolated plate cells) over 150 My', async ({ page }) => {
  test.setTimeout(240_000);
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
    while (sim.geoMy < 150) { sim.runTicks(64); await frame(); }
    const pid = new Uint32Array(await f.read(r, 'plateId'));
    let iso = 0;
    for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
      const me = pid[x + z * 256];
      let own = 0;
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) if (pid[((x + dx + 256) % 256) + ((z + dz + 256) % 256) * 256] === me) own++;
      if (own <= 1) iso++;
    }
    return { isoFrac: iso / 65536 };
  });
  expect(res.isoFrac).toBeLessThan(0.002);
});
