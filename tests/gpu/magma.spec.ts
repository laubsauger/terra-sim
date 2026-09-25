import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// V3: magma and lava are budget members. Every fill unit melted out of the reservoir, emplaced in a chamber,
// erupted, flowed, frozen, crystallized — or destroyed with a subducted column and credited back — must add
// up exactly, or crust mass drifts over millions of ticks. The scenario exercises all paths: a plume under an
// ocean plate that is subducting under a continent (chambers ride into the trench), arc volcanism, ridge lava.
test('V3: crust + reservoir + magma + lava exactly conserved through melt, eruptions, flow, subduction; V7 finite', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const r = await makeRenderer();
    const w = twoPlateWorld([3, 0], [0, 0]);
    w.mantleReservoir = 50_000_000;
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [{ x: 80, z: 128, r: 9, age: 20, life: 1e9 }], evolve: false } });
    const b0 = await rig.budget();
    const totals: number[] = [];
    const mismatches: number[] = [];
    for (let k = 0; k < 4; k++) {
      rig.step(200);
      const b = await rig.budget();
      totals.push(b.total);
      mismatches.push(b.cacheMismatch);
    }
    const st = await rig.stats();
    const b1 = await rig.budget();
    let finite = true;
    for (const n of ['mantle', 'heatFlow', 'crustTemp', 'arc', 'surfY']) {
      const a = await rig.readF(n);
      for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) { finite = false; break; }
    }
    return { t0: b0.total, totals, mismatches, chambers: b1.chambers, pending: b1.pending, lava: b1.lava, st, finite };
  });
  for (const t of res.totals) expect(t).toBe(res.t0);
  // magCol.x (what eruptions and the tectonics fix-up trust) must equal the MAGMA voxels, column by column
  for (const m of res.mismatches) expect(m).toBe(0);
  expect(res.finite).toBe(true);
  expect(res.st.eruptions).toBeGreaterThan(0); // episodes
  expect(res.st.returned).toBeGreaterThan(0);   // chambers carried into the trench were credited back
  expect(res.st.solidified).toBeGreaterThan(0);
  expect(res.st.frozen).toBeGreaterThan(0);
  expect(res.chambers + res.pending).toBeGreaterThan(0);
});

// A plume fixed in the mantle frame under a moving plate must leave a line of volcanic edifices carried
// downstream with the plate, youngest over the plume and oldest farthest away (Hawaii–Emperor). If chambers
// or rock did not ride with the plate, or melt did not follow the mantle frame, there would be one blob.
test('hotspot under a moving plate leaves a chain, oldest farthest; hotspot rock stays oceanic', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    // both plates drift east at 2 cells/My: rigid translation, no convergence, no ridges
    const w = twoPlateWorld([2, 0], [2, 0]);
    w.mantleReservoir = 50_000_000;
    const PX = 100;
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [{ x: PX, z: 128, r: 9, age: 20, life: 1e9 }], evolve: false } });
    const profile = async () => {
      const s = await rig.readF('surfY');
      const on = new Array(256).fill(0), off = new Array(256).fill(0);
      for (let x = 0; x < 256; x++) for (let z = 0; z < 256; z++) {
        const e = s[L.colIdx(x, z)]! - 61; // ocean floor was flat at 61
        if (Math.abs(z - 128) <= 10) on[x] = Math.max(on[x], e); else if (Math.abs(z - 128) > 40) off[x] = Math.max(off[x], e);
      }
      return { on, off };
    };
    const farEnd = (on: number[]) => { let x = PX; while (x < 250 && Math.max(on[x + 1]!, on[x + 2]!, on[x + 3]!) >= 1) x++; return x; };
    rig.step(300); // 15 My → plate moved ~30 cells
    const p1 = await profile();
    rig.step(300); // 30 My → ~60 cells
    const p2 = await profile();
    // material check along the chain: everything above the old floor is oceanic (no continental flag)
    const vox = await rig.readU('vox');
    let volcanic = 0, flagged = 0;
    for (let x = PX; x < PX + 50; x++) for (let z = 118; z <= 138; z++) for (let y = 61; y < L.NY; y++) {
      const v = vox[L.voxIdx(x, y, z)]!;
      if (L.voxMat(v) === L.Mat.AIR) continue;
      volcanic++;
      if (L.voxFlags(v) & L.FLAG_CONTINENTAL) flagged++;
    }
    const s = p2.on.slice(PX + 4, PX + 50);
    return {
      end1: farEnd(p1.on), end2: farEnd(p2.on),
      chainFrac: s.filter((e: number) => e >= 1).length / s.length,
      peak: Math.max(...p2.on.slice(PX, PX + 60)),
      upstream: Math.max(...p2.on.slice(62, PX - 12)),
      offAxis: Math.max(...p2.off.slice(62, 186)),
      volcanic, flagged,
    };
  });
  console.log('chain', JSON.stringify(res));
  // chain extends downstream by the plate displacement between snapshots (30 cells): oldest part moved farthest
  expect(res.end2 - res.end1).toBeGreaterThan(22);
  expect(res.end2 - res.end1).toBeLessThan(38);
  expect(res.end2).toBeGreaterThan(140);
  expect(res.chainFrac).toBeGreaterThan(0.7);
  expect(res.peak).toBeGreaterThan(3);
  expect(res.upstream).toBeLessThan(1);  // nothing upstream of the plume
  expect(res.offAxis).toBeLessThan(1);   // a line, not a sheet
  expect(res.volcanic).toBeGreaterThan(500);
  expect(res.flagged).toBe(0);           // hotspot basalt/gabbro stays oceanic
});

// Arc magmas form where the slab reaches melting depth: a few cells inland on the overriding plate. Arc
// andesite carries FLAG_CONTINENTAL because arcs are how continents regrow from the mantle reservoir (V3
// long-term balance). The subducting ocean must get no melt at all.
test('arc volcanism on the overriding continent 3-8 cells inland, none on the subducting ocean', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const w = twoPlateWorld([3, 0], [0, 0]); // trench at x = 128 (continent wins)
    w.mantleReservoir = 50_000_000;
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [], evolve: false } });
    rig.step(600);
    const vox = await rig.readU('vox');
    const magCol = await rig.readU('magCol');
    const arcRock = new Array(256).fill(0);
    let andesite = 0, andesiteFlagged = 0, oceanMagma = 0, oceanPending = 0;
    for (let y = 0; y < L.NY; y++) for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
      const v = vox[L.voxIdx(x, y, z)]!;
      const m = L.voxMat(v);
      if (m === L.Mat.ANDESITE) { andesite++; if (L.voxFlags(v) & L.FLAG_CONTINENTAL) andesiteFlagged++; }
      if (m === L.Mat.ANDESITE || m === L.Mat.MAGMA) arcRock[x]++;
      if (m === L.Mat.MAGMA && x >= 20 && x < 128) oceanMagma++;
    }
    for (let z = 0; z < 256; z++) for (let x = 20; x < 128; x++) oceanPending += magCol[L.colIdx(x, z) * 2 + 1]!;
    const total = arcRock.slice(128, 256).reduce((a: number, b: number) => a + b, 0);
    // chambers sit 3-8 cells inland; arc lava then flows a few cells further, so rock spreads to ~x 145
    const near = arcRock.slice(130, 146).reduce((a: number, b: number) => a + b, 0);
    const far = arcRock.slice(150, 256).reduce((a: number, b: number) => a + b, 0);
    let peakX = 128; for (let x = 128; x < 256; x++) if (arcRock[x] > arcRock[peakX]) peakX = x;
    return { total, near, far, peakX, andesite, andesiteFlagged, oceanMagma, oceanPending, profile: arcRock.slice(126, 142) };
  });
  console.log('arc', JSON.stringify(res));
  expect(res.total).toBeGreaterThan(2000);
  expect(res.peakX).toBeGreaterThanOrEqual(131);
  expect(res.peakX).toBeLessThanOrEqual(137);
  expect(res.near / res.total).toBeGreaterThan(0.8);
  expect(res.far / res.total).toBeLessThan(0.05);
  expect(res.oceanMagma).toBe(0);
  expect(res.oceanPending).toBe(0);
  expect(res.andesite).toBeGreaterThan(0);
  expect(res.andesiteFlagged).toBe(res.andesite);
});

// Lava is how volcanic edifices get their shape: it must run downhill, stop as it cools, and turn into rock
// that raises the surface near the vent — exactly converting lava mass into crust mass (V3).
test('lava flows downhill, cools, solidifies into rock; surface rises near vent; lava → 0; exact mass', async ({ page }) => {
  test.setTimeout(120_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { makeMagmaRig, hillWorld } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const M = await import('/src/sim/magmaModel.ts');
    const r = await makeRenderer();
    const rig = await makeMagmaRig(r, hillWorld(), { mantle: { plumes: [], evolve: false } });
    const only = { tectonics: false, mantle: false, magma: false };
    const VX = 100, VZ = 128; // west flank of a hill centred at x = 128: downhill is −x
    const s0 = await rig.readF('surfY');
    const b0 = (await rig.budget()).total;
    // eruption-sized pulses (one layer each), like a vent erupting every tick
    for (let k = 0; k < 40; k++) { rig.lava.inject(r, L.colIdx(VX, VZ), 255); rig.step(1, only); }
    const b1 = (await rig.budget()).total;
    const lv = await rig.readU('lava');
    let m = 0, mx = 0, west = 0, east = 0, tMax = 0;
    for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
      const c = L.colIdx(x, z), a = lv[c * 2]!;
      if (!a) continue;
      m += a; mx += a * x;
      if (x < VX) west += a; else if (x > VX) east += a;
      tMax = Math.max(tMax, M.lavaTempC(lv[c * 2 + 1]!));
    }
    const early = { centroid: mx / Math.max(m, 1), west, east, tMax, mass: m };
    const mid = (await rig.budget()).total;
    rig.step(600, only);
    const bEnd = await rig.budget();
    const s1 = await rig.readF('surfY');
    let near = 0, farDelta = 0, rise = 0, rx = 0;
    for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
      const c = L.colIdx(x, z), d = s1[c]! - s0[c]!;
      rise += d; rx += d * x;
      const dx = x - VX, dz = z - VZ;
      if (dx * dx + dz * dz <= 36) near = Math.max(near, d);
      if (dx * dx + dz * dz > 60 * 60) farDelta = Math.max(farDelta, Math.abs(d));
    }
    return { b0, b1, mid, end: bEnd.total, lavaEnd: bEnd.lava, early, near, farDelta, rise, rockCentroid: rx / rise };
  });
  console.log('lava', JSON.stringify(res));
  expect(res.b1).toBe(res.b0);       // inject draws from the reservoir
  expect(res.mid).toBe(res.b0);
  expect(res.end).toBe(res.b0);
  expect(res.early.mass).toBeGreaterThan(0);
  expect(res.early.centroid).toBeLessThan(100 - 1);   // live lava moved downhill
  expect(res.early.west).toBeGreaterThan(res.early.east * 1.5);
  expect(res.rockCentroid).toBeLessThan(100 - 1);     // and so did the rock it left
  expect(res.early.tMax).toBeLessThan(1200);          // cooling
  expect(res.lavaEnd).toBe(0);                        // everything froze
  expect(res.near).toBeGreaterThan(0.5);              // rock built up around the vent
  expect(res.farDelta).toBe(0);
  expect(res.rise).toBeGreaterThan(40 * 0.95);        // ≈ 40 layers of rock: Σ surface rise over all columns
  expect(res.rise).toBeLessThan(40 * 1.05);
});

// A lava flow is a tongue that grows downhill from its source. Routing is deterministic steepest descent with
// channel memory, so a steadily fed flow's front (lava or fresh rock) must advance monotonically downhill as a
// narrow tongue, not jump around.
test('flow front advances monotonically downhill as a coherent tongue', async ({ page }) => {
  test.setTimeout(120_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { makeMagmaRig, hillWorld } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const rig = await makeMagmaRig(r, hillWorld(), { mantle: { plumes: [], evolve: false } });
    const only = { tectonics: false, mantle: false, magma: false };
    const VX = 100, VZ = 128; // west flank of a hill centred at x = 128: downhill is −x
    const s0 = await rig.readF('surfY');
    const b0 = (await rig.budget()).total;
    const fronts: number[] = [];
    let width = 0, length = 0;
    for (let t = 0; t < 60; t++) {
      rig.lava.inject(r, L.colIdx(VX, VZ), 128); // steady effusion
      rig.step(1, only);
      if (t % 5 === 4) {
        const lv = await rig.readU('lava');
        const s = await rig.readF('surfY');
        let front = 0, zMin = 999, zMax = -1, xMax = -999;
        for (let z = 0; z < 256; z++) for (let x = 0; x < 256; x++) {
          const c = L.colIdx(x, z);
          if (lv[c * 2]! === 0 && s[c]! - s0[c]! < 0.02) continue;
          front = Math.max(front, VX - x); xMax = Math.max(xMax, x);
          zMin = Math.min(zMin, z); zMax = Math.max(zMax, z);
        }
        fronts.push(front);
        width = zMax - zMin + 1; length = front + (xMax - VX) + 1;
      }
    }
    return { fronts, width, length, drift: (await rig.budget()).total - b0 };
  });
  console.log('front', JSON.stringify(res));
  for (let k = 1; k < res.fronts.length; k++) expect(res.fronts[k]!).toBeGreaterThanOrEqual(res.fronts[k - 1]!);
  expect(res.fronts.at(-1)! - res.fronts[0]!).toBeGreaterThanOrEqual(6); // it keeps advancing
  expect(res.fronts.at(-1)!).toBeGreaterThanOrEqual(12);
  expect(res.width).toBeLessThan(res.length);                            // a tongue, not a blob
  expect(res.drift).toBe(0);                                             // V3 (inject draws the reservoir)
});

// An eruption episode reads as a volcano: build-up, a sustained active phase with a persistent lava lake in the
// vent (fed every tick, not flickering), waning, then a summit crater below the rim — with exact mass throughout.
test('eruption episode: build-up → active lake → waning → summit crater; exact V3', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { makeMagmaRig, hillWorld } = await import('/src/sim/magmaRig.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const w = hillWorld();
    const VX = 128, VZ = 128; // summit: surface ≈ 100, top voxel y = 99
    for (let y = 86; y <= 93; y++) w.vox[L.voxIdx(VX, y, VZ)] = L.packVoxel(L.Mat.MAGMA, 255, 0, 0);
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [], evolve: false } });
    const vent = L.colIdx(VX, VZ);
    const b0 = (await rig.budget()).total;
    const s0 = await rig.readF('surfY');
    const phases: number[] = [];
    let activeTicks = 0, activeNoLake = 0, maxBuildAct = 0, activeActMin = 1;
    for (let t = 0; t < 700; t++) {
      rig.step(1, { tectonics: false });
      const v = await rig.readF('volcano');
      const ph = v[vent * 4 + 1]!;
      if (phases.at(-1) !== ph) phases.push(ph);
      if (ph === 1) maxBuildAct = Math.max(maxBuildAct, v[vent * 4]!);
      if (ph === 2) {
        activeTicks++;
        activeActMin = Math.min(activeActMin, v[vent * 4]!);
        if ((await rig.readU('lava'))[vent * 2]! === 0) activeNoLake++;
      }
      if (phases.length >= 5 && ph === 0) break;
    }
    const s = await rig.readF('surfY');
    let rimMin = 1e9; for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) rimMin = Math.min(rimMin, s[L.colIdx(VX + dx!, VZ + dz!)]!);
    let flowRise = 0; for (let c = 0; c < L.NCOL; c++) { const x = c & 255, z = c >> 8; if (Math.max(Math.abs(x - VX), Math.abs(z - VZ)) > 3) flowRise += Math.max(0, s[c]! - s0[c]!); }
    const b1 = await rig.budget();
    return { phases, activeTicks, activeNoLake, maxBuildAct, activeActMin, vent: s[vent]!, rimMin, flowRise, drift: b1.total - b0, st: await rig.stats() };
  });
  console.log('episode', JSON.stringify(res));
  expect(res.phases).toEqual([1, 2, 3, 4, 0]);   // lifecycle order, one episode
  expect(res.phases).toContain(2);
  expect(res.phases.at(-1)).toBe(0);            // episode completed
  expect(res.activeTicks).toBeGreaterThan(40);  // several My of sustained activity, not a flicker
  expect(res.activeNoLake).toBe(0);             // the vent holds a lava lake every active tick
  expect(res.activeActMin).toBe(1);
  expect(res.maxBuildAct).toBeLessThan(0.31);
  expect(res.vent).toBeLessThan(res.rimMin - 0.5); // summit crater below the rim
  expect(res.flowRise).toBeGreaterThan(0.5);    // lava overflowed the crater rim and built flows down the flanks
  expect(res.st.eruptions).toBe(1);
  expect(res.drift).toBe(0);
});

// V2: identical inputs → identical state (integer atomics only, dither from (column, tick) hashes).
test('V2: magma/lava/mantle deterministic across runs', async ({ page }) => {
  test.setTimeout(180_000);
  const h = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const { hashFields } = await import('/src/sim/hash.ts');
    const r = await makeRenderer();
    const run = async () => {
      const w = twoPlateWorld([3, 1], [-0.5, 0]);
      w.mantleReservoir = 50_000_000;
      const rig = await makeMagmaRig(r, w, { mantle: { plumes: [{ x: 70, z: 60, r: 9, age: 20, life: 1e9 }] } });
      rig.step(240);
      return hashFields(r, rig.f, ['vox', 'lava', 'magCol', 'arc', 'mantle', 'crustTemp', 'heatFlow', 'counters', 'magmaCtr']);
    };
    return [await run(), await run()];
  });
  expect(h[1]).toBe(h[0]);
});

// Budget: all magma/lava/mantle kernels well under ~1.5 ms per tick on average (V9 sim ≤ 5 ms/frame).
test('perf: magma + lava + mantle average cost per tick', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const r = await makeRenderer();
    const w = twoPlateWorld([3, 0], [0, 0]);
    w.mantleReservoir = 50_000_000;
    const rig = await makeMagmaRig(r, w, { mantle: { plumes: [{ x: 80, z: 128, r: 9, age: 20, life: 1e9 }], evolve: false } });
    rig.step(200); // warm up: pipelines compiled, chambers active
    await rig.readU('magmaCtr');
    const time = async (only: Parameters<typeof rig.step>[1]) => {
      const t0 = performance.now();
      rig.step(400, only);
      await rig.readU('magmaCtr');
      return (performance.now() - t0) / 400;
    };
    const base = await time({ tectonics: false, mantle: false, magma: false, lava: false }); // derive only
    const full = await time({ tectonics: false });
    return { base, full, mine: full - base };
  });
  console.log(`magma+lava+mantle ≈ ${res.mine.toFixed(3)} ms/tick (wall, incl. dispatch overhead); derive-only ${res.base.toFixed(3)} ms/tick`);
  expect(res.mine).toBeLessThan(3); // loose wall-clock bound (headless, includes CPU dispatch cost)
});

// Regression (holes through crust + mantle band, found in integration): a full-sim run must never sink a column
// below the deepest crust base the world started with, nor lose the mantle floor at y = 0. Root causes were in
// crustFlow (a crust-less receiver has base = NY, so flow wrote GNEISS at the grid ceiling → floating stacks that
// isostasy sank through the mantle) and uncapped thermal subsidence of stripped, ancient columns.
test('full sim 60 My: no column below the initial deepest crust base; y = 0 always PERIDOTITE', async ({ page }) => {
  test.setTimeout(600_000);
  const res = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const { generateWorld } = await import('/src/sim/worldgen.ts');
    const L = await import('/src/sim/layout.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = generateWorld(3, { plates: 7 });
    let floor = L.NY;
    for (let c = 0; c < L.NCOL; c++) for (let y = 0; y < L.NY; y++) {
      const m = L.voxMat(w.vox[c + y * L.NCOL]!);
      if (m !== L.Mat.AIR && m !== L.Mat.PERIDOTITE && m !== L.Mat.MAGMA) { floor = Math.min(floor, y); break; }
    }
    uploadWorld(f, w);
    const sim = new Sim(r, f, w, new Params(), 0.05);
    const frame = () => new Promise((res) => setTimeout(res, 0));
    const checks: { my: number; minSurf: number; below: number; y0bad: number }[] = [];
    let next = 200;
    while (sim.tick < 1200) {
      sim.runTicks(20); await frame();
      if (sim.tick >= next) {
        next += 200;
        const s = new Float32Array(await f.read(r, 'surfY'));
        const vox = new Uint32Array(await f.read(r, 'vox'), 0, L.NCOL); // layer y = 0
        let minSurf = 1e9, below = 0, y0bad = 0;
        for (let c = 0; c < L.NCOL; c++) {
          minSurf = Math.min(minSurf, s[c]!); if (s[c]! < floor) below++;
          if (L.voxMat(vox[c]!) !== L.Mat.PERIDOTITE) y0bad++;
        }
        checks.push({ my: Math.round(sim.geoMy), minSurf, below, y0bad });
      }
    }
    return { floor, checks };
  });
  console.log('holes regression', JSON.stringify(res));
  for (const c of res.checks) {
    expect(c.below, `columns below initial crust floor ${res.floor} at ${c.my} My`).toBe(0);
    expect(c.y0bad, `y=0 not PERIDOTITE at ${c.my} My`).toBe(0);
  }
});

// Long-run balance report on a generated world (BALANCE=1): reservoir inflow from tectonics (subduction +
// delamination − ridges) vs outflow through volcanism. Informational; prints rates per My.
test('balance report (BALANCE=1)', async ({ page }) => {
  test.skip(!process.env.BALANCE);
  test.setTimeout(600_000);
  const rows = await page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { generateWorld } = await import('/src/sim/worldgen.ts');
    const { makeMagmaRig } = await import('/src/sim/magmaRig.ts');
    const r = await makeRenderer();
    const out: string[] = [];
    for (const seed of [1]) {
      const rig = await makeMagmaRig(r, generateWorld(seed), { isoEvery: 4 });
      const b0 = await rig.budget();
      let prev = { res: b0.reservoir, st: await rig.stats() };
      for (let k = 1; k <= 5; k++) {
        rig.step(1000); // 50 My
        const b = await rig.budget();
        const st = await rig.stats();
        const my = 50;
        const arc = await rig.readF('arc');
        const ctrM = new Int32Array(await rig.f.read(r, 'magmaCtr'));
        let sw = 0, nw = 0, sz = 0, nTrench = 0, arcRate = 0, nArc = 0;
        const MM = await import('/src/sim/magmaModel.ts');
        for (let c = 0; c < 65536; c++) {
          sw += arc[c * 4 + 3]!; if (arc[c * 4 + 3]! > 0.05) nw++;
          sz += arc[c * 4 + 2]!; if (arc[c * 4]! === 0) nTrench++;
          const wgt = MM.arcWindow(arc[c * 4]!); if (wgt > 0 && arc[c * 4 + 1]! > 0) { nArc++; arcRate += Math.min(arc[c * 4 + 1]!, 32) * wgt; }
        }
        out.push(`  Σz=${sz.toFixed(0)} (≈ events/My × τ) trenchCols=${nTrench} arcCols=${nArc} arcWeight=${arcRate.toFixed(0)} E_EMA=${ctrM[10]! / 256} K=${ctrM[12]! / 1024}`);
        const ctrs = new Int32Array(await rig.f.read(r, 'counters'));
        let created = 0; for (let p = 0; p < 16; p++) created += ctrs[16 + p * 4 + 2]!;
        // pending histogram: where is stuck melt?
        const info = await rig.readU('colInfo');
        let pendCols = 0, pendFull = 0, pendThin = 0;
        for (let c = 0; c < 65536; c++) { const p = b.magCol[c * 2 + 1]!; if (p >= 255) { pendCols++; if (p >= 1000) pendFull++; const base = info[c * 2]! & 255, top = (info[c * 2]! >> 8) & 255; if (top - base < 4) pendThin++; } }
        out.push(`  Σw=${sw.toFixed(0)} nW=${nw} ridgeColsCreated(cum)=${created} pendCols≥255=${pendCols} atCap=${pendFull} thin=${pendThin}`);
        const drawn = (st.drawn - prev.st.drawn) / my, returned = (st.returned - prev.st.returned) / my;
        const tecNet = (b.reservoir - prev.res) / my + drawn - returned; // tectonics-only reservoir change
        out.push(`seed ${seed} t=${k * 50}My res=${(b.reservoir / 255).toFixed(0)}vox tecNet=${(tecNet / 255).toFixed(0)} vox/My volcanicDraw=${(drawn / 255).toFixed(0)} vox/My returned=${(returned / 255).toFixed(0)} vox/My eruptions/My=${((st.eruptions - prev.st.eruptions) / my).toFixed(0)} frozen=${((st.frozen - prev.st.frozen) / my / 255).toFixed(0)} solid=${((st.solidified - prev.st.solidified) / my / 255).toFixed(0)} pending=${(b.pending / 255).toFixed(0)}vox chambers=${(b.chambers / 255).toFixed(0)}vox lava=${(b.lava / 255).toFixed(1)}vox drift=${b.total - b0.total}`);
        prev = { res: b.reservoir, st };
      }
    }
    return out;
  });
  console.log(rows.join('\n'));
});
