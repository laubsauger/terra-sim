import { test, expect, type Page } from '@playwright/test';

// Climate + biome passes (T28-T31) on a synthetic world: ocean everywhere except a continent
// x ∈ [64,192) spanning all latitudes, with two N-S ridges (x = 100 and x = 180, +24 voxels).
// Mid-latitude westerlies (u > 0) cross ridge 1 from the west; trade easterlies (u < 0) cross ridge 2
// from the east, leaving the hot interior between the ridges in rain shadow.

interface Metrics {
  equiv: { water: number; vapor: number; ice: number; temp: number; precip: number; biomeMismatch: number };
  w0: number; w1: number;
  seamT: { z0: number; z1: number; zLast: number };
  warmT: number; coldT: number;
  lowT: number; ridgeT: number;
  westerly: { windward: number; lee: number };
  trade: { windward: number; lee: number };
  iceCold: number; iceWarm: number; iceColdCols: number;
  hotDry: { desert: number; total: number };
  coldBiomes: { ice: number; desert: number; rainforest: number; land: number };
  iceAgeDrop: number;
  nonFinite: number;
  msPerTick: number;
}

let m: Metrics;

async function runWorld(page: Page): Promise<Metrics> {
  await page.goto('/tests/gpu/support/blank.html');
  return page.evaluate(async () => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { Params } = await import('/src/core/params.ts');
    const L = await import('/src/sim/layout.ts');
    const { registerSimFields } = await import('/src/sim/fields.ts');
    const { createClimatePass } = await import('/src/sim/climate.ts');
    const { createBiomePass } = await import('/src/sim/biome.ts');
    const CM = await import('/src/sim/climateModel.ts');
    const BM = await import('/src/sim/biomeModel.ts');
    const { NX, NZ, NCOL, colIdx } = L;

    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();

    const R1 = 100, R2 = 180;
    const surfY = f.cpuArray('surfY') as Float32Array, water = f.cpuArray('water') as Float32Array;
    for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
      const i = colIdx(x, z);
      if (x >= 64 && x < 192) {
        surfY[i] = 84 + 24 * Math.exp(-((x - R1) ** 2) / 32) + 24 * Math.exp(-((x - R2) ** 2) / 32);
        water[i] = 0;
      } else { surfY[i] = 60; water[i] = 16; }
    }
    // CPU reference state from the same start
    const cs = CM.makeClimateState(); const ck = CM.makeClimateScratch();
    cs.surfY.set(surfY); cs.water.set(water);
    const bs = { surfTemp: cs.surfTemp, precip: cs.precip, surfY: cs.surfY, water: cs.water, ice: cs.ice,
      climAvg: new Float32Array(2 * NCOL), veg: new Float32Array(NCOL), biome: new Uint32Array(NCOL) };
    f.markDirty('surfY'); f.markDirty('water');

    const params = new Params();
    const climate = createClimatePass(f, params);
    const biome = createBiomePass(f);
    const readF = async (n: string) => new Float32Array(await f.read(r, n));
    const total = async () => {
      const [w, v, ic] = [await readF('water'), await readF('vapor'), await readF('ice')];
      let t = 0; for (let i = 0; i < NCOL; i++) t += w[i]! + v[i]! + ic[i]!;
      return t;
    };
    const step = () => { climate.step(r, 0.05); biome.step(r); };

    const w0 = await total();
    // 1. GPU mirrors the CPU reference (20 ticks)
    const EQ = 20;
    let seamT = { z0: 0, z1: 0, zLast: 0 };
    for (let t = 0; t < EQ; t++) {
      step(); CM.climateStepCpu(cs, ck, CM.DEFAULT_CLIMATE_UNIFORMS); BM.biomeStepCpu(bs, L.Y_SEA_NOMINAL);
      if (t === 0) { // open ocean at sea level, before rain changes water depth (lapse)
        const T0 = await readF('surfTemp');
        seamT = { z0: T0[colIdx(20, 0)]!, z1: T0[colIdx(20, 1)]!, zLast: T0[colIdx(20, NZ - 1)]! };
      }
    }
    const maxRel = (g: Float32Array, c: Float32Array, floor: number) => {
      let e = 0; for (let i = 0; i < g.length; i++) e = Math.max(e, Math.abs(g[i]! - c[i]!) / Math.max(Math.abs(c[i]!), floor));
      return e;
    };
    const gBiome = new Uint32Array(await f.read(r, 'biome'));
    let biomeMismatch = 0; for (let i = 0; i < NCOL; i++) if (gBiome[i] !== bs.biome[i]) biomeMismatch++;
    const equiv = {
      water: maxRel(await readF('water'), cs.water, 1),
      vapor: maxRel(await readF('vapor'), cs.vapor, 1e-2),
      ice: maxRel(await readF('ice'), cs.ice, 1e-2),
      temp: maxRel(await readF('surfTemp'), cs.surfTemp, 1),
      precip: maxRel(await readF('precip'), cs.precip, 1e-4),
      biomeMismatch,
    };
    // 2. long run: 1000 ticks total for the conservation window, then on to 2000 for climate equilibrium
    const t0 = performance.now();
    for (let t = EQ; t < 1000; t++) step();
    const w1 = await total(); // readback waits for the GPU → upper bound incl. submit + 3 readbacks
    const msPerTick = (performance.now() - t0) / (1000 - EQ);
    for (let t = 1000; t < 2000; t++) step();

    const T = await readF('surfTemp'), ice = await readF('ice'), avg = await readF('climAvg');
    const bio = new Uint32Array(await f.read(r, 'biome'));
    const pAvg = (x: number, z: number) => avg[2 * colIdx(x, z) + 1]!;
    const dLat = (z: number) => Math.min(z, NZ - z); // cells from the warm-belt centre
    const OCEAN_X = 20;

    // precip (long-term) either side of a ridge, averaged over rows and 3..8 cells off the crest
    const flank = (ridge: number, dir: number, rows: number[]) => {
      let s = 0, n = 0;
      for (const z of rows) for (let d = 3; d <= 8; d++) { s += pAvg(ridge + dir * d, z); n++; }
      return s / n;
    };
    const rowsWhere = (pred: (z: number) => boolean) => [...Array(NZ).keys()].filter(pred);
    const westerlyRows = rowsWhere((z) => CM.windU(z) > 0.25);
    const tradeRows = rowsWhere((z) => CM.windU(z) < -0.15 && dLat(z) < 40);

    let iceCold = 0, iceWarm = 0, iceColdCols = 0;
    const cold = { ice: 0, desert: 0, rainforest: 0, land: 0 };
    const hotDry = { desert: 0, total: 0 };
    let nonFinite = 0;
    for (const n of ['water', 'vapor', 'ice', 'surfTemp', 'precip', 'climAvg', 'veg']) {
      const a = await readF(n); for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) nonFinite++;
    }
    for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
      const i = colIdx(x, z), d = dLat(z);
      if (d > NZ / 3) { // latitude > 60°
        iceCold += ice[i]!; if (ice[i]! > 1e-3) iceColdCols++;
        if (bio[i] === BM.Biome.ICE) cold.ice++;
        if (x >= 64 && x < 192) { cold.land++; if (bio[i] === BM.Biome.DESERT) cold.desert++; if (bio[i] === BM.Biome.RAINFOREST) cold.rainforest++; }
      }
      if (d < 25) iceWarm += ice[i]!; // latitude < 17°
      // subtropical interior (latitude 11°..28°) in the lee of ridge 2 (trades): x 110..170
      if (d >= 16 && d < 40 && x >= 110 && x < 170) { hotDry.total++; if (bio[i] === BM.Biome.DESERT) hotDry.desert++; }
    }

    // 3. ice-age forcing actually cools the surface
    let tBefore = 0; for (let i = 0; i < NCOL; i++) tBefore += T[i]!;
    climate.setIceAge(1);
    climate.step(r, 0.05);
    const T2 = await readF('surfTemp');
    let tAfter = 0; for (let i = 0; i < NCOL; i++) tAfter += T2[i]!;
    climate.setIceAge(0);

    return {
      equiv, w0, w1, msPerTick,
      seamT,
      warmT: T[colIdx(OCEAN_X, 0)]!, coldT: T[colIdx(OCEAN_X, NZ / 2)]!,
      lowT: T[colIdx(140, 64)]!, ridgeT: T[colIdx(R1, 64)]!,
      westerly: { windward: flank(R1, -1, westerlyRows), lee: flank(R1, +1, westerlyRows) },
      trade: { windward: flank(R2, +1, tradeRows), lee: flank(R2, -1, tradeRows) },
      iceCold, iceWarm, iceColdCols, hotDry, coldBiomes: cold,
      iceAgeDrop: (tBefore - tAfter) / NCOL,
      nonFinite,
    };
  });
}

test.describe.serial('climate + biome on a synthetic ridge world', () => {
  test.beforeAll(async ({ browser }) => {
    const page = await browser.newPage();
    m = await runWorld(page);
    await page.close();
    console.log(`climate+biome GPU: ${m.msPerTick.toFixed(3)} ms/tick (upper bound)`, JSON.stringify(m));
  });

  // The GPU kernels are the sim; the CPU model is what the tuning and unit tests reason about.
  // If they drift apart, every tuned constant and CPU-side test stops meaning anything.
  test('GPU passes match the CPU reference model', () => {
    expect(m.equiv.water).toBeLessThan(1e-4);
    expect(m.equiv.vapor).toBeLessThan(1e-3);
    expect(m.equiv.ice).toBeLessThan(1e-3);
    expect(m.equiv.temp).toBeLessThan(1e-4);
    expect(m.equiv.precip).toBeLessThan(1e-2);
    expect(m.equiv.biomeMismatch).toBeLessThan(20); // only threshold-edge columns may differ
  });

  // V4: evaporation, rain, snow, melt, advection and glacier flow only move water between
  // water/vapor/ice. Any leak drains or floods the oceans over millions of ticks.
  test('V4: Σwater + Σvapor + Σice conserved over 1000 ticks', () => {
    expect(Math.abs(m.w1 - m.w0) / m.w0).toBeLessThan(1e-5);
  });

  // V1: latitude is periodic; the z seam must be as smooth as any other row step,
  // and the warm belt (z=0) must be warmer than the cold belt (z=NZ/2).
  test('periodic latitude: seamless across the z wrap, warm belt warmer than cold belt', () => {
    const interior = Math.abs(m.seamT.z1 - m.seamT.z0);
    expect(Math.abs(m.seamT.zLast - m.seamT.z0)).toBeLessThanOrEqual(interior + 1e-4);
    expect(m.warmT - m.coldT).toBeGreaterThan(30);
  });

  // Lapse rate: mountain tops must be colder than lowlands at the same latitude (snowcaps, alpine).
  test('higher altitude is colder', () => {
    expect(m.lowT - m.ridgeT).toBeGreaterThan(10);
  });

  // Orographic precip + rain shadow: the windward flank of a ridge gets wetter than the lee,
  // for westerlies (wind from -x) and trades (wind from +x) alike.
  test('windward flank gets more precip than the lee (rain shadow)', () => {
    expect(m.westerly.windward).toBeGreaterThan(5 * m.westerly.lee);
    expect(m.trade.windward).toBeGreaterThan(5 * m.trade.lee);
  });

  // Snow/ice: cold belt accumulates ice from snowfall; the warm belt melts everything.
  test('cold belt accumulates ice, warm belt stays ice-free', () => {
    expect(m.iceCold).toBeGreaterThan(1);
    expect(m.iceColdCols).toBeGreaterThan(500);
    expect(m.iceWarm).toBe(0);
  });

  // Whittaker-like classification: hot + dry (rain shadow) → desert; cold → ice/tundra, never
  // rainforest; the biome map drives render splats and flora, so implausible ids are visible bugs.
  test('biome ids plausible: desert in hot rain shadow, ice in the cold belt', () => {
    expect(m.hotDry.desert / m.hotDry.total).toBeGreaterThan(0.5);
    expect(m.coldBiomes.ice).toBeGreaterThan(0);
    expect(m.coldBiomes.rainforest).toBe(0);
  });

  // The event scheduler drives ice ages through setIceAge; it must actually cool the planet.
  test('ice-age forcing cools the surface by ICE_AGE_DT', () => {
    expect(m.iceAgeDrop).toBeGreaterThan(7.9);
    expect(m.iceAgeDrop).toBeLessThan(8.1);
  });

  // V7: every climate field stays finite.
  test('V7: no NaN/Inf in climate fields', () => {
    expect(m.nonFinite).toBe(0);
  });
});

// A normal generated world run through the full Sim: the biome map must be a real mix, not
// forest-vs-desert, and the GPU classifier must agree with classifyBiomeVeg on real data.
interface GenMetrics { land: number; share: Record<string, number>; mismatch: number; badVeg: number; distinct: number }
let g: GenMetrics;

test.describe.serial('biomes on a generated world (full sim, 1500 ticks)', () => {
  test.beforeAll(async ({ browser }) => {
    test.setTimeout(180_000);
    const page = await browser.newPage();
    await page.goto('/tests/gpu/support/blank.html');
    g = await page.evaluate(async () => {
      const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
      const { GpuFields } = await import('/src/core/gpu.ts');
      const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
      const { Sim } = await import('/src/sim/sim.ts');
      const { Params } = await import('/src/core/params.ts');
      const { generateWorld } = await import('/src/sim/worldgen.ts');
      const BM = await import('/src/sim/biomeModel.ts');
      const { NX, NZ, NCOL } = await import('/src/sim/layout.ts');
      const r = await makeRenderer();
      const f = new GpuFields(); registerSimFields(f); f.freeze();
      const w = generateWorld(1, { plates: 7 });
      uploadWorld(f, w);
      const sim = new Sim(r, f, w, new Params(), 0.05);
      const frame = () => new Promise((res) => setTimeout(res, 0));
      let done = 0;
      while (done < 1500) { done += sim.runTicks(20); await frame(); }
      // biome is the last pass of a tick and 1500 is not a stats-window end → fields are as the kernel saw them
      const rf = async (n: string) => new Float32Array(await f.read(r, n));
      const [surfY, water, ice, avg, veg] = [await rf('surfY'), await rf('water'), await rf('ice'), await rf('climAvg'), await rf('veg')];
      const biome = new Uint32Array(await f.read(r, 'biome'));
      const sea = sim.biome.uniforms.seaLevel.value;
      const names = Object.fromEntries(Object.entries(BM.Biome).map(([k, v]) => [v, k]));
      const cnt: Record<string, number> = {};
      let land = 0, mismatch = 0, badVeg = 0;
      const deep = (x: number, z: number) => water[(x & (NX - 1)) + (z & (NZ - 1)) * NX]! > BM.BIOME.OCEAN_MIN;
      for (let i = 0; i < NCOL; i++) {
        const x = i & (NX - 1), z = i >> 8;
        const coastal = deep(x + 1, z) || deep(x - 1, z) || deep(x, z + 1) || deep(x, z - 1);
        const c = BM.classifyBiomeVeg(avg[2 * i]!, avg[2 * i + 1]!, surfY[i]! - sea, water[i]!, ice[i]!, coastal);
        if (c.biome !== biome[i]) mismatch++;
        const b = biome[i]!;
        if (b === BM.Biome.OCEAN || b === BM.Biome.ICE) continue;
        land++;
        cnt[names[b]!] = (cnt[names[b]!] ?? 0) + 1;
        if (!(veg[i]! >= 0 && veg[i]! <= 1)) badVeg++; // veg lags its target (relaxation), so only its range is checked
      }
      const share: Record<string, number> = {};
      for (const [k, v] of Object.entries(cnt)) share[k] = v / land;
      return { land, share, mismatch, badVeg, distinct: Object.values(share).filter((s) => s >= 0.02).length };
    });
    await page.close();
    console.log('generated world biome shares', JSON.stringify(g));
  });

  test('GPU biome ids match classifyBiomeVeg on the real fields', () => {
    expect(g.mismatch).toBeLessThan(0.002 * 65536); // threshold-edge float differences only
    expect(g.badVeg).toBe(0);
  });

  // User-facing variety: rain-shadow interiors must grade through semi-arid belts
  // (steppe / shrubland / cold desert) instead of jumping from forest to desert.
  test('meaningful biome mix with semi-arid transition belts', () => {
    expect(g.land).toBeGreaterThan(10_000);
    expect(g.distinct).toBeGreaterThanOrEqual(8);
    const semi = (g.share.STEPPE ?? 0) + (g.share.SHRUBLAND ?? 0) + (g.share.COLD_DESERT ?? 0);
    expect(semi).toBeGreaterThan(0.08);
    expect(g.share.STEPPE ?? 0).toBeGreaterThan(0.02);
    expect(g.share.SHRUBLAND ?? 0).toBeGreaterThan(0.02);
    expect(g.share.DESERT ?? 0).toBeLessThan(0.3);
  });
});
