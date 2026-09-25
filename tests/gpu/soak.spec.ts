import { test, expect } from '@playwright/test';

// §T.36 / V8: the vivarium must stay alive and interesting unattended. Run 5000 My at max speed with no
// rendering and check the world never dries out, drowns, flattens or freezes. Gated by SOAK=1 (npm run soak).
const SOAK_MY = Number(process.env.SOAK_MY ?? 5000);
const RELIEF_MIN = 12; // layers between sea level and the 99th-percentile land height

test.skip(!process.env.SOAK, 'soak runs only with SOAK=1');

test('5000 My soak keeps the world balanced (V8)', async ({ page }) => {
  test.setTimeout(0);
  await page.goto('/tests/gpu/support/blank.html');
  const res = await page.evaluate(async ({ SOAK_MY, seed }) => {
    const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { Sim } = await import('/src/sim/sim.ts');
    const { Params } = await import('/src/core/params.ts');
    const { generateWorld } = await import('/src/sim/worldgen.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = generateWorld(seed, { plates: 7 });
    uploadWorld(f, w);
    const sim = new Sim(r, f, w, new Params(), 0.05);
    const frame = () => new Promise((res) => setTimeout(res, 0));
    const samples: { my: number; land: number; ocean: number; relief: number; plates: number; finite: boolean; phase: string; islands: number; water: number; res: number }[] = [];
    const waterTotal = async () => { let t = 0; for (const n of ['water', 'vapor', 'ice']) for (const v of new Float32Array(await f.read(r, n))) t += v; return t; };
    const W0 = await waterTotal();
    let minPlates = 99, maxPlates = 0;
    const sample = async () => {
      const s = new Float32Array(await f.read(r, 'surfY'));
      const wa = new Float32Array(await f.read(r, 'water'));
      const ci = new Uint32Array(await f.read(r, 'colInfo'));
      let islands = 0; // land on oceanic crust: hotspot chains, island arcs, ridge highs
      for (let i = 0; i < wa.length; i++) if (wa[i]! < 0.5 && !((ci[i * 2]! >> 16) & 1)) islands++;
      const sea = sim.stats?.seaLevel ?? 76;
      let land = 0, ocean = 0, finite = true;
      const heights: number[] = [];
      for (let i = 0; i < s.length; i++) {
        if (!Number.isFinite(s[i]!) || !Number.isFinite(wa[i]!)) finite = false;
        if (wa[i]! < 0.5) { land++; heights.push(s[i]!); }
        if (wa[i]! > 3) ocean++;
      }
      heights.sort((a, b) => a - b);
      const p99 = heights.length ? heights[Math.floor(heights.length * 0.99)]! : sea;
      const res = new Int32Array(await f.read(r, 'counters'))[0]! / 255 / s.length; // mantle reservoir, layers per column
      samples.push({ my: sim.geoMy, land: land / s.length, ocean: ocean / s.length, relief: p99 - sea, plates: sim.alivePlates(), finite, phase: sim.wilson.state.phase, islands,
        water: Math.abs((await waterTotal()) - W0) / W0, res });
    };
    const t0 = performance.now();
    let next = 0;
    while (sim.geoMy < SOAK_MY) {
      sim.runTicks(64);
      minPlates = Math.min(minPlates, sim.alivePlates()); maxPlates = Math.max(maxPlates, sim.alivePlates());
      if (sim.geoMy >= next) { await sample(); next += 50; }
      await frame();
    }
    const magma = await sim.magma.readStats(r);
    return {
      eruptions: magma.eruptions,
      samples, minPlates, maxPlates, cycles: sim.wilson.state.cycles, unhandled: sim.unhandledEvents,
      events: sim.events.log.length, seconds: (performance.now() - t0) / 1000,
    };
  }, { SOAK_MY, seed: Number(process.env.SOAK_SEED ?? 3) });

  console.log(`soak ${SOAK_MY} My in ${res.seconds.toFixed(0)} s, eruptions=${res.eruptions}, cycles=${res.cycles}, plates ${res.minPlates}-${res.maxPlates}, unhandled events=${res.unhandled}`);
  for (const s of res.samples.filter((_, i) => i % 10 === 0)) console.log(JSON.stringify(s));

  // The collapse the user saw (land 27% → 7%, water −11%, reservoir 7 layers/col) starts in the first
  // 100 My, so balance holds from the first samples, not only after a long spin-up.
  const early = res.samples.filter((s) => s.my > 20);
  for (const s of early) {
    expect(s.water, `water drift at ${s.my.toFixed(0)} My`).toBeLessThan(1e-3); // V4
    expect(Math.abs(s.res), `reservoir at ${s.my.toFixed(0)} My`).toBeLessThan(3); // crust is not parked in the mantle (V3)
    expect(s.land, `land at ${s.my.toFixed(0)} My`).toBeGreaterThanOrEqual(0.12);
  }
  const late = res.samples.filter((s) => s.my > 200); // allow spin-up
  for (const s of late) {
    expect(s.finite).toBe(true);                               // V7
    expect(s.land).toBeGreaterThanOrEqual(0.15);                // never drowns
    expect(s.land).toBeLessThanOrEqual(0.6);                    // never dries out
    expect(s.ocean).toBeGreaterThanOrEqual(0.3);
    expect(s.relief).toBeGreaterThanOrEqual(RELIEF_MIN);        // never flattens
  }
  expect(res.minPlates).toBeGreaterThanOrEqual(3);             // V6
  expect(res.maxPlates).toBeLessThanOrEqual(12);
  expect(res.cycles).toBeGreaterThanOrEqual(1);                // at least one full Wilson cycle
  expect(res.unhandled).toBe(0);                               // every event kind has a handler
  expect(res.eruptions).toBeGreaterThanOrEqual(SOAK_MY / 50);  // ≥1 eruption per 50 My
  // volcanic islands must keep emerging, not just once from worldgen
  expect(late.filter((s) => s.islands >= 20).length).toBeGreaterThanOrEqual(late.length * 0.5);
});
