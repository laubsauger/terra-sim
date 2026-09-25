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
    const samples: { my: number; land: number; ocean: number; relief: number; plates: number; finite: boolean; phase: string }[] = [];
    let minPlates = 99, maxPlates = 0;
    const sample = async () => {
      const s = new Float32Array(await f.read(r, 'surfY'));
      const wa = new Float32Array(await f.read(r, 'water'));
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
      samples.push({ my: sim.geoMy, land: land / s.length, ocean: ocean / s.length, relief: p99 - sea, plates: sim.alivePlates(), finite, phase: sim.wilson.state.phase });
    };
    const t0 = performance.now();
    let next = 0;
    while (sim.geoMy < SOAK_MY) {
      sim.runTicks(64);
      minPlates = Math.min(minPlates, sim.alivePlates()); maxPlates = Math.max(maxPlates, sim.alivePlates());
      if (sim.geoMy >= next) { await sample(); next += 50; }
      await frame();
    }
    return {
      samples, minPlates, maxPlates, cycles: sim.wilson.state.cycles, unhandled: sim.unhandledEvents,
      events: sim.events.log.length, seconds: (performance.now() - t0) / 1000,
    };
  }, { SOAK_MY, seed: Number(process.env.SOAK_SEED ?? 3) });

  console.log(`soak ${SOAK_MY} My in ${res.seconds.toFixed(0)} s, cycles=${res.cycles}, plates ${res.minPlates}-${res.maxPlates}, unhandled events=${res.unhandled}`);
  for (const s of res.samples.filter((_, i) => i % 10 === 0)) console.log(JSON.stringify(s));

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
});
