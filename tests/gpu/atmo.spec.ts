// §T.41/§T.42 atmosphere: clouds follow the sim's vapor/precip and shade the ground under them,
// plumes rise from hot lava, hydrothermal vents sit on young crust, rain falls only from raining
// columns (snow where it is cold), impacts and flood basalts get their FX. The atmosphere only reads
// sim fields (V15), so most tests hand-paint vapor/precip/surfTemp/lava/crustAge on a real worldgen
// terrain; screenshots use a warmed-up Sim and the full look pipeline.
import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const OUT = process.env.ATMO_SHOTS ?? 'test-results/atmo';
test.use({ viewport: { width: 1280, height: 800 } });
test.setTimeout(240_000);

// Other layers are edited concurrently; only GPU-level problems and errors from this module fail a test.
const GPU_PROBLEM = /webgpu|validation|wgsl|shader|pipeline|binding|uncaptured/i;
const MINE = /atmo|cloud|plume|rain|curtain|lightning/i;

function watchProblems(page: Page): string[] {
  const problems: string[] = [];
  page.on('pageerror', (e) => { if (MINE.test(e.message + (e.stack ?? ''))) problems.push('pageerror: ' + e.message); });
  page.on('console', (m) => {
    const t = m.type();
    if (/favicon/.test(m.location().url)) return;
    if ((t === 'error' || t === 'warning') && (GPU_PROBLEM.test(m.text()) || MINE.test(m.text()))) problems.push(`${t}: ${m.text()}`);
  });
  return problems;
}

async function open(page: Page, query: string) {
  await page.goto(`/tests/gpu/support/atmo.html?${query}`);
  await page.waitForFunction(() => (window as any).atReady === true, null, { timeout: 120_000 });
}

// Painted layout (tests/gpu/support/atmoWorld.ts)
const WET = { x0: 24, x1: 104, z0: 40, z1: 120 };
const STORM = { x0: 56, x1: 88, z0: 64, z1: 96 };
const SNOWY = { x0: 150, x1: 200, z0: 150, z1: 200 };
const cellW = (c: number) => (c + 0.5) * (4 / 256) - 2; // column → world
const colOf = (w: number) => Math.floor((w + 2) / (4 / 256));
const inBox = (b: typeof WET, x: number, z: number, m = 0) => x >= b.x0 - m && x <= b.x1 + m && z >= b.z0 - m && z <= b.z1 + m;
const PRECIP_RAIN = 8e-4;

interface Cell { x: number; z: number; cov: number; storm: number; base: number; ground: number }

test('clouds form over saturated air, not over dry air, and shade the ground under them', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'world=paint&look=1');
  await page.evaluate(() => (window as any).at.paint({}));
  await page.evaluate(() => (window as any).at.setTime(0.5)); // noon: short, testable shadows
  await page.evaluate(() => (window as any).at.frames(60, 1 / 30));
  const cells: Cell[] = await page.evaluate(() => (window as any).at.clouds());
  // Inside the saturated box (away from its blurred rim) the coverage noise leaves clumpy cells with
  // gaps; outside, 30+ columns away from any moist air, there is nothing (V15: clouds follow the sim).
  const wet = cells.filter((c) => inBox(WET, colOf(c.x), colOf(c.z), -14));
  const dry = cells.filter((c) => !inBox(WET, colOf(c.x), colOf(c.z), 30) && !inBox(SNOWY, colOf(c.x), colOf(c.z), 30));
  const frac = (a: Cell[]) => a.filter((c) => c.cov > 0.05).length / Math.max(1, a.length);
  console.log(`clouds: wet cells ${(100 * frac(wet)).toFixed(0)} % covered (${wet.length}), dry ${(100 * frac(dry)).toFixed(1)} % (${dry.length}); whole sky ${(100 * frac(cells)).toFixed(0)} %`);
  expect(wet.length).toBeGreaterThan(20);
  expect(frac(wet), 'saturated air grows clouds').toBeGreaterThan(0.3);
  const solid = wet.filter((c) => c.cov > 0.5).length / wet.length;
  console.log(`solid (cov > 0.5) share of the saturated box: ${(100 * solid).toFixed(0)} %`);
  expect(solid, 'clumpy with gaps, never a solid blanket, even in saturated air').toBeLessThan(0.8);
  expect(Math.max(...dry.map((c) => c.cov)), 'dry air stays clear').toBeLessThan(0.01);
  expect(frac(cells), 'moderate sky coverage').toBeLessThan(0.35);
  // storm cells only where precip is heavy
  const stormy = cells.filter((c) => c.storm > 0.3);
  expect(stormy.length, 'heavy precip builds storm cells').toBeGreaterThan(0);
  for (const c of stormy) expect(inBox(STORM, colOf(c.x), colOf(c.z), 16) || inBox(SNOWY, colOf(c.x), colOf(c.z), 16)).toBe(true);

  // cloudShadowAt (the exported TSL the terrain multiplies its sun term with): dark under the thickest
  // cells along the key light, exactly 1 under clear sky.
  const sp: { key: number[]; planeY: number; on: number } = await page.evaluate(() => (window as any).at.shadowParams());
  expect(sp.on).toBe(1);
  const [kx, ky, kz] = sp.key as [number, number, number];
  const thick = [...wet].sort((a, b) => b.cov - a.cov).slice(0, 6);
  const gy = thick[0]!.ground;
  const t = (sp.planeY - gy) / ky;
  const under = thick.map((c) => [c.x - kx * t, gy, c.z - kz * t]);
  const clear = dry.slice(0, 8).map((c) => [c.x - kx * t, gy, c.z - kz * t]).filter(([x, , z]) => Math.abs(x!) < 1.9 && Math.abs(z!) < 1.9);
  const sUnder: number[] = await page.evaluate((p) => (window as any).at.shadowAt(p), under);
  const sClear: number[] = await page.evaluate((p) => (window as any).at.shadowAt(p), clear);
  console.log('cloud shadow under thick cells', sUnder.map((v) => v.toFixed(2)).join(' '), '| clear', sClear.map((v) => v.toFixed(2)).join(' '));
  expect(Math.min(...sUnder), 'direct sun is dimmed under a thick cloud').toBeLessThan(0.8);
  for (const v of sClear) expect(v).toBeGreaterThan(0.99);

  mkdirSync(OUT, { recursive: true });
  await page.evaluate(() => (window as any).at.hero());
  await page.evaluate(() => (window as any).at.frames(20, 1 / 30));
  await page.screenshot({ path: `${OUT}/painted-noon.png` });
  expect(problems, problems.join('\n')).toEqual([]);
});

test('an erupting vent (volcano.x) spawns an ash column that rises, plus fountain spray and bombs', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'world=paint&look=0');
  const vent = { x: 200, z: 60 };
  await page.evaluate((v) => (window as any).at.paint({ lava: v }), vent);
  await page.evaluate(() => (window as any).at.frames(30, 1 / 30));
  const a = await page.evaluate(() => (window as any).at.particles());
  await page.evaluate(() => (window as any).at.frames(60, 1 / 30));
  const b = await page.evaluate(() => (window as any).at.particles());
  const vx = cellW(vent.x), vz = cellW(vent.z);
  // one vent cell, at the painted pool
  // the 5-column pool straddles at most 2×2 of the 4×4-column scan cells: ≤ 4 vents, all at the pool
  expect(b.ventCount, 'the hot pool registers as a vent').toBeGreaterThanOrEqual(1);
  expect(b.ventCount).toBeLessThanOrEqual(4);
  for (let i = 0; i < b.ventCount; i++) expect(Math.hypot(b.vents[i * 4] - vx, b.vents[i * 4 + 2] - vz), 'vent at the lava column').toBeLessThan(0.06);
  const plume = (s: typeof b) => s.live.filter((p: { kind: number }) => p.kind < 1.5);
  const ejecta = b.live.filter((p: { kind: number }) => Math.abs(p.kind - 7) < 0.5 || Math.abs(p.kind - 8) < 0.5);
  console.log(`ejecta (fountain + bombs): ${ejecta.length}`);
  expect(ejecta.length, 'an active vent throws fountain spray and bombs').toBeGreaterThan(5);
  const pa = plume(a), pb = plume(b);
  console.log(`plume: ${pa.length} → ${pb.length} particles; mean height above vent ${(avg(pa.map((p: any) => p.y)) - b.vents[1]).toFixed(3)} → ${(avg(pb.map((p: any) => p.y)) - b.vents[1]).toFixed(3)}`);
  expect(pb.length, 'plume particles spawned').toBeGreaterThan(40);
  // near the vent horizontally (wind bends it downwind, but it starts at the vent)
  const young = pb.filter((p: { age: number }) => p.age < 0.5);
  for (const p of young) expect(Math.hypot(p.x - vx, p.z - vz)).toBeLessThan(0.1);
  // and it rises: the plume's mean height grows as it develops, fresh particles move up
  expect(avg(pb.map((p: any) => p.y))).toBeGreaterThan(avg(pa.map((p: any) => p.y)) + 0.02);
  expect(avg(young.map((p: any) => p.vy))).toBeGreaterThan(0.05);
  // no plume particle anywhere else on the block
  for (const p of pb) expect(Math.abs(p.z - vz), `particle at z ${p.z.toFixed(2)}`).toBeLessThan(1.2);
  expect(problems, problems.join('\n')).toEqual([]);
});

test('hydrothermal vents sit on young crust: bubbles and smokers under water stay below the surface', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'world=paint&look=0');
  const rift = { z: 200, x0: 0, x1: 255 };
  await page.evaluate((r) => (window as any).at.paint({ rift: r }), rift);
  await page.evaluate(() => (window as any).at.frames(120, 1 / 30));
  const s = await page.evaluate(() => (window as any).at.particles());
  const isUnder = (p: any) => p.kind > 3.5 && p.kind < 6.5; // bubble, smoker, turbid
  console.log(`hydrothermal: ${s.hydroCount} vents, ${s.live.filter(isUnder).length} underwater particles`);
  expect(s.hydroCount, 'young crust hosts vents').toBeGreaterThan(0);
  // vents only on the rift rows (4×4-column scan cells), a sparse subset (deterministic hash)
  const n = Math.min(s.hydroCount, 64);
  for (let i = 0; i < n; i++) expect(Math.abs(colOf(s.hydro[i * 4 + 2]) - rift.z), 'vent on the rift').toBeLessThanOrEqual(2);
  // the 3-row rift touches 2 × 64 scan cells; a deterministic ~30 % subset hosts vents
  expect(s.hydroCount, 'sparse').toBeLessThan(0.5 * 128);
  const under = s.live.filter(isUnder);
  for (const p of under) expect(p.y, 'dissipates before the surface').toBeLessThanOrEqual(p.ceil + 1e-4);
  for (const p of under) expect(Math.abs(colOf(p.z) - rift.z)).toBeLessThanOrEqual(4);
  expect(problems, problems.join('\n')).toEqual([]);
});

test('rain falls only from precipitating columns; snow where it is freezing', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'world=paint&look=0');
  await page.evaluate(() => (window as any).at.paint({}));
  await page.evaluate(() => (window as any).at.frames(90, 1 / 30));
  const drops: { x: number; y: number; z: number; snow: boolean; col: number }[] = await page.evaluate(() => (window as any).at.rain());
  const rain = drops.filter((d) => !d.snow), snow = drops.filter((d) => d.snow);
  console.log(`rain ${rain.length}, snow ${snow.length}`);
  expect(rain.length).toBeGreaterThan(20);
  expect(snow.length).toBeGreaterThan(5);
  // painted precip: STORM 5e-3 (rain), SNOWY 2e-3 at −8 °C (snow), WET 2e-4 < PRECIP_RAIN, elsewhere 0
  expect(2e-4).toBeLessThan(PRECIP_RAIN);
  for (const d of drops) {
    const cx = d.col % 256, cz = Math.floor(d.col / 256);
    if (d.snow) expect(inBox(SNOWY, cx, cz), `snow from column ${cx},${cz}`).toBe(true);
    else expect(inBox(STORM, cx, cz), `rain from column ${cx},${cz}`).toBe(true);
  }
  expect(problems, problems.join('\n')).toEqual([]);
});

test('storm lightning, meteor haze (strike itself is src/fx), flood-basalt haze', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'world=paint&look=0');
  await page.evaluate(() => (window as any).at.paint({}));
  await page.evaluate(() => (window as any).at.frames(100, 1 / 10)); // 10 s of storm
  const st = await page.evaluate(() => (window as any).at.stats());
  console.log('lightning strikes in 10 s:', st.lightning);
  expect(st.lightning, 'storm cells flash').toBeGreaterThan(0);
  // meteor: the cinematic strike (flash, curtain, dust column) is src/fx; atmosphere adds only a regional
  // haze, so the two never double up (user: 'random grey poof out of nowhere')
  await page.evaluate(() => (window as any).at.trigger({ kind: 'meteor', x: 190, z: 20, magnitude: 1 }));
  await page.evaluate(() => (window as any).at.frames(20, 1 / 30));
  const m = await page.evaluate(() => (window as any).at.particles());
  expect(m.live.filter((p: any) => Math.abs(p.kind - 2) < 0.5).length, 'no atmosphere dust poof').toBe(0);
  expect(m.live.filter((p: any) => Math.abs(p.kind - 3) < 0.5).length, 'regional haze').toBeGreaterThan(5);
  expect((await page.evaluate(() => (window as any).at.stats())).flashes).toBe(0);
  await page.evaluate(() => (window as any).at.trigger({ kind: 'floodBasalt', x: 120, z: 200, magnitude: 1 }));
  await page.evaluate(() => (window as any).at.frames(60, 1 / 30));
  const f = await page.evaluate(() => (window as any).at.particles());
  const haze = f.live.filter((p: any) => Math.abs(p.kind - 3) < 0.5);
  expect(haze.length, 'wide low haze').toBeGreaterThan(50);
  const hx = cellW(120), hz = cellW(200);
  expect(Math.max(...haze.map((p: any) => Math.hypot(p.x - hx, p.z - hz))), 'haze spreads wide').toBeGreaterThan(0.25);
  expect(problems, problems.join('\n')).toEqual([]);
});

test('perf (1440p) and screenshots on a warmed-up sim world', async ({ page }) => {
  const problems = watchProblems(page);
  await page.setViewportSize({ width: 2560, height: 1440 });
  await open(page, 'world=sim&ticks=400&look=1');
  await page.evaluate(() => (window as any).at.frames(60, 1 / 30));
  const perf: Record<string, unknown> = {};
  for (const hq of [true, false]) {
    await page.evaluate((v) => (window as any).at.setQuality(v), hq);
    await page.evaluate(() => (window as any).at.frames(20, 1 / 30));
    perf[hq ? 'high' : 'low'] = await page.evaluate(() => (window as any).at.perfParts(40));
  }
  // GPU timestamps: atmosphere compute passes, and the cloud march render pass (the full-res composite
  // is one texture fetch pair per covered pixel). Logged for the V9 budget; not a hard gate here.
  console.log('atmo perf', JSON.stringify(perf));
  await page.evaluate(() => (window as any).at.setQuality(true));
  await page.setViewportSize({ width: 1280, height: 800 });
  mkdirSync(OUT, { recursive: true });
  const shots: [string, number[] | null, number[] | null, number][] = [
    ['hero', null, null, 0.705],
    ['close', [2.6, 1.6, 2.9], [0, 0.1, 0], 0.705],
    ['noon', null, null, 0.5],
    ['top', [0.3, 7.5, 1.2], [0, 0, 0], 0.5],
  ];
  for (const [name, pos, target, tod] of shots) {
    await page.evaluate((v) => (window as any).at.setTime(v), tod);
    if (pos) await page.evaluate(([p, t]) => (window as any).at.setCam(p, t), [pos, target]);
    else await page.evaluate(() => (window as any).at.hero());
    await page.evaluate(() => (window as any).at.frames(40, 1 / 30));
    await page.screenshot({ path: `${OUT}/sim-${name}.png` });
  }
  const p = await page.evaluate(() => (window as any).at.particles());
  if (p.ventCount > 0) {
    const [x, y, z] = p.vents;
    await page.evaluate(() => (window as any).at.setTime(0.705));
    await page.evaluate(([vx, vy, vz]) => (window as any).at.setCam([vx + 1.3, vy + 0.55, vz + 1.5], [vx, vy + 0.3, vz]), [x, y, z]);
    await page.evaluate(() => (window as any).at.frames(60, 1 / 30));
    await page.screenshot({ path: `${OUT}/sim-plume.png` });
  }
  expect(problems, problems.join('\n')).toEqual([]);
});

function avg(a: number[]): number { return a.reduce((s, v) => s + v, 0) / Math.max(1, a.length); }
