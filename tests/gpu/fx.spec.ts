// Earthquake / tsunami FX (src/fx): FX only (V15), so the tests drive them with debug quakes on a warmed-up
// Sim world and read their GPU state back. What matters and why:
//  - quakes run on a REAL-time schedule (geologic rates at sim speed would be a permanent tremor), a
//    notable one every ~12–40 s on an active world, none when paused or without boundary activity;
//  - a megathrust under the open ocean (or an ocean impact) makes a tsunami whose front spreads,
//    slows over shallow water (∝ √depth) and reaches the coasts, where it runs up only onto low land;
//  - a land quake shakes dust off the slopes but never makes a tsunami;
//  - the camera shake is small, decays, can be turned off and never leaves the camera displaced;
//  - a meteor strike streaks in, hits on schedule, calls back exactly once at the impact frame (the god tool
//    enqueues the sim's crater there) and throws its ejecta downrange; the sim's own meteors (event log) get a
//    short streak instead of a second sequence; an ocean impact raises a tsunami;
//  - the impact reads like the plumes, not like a grey ball hanging over the map: a brief fireball, a dust
//    column whose height follows the impact size (a small strike stays small) and that is gone a few
//    seconds later (nothing but the entry smoke tail outlives it);
//  - idle FX cost nothing (hidden, no dispatches).
import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { QuakeScheduler, QUAKE, TSUNAMI, SCORCH_S, SCORCH_WET_S, activity, type QuakeSites, type QuakePick } from '../../src/fx/fxModel';

const OUT = process.env.FX_SHOTS ?? 'test-results/fx';
test.use({ viewport: { width: 1280, height: 800 } });
test.setTimeout(240_000);

// Other layers are edited concurrently; GPU-level problems and anything from src/fx fail a test.
const GPU_PROBLEM = /webgpu|validation|wgsl|shader|pipeline|binding|uncaptured/i;
function watchProblems(page: Page): string[] {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message + ' ' + (e.stack ?? '')));
  page.on('console', (m) => {
    const t = m.type();
    if (/favicon/.test(m.location().url)) return;
    if (t === 'error' || (t === 'warning' && (GPU_PROBLEM.test(m.text()) || /src\/fx\//.test(m.location().url)))) problems.push(`${t}: ${m.text()}`);
  });
  return problems;
}

async function open(page: Page, query = '') {
  await page.goto(`/tests/gpu/support/fx.html?ticks=400&${query}`);
  await page.waitForFunction(() => (window as any).fxReady === true, null, { timeout: 180_000 });
  await page.evaluate(() => (window as any).fxp.frames(5)); // the render display columns the FX read are filled per frame
}
const fxp = (page: Page) => ({
  frames: (n: number, dt = 1 / 60) => page.evaluate(([n, dt]) => (window as any).fxp.frames(n, dt), [n, dt]),
  sites: () => page.evaluate(() => (window as any).fxp.sites()),
  trigger: (x: number, z: number, m: number) => page.evaluate(([x, z, m]) => (window as any).fxp.trigger(x, z, m), [x, z, m]),
  tsu: (x: number, z: number) => page.evaluate(([x, z]) => (window as any).fxp.tsu(x, z), [x, z]),
  dust: () => page.evaluate(() => (window as any).fxp.dust()),
  stats: () => page.evaluate(() => (window as any).fxp.stats()),
});

/** Deterministic rng for the scheduler runs (mulberry32). */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
/** Run the scheduler for `seconds` of real time at 60 fps; returns the quakes with their fire times. */
function run(sites: QuakeSites, speed: number, seconds: number, seed = 1) {
  const s = new QuakeScheduler(rng(seed));
  const out: (QuakePick & { t: number })[] = [];
  const dt = 1 / 60;
  for (let i = 0; i < seconds * 60; i++) { const q = s.step(dt, sites, speed); if (q) out.push({ ...q, t: i * dt }); }
  return out;
}
const gaps = (qs: { t: number }[]) => qs.slice(1).map((q, i) => q.t - qs[i]!.t);

test('scheduler: real-time rate bounds, scaled by boundary activity and sim speed', () => {
  const busy: QuakeSites = { subduct: { x: 40, z: 90, events: 220 }, collide: { x: 180, z: 30, events: 120 } };
  const moderate: QuakeSites = { subduct: { x: 40, z: 90, events: 12 }, collide: null };
  const faint: QuakeSites = { subduct: null, collide: { x: 180, z: 30, events: 3 } };
  const hour = 3600;
  const qBusy = run(busy, 1, hour), qMod = run(moderate, 1, hour), qSlow = run(busy, 0.002, hour), qFaint = run(faint, 1, hour);
  const gB = gaps(qBusy), gM = gaps(qMod);
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  console.log(`scheduler/h: busy ${qBusy.length} (gap ${Math.min(...gB).toFixed(1)}–${Math.max(...gB).toFixed(1)} s, mean ${mean(gB).toFixed(1)}), moderate ${qMod.length} (activity ${activity(moderate).toFixed(2)}, gap ≤ ${Math.max(...gM).toFixed(1)}), slow sim ${qSlow.length}, faint ${qFaint.length}`);
  // never closer than MIN_GAP_S, an active world gets its quake by MAX_GAP_S at the latest
  for (const g of [...gB, ...gM, ...gaps(qSlow), ...gaps(qFaint)]) expect(g).toBeGreaterThanOrEqual(QUAKE.MIN_GAP_S - 1e-9);
  for (const g of [...gB, ...gM]) expect(g).toBeLessThanOrEqual(QUAKE.MAX_GAP_S + 1 / 60 + 1e-9);
  expect(mean(gB), 'busy boundaries: quakes come soon after the minimum gap').toBeLessThan(24);
  // probability scales with activity and speed; a quiet world stays mostly quiet
  expect(qMod.length).toBeLessThan(qBusy.length);
  expect(qSlow.length).toBeLessThan(qBusy.length);
  expect(qFaint.length).toBeLessThan(qMod.length);
  // no boundary activity, or a paused geo clock: the ground is still
  expect(run({ subduct: null, collide: null }, 1, hour).length).toBe(0);
  expect(run(busy, 0, hour).length).toBe(0);
  // magnitudes: M5–M9 style, megathrusts are the big ones, collision (mountain) quakes stay below ~M8.3
  const all = [...qBusy, ...run(busy, 1, 6 * hour, 7)];
  const mags = all.map((q) => q.mag).sort((a, b) => a - b);
  const median = mags[mags.length >> 1]!;
  console.log(`magnitudes: n ${mags.length}, median ${median.toFixed(2)}, p95 ${mags[Math.floor(mags.length * 0.95)]!.toFixed(2)}, max ${mags[mags.length - 1]!.toFixed(2)}, ≥${TSUNAMI.MIN_MAG} subduct ${all.filter((q) => q.kind === 'subduct' && q.mag >= TSUNAMI.MIN_MAG).length}`);
  expect(mags[0]!).toBeGreaterThanOrEqual(5);
  expect(mags[mags.length - 1]!).toBeLessThanOrEqual(9.3);
  expect(median).toBeGreaterThan(5.8);
  expect(median).toBeLessThan(7.2);
  expect(all.filter((q) => q.kind === 'collide').every((q) => q.mag <= 8.3)).toBe(true);
  expect(all.filter((q) => q.mag >= 8.4).every((q) => q.kind === 'subduct')).toBe(true);
  expect(all.some((q) => q.kind === 'subduct' && q.mag >= TSUNAMI.MIN_MAG), 'megathrusts big enough for a tsunami do happen').toBe(true);
  // epicentres stay at the reported sites (± jitter, torus)
  const near = (a: number, b: number) => Math.min(Math.abs(a - b), 256 - Math.abs(a - b)) <= QUAKE.JITTER;
  for (const q of all) {
    const s = q.kind === 'subduct' ? busy.subduct! : busy.collide!;
    expect(near(q.x, s.x) && near(q.z, s.z)).toBe(true);
  }
});

test('ocean megathrust: tsunami ring grows, slows over shallow water and runs up onto low coasts', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'quality=low');
  const f = fxp(page);
  const { ocean } = await f.sites();
  expect(ocean.x, 'the warmed-up world has open ocean').toBeGreaterThanOrEqual(0);
  const q = await f.trigger(ocean.x, ocean.z, 8.8);
  console.log(`ocean quake at (${ocean.x}, ${ocean.z}) depth ${q.waterDepth.toFixed(1)}, ${ocean.toLand} cells off the coast`);
  expect(q.tsunami).toBe(true);
  expect(q.waterDepth).toBeGreaterThan(TSUNAMI.MIN_DEPTH);
  await f.frames(60);
  const r1 = await f.tsu(ocean.x, ocean.z);
  await f.frames(120);
  const r2 = await f.tsu(ocean.x, ocean.z);
  let r3 = r2;
  for (let i = 0; i < 12 && (r3.coastReached === 0 || r3.reachedLand === 0); i++) { await f.frames(60); r3 = await f.tsu(ocean.x, ocean.z); }
  await f.frames(110, 0.1); // let the front cross the shelves
  const r4 = await f.tsu(ocean.x, ocean.z);
  console.log(`tsunami front: ${r1.front.toFixed(1)} cells @${r1.age.toFixed(1)} s → ${r2.front.toFixed(1)} @${r2.age.toFixed(1)} s; @${r3.age.toFixed(1)} s coast columns ${r3.coastReached}, land run-up ${r3.reachedLand}; @${r4.age.toFixed(1)} s run-up ${r4.reachedLand} (max ${r4.maxLandAlt.toFixed(2)} layers above sea), slowness deep ${r4.slowDeep.toFixed(3)} (${r4.nDeep}) vs shallow ${r4.slowShallow.toFixed(3)} (${r4.nShallow}) s/cell, measured/expected ${r4.relDeep.toFixed(2)} / ${r4.relShallow.toFixed(2)}`);
  expect(r1.active && r2.active).toBe(true);
  expect(r1.front).toBeGreaterThan(4);
  expect(r2.front, 'the ring expands').toBeGreaterThan(r1.front + 10);
  // deep-water speed follows C0·√(depth/DREF): a 16-layer ocean runs ≈ 13 cells/s
  expect(r2.front / r2.age).toBeGreaterThan(0.5 * TSUNAMI.C0);
  expect(r2.front / r2.age).toBeLessThan(1.5 * TSUNAMI.C0);
  expect(r3.coastReached, 'the wave reaches a coast').toBeGreaterThan(0);
  expect(r3.reachedLand, 'and runs up onto the land').toBeGreaterThan(0);
  // run-up is limited to low coastal land (RUNUP_K × the deep-water amplitude, here < ~5 layers)
  expect(r4.maxLandAlt).toBeLessThan(TSUNAMI.RUNUP_K * 2.2 + 0.5);
  // shallow-water speed ∝ √depth: the local slowness |∇T| matches 1/(C0·√(depth/DREF)) in deep and in shallow
  // water (open 3×3 patches), so the front slows (and bunches up) over the shelves
  expect(r4.nShallow, 'the front crossed open shallow water').toBeGreaterThan(8);
  expect(r4.relDeep).toBeGreaterThan(0.85); expect(r4.relDeep).toBeLessThan(1.15);
  expect(r4.relShallow).toBeGreaterThan(0.7); expect(r4.relShallow).toBeLessThan(1.3);
  expect(r4.slowShallow).toBeGreaterThan(r4.slowDeep * 1.3); // < 5 vs > 8 layers deep
  expect(problems).toEqual([]);
});

test('land quake: dust rises from the slopes within the radius and settles, no tsunami', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'quality=low');
  const f = fxp(page);
  const { land } = await f.sites();
  const q = await f.trigger(land.x, land.z, 7.5);
  expect(q.tsunami).toBe(false);
  expect(await page.evaluate(() => (window as any).fxp.tsunamiActive())).toBe(false);
  await f.frames(90);
  const d = await f.dust();
  const [ex, ez] = [q.position[0], q.position[2]];
  const wrapD = (a: number, b: number) => { const t = Math.abs(a - b) % 4; return Math.min(t, 4 - t); };
  const far = d.pts.filter((p: number[]) => Math.hypot(wrapD(p[0]!, ex), wrapD(p[2]!, ez)) > (26 + 24 * 2.5) * 4 / 256 + 0.05);
  console.log(`land quake M7.5 at (${land.x}, ${land.z}) alt ${land.alt.toFixed(1)}: dust alive ${d.alive}, lifted ${d.lifted}, outside the radius ${far.length}`);
  expect(d.lifted, 'dust is shaken loose').toBeGreaterThan(100);
  expect(far.length, 'only within the quake radius').toBe(0);
  expect((await f.stats()).tsunamis).toBe(0);
  // it settles: all puffs gone within ~12 s
  await f.frames(60, 0.2);
  expect((await f.dust()).alive).toBe(0);
  expect(problems).toEqual([]);
});

test('ocean meteor impacts (sim event log) spawn a tsunami; the shake decays and never displaces the camera', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'quality=low');
  const f = fxp(page);
  const { ocean } = await f.sites();
  const cam0: number[] = await page.evaluate(() => (window as any).fxp.camPos());
  await page.evaluate(([x, z]) => (window as any).fxp.godMeteor(x, z, 1.2), [ocean.x, ocean.z]);
  await f.frames(3);
  await f.frames(30); // the short streak of a meteor the sim already applied lands within ~0.3 s
  const quakes = await page.evaluate(() => (window as any).fxp.quakes);
  expect(quakes.length).toBe(1);
  expect(quakes[0].kind).toBe('impact');
  expect(quakes[0].tsunami).toBe(true);
  // shake: visible but small (≤ ~1 % of the view distance), then gone without drift
  let maxOff = 0;
  const dist = Math.hypot(cam0[0]! - cam0[3]!, cam0[1]! - cam0[4]!, cam0[2]! - cam0[5]!);
  for (let i = 0; i < 20; i++) {
    await f.frames(3);
    const c: number[] = await page.evaluate(() => (window as any).fxp.camPos());
    maxOff = Math.max(maxOff, Math.hypot(c[0]! - cam0[0]!, c[1]! - cam0[1]!, c[2]! - cam0[2]!));
    // the orbit target moves with the camera: a translation, the view direction never swings
    expect(Math.abs(c[0]! - c[3]! - (cam0[0]! - cam0[3]!))).toBeLessThan(1e-6);
  }
  await f.frames(60, 0.2);
  const c1: number[] = await page.evaluate(() => (window as any).fxp.camPos());
  const drift = Math.hypot(c1[0]! - cam0[0]!, c1[1]! - cam0[1]!, c1[2]! - cam0[2]!);
  console.log(`shake: max offset ${(maxOff / dist * 100).toFixed(2)} % of view distance, final drift ${drift.toExponential(1)}`);
  expect(maxOff).toBeGreaterThan(1e-4 * dist);
  expect(maxOff).toBeLessThan(0.013 * dist);
  expect(drift).toBeLessThan(1e-5);
  // shake off: a new quake leaves the camera alone
  await page.evaluate(() => (window as any).fxp.setShake(false));
  await f.trigger(ocean.x, ocean.z, 9);
  await f.frames(20);
  const c2: number[] = await page.evaluate(() => (window as any).fxp.camPos());
  expect(Math.hypot(c2[0]! - cam0[0]!, c2[1]! - cam0[1]!, c2[2]! - cam0[2]!)).toBeLessThan(1e-5);
  expect(problems).toEqual([]);
});

test('meteor strike: streak, impact on schedule with one sim event, ejecta downrange, ocean impact → tsunami', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'quality=low');
  const f = fxp(page);
  const { ocean, land } = await f.sites();
  const events = () => page.evaluate(() => (window as any).fxp.events());
  const parts = (px = 0, pz = 0, py = 0) => page.evaluate(([x, z, y]) => (window as any).fxp.meteorParticles(x, z, y), [px, pz, py]);
  const e0 = await events();
  // god-tool flow: travel azimuth 0 = +x; the sim event is enqueued in onImpact
  await page.evaluate(([x, z]) => (window as any).fxp.strikeAsync(x, z, 0, 1.0), [land.x, land.z]);
  await f.frames(3);
  await f.frames(60); // ~1.05 s: still in the air
  const mid = await parts();
  expect(await events(), 'no crater before the flash').toBe(e0);
  expect(mid.byKind[0], 'smoke tail laid along the path').toBeGreaterThan(50);
  expect(mid.byKind[1]).toBe(0);
  expect(mid.light).toBeGreaterThan(1); // the head lights the ground on its way in
  await f.frames(33); // past the 1.5 s impact
  const q = await page.evaluate(() => (window as any).fxp.strikeResult());
  expect(q.kind).toBe('impact');
  expect(q.tsunami).toBe(false);
  expect(await events(), 'exactly one sim meteor event, at the impact').toBe(e0 + 1);
  await f.frames(30);
  const after = await parts(q.position[0], q.position[2]);
  console.log(`meteor on land: particles by kind ${JSON.stringify(after.byKind)}, curtain centroid offset (${after.curtain[0].toFixed(3)}, ${after.curtain[1].toFixed(3)}) world, travel +x`);
  expect(after.byKind[1], 'ejecta curtain').toBeGreaterThan(150);
  expect(after.byKind[2], 'incandescent ejecta').toBeGreaterThan(40);
  expect(after.byKind[3], 'rising column').toBeGreaterThan(20);
  expect(after.byKind[5], 'fireball').toBeGreaterThan(10);
  // oblique impact: the curtain is thrown downrange (+x), not uprange
  expect(after.curtain[0]).toBeGreaterThan(0.02);
  expect(after.curtain[0]).toBeGreaterThan(3 * Math.abs(after.curtain[1]));
  // the log entry of our own strike does not replay a second sequence
  await f.frames(20);
  expect((await page.evaluate(() => (window as any).fxp.quakes)).length).toBe(1);
  // ~2.9 s after the impact: the fireball was brief, the column stands at a plume-like height (world units;
  // the volcanic ash columns top out at ~0.2–0.5)
  await f.frames(100);
  const col = await parts(q.position[0], q.position[2], q.position[1]);
  expect(col.byKind[5], 'the fireball is over within ~1.5 s').toBe(0);
  expect(col.colTop, 'the column rises').toBeGreaterThan(0.15);
  expect(col.colTop, 'but stays a plume, not a sky-high cloud').toBeLessThan(0.6);
  // ~7 s: curtain, column, fireball and secondary puffs are all gone (no lingering dust cloud)
  await f.frames(250);
  const late = await parts();
  expect([late.byKind[1], late.byKind[3], late.byKind[4], late.byKind[5]], 'impact dust fades within a few seconds').toEqual([0, 0, 0, 0]);
  // … while the charred crater and ejecta blanket stay on the ground for most of a minute (~30 s now)
  await f.frames(Math.round(23 / 0.25), 0.25);
  const ground = (await page.evaluate(() => (window as any).fxp.visible())).find(([n]: [string]) => n === 'quakeGround');
  expect(ground?.[1], 'the scorch still shows ~30 s after the impact').toBe(true);
  // a meteor the sim applied on its own: a short streak that lands right away
  await page.evaluate(([x, z]) => (window as any).fxp.godMeteor(x, z, 1.1), [ocean.x, ocean.z]);
  await f.frames(3);
  await f.frames(30);
  const qs = await page.evaluate(() => (window as any).fxp.quakes);
  expect(qs.length).toBe(2);
  expect(qs[1].kind).toBe('impact');
  expect(qs[1].tsunami, 'ocean impact raises a tsunami').toBe(true);
  expect(await page.evaluate(() => (window as any).fxp.tsunamiActive())).toBe(true);
  // the whole show winds down: nothing drawn, nothing dispatched
  await f.frames(Math.ceil((SCORCH_S + 4) / 0.25), 0.25); // the impact scorch is the last to go
  const vis = await page.evaluate(() => (window as any).fxp.visible());
  expect(vis.every(([, v]: [string, boolean]) => !v), JSON.stringify(vis)).toBe(true);
  const mv = await page.evaluate(() => (window as any).fxp.meteorVisible());
  expect(mv.every(([, v]: [string, boolean]) => !v), JSON.stringify(mv)).toBe(true);
  expect(await page.evaluate(() => (window as any).fxp.updateCalls())).toBe(0);
  expect(problems).toEqual([]);
});

test('meteor impact size follows the strike: a small meteor raises a small column, a big one a plume-sized one', async ({ page }) => {
  const problems = watchProblems(page);
  // same site on a fresh world each time (a second strike would land in the first one's crater)
  const column = async (mag: number, radius: number) => {
    await open(page, 'quality=low');
    const { land } = await fxp(page).sites();
    await page.evaluate(([x, z, m, r]) => (window as any).fxp.strikeAsync(x, z, 0, m, r), [land.x, land.z, mag, radius]);
    await fxp(page).frames(3);
    await fxp(page).frames(93); // past the 1.5 s entry
    const q = await page.evaluate(() => (window as any).fxp.strikeResult());
    await fxp(page).frames(140); // ~2.4 s after the impact: the column stands, the cap has formed
    return page.evaluate(([x, z, y]) => (window as any).fxp.meteorParticles(x, z, y), [q.position[0], q.position[2], q.position[1]]);
  };
  const small = await column(0.25, 4), big = await column(1, 12);
  console.log(`column top above the impact: M0.25 r4 ${small.colTop.toFixed(3)} (${small.byKind[3]} puffs), M1 r12 ${big.colTop.toFixed(3)} (${big.byKind[3]} puffs) world`);
  expect(small.byKind[3], 'a small strike still raises a little column').toBeGreaterThan(5);
  expect(small.colTop, 'small impacts stay small').toBeLessThan(0.6 * big.colTop);
  expect(small.byKind[3] + small.byKind[1], 'and throw less dust').toBeLessThan(0.6 * (big.byKind[3] + big.byKind[1]));
  expect(problems).toEqual([]);
});

test('an ocean impact marks the seabed briefly (faint glow, silt), then clears; no long scorch', async ({ page }) => {
  // user: 'shouldn't we get some sort of discoloration underwater too and a subtle glow maybe? just shorter'
  const problems = watchProblems(page);
  await open(page, 'quality=low');
  const f = fxp(page);
  const { ocean } = await f.sites();
  await page.evaluate(([x, z]) => (window as any).fxp.strikeAsync(x, z, 0.7, 1.1), [ocean.x, ocean.z]);
  await f.frames(3);
  await f.frames(93); // past the entry
  const q = await page.evaluate(() => (window as any).fxp.strikeResult());
  expect(q.waterDepth, 'an open-ocean strike').toBeGreaterThan(1);
  const ground = async () => (await page.evaluate(() => (window as any).fxp.visible())).find(([n]: [string]) => n === 'quakeGround')?.[1];
  await f.frames(Math.round(9 / 0.25), 0.25); // ~9 s: the shock ring is long gone, the seabed mark remains
  expect(await ground(), 'the seabed mark shows ~9 s after the impact').toBe(true);
  await f.frames(Math.round((SCORCH_WET_S - 9 + 3) / 0.25), 0.25);
  expect(await ground(), 'and clears by SCORCH_WET_S (much sooner than a land scar)').toBe(false);
  expect(problems).toEqual([]);
});

test('meteor strike with the FX disabled still calls back at once (the god tool always gets its crater)', async ({ page }) => {
  await open(page, 'quality=low');
  const { land } = await fxp(page).sites();
  const e0 = await page.evaluate(() => (window as any).fxp.events());
  await page.evaluate(() => (window as any).fxp.setEnabled(false));
  await page.evaluate(([x, z]) => (window as any).fxp.strikeAsync(x, z, 1, 1), [land.x, land.z]);
  await page.evaluate(() => (window as any).__strike);
  expect(await page.evaluate(() => (window as any).fxp.events())).toBe(e0 + 1);
});

test('perf: active FX cost and idle FX hidden with no dispatches; hero-view meteor + tsunami frame sequences', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page);
  const f = fxp(page);
  const { ocean, land } = await f.sites();
  mkdirSync(OUT, { recursive: true });
  await page.evaluate(() => (window as any).fxp.setShake(false)); // sharp stills
  await f.frames(10);
  await page.screenshot({ path: `${OUT}/fx_hero_before.png` });
  // meteor on land: entry streak → flash → ejecta → column → aftermath
  await page.evaluate(([x, z]) => (window as any).fxp.strikeAsync(x, z, 2.4, 1.2), [land.x, land.z]);
  for (const [i, n] of [45, 30, 18, 12, 12, 20, 30, 60, 120].entries()) { await f.frames(n); await page.screenshot({ path: `${OUT}/fx_meteor_${i}.png` }); }
  // ocean impact → tsunami
  await page.evaluate(([x, z]) => (window as any).fxp.godMeteor(x, z, 1.3), [ocean.x, ocean.z]);
  for (let i = 0; i < 8; i++) { await f.frames(60); await page.screenshot({ path: `${OUT}/fx_tsunami_${i}.png` }); }
  // GPU cost while active: an ocean meteor impact 0.5 s in (flash, fireball, ejecta, column, ground ring,
  // tsunami). Wall-clock A/B, each layer drawn 8× to stand above the frame-time noise; sweeps back to back.
  await page.evaluate(([x, z]) => (window as any).fxp.strikeAsync(x, z, 2.2, 1.2), [ocean.x, ocean.z]);
  await f.frames(120);
  const amp: Record<string, { perCopy: number }> = {};
  for (const name of ['tsunami', 'meteorDust', 'meteorEmbers', 'quakeGround', 'meteorTrail']) amp[name] = await page.evaluate((n) => (window as any).fxp.benchAmp(n, 8), name);
  const sweeps: number = await page.evaluate(() => (window as any).fxp.benchSweeps(200));
  const total = sweeps + Object.values(amp).reduce((s, a) => s + a.perCopy, 0);
  const gpu: string = await page.evaluate(() => (window as any).fxp.gpu);
  console.log(`fx active @1280×800 on ${gpu}: ${Object.entries(amp).map(([k, v]) => `${k} ${v.perCopy.toFixed(3)}`).join(', ')}, tsunami sweeps ${sweeps.toFixed(3)} → ≈ ${total.toFixed(2)} ms`);
  expect(total, 'active budget (≤ 0.7 ms target; loose bound for a shared, noisy GPU)').toBeLessThan(1.5);
  // idle: once everything has played out, nothing is drawn or dispatched
  await f.frames(Math.ceil((SCORCH_S + 6) / 0.25), 0.25); // (the impact scorch stays longest)
  const vis = await page.evaluate(() => (window as any).fxp.visible());
  expect(vis.every(([, v]: [string, boolean]) => !v), `hidden when idle: ${JSON.stringify(vis)}`).toBe(true);
  expect((await f.stats()).active).toBe(false);
  expect(await page.evaluate(() => (window as any).fxp.updateCalls()), 'no compute dispatches when idle').toBe(0);
  expect(problems).toEqual([]);
});
