// §T.45 tiny life: flora follows biome/veg and never sits underwater or on cliffs, grows/shrinks
// instead of popping; birds stay in the block above terrain; critters never enter water; fish stay
// in shallow water under the surface. Life only reads sim fields (V15), so the tests hand-paint
// biome/veg on a real worldgen + derive terrain.
import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const OUT = process.env.LIFE_SHOTS ?? 'test-results/life';
test.use({ viewport: { width: 1280, height: 800 } });
test.setTimeout(240_000);

const GPU_PROBLEM = /webgpu|validation|wgsl|shader|pipeline|binding|uncaptured/i;
// Biome ids (src/sim/biomeModel.ts)
const B = { OCEAN: 0, ICE: 1, TUNDRA: 2, TAIGA: 3, TEMPERATE_FOREST: 4, GRASSLAND: 5, DESERT: 6, SAVANNA: 7, RAINFOREST: 8, BEACH: 9, ALPINE: 10, STEPPE: 11, SHRUBLAND: 12, COLD_DESERT: 13 };
// Species ids (src/life/lifeModel.ts Sp)
const SP = { PINE: 3, SHRUB_TUNDRA: 10, GRASS_ALPINE: 12, SHRUB_SAGE: 13, CYPRESS: 14, GRASS_STEPPE: 15, SHRUB_STEPPE: 16, HARDY_TREE: 17, CUSHION: 18,
  F_GRASS: 19, F_GRASS_TALL: 20, F_SEED: 21, F_CLOVER: 22, F_FLOWER: 23, F_STRAW: 24, FEATHER: 25, REED: 26, TUSSOCK: 27, SAGEBRUSH: 28, BUSH_GREEN: 29, BOULDER: 30, STONE: 31, HEDGE: 32 };
const MISMATCH_OK = 16;
const LARGE = ['tree', 'pine', 'cactus', 'acacia', 'cypress', 'hedge'];

/**
 * Collect page errors and WebGPU/shader warnings. `foreignOk`: errors that name another agent's
 * render file (full-look screenshot run) are logged, not failed on; the data tests run bare.
 */
function watchProblems(page: Page, foreignOk = false): string[] {
  const problems: string[] = [];
  const add = (msg: string) => {
    if (foreignOk && /src\/(render|atmo)\/|\b(sides|terrain|water|sky|post|backdrop|clouds|plumes|rain)\.ts/.test(msg) && !/src\/life\//.test(msg)) {
      console.log('[not life] ' + msg.slice(0, 200));
      return;
    }
    problems.push(msg);
  };
  page.on('pageerror', (e) => add('pageerror: ' + e.message + ' ' + (e.stack ?? '')));
  page.on('console', (m) => {
    const t = m.type();
    if (/favicon/.test(m.location().url)) return;
    if (t === 'error' || (t === 'warning' && GPU_PROBLEM.test(m.text()))) add(`${t}: ${m.text()} @${m.location().url}`);
  });
  return problems;
}

async function open(page: Page, query: string) {
  await page.goto(`/tests/gpu/support/life.html?${query}`);
  await page.waitForFunction(() => (window as any).ltReady === true, null, { timeout: 90_000 });
}

type Kinds = Record<string, { live: number; drawn: number; listOk: boolean; byBiome: Record<number, number> }>;
interface Zone { cols: number; trees: number; scaleSum: number; pines: number; tempTrees: number; line: number }
interface FloraSummary { live: number; growing: number; shrinking: number; kinds: Kinds; perSpecies: Record<number, number>; biomeCols: Record<number, number>; zone: Zone[];
  bad: { wrongBiome: number; wet: number; steep: number; cpuMismatch: number; yMismatch: number; aboveBare: number; examples: string[] } }

test('flora tracks biome, avoids water and cliffs, matches CPU reference, morphs smoothly', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'paint=stripes&look=0&bare=1');
  await page.evaluate(() => (window as any).lt.frames(3));
  await page.evaluate(() => (window as any).lt.refresh(true));
  const a: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));

  // GPU kernel == CPU reference (lifeModel.floraDecide) for every slot: placement rules are exactly the documented ones.
  // The patch noise is f32 on the GPU and f64 on the CPU, so a slot sitting exactly on a threshold may
  // flip: allow ≤ 16 of the 368,640 slots.
  expect(a.bad.cpuMismatch, a.bad.examples.join('\n')).toBeLessThanOrEqual(MISMATCH_OK);
  expect(a.bad.yMismatch, 'plant base height = terrain as drawn, sunk on slopes\n' + a.bad.examples.join('\n')).toBe(0);
  // Nothing underwater or on slopes above the threshold.
  expect(a.bad.wet).toBe(0);
  expect(a.bad.steep).toBe(0);
  expect(a.bad.wrongBiome, a.bad.examples.join('\n')).toBe(0);
  expect(a.bad.aboveBare, 'bare rock and snow above the plant limit').toBe(0);
  // Indirect draw args and compacted lists hold exactly the live slots (what is drawn is what the state says).
  for (const [name, k] of Object.entries(a.kinds)) {
    expect(k.listOk, `${name} list/args consistent (${k.drawn} drawn, ${k.live} live)`).toBe(true);
  }
  const k = a.kinds;
  // Plants per column of their biome (bands differ in land area, so compare densities).
  const dens = (kind: string, biome: number) => (k[kind]!.byBiome[biome] ?? 0) / Math.max(1, a.biomeCols[biome] ?? 0);
  for (const b of [B.TEMPERATE_FOREST, B.RAINFOREST, B.TAIGA, B.GRASSLAND, B.SAVANNA, B.DESERT]) {
    expect(a.biomeCols[b] ?? 0, `stripes world has enough land of biome ${b}`).toBeGreaterThan(400);
  }
  // Trees where forest is (large slots: ~0.39 per column × veg × rule density, minus slope/wet/treeline culls
  // on this mountainous seed).
  expect(dens('tree', B.TEMPERATE_FOREST)).toBeGreaterThan(0.04);
  expect(dens('tree', B.RAINFOREST)).toBeGreaterThan(0.035);
  expect(dens('pine', B.TAIGA)).toBeGreaterThan(0.06);
  expect(dens('acacia', B.SAVANNA)).toBeGreaterThan(0.004);
  expect(dens('cactus', B.DESERT)).toBeGreaterThan(0.015);
  // Sparse lone trees on grassland, far below forest density.
  expect(dens('tree', B.GRASSLAND)).toBeLessThan(dens('tree', B.TEMPERATE_FOREST) * 0.2);
  // Cacti only in desert; no plant of any kind in ocean, ice or beach; alpine is meadow (no trees).
  expect(Object.keys(k.cactus!.byBiome).map(Number)).toEqual([B.DESERT]);
  for (const [name, kind] of Object.entries(k)) {
    for (const bad of [B.OCEAN, B.ICE, B.BEACH]) expect(kind.byBiome[bad] ?? 0, `${name} on biome ${bad}`).toBe(0);
    if (LARGE.includes(name)) expect(kind.byBiome[B.ALPINE] ?? 0, `${name} on alpine`).toBe(0);
  }
  expect(a.biomeCols[B.ALPINE] ?? 0, 'test world has alpine ground').toBeGreaterThan(100);
  expect(dens('grass', B.ALPINE), 'alpine meadow tufts').toBeGreaterThan(0.05);

  // Budget: ≤ ~60k instances in total.
  expect(a.live).toBeLessThanOrEqual(62_500);

  // Rainforest is denser than temperate forest on the same ground (veg 1 vs 0.85, rule 1.05 vs 0.9).
  await page.evaluate(() => { const lt = (window as any).lt; lt.repaint(4, 8); lt.refresh(true); });
  const rain: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));
  expect(rain.bad.cpuMismatch).toBeLessThanOrEqual(MISMATCH_OK);
  const rainTrees = rain.kinds.tree!.byBiome[B.RAINFOREST]! - k.tree!.byBiome[B.RAINFOREST]!;
  expect(rainTrees).toBeGreaterThan(k.tree!.byBiome[B.TEMPERATE_FOREST]! * 1.25);
  await page.evaluate(() => { const lt = (window as any).lt; lt.reset(); lt.refresh(true); });

  // Treeline: all land temperate forest, so altitude is the only difference. Bins: lowland < TREE_A0,
  // lower half of the ramp, upper half, ≥ TREE_A1.
  await page.evaluate(() => { const lt = (window as any).lt; lt.paintAll(4); lt.refresh(true); });
  const tl: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));
  expect(tl.bad.cpuMismatch, tl.bad.examples.join('\n')).toBeLessThanOrEqual(MISMATCH_OK);
  expect(tl.bad.wrongBiome, tl.bad.examples.join('\n')).toBe(0);
  const z = tl.zone;
  for (const i of [0, 1, 2]) expect(z[i]!.cols, `forest columns in altitude bin ${i}`).toBeGreaterThan(300);
  const zd = (i: number, f: keyof Zone) => (z[i]![f] as number) / z[i]!.cols;
  // forest thins with height ...
  expect(zd(1, 'trees')).toBeLessThan(zd(0, 'trees') * 0.85);
  expect(zd(2, 'trees')).toBeLessThan(zd(0, 'trees') * 0.4);
  // ... trees get smaller toward the line ...
  expect(z[2]!.scaleSum / z[2]!.trees).toBeLessThan((z[0]!.scaleSum / z[0]!.trees) * 0.85);
  // ... pines take over from broadleaf higher up ...
  expect(z[2]!.pines / z[2]!.tempTrees).toBeGreaterThan(z[0]!.pines / z[0]!.tempTrees + 0.3);
  // ... and low shrubs + tufts replace trees near the line.
  // (patch clumping thins them too, so compare as a ratio plus a floor)
  expect(zd(2, 'line')).toBeGreaterThan(zd(0, 'line') * 2 + 0.03);
  await page.evaluate(() => { const lt = (window as any).lt; lt.reset(); lt.refresh(true); });

  // Low quality tier: a fixed half of the slots (by hash, so the same plants stay), CPU reference still exact.
  await page.evaluate(() => { const lt = (window as any).lt; lt.life.setHighQuality(false); lt.refresh(true); });
  const low: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true, quality: 0.5 }));
  expect(low.bad.cpuMismatch, low.bad.examples.join('\n')).toBeLessThanOrEqual(MISMATCH_OK);
  expect(low.live).toBeGreaterThan(a.live * 0.4);
  expect(low.live).toBeLessThan(a.live * 0.6);
  await page.evaluate(() => { const lt = (window as any).lt; lt.life.setHighQuality(true); lt.refresh(true); });

  // New climate biome ids (11-13) and grassland, each painted over all land.
  interface Clumps { cells: number; mean: number; p95: number; bareShare: number; flowerMean: number; flowerP95: number; flowerCellsNone: number; fineLive: number; species: Record<number, number> }
  const per: Record<number, FloraSummary> = {};
  const clump: Record<number, Clumps> = {};
  for (const bid of [B.GRASSLAND, B.STEPPE, B.SHRUBLAND, B.COLD_DESERT]) {
    await page.evaluate((b) => { const lt = (window as any).lt; lt.paintAll(b); lt.refresh(true); }, bid);
    const r: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));
    expect(r.bad.cpuMismatch, r.bad.examples.join('\n')).toBeLessThanOrEqual(MISMATCH_OK);
    expect(r.bad.wrongBiome, r.bad.examples.join('\n')).toBe(0);
    expect(r.bad.wet).toBe(0);
    for (const n of Object.keys(r.kinds)) expect(r.kinds[n]!.listOk, `${n} list`).toBe(true);
    per[bid] = r;
    clump[bid] = await page.evaluate(() => (window as any).lt.clumps());
  }
  const has = (bid: number, sp: number) => per[bid]!.perSpecies[sp] ?? 0;
  // Meadows are clumped: dense drifts (95th-percentile cell ≥ 2.5× the mean), bare spots, flower
  // patches (most flowers in a few cells, many cells without any).
  for (const bid of [B.GRASSLAND, B.STEPPE]) {
    const c = clump[bid]!;
    expect(c.cells).toBeGreaterThan(80);
    expect(c.p95, `hotspots biome ${bid}`).toBeGreaterThan(c.mean * 2.5);
    expect(c.bareShare, `bare spots biome ${bid}`).toBeGreaterThan(0.1);
    expect(c.flowerP95).toBeGreaterThan(c.flowerMean * 3);
    expect(c.flowerCellsNone).toBeGreaterThan(0.2);
  }
  // Grassland variety: four ground-cover grasses/clover + wildflowers, bushes, boulders, hedges, lone trees, reeds.
  for (const sp of [SP.F_GRASS, SP.F_GRASS_TALL, SP.F_SEED, SP.F_CLOVER, SP.F_FLOWER]) expect(has(B.GRASSLAND, sp), `grassland species ${sp}`).toBeGreaterThan(300);
  for (const sp of [SP.BUSH_GREEN, SP.BOULDER, SP.HEDGE, SP.REED]) expect(has(B.GRASSLAND, sp), `grassland species ${sp}`).toBeGreaterThan(5);
  expect(clump[B.GRASSLAND]!.fineLive).toBeGreaterThan(5000);
  // Steppe variety: straw tufts, feather grass, tussocks, sagebrush, stones.
  for (const sp of [SP.F_STRAW, SP.FEATHER, SP.TUSSOCK, SP.SAGEBRUSH, SP.STONE]) expect(has(B.STEPPE, sp), `steppe species ${sp}`).toBeGreaterThan(50);
  const pc = (bid: number, sp: number) => (per[bid]!.perSpecies[sp] ?? 0) / per[bid]!.biomeCols[bid]!;
  // steppe: the odd windswept hardy tree
  expect(per[B.STEPPE]!.perSpecies[SP.HARDY_TREE] ?? 0).toBeGreaterThan(5);
  expect(pc(B.STEPPE, SP.HARDY_TREE)).toBeLessThan(0.02);
  // shrubland: dense scrub, a few cypress columns
  expect(pc(B.SHRUBLAND, SP.SHRUB_SAGE)).toBeGreaterThan(0.08);
  expect(per[B.SHRUBLAND]!.perSpecies[SP.CYPRESS] ?? 0).toBeGreaterThan(20);
  // cold desert: very sparse cushions and stones, nothing else
  expect(pc(B.COLD_DESERT, SP.CUSHION)).toBeGreaterThan(0.005);
  expect(pc(B.SHRUBLAND, SP.SHRUB_SAGE)).toBeGreaterThan(pc(B.COLD_DESERT, SP.CUSHION) * 3);
  expect(Object.keys(per[B.COLD_DESERT]!.perSpecies).map(Number).sort((x, y) => x - y)).toEqual([SP.CUSHION, SP.STONE]);
  await page.evaluate(() => { const lt = (window as any).lt; lt.reset(); lt.refresh(true); });

  // Sim evolves: temperate forest turns to desert. Right after the refresh the trees must still be
  // there, shrinking (target 0), not replaced by cacti in one frame.
  const treesBefore = k.tree!.byBiome[B.TEMPERATE_FOREST]!;
  await page.evaluate(() => (window as any).lt.repaint(4, 6));
  await page.evaluate(() => (window as any).lt.refresh(false));
  const b: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: false }));
  expect(b.kinds.tree!.live, 'trees still standing mid-transition').toBeGreaterThanOrEqual(treesBefore);
  expect(b.shrinking).toBeGreaterThanOrEqual(treesBefore);
  // After the ramp, the next refresh swaps species: trees gone from the old forest, cacti growing there.
  await page.evaluate(() => { (window as any).lt.advance(3); (window as any).lt.refresh(false); });
  const c: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: false }));
  expect(c.kinds.tree!.live).toBeLessThan(k.tree!.live - treesBefore * 0.9);
  expect(c.kinds.cactus!.live, 'cacti now grow on the former forest').toBeGreaterThan(k.cactus!.live + 10);
  expect(c.growing).toBeGreaterThan(0);
  expect(c.bad.wrongBiome).toBe(0);

  await page.evaluate(() => (window as any).lt.frames(3));
  expect(problems).toEqual([]);
});

test('birds stay in the block above terrain; critters stay dry on grazing land; fish stay in shallow water', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'paint=stripes&look=0&bare=1');
  await page.evaluate(async () => { const lt = (window as any).lt; await lt.frames(3); await lt.life.whenReady(); });
  const r = await page.evaluate(() => (window as any).lt.creatures(60));
  // sparse on purpose: ≤ 2 flocks of 6-10, ≤ 4 herds of 2-4 on big grazing patches only
  expect(r.counts.birds).toBeGreaterThanOrEqual(6);
  expect(r.counts.birds).toBeLessThanOrEqual(20);
  expect(r.counts.critters).toBeGreaterThanOrEqual(2);
  expect(r.counts.critters).toBeLessThanOrEqual(16);
  expect(r.counts.fish).toBeGreaterThan(20);
  const v = r.violations;
  expect(v.birdOut, v.examples.join('\n')).toBe(0);
  expect(v.birdLow, v.examples.join('\n')).toBe(0);
  expect(v.birdHigh).toBe(0);
  expect(v.critterWet, v.examples.join('\n')).toBe(0);
  expect(v.critterBiome).toBe(0);
  expect(v.critterSmallPatch, 'no herds on tiny grazing islands').toBe(0);
  expect(v.fishOut).toBe(0);
  expect(v.fishAbove).toBe(0);
  // they actually move (a frozen flock passes every bound trivially)
  expect(r.path.bird / r.counts.birds).toBeGreaterThan(5);
  expect(r.path.critter / r.counts.critters).toBeGreaterThan(0.05);
  expect(r.path.fish / r.counts.fish).toBeGreaterThan(0.2);
  await page.evaluate(() => (window as any).lt.frames(3));
  expect(problems).toEqual([]);
});

test('life screenshots with the full look (natural biomes)', async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  const problems = watchProblems(page, true);
  await open(page, 'paint=natural');
  await page.evaluate(async () => { const lt = (window as any).lt; await lt.frames(2); await lt.life.whenReady(); lt.refresh(true); await lt.frames(20); });
  const info = await page.evaluate(() => (window as any).lt.drawInfo());
  console.log('draw info (look + life)', JSON.stringify(info));
  for (const v of ['hero', 'tree', 'pine', 'acacia', 'cactus', 'cypress', 'shrub', 'treeline', 'critter', 'fish', 'fishlow', 'bird', 'top']) {
    const ok = await page.evaluate(async (n) => { const lt = (window as any).lt; const r = await lt.view(n); if (r) await lt.frames(12); return r; }, v);
    if (ok) await page.screenshot({ path: `${OUT}/${v}.png` });
  }
  // Meadow and steppe close-ups / mid distance on all-land grassland and steppe, at noon.
  for (const [bid, views] of [[5, ['meadow', 'meadowmid']], [11, ['steppe', 'steppemid']]] as const) {
    await page.evaluate(async (b) => { const lt = (window as any).lt; lt.look?.setTimeOfDay?.(0.42); lt.paintAll(b); lt.refresh(true); await lt.frames(3); }, bid);
    for (const v of views) {
      const ok = await page.evaluate(async (n) => { const lt = (window as any).lt; const r = await lt.view(n); if (r) await lt.frames(12); return r; }, v);
      if (ok) await page.screenshot({ path: `${OUT}/${v}.png` });
    }
  }
  expect(problems).toEqual([]);
});
