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
const B = { OCEAN: 0, ICE: 1, TUNDRA: 2, TAIGA: 3, TEMPERATE_FOREST: 4, GRASSLAND: 5, DESERT: 6, SAVANNA: 7, RAINFOREST: 8, BEACH: 9, ALPINE: 10 };

function watchProblems(page: Page): string[] {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => {
    const t = m.type();
    if (/favicon/.test(m.location().url)) return;
    if (t === 'error' || (t === 'warning' && GPU_PROBLEM.test(m.text()))) problems.push(`${t}: ${m.text()}`);
  });
  return problems;
}

async function open(page: Page, query: string) {
  await page.goto(`/tests/gpu/support/life.html?${query}`);
  await page.waitForFunction(() => (window as any).ltReady === true, null, { timeout: 90_000 });
}

type Kinds = Record<string, { live: number; drawn: number; listOk: boolean; byBiome: Record<number, number> }>;
interface FloraSummary { live: number; growing: number; shrinking: number; kinds: Kinds; perSpecies: Record<number, number>; biomeCols: Record<number, number>;
  bad: { wrongBiome: number; wet: number; steep: number; cpuMismatch: number; yMismatch: number; examples: string[] } }

test('flora tracks biome, avoids water and cliffs, matches CPU reference, morphs smoothly', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'paint=stripes&look=0');
  await page.evaluate(() => (window as any).lt.frames(3));
  await page.evaluate(() => (window as any).lt.refresh(true));
  const a: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));

  // GPU kernel == CPU reference (lifeModel.floraDecide) for every slot: placement rules are exactly the documented ones.
  expect(a.bad.cpuMismatch, a.bad.examples.join('\n')).toBe(0);
  expect(a.bad.yMismatch, 'plant base height = bilinear surfY').toBe(0);
  // Nothing underwater or on slopes above the threshold.
  expect(a.bad.wet).toBe(0);
  expect(a.bad.steep).toBe(0);
  expect(a.bad.wrongBiome).toBe(0);
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
  // Trees where forest is (large slots: ~0.39 per column × veg × rule density, minus culls).
  expect(dens('tree', B.TEMPERATE_FOREST)).toBeGreaterThan(0.12);
  expect(dens('tree', B.RAINFOREST)).toBeGreaterThan(0.12);
  expect(dens('pine', B.TAIGA)).toBeGreaterThan(0.12);
  expect(dens('acacia', B.SAVANNA)).toBeGreaterThan(0.01);
  expect(dens('grass', B.GRASSLAND)).toBeGreaterThan(0.15);
  expect(dens('cactus', B.DESERT)).toBeGreaterThan(0.02);
  // Sparse lone trees on grassland, far below forest density.
  expect(dens('tree', B.GRASSLAND)).toBeLessThan(dens('tree', B.TEMPERATE_FOREST) * 0.2);
  // Cacti only in desert; no plant of any kind in ocean, ice, alpine or beach.
  expect(Object.keys(k.cactus!.byBiome).map(Number)).toEqual([B.DESERT]);
  for (const [name, kind] of Object.entries(k)) {
    for (const bad of [B.OCEAN, B.ICE, B.ALPINE, B.BEACH]) expect(kind.byBiome[bad] ?? 0, `${name} on biome ${bad}`).toBe(0);
  }
  // Budget: ≤ ~60k instances in total.
  expect(a.live).toBeLessThanOrEqual(62_500);

  // Rainforest is denser than temperate forest on the same ground (veg 1 vs 0.85, rule 1.05 vs 0.9).
  await page.evaluate(() => { const lt = (window as any).lt; lt.repaint(4, 8); lt.refresh(true); });
  const rain: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true }));
  expect(rain.bad.cpuMismatch).toBe(0);
  const rainTrees = rain.kinds.tree!.byBiome[B.RAINFOREST]! - k.tree!.byBiome[B.RAINFOREST]!;
  expect(rainTrees).toBeGreaterThan(k.tree!.byBiome[B.TEMPERATE_FOREST]! * 1.25);
  await page.evaluate(() => { const lt = (window as any).lt; lt.reset(); lt.refresh(true); });

  // Low quality tier: a fixed half of the slots (by hash, so the same plants stay), CPU reference still exact.
  await page.evaluate(() => { const lt = (window as any).lt; lt.life.setHighQuality(false); lt.refresh(true); });
  const low: FloraSummary = await page.evaluate(() => (window as any).lt.flora({ snapped: true, quality: 0.5 }));
  expect(low.bad.cpuMismatch, low.bad.examples.join('\n')).toBe(0);
  expect(low.live).toBeGreaterThan(a.live * 0.4);
  expect(low.live).toBeLessThan(a.live * 0.6);
  await page.evaluate(() => { const lt = (window as any).lt; lt.life.setHighQuality(true); lt.refresh(true); });

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
  expect(c.kinds.cactus!.live).toBeGreaterThan(k.cactus!.live * 1.5);
  expect(c.growing).toBeGreaterThan(0);
  expect(c.bad.wrongBiome).toBe(0);

  await page.evaluate(() => (window as any).lt.frames(3));
  expect(problems).toEqual([]);
});

test('birds stay in the block above terrain; critters stay dry on grazing land; fish stay in shallow water', async ({ page }) => {
  const problems = watchProblems(page);
  await open(page, 'paint=stripes&look=0');
  await page.evaluate(async () => { const lt = (window as any).lt; await lt.frames(3); await lt.life.whenReady(); });
  const r = await page.evaluate(() => (window as any).lt.creatures(60));
  expect(r.counts.birds).toBeGreaterThanOrEqual(3 * 8);
  expect(r.counts.critters).toBeGreaterThan(20);
  expect(r.counts.fish).toBeGreaterThan(20);
  const v = r.violations;
  expect(v.birdOut, v.examples.join('\n')).toBe(0);
  expect(v.birdLow, v.examples.join('\n')).toBe(0);
  expect(v.birdHigh).toBe(0);
  expect(v.critterWet, v.examples.join('\n')).toBe(0);
  expect(v.critterBiome).toBe(0);
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
  const problems = watchProblems(page);
  await open(page, 'paint=natural');
  await page.evaluate(async () => { const lt = (window as any).lt; await lt.frames(2); await lt.life.whenReady(); lt.refresh(true); await lt.frames(20); });
  const info = await page.evaluate(() => (window as any).lt.drawInfo());
  console.log('draw info (look + life)', JSON.stringify(info));
  for (const v of ['hero', 'tree', 'pine', 'acacia', 'cactus', 'critter', 'fish', 'bird', 'top']) {
    const ok = await page.evaluate(async (n) => { const lt = (window as any).lt; const r = await lt.view(n); if (r) await lt.frames(12); return r; }, v);
    if (ok) await page.screenshot({ path: `${OUT}/${v}.png` });
  }
  expect(problems).toEqual([]);
});
