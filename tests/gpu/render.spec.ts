import { test, expect, type Page } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const OUT = 'test-results/render';
test.use({ viewport: { width: 1280, height: 800 } });
test.setTimeout(120_000);

type Pt = { px: number; py: number; y?: number };
interface FrontPts { strata: Pt[]; skyOverOcean: Pt; rockUnderLand: Pt }
interface Analysis { samples: number[][]; bgPixel: number[]; lumStd: number; nonBgFrac: number }

const dist = (a: number[], b: number[]) => Math.abs(a[0]! - b[0]!) + Math.abs(a[1]! - b[1]!) + Math.abs(a[2]! - b[2]!);
/** Greedy clustering: number of clearly different colours in a list. */
const distinct = (cols: number[][], thr = 30) => {
  const reps: number[][] = [];
  for (const c of cols) if (!reps.some((r) => dist(r, c) < thr)) reps.push(c);
  return reps.length;
};

async function shoot(page: Page, view: string, pts: Pt[] = []): Promise<Analysis> {
  const png = await page.screenshot({ path: `${OUT}/${view}.png` });
  return page.evaluate(([b64, p]) => (window as any).rt.analyze(b64, p), [png.toString('base64'), pts] as const);
}

const GPU_PROBLEM = /webgpu|validation|wgsl|shader|pipeline|binding|uncaptured/i;

test('terrain + side cuts render a strata diorama without GPU errors', async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => {
    const t = m.type();
    if (/favicon/.test(m.location().url)) return;
    if (t === 'error' || (t === 'warning' && GPU_PROBLEM.test(m.text()))) problems.push(`${t}: ${m.text()}`);
  });

  await page.goto('/tests/gpu/support/render.html');
  await page.waitForFunction(() => (window as any).rt?.ready === true, null, { timeout: 60_000 });

  // (2) The beauty shot is a picture, not a blank or flat-filled canvas.
  await page.evaluate(() => (window as any).rt.view('beauty'));
  const beauty = await shoot(page, 'beauty');
  expect(beauty.lumStd, 'luminance spread of beauty shot').toBeGreaterThan(20);
  expect(beauty.nonBgFrac, 'fraction of frame covered by the block').toBeGreaterThan(0.2);
  for (const v of ['corner', 'close', 'top']) { await page.evaluate((n) => (window as any).rt.view(n), v); await shoot(page, v); }

  // Horizontal view of the +Z cut face.
  const pts: FrontPts = await page.evaluate(() => (window as any).rt.view('front'));
  const all = [...pts.strata, pts.skyOverOcean, pts.rockUnderLand];
  const front = await shoot(page, 'front', all);
  const strataCols = front.samples.slice(0, pts.strata.length);
  const [sky, rock] = front.samples.slice(pts.strata.length) as [number[], number[]];

  // (3) A vertical line down the face crosses several rock layers: colours must differ with depth,
  // otherwise the face is not reading the voxels (or reads the wrong ping-pong buffer).
  expect(distinct(strataCols), `strata colours ${JSON.stringify(strataCols)}`).toBeGreaterThanOrEqual(5);

  // (4) The face quad spans the full grid height; above a low (ocean) column it must be discarded so
  // the background shows, while at the same height over land it is solid rock.
  expect(dist(sky, front.bgPixel), `above ocean ${sky} vs bg ${front.bgPixel}`).toBeLessThan(15);
  expect(dist(rock, front.bgPixel), `under land ${rock} vs bg ${front.bgPixel}`).toBeGreaterThan(40);

  // (5) Ping-pong: flipping 'vox' parity must change what the materials read, without rebuilding them.
  // Buffer 0 holds a solid-MAGMA decoy, so after the flip every sample glows orange; before it, most did not.
  // Magma is HDR-emissive (it has to bloom), so after tone mapping it nearly clips the red channel
  // and reads hot orange; lit rock layers stay well below that.
  const magmaLike = (c: number[]) => c[0]! >= 235 && c[0]! - c[2]! > 90 && c[0]! > c[1]! + 20;
  expect(strataCols.filter(magmaLike).length, 'real world is mostly not magma').toBeLessThan(pts.strata.length / 3);
  await page.evaluate(() => (window as any).rt.setVoxParity(0));
  await page.evaluate(() => (window as any).rt.view('front'));
  const decoy = await shoot(page, 'front-decoy', pts.strata);
  expect(decoy.samples.every(magmaLike), `decoy colours ${JSON.stringify(decoy.samples)}`).toBe(true);

  // (1) No WebGPU validation / shader / page errors anywhere along the way.
  expect(problems).toEqual([]);
});

interface LookAnalysis { samples: number[][]; mean: number; lumStd: number; blackFrac: number; speckles: number }
const lumOf = (c: number[]) => 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;

// The shipped look: post pipeline (tone map, grade, AO, AA, bloom, DOF on high) over the diorama
// staging. A broken pass tends to show up as a black / flat frame, a lost backdrop, or NaN specks.
for (const quality of ['high', 'low'] as const) {
  test(`post pipeline (${quality}) renders the diorama on its plinth over a studio backdrop, no NaN specks`, async ({ page }) => {
    mkdirSync(OUT, { recursive: true });
    const problems: string[] = [];
    page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
    page.on('console', (m) => {
      const t = m.type();
      if (/favicon/.test(m.location().url)) return;
      if (t === 'error' || (t === 'warning' && GPU_PROBLEM.test(m.text()))) problems.push(`${t}: ${m.text()}`);
    });
    await page.goto(`/tests/gpu/support/renderLook.html?quality=${quality}`);
    await page.waitForFunction(() => (window as any).rl?.ready === true, null, { timeout: 60_000 });
    await page.evaluate(() => (window as any).rl.frames(40)); // TRAA / SSGI history settles
    const pts = await page.evaluate(() => (window as any).rl.points());
    const png = await page.screenshot({ path: `${OUT}/look-${quality}.png` });
    const a: LookAnalysis = await page.evaluate(([b64, p]) => (window as any).rl.analyze(b64, p),
      [png.toString('base64'), [pts.plinth, pts.backdrop, { px: pts.backdrop.px, py: 800 - 24 }]] as const);
    const [plinth, bgTop, bgBottom] = a.samples as [number[], number[], number[]];

    // A lit picture, not a black or flat-filled frame.
    expect(a.mean, 'mean luminance').toBeGreaterThan(18);
    expect(a.lumStd, 'luminance spread').toBeGreaterThan(15);
    // Studio backdrop: never flat black, and a gradient (vignette + sweep), not a single colour.
    expect(lumOf(bgTop), `backdrop ${bgTop}`).toBeGreaterThan(6);
    expect(Math.abs(lumOf(bgTop) - lumOf(bgBottom)), `backdrop gradient ${bgTop} vs ${bgBottom}`).toBeGreaterThan(2);
    // The plinth is there and reads apart from the backdrop.
    expect(dist(plinth, bgTop), `plinth ${plinth} vs backdrop ${bgTop}`).toBeGreaterThan(12);
    // NaN/Inf anywhere in the HDR chain leaves dead black pixels (TRAA / bloom / DOF spread them).
    expect(a.speckles, 'isolated dead pixels').toBeLessThanOrEqual(3);
    expect(a.blackFrac, 'pure-black fraction').toBeLessThan(0.002);
    expect(problems).toEqual([]);
  });
}

// Regression (B: faces flicker to uniform brown every other sim tick): the sim swaps the 'vox'
// ping-pong each run, so faces must read identical strata on both parities, frame after frame.
test('cut faces show the same strata on both vox parities across many swaps', async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon/.test(m.location().url)) problems.push(m.text()); });
  await page.goto('/tests/gpu/support/render.html');
  await page.waitForFunction(() => (window as any).rt?.ready === true, null, { timeout: 60_000 });
  await page.evaluate(() => (window as any).rt.mirrorVox());
  const pts: FrontPts = await page.evaluate(() => (window as any).rt.view('front'));
  let ref: number[][] | null = null;
  for (let i = 0; i < 8; i++) {
    await page.evaluate((p) => { (window as any).rt.setVoxParity(p); return (window as any).rt.frame(); }, i & 1);
    const a = await shoot(page, `swap-${i}`, pts.strata);
    // every frame: real strata (several distinct layers), nothing magma-like or washed to one colour
    expect(distinct(a.samples), `frame ${i} parity ${i & 1}: ${JSON.stringify(a.samples)}`).toBeGreaterThanOrEqual(5);
    if (!ref) ref = a.samples;
    else a.samples.forEach((c, k) => expect(dist(c, ref![k]!), `frame ${i} sample ${k}`).toBeLessThan(24));
  }
  expect(problems).toEqual([]);
});

// Plates move in whole cells in the sim; the display must not jump (user: "mountains jump after a
// move… the whole tick moves one grid unit at once"). Sub-cell advection + remap on tectonics runs
// keep the rendered peak moving smoothly and monotonically.
test('rendered mountain moves continuously across whole-cell plate shifts', async ({ page }) => {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon/.test(m.location().url)) problems.push(m.text()); });
  await page.goto('/tests/gpu/support/renderMotion.html');
  await page.waitForFunction(() => (window as any).rm?.ready === true, null, { timeout: 60_000 });
  // 60 fps at 1 My/s with a fast plate (4 cells/My): ~0.067 cell per frame, 4 shifts in 70 frames
  const { peaks, shifts } = await page.evaluate(() => (window as any).rm.run(70, 1 / 60, 4));
  expect(shifts.length, 'whole-cell shifts happened').toBeGreaterThanOrEqual(4);
  for (let i = 1; i < peaks.length; i++) {
    const step = peaks[i] - peaks[i - 1];
    expect(step, `frame ${i} step ${step.toFixed(3)} (shift frames ${shifts})`).toBeGreaterThanOrEqual(-0.02);
    expect(step, `frame ${i} step ${step.toFixed(3)} (shift frames ${shifts})`).toBeLessThan(0.3);
  }
  expect(peaks.at(-1)! - peaks[0]!, 'total advance ≈ 70 frames × 4/60 cell').toBeGreaterThan(3.5);
  // Both plateId ping-pong halves hold the same plates here, so the parity must not move anything
  // (regression: a branch-cached index read garbage plate ids on one parity → faces flickered).
  const pp: number[] = await page.evaluate(() => (window as any).rm.parityPeaks());
  for (const v of pp) expect(Math.abs(v - pp[0]!), `parity peaks ${pp}`).toBeLessThan(0.02);
  expect(problems).toEqual([]);
});

// Plates step only on tectonics runs (every 4 ticks), i.e. every few frames at normal speed; the user saw
// plates jerk on every run. The display must spread each step over the frames until the next one.
test('plates stepping every few frames still glide smoothly on screen', async ({ page }) => {
  await page.goto('/tests/gpu/support/renderMotion.html');
  await page.waitForFunction(() => (window as any).rm?.ready === true, null, { timeout: 60_000 });
  // 4 cells/s, one 0.33-cell step every 5 frames: an unsmoothed display jumps 0.33 then holds 4 frames
  const { peaks } = await page.evaluate(() => (window as any).rm.run(90, 1 / 60, 4, false, 5));
  const steps = peaks.slice(21).map((p: number, i: number) => p - peaks[20 + i]); // after the easing settles
  const mean = steps.reduce((a: number, b: number) => a + b, 0) / steps.length;
  expect(mean, 'average speed ≈ 4/60 cell per frame').toBeGreaterThan(0.05);
  for (const s of steps) {
    expect(s, `steps ${steps.map((v: number) => v.toFixed(3))}`).toBeLessThan(0.16);
    expect(s, `steps ${steps.map((v: number) => v.toFixed(3))}`).toBeGreaterThan(0.01); // never holds still
  }
});

// Regression (faces alternated between two cross-sections every few frames): plate offsets hovering
// around ±0.5 cell perpendicular to a face must not switch the face to another column / the far side.
test('cut face stays stable while plate offsets hover around half a cell', async ({ page }) => {
  const problems: string[] = [];
  page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon/.test(m.location().url)) problems.push(m.text()); });
  await page.goto('/tests/gpu/support/render.html');
  await page.waitForFunction(() => (window as any).rt?.ready === true, null, { timeout: 60_000 });
  await page.evaluate(() => (window as any).rt.markNeighbourRows()); // any column switch would show magma
  const pts: FrontPts = await page.evaluate(() => (window as any).rt.view('front'));
  let ref: number[][] | null = null;
  for (const oz of [0.45, 0.55, 0.45, -0.55, -0.45, 0.55]) {
    await page.evaluate((o) => { (window as any).rt.setPlateOffset(0, o); return (window as any).rt.frame(); }, oz);
    const a = await shoot(page, `offset-${oz}`, pts.strata);
    if (!ref) ref = a.samples;
    else a.samples.forEach((c, k) => expect(dist(c, ref![k]!), `offset ${oz} sample ${k}`).toBeLessThan(30));
  }
  expect(problems).toEqual([]);
});

// Regression (flicker of terrain, shadows and cut-face layers every few frames): life/flora called
// updateRenderColumns() without dt on every refresh, which snapped the eased display to the raw sim.
test('foreign display refreshes between frames do not snap the eased display', async ({ page }) => {
  await page.goto('/tests/gpu/support/renderMotion.html');
  await page.waitForFunction(() => (window as any).rm?.ready === true, null, { timeout: 60_000 });
  const { peaks, heights } = await page.evaluate(() => (window as any).rm.run(70, 1 / 60, 4, true));
  for (let i = 1; i < peaks.length; i++) {
    const step = peaks[i] - peaks[i - 1];
    expect(step, `frame ${i} step ${step.toFixed(3)}`).toBeGreaterThanOrEqual(-0.02);
    expect(step, `frame ${i} step ${step.toFixed(3)}`).toBeLessThan(0.3);
    // a 12-layer in-place uplift eases in over ~0.5 s (τ = 0.45 s): < 1 layer per frame, never a snap
    const dh = heights[i] - heights[i - 1];
    expect(Math.abs(dh), `frame ${i} height step ${dh.toFixed(2)}`).toBeLessThan(1.0);
  }
  expect(heights.at(-1)! - heights[0]!, 'uplift arrived').toBeGreaterThan(12);
});

// Regression (translucent water curtains up the mountains, flickering every tick): rain films of a few
// hundredths of a layer on land counted as fully wet, so the sheet level on a mountain became its top
// and the sheet interpolated up the slope from the sea. Films must never lift the sheet; real water
// (the sea, a stream) must still draw.
test('rain films on land never lift the water sheet; sea and streams still draw', async ({ page }) => {
  await page.goto('/tests/gpu/support/renderMotion.html');
  await page.waitForFunction(() => (window as any).rm?.ready === true, null, { timeout: 60_000 });
  for (const film of [0.01, 0.03]) {
    const { h, lv, sea } = await page.evaluate((f) => (window as any).rm.waterProfile(f), film);
    let land = 0, sheetOverLand = 0, openSea = 0;
    for (let i = 0; i < h.length; i++) {
      if (h[i] > sea + 0.3) { land++; if (lv[i] > h[i] + 0.02) sheetOverLand++; }
      if (h[i] < sea - 2) { openSea++; expect(Math.abs(lv[i] - sea), `film ${film}: sea level at sample ${i}`).toBeLessThan(0.05); }
    }
    expect(land, 'profile crosses the island').toBeGreaterThan(40);
    expect(openSea, 'profile crosses open sea').toBeGreaterThan(200);
    expect(sheetOverLand, `film ${film}: land samples with the sheet above the ground`).toBe(0);
  }
  const { h, lv, sea, centre } = await page.evaluate(() => (window as any).rm.waterProfile(0, true));
  let flank = 0, drawn = 0;
  for (let i = centre + 4; i < h.length; i++) if (h[i] > sea + 1) { flank++; if (lv[i] > h[i] + 0.2) drawn++; }
  expect(flank, 'stream flank samples').toBeGreaterThan(20);
  expect(drawn / flank, `stream drawn on ${drawn}/${flank} flank samples`).toBeGreaterThan(0.8);
});

// User: "cutting away with the handles leaves the sides rendering differently and missing the mantle magma
// effect". The slice caps share the outer faces' shading: a cap on the same column row as the +Z face must
// look the same pixel for pixel (strata, smoothing, mantle glow), not flat voxel blocks.
test('a slice cap shows the same cross-section as the outer face on that row', async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  await page.goto('/tests/gpu/support/render.html');
  await page.waitForFunction(() => (window as any).rt?.ready === true, null, { timeout: 60_000 });
  const pts: FrontPts = await page.evaluate(() => (window as any).rt.view('front'));
  const all = [...pts.strata, pts.rockUnderLand];
  const face = await shoot(page, 'cap-face', all);
  await page.evaluate(() => { (window as any).rt.showCap(true); return (window as any).rt.frame(); });
  const cap = await shoot(page, 'cap-cap', all);
  expect(distinct(cap.samples), `cap strata ${JSON.stringify(cap.samples)}`).toBeGreaterThanOrEqual(5);
  cap.samples.forEach((c, k) => expect(dist(c, face.samples[k]!), `sample ${k} (y ${all[k]!.y}): cap ${c} face ${face.samples[k]}`).toBeLessThan(18));
});

// User: "see plate boundaries into the depth … in the cut-outs on the sides". With the Tectonics layer on,
// a boundary crossing the face draws a line down the cut (divergent: straight, cyan like the map lines);
// with the layer off the face is untouched.
test('plate boundaries run down the cut face only while the Tectonics layer is on', async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  await page.goto('/tests/gpu/support/render.html');
  await page.waitForFunction(() => (window as any).rt?.ready === true, null, { timeout: 60_000 });
  const meta = await page.evaluate(() => (window as any).rt.meta);
  await page.evaluate(() => (window as any).rt.view('front'));
  const SPLIT = 64; // under the land crest (LAND_X)
  const pts = await page.evaluate(([s, top]) => {
    const rt = (window as any).rt, CELL = 4 / 256, x0 = (s + 0.5) * CELL - 2 - CELL / 2; // half-way between columns s-1 and s
    return [rt.project(x0, top - 12), rt.project(x0, top - 30), rt.project(x0 + 12 * CELL, top - 12)];
  }, [SPLIT, meta.landSurf]);
  const run = async (opacity: number, tag: string) => {
    await page.evaluate(([s, o]) => { (window as any).rt.setPlates(s, [-1, 0], [1, 0], o); return (window as any).rt.frame(); }, [SPLIT, opacity]);
    return (await shoot(page, `tec-${tag}`, pts)).samples;
  };
  const off = await run(0, 'off');
  const on = await run(1, 'on');
  for (const k of [0, 1]) {
    expect(dist(on[k]!, off[k]!), `boundary pixel ${k}: on ${on[k]} off ${off[k]}`).toBeGreaterThan(60);
    expect(on[k]![2]!, `divergent line is cyan: ${on[k]}`).toBeGreaterThan(on[k]![0]!);
  }
  // away from the boundary only the faint plate tint changes
  expect(dist(on[2]!, off[2]!), `off-boundary pixel: on ${on[2]} off ${off[2]}`).toBeLessThan(dist(on[0]!, off[0]!) / 2);
  const offAgain = await run(0, 'off2');
  offAgain.forEach((c, k) => expect(dist(c, off[k]!), `layer off again, pixel ${k}`).toBeLessThan(12));
});
