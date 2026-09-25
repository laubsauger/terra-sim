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
