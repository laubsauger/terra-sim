import { test, expect, type Page } from '@playwright/test';

// §I.overlays / §I.keys / T47: data overlays 1-9, legend, Tectonics layer (T), plate boundary classification, arrows.

const collectErrors = (page: Page) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`${e.message} @ ${(e.stack ?? '').split('\n').slice(1, 3).join(' | ')}`));
  page.on('console', (m) => {
    const t = m.text();
    if (/status of 404/.test(t)) return;
    if (m.type() === 'error' || /WGSL|validation error|Invalid (ShaderModule|RenderPipeline|ComputePipeline|BindGroup)/i.test(t)) errors.push(t.slice(0, 400));
  });
  return errors;
};

// Every overlay key must render (sheet shader + prep kernel compile and run on the live sim) and the legend must say
// what is shown and in which unit: the legend is how a player reads the colours.
test('overlay keys 1-9 render cleanly with the right legend; 0 / same key turn off; T toggles plate lines', async ({ page }) => {
  test.setTimeout(120_000);
  const errors = collectErrors(page);
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const defs = await page.evaluate(() => (window as unknown as { terra: { overlays: { defs: { key: number; title: string; bar?: { unit: string }; bar2?: { unit: string }; swatches?: { name: string }[] }[] } } })
    .terra.overlays.defs.map((d) => ({ key: d.key, title: d.title, unit: d.bar?.unit ?? d.bar2?.unit ?? null, swatch: d.swatches?.[0]?.name ?? null })));
  expect(defs.map((d) => d.key)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const legend = page.locator('.terra-legend');
  for (const d of defs) {
    await page.keyboard.press(String(d.key));
    await expect(legend).toHaveClass(/show/);
    await expect(legend.locator('.tl-title')).toHaveText(d.title);
    if (d.unit) await expect(legend.locator('.tl-unit').first()).toHaveText(d.unit);
    if (d.swatch) await expect(legend.locator('.tl-sw')).toContainText(d.swatch);
    await page.waitForTimeout(300); // a few frames: prep + sheet draw on the live sim
  }
  // units a reader relies on
  const units = Object.fromEntries(defs.map((d) => [d.key, d.unit]));
  expect(units[2]).toBe('My');
  expect(units[3]).toBe('°C');
  expect(units[4]).toBe('mW/m²');
  await page.keyboard.press('9'); // same key again → off
  await expect(legend).not.toHaveClass(/show/);
  await page.keyboard.press('2');
  await expect(legend.locator('.tl-title')).toHaveText('Crust age');
  // hover readout: the value under the cursor shows in the legend
  const box = (await page.locator('canvas').first().boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.55);
  await page.mouse.move(box.x + box.width * 0.5 + 3, box.y + box.height * 0.55 + 2);
  await expect(legend.locator('.tl-read b')).toContainText('My', { timeout: 5000 });
  await page.keyboard.press('0');
  await expect(legend).not.toHaveClass(/show/);

  // Tectonics layer: game-UI toggle, independent of the data overlays
  const pill = page.locator('.terra-tec-pill');
  await expect(pill).toBeVisible();
  const on0 = await page.evaluate(() => (window as unknown as { terra: { overlays: { tectonics: boolean } } }).terra.overlays.tectonics);
  await page.keyboard.press('t');
  await expect(pill).toHaveAttribute('aria-pressed', String(!on0));
  await pill.click();
  await expect(pill).toHaveAttribute('aria-pressed', String(on0));
  if (!on0) await page.keyboard.press('t');
  await page.keyboard.press('5'); // coexists with a data overlay
  await page.waitForTimeout(400);
  expect(await page.evaluate(() => (window as unknown as { terra: { overlays: { tectonics: boolean; current: number } } }).terra.overlays.tectonics)).toBe(true);
  expect(errors).toEqual([]);
});

/** Synthetic two-plate world (tecWorld.ts) with overlays built on it, in a blank page. */
async function withTwoPlates<T>(page: Page, vel0: [number, number], vel1: [number, number], body: string): Promise<T> {
  await page.goto('/tests/gpu/support/blank.html');
  return page.evaluate(async ({ vel0, vel1, body }) => {
    const { makeRenderer, THREE } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createDerivePass } = await import('/src/sim/derive.ts');
    const { createOverlays } = await import('/src/overlay/overlays.ts');
    const { updateAllRenderColumns } = await import('/src/render/space.ts');
    const r = await makeRenderer();
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld(vel0, vel1);
    uploadWorld(f, w);
    createDerivePass(f).run(r);
    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    // plate 0 spans x ∈ [0,128), plate 1 x ∈ [128,256): centroids at the middle of each half
    const centroids: [number, number][] = [[64, 128], [192, 128]];
    const ov = createOverlays({ fields: f, scene, renderer: r, camera: cam, source: {
      plates: () => w.plates, centroids: () => centroids, seaLevel: () => 76, geoMy: () => 0,
    } });
    ov.tectonics = true;
    ov.frame(0.016);
    updateAllRenderColumns(r);
    const fn = new Function('ctx', `return (async () => { ${body} })();`);
    return fn({ r, f, w, ov, scene, THREE, updateAllRenderColumns });
  }, { vel0, vel1, body }) as Promise<T>;
}

// Boundary colour must follow the physics: plate 0 drives east into plate 1 → the seam at x = 128 is convergent
// (red), the seam across the torus edge at x = 0 is divergent (cyan). Sliding past each other → transform (yellow).
test('plate boundaries classify convergent / divergent / transform from relative plate motion', async ({ page }) => {
  type Res = { conv: number[][]; div: number[][]; inner: number[] };
  const body = `
    const T = await ctx.ov.readAll('tec');
    const at = (x, z) => { const i = (x + z * 256) * 4; return [T[i], T[i + 1], T[i + 2], T[i + 3]]; };
    const conv = [], div = [];
    for (const z of [10, 100, 200]) { conv.push(at(127, z), at(128, z)); div.push(at(255, z), at(0, z)); }
    return { conv, div, inner: at(64, 50) };`;
  const head = await withTwoPlates<Res>(page, [4, 0], [0, 0], body);
  for (const c of head.conv) { expect(c[1]).toBe(1); expect(c[2]).toBeGreaterThan(0.9); expect(c[3]).toBeCloseTo(4, 3); }
  for (const c of head.div) { expect(c[1]).toBe(1); expect(c[2]).toBeLessThan(-0.9); }
  expect(head.inner[1]).toBe(0); // interior columns carry no boundary
  expect([head.conv[0]![0], head.conv[1]![0]]).toEqual([0, 1]); // plate ids either side
  const slide = await withTwoPlates<Res>(page, [0, 4], [0, 0], body);
  for (const c of [...slide.conv, ...slide.div]) { expect(c[1]).toBe(1); expect(Math.abs(c[2]!)).toBeLessThan(0.2); }
});

// The arrow is the only cue for plate motion: its head must point where the plate goes, and faster = longer.
test('plate arrows point along plate velocities (rendered top-down) and scale with speed', async ({ page }) => {
  type Res = { st: { heading: number; length: number; visible: boolean; x: number; z: number }[]; img: { axisDeg: number; headAhead: number; n: number }[] };
  const res = await withTwoPlates<Res>(page, [3, 0], [0, -1], `
    const { r, ov, THREE } = ctx;
    for (let i = 0; i < 120; i++) ov.frame(0.05); // let the eased arrows settle
    ctx.updateAllRenderColumns(r);
    const sc = new THREE.Scene();
    sc.add(ov.objects.arrows);
    const S = 512;
    const cam = new THREE.OrthographicCamera(-2, 2, 2, -2, 0.1, 50);
    cam.position.set(0, 20, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
    const rt = new THREE.RenderTarget(S, S);
    r.setRenderTarget(rt);
    r.setClearColor(0x000000, 1);
    r.render(sc, cam);
    const px = await r.readRenderTargetPixelsAsync(rt, 0, 0, S, S);
    r.setRenderTarget(null);
    const st = ov.arrows.slice(0, 2).map((s) => ({ heading: s.heading, length: s.length, visible: s.visible, x: s.x, z: s.z }));
    // world (x, z) of a pixel; readback rows start at the bottom of the image (world +z at the bottom here)
    const img = st.map((s) => {
      const hx = Math.cos(s.heading), hz = Math.sin(s.heading);
      const pts = [];
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const o = (y * S + x) * 4;
        if (px[o] + px[o + 1] + px[o + 2] < 240) continue; // cream body only (casing is near-black)
        const wx = (x + 0.5) / S * 4 - 2, wz = (y + 0.5) / S * 4 - 2;
        if (Math.hypot(wx - s.x, wz - s.z) > s.length * 0.75) continue;
        pts.push([wx - s.x, wz - s.z]);
      }
      // principal axis
      let sxx = 0, szz = 0, sxz = 0;
      for (const [a, b] of pts) { sxx += a * a; szz += b * b; sxz += a * b; }
      const axis = 0.5 * Math.atan2(2 * sxz, sxx - szz);
      let d = Math.abs(axis - Math.atan2(hz, hx)) % Math.PI; d = Math.min(d, Math.PI - d);
      // the head is the only part wider than the shaft: its pixels must sit ahead of the centre
      const W = 0.105;
      const wide = pts.filter(([a, b]) => Math.abs(-a * hz + b * hx) > 0.4 * W);
      const headAhead = wide.reduce((acc, [a, b]) => acc + a * hx + b * hz, 0) / Math.max(1, wide.length);
      return { axisDeg: d * 180 / Math.PI, headAhead, n: pts.length };
    });
    return { st, img };`);
  const [a, b] = res.st;
  expect(a!.visible && b!.visible).toBe(true);
  expect(a!.heading).toBeCloseTo(0, 2);              // (3, 0) → +x
  expect(b!.heading).toBeCloseTo(-Math.PI / 2, 2);   // (0, -1) → −z
  expect(a!.length).toBeGreaterThan(b!.length);      // faster plate, longer arrow
  for (const [i, im] of res.img.entries()) {
    expect(im.n, `arrow ${i} pixels`).toBeGreaterThan(200);
    expect(im.axisDeg, `arrow ${i} axis`).toBeLessThan(8);
    expect(im.headAhead, `arrow ${i} head ahead`).toBeGreaterThan(res.st[i]!.length * 0.15);
  }
});

// Legend swatches are only honest if the post chain shows the same colour: the overlay undoes ACES + exposure.
test('overlay colours survive the post tone map (legend colour == screen colour)', async ({ page }) => {
  await page.goto('/tests/gpu/support/blank.html');
  const worst = await page.evaluate(async () => {
    const { makeRenderer, TSL, THREE } = await import('/tests/gpu/support/harness.ts');
    const { untonemap } = await import('/src/overlay/overlayTsl.ts');
    const { hexToLinear, RAMPS, PLATE_COLORS, BIOME_COLORS, BOUNDARY } = await import('/src/overlay/colormaps.ts');
    const r = await makeRenderer();
    const hex = [...Object.values(RAMPS).flatMap((x) => x.stops), ...PLATE_COLORS, ...BIOME_COLORS, BOUNDARY.convergent, BOUNDARY.divergent, BOUNDARY.transform];
    const out = TSL.instancedArray(hex.length, 'vec4');
    const inp = TSL.uniformArray(hex.map((h) => new THREE.Vector3(...hexToLinear(h))), 'vec3');
    const e = 1.1; // post.ts exposure
    r.compute(TSL.Fn(() => {
      const h = untonemap(inp.element(TSL.instanceIndex) as never, TSL.float(e));
      out.element(TSL.instanceIndex).assign(TSL.vec4(TSL.acesFilmicToneMapping(h as never, TSL.float(e)) as never, 0));
    })().compute(hex.length));
    const a = new Float32Array(await r.getArrayBufferAsync(out.value));
    let w = 0;
    hex.forEach((h, i) => { const c = hexToLinear(h); for (let k = 0; k < 3; k++) w = Math.max(w, Math.abs(a[4 * i + k]! - c[k]!)); });
    return w;
  });
  expect(worst).toBeLessThan(0.02);
});

// Plate colours follow plate identity; a split child gets a colour related to (but distinct from) its parent.
test('plate colours are stable per plate and split children inherit a related colour', async ({ page }) => {
  await page.goto('/tests/gpu/support/blank.html');
  const res = await page.evaluate(async () => {
    const { makeRenderer, THREE } = await import('/tests/gpu/support/harness.ts');
    const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
    const { GpuFields } = await import('/src/core/gpu.ts');
    const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
    const { createOverlays } = await import('/src/overlay/overlays.ts');
    const { hexToLinear, PLATE_COLORS } = await import('/src/overlay/colormaps.ts');
    const r = await makeRenderer();
    const f = new GpuFields(); registerSimFields(f); f.freeze();
    const w = twoPlateWorld([1, 0], [0, 0]); uploadWorld(f, w);
    const log: { op: { kind: string; plate?: number; into?: number } }[] = [];
    const ov = createOverlays({ fields: f, scene: new THREE.Scene(), renderer: r, camera: new THREE.PerspectiveCamera(), source: {
      plates: () => w.plates, centroids: () => null, seaLevel: () => 76, geoMy: () => 0, lineage: () => log as never,
    } });
    ov.tectonics = true;
    ov.frame(0.016);
    const before = ov.plateColors.slice();
    log.push({ op: { kind: 'split', plate: 1, into: 5, nx: 1, nz: 0, cx: 0, cz: 0 } as never });
    ov.frame(0.016);
    const after = ov.plateColors.slice();
    const lab = (h: string) => { const [R, G, B] = hexToLinear(h); const l = Math.cbrt(0.4122 * R + 0.5363 * G + 0.0514 * B), m = Math.cbrt(0.2119 * R + 0.6807 * G + 0.1074 * B), s = Math.cbrt(0.0883 * R + 0.2817 * G + 0.63 * B);
      return [0.2105 * l + 0.7936 * m - 0.0041 * s, 1.978 * l - 2.4286 * m + 0.4506 * s, 0.0259 * l + 0.7828 * m - 0.8087 * s]; };
    const dE = (a: string, b: string) => { const x = lab(a), y = lab(b); return 100 * Math.hypot(x[0]! - y[0]!, x[1]! - y[1]!, x[2]! - y[2]!); };
    return { same: before.every((c, i) => i === 5 || after[i] === c), initial: before[5] === PLATE_COLORS[5], child: after[5], parent: after[1],
      dChildParent: dE(after[5]!, after[1]!), dChildOld: dE(after[5]!, PLATE_COLORS[5]!) };
  });
  expect(res.initial).toBe(true);
  expect(res.same).toBe(true);                    // nobody else is recoloured
  expect(res.dChildParent).toBeGreaterThan(6);    // tellable apart …
  expect(res.dChildParent).toBeLessThan(res.dChildOld); // … but closer to the parent than the slot's unrelated default
});

// Users compare plate speeds by arrow length: a plate 3× faster must look clearly longer, not nearly equal
// (√ scaling showed 0.9 and 3.0 cm/yr as 0.52 vs 0.72), and the fastest arrow must not dwarf the block.
test('arrow length is proportional to plate speed', async ({ page }) => {
  await page.goto('/tests/gpu/support/blank.html');
  const r = await page.evaluate(async () => {
    const { arrowLength, MIN_LEN, MAX_LEN } = await import('/src/overlay/arrows.ts');
    return { slow: arrowLength(0.47), fast: arrowLength(1.5), max: arrowLength(99), min: MIN_LEN, top: MAX_LEN };
  });
  expect((r.fast - r.min) / (r.slow - r.min)).toBeCloseTo(1.5 / 0.47, 1);
  expect(r.fast / r.slow).toBeGreaterThan(1.6);
  expect(r.max).toBeLessThanOrEqual(0.6); // block is 2 world units wide
});

// The Overlays panel toggles must act on what is on screen by default: the Tectonics layer alone, without a data
// overlay (they used to apply only to data overlays, so with just the plate lines on they did nothing).
test('arrow, legend and declutter toggles act on the Tectonics layer alone; Plates off by default', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const state = () => page.evaluate(() => {
    const t = (window as any).terra;
    return { arrows: t.overlays.objects.arrows.visible, life: t.life.object.visible, tags: [...document.querySelectorAll('.terra-plate-tag')].some((e: any) => e.style.display !== 'none') };
  });
  expect(await page.evaluate(() => (window as any).terra.overlays.tectonics), 'plate lines + arrows start off (user)').toBe(false);
  await page.evaluate(() => { const o = (window as any).terra.overlays; o.select(0); o.tectonics = true; });
  await expect.poll(async () => (await state()).arrows, { timeout: 5000 }).toBe(true);
  const on = await state();
  expect(on.life, 'declutter is off by default: life stays under the plate lines').toBe(true);
  await page.evaluate(() => { const o = (window as any).terra.overlays; o.arrowsVisible = false; o.declutter = true; });
  await page.waitForTimeout(400);
  const off = await state();
  expect(off.arrows).toBe(false);
  expect(off.life).toBe(false);
  expect(off.tags).toBe(false);
  await page.evaluate(() => { const o = (window as any).terra.overlays; o.arrowsVisible = true; o.declutter = false; o.legendVisible = false; });
  await page.waitForTimeout(400);
  const back = await state();
  expect(back.arrows).toBe(true);
  expect(back.life).toBe(true);
  expect(back.tags, 'legend off hides the plate speed tags').toBe(false);
});

// 'hide life & clouds' is a plain switch: it must work with no overlay and no plate lines on.
test('declutter hides life with no overlay on and restores it', async ({ page }) => {
  await page.goto('/?seed=4');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const life = () => page.evaluate(() => (window as any).terra.life.object.visible as boolean);
  await page.evaluate(() => { const o = (window as any).terra.overlays; o.select(0); o.tectonics = false; o.declutter = true; });
  expect(await life()).toBe(false);
  await page.evaluate(() => { (window as any).terra.overlays.declutter = false; });
  expect(await life()).toBe(true);
});
