import { test, expect } from '@playwright/test';

// §I.probe: what the card reports must be the voxel truth of the clicked column, top and cut face alike.
test('probe picks top and side, reports column matching GPU data', async ({ page }) => {
  await page.goto('/?seed=5&speed=0.001');
  await page.waitForFunction(() => (window as unknown as { terraReady?: boolean }).terraReady === true, null, { timeout: 30_000 });
  const res = await page.evaluate(async () => {
    const T = await import('/src/ui/probe.ts');
    const { THREE } = await import('/tests/gpu/support/harness.ts');
    const L = await import('/src/sim/layout.ts');
    const S = await import('/src/render/space.ts');
    const { Probe } = await import('/src/sim/probe.ts');
    const t = (window as any).terra;
    const surf = new Float32Array(await t.fields.read(t.stage.renderer, 'surfY'));
    // straight-down ray over column (40, 90)
    const x = 40, z = 90;
    const top = T.pickColumn(new THREE.Ray(new THREE.Vector3(S.cellToWorld(x), 5, S.cellToWorld(z)), new THREE.Vector3(0, -1, 0)), surf);
    // horizontal ray into the +X face at column z=100, 10 layers below the surface
    const yv = surf[L.colIdx(L.NX - 1, 100)]! - 10;
    const side = T.pickColumn(new THREE.Ray(new THREE.Vector3(5, S.voxelToWorldY(yv), S.cellToWorld(100)), new THREE.Vector3(-1, 0, 0)), surf);
    const p = new Probe(t.fields);
    const r = await p.read(t.stage.renderer, x, z, { veg: 0, ice: 0, vapor: 0, sedSusp: 0 });
    return { top, side, yv: Math.floor(yv), probeSurf: r.surfY, surf: surf[L.colIdx(x, z)], topSolid: r.voxels[Math.ceil(r.surfY) - 1]!.mat, above: r.voxels[Math.ceil(r.surfY)]!.mat };
  });
  expect(res.top).toMatchObject({ x: 40, z: 90, face: 'top' });
  expect(res.side).toMatchObject({ x: 255, z: 100, face: 'side', y: res.yv });
  expect(res.probeSurf).toBe(res.surf);
  expect(res.topSolid).not.toBe(0); // solid at surface
  expect(res.above).toBe(0);        // air above
});
