import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// Shared in-page setup: a two-plate synthetic world with the full Sim, plus helpers.
const SETUP = async () => {
  const { makeRenderer } = await import('/tests/gpu/support/harness.ts');
  const { twoPlateWorld } = await import('/tests/gpu/support/tecWorld.ts');
  const { GpuFields } = await import('/src/core/gpu.ts');
  const { registerSimFields, uploadWorld } = await import('/src/sim/fields.ts');
  const { Sim } = await import('/src/sim/sim.ts');
  const { Params } = await import('/src/core/params.ts');
  const { hashFields } = await import('/src/sim/hash.ts');
  const save = await import('/src/core/save.ts');
  const { NanGuard } = await import('/src/core/nanGuard.ts');
  const r = await makeRenderer();
  const mk = (vel0: [number, number], vel1: [number, number]) => {
    const f = new GpuFields();
    registerSimFields(f);
    f.freeze();
    const w = twoPlateWorld(vel0, vel1);
    uploadWorld(f, w);
    const params = new Params();
    const sim = new Sim(r, f, w, params, 0.05);
    const clock = { geoTime: 0 };
    return { f, sim, params, clock, refs: { renderer: r, fields: f, sim, params, clock } };
  };
  const frame = () => new Promise((res) => setTimeout(res, 0));
  // chunked like the frame loop; stalls on the window readback barrier are waited out
  const runTo = async (sim: InstanceType<typeof Sim>, target: number) => {
    while (sim.tick < target) { sim.runTicks(Math.min(7, target - sim.tick)); await frame(); }
  };
  const hash = (f: InstanceType<typeof GpuFields>) => hashFields(r, f);
  const cpu = async (sim: InstanceType<typeof Sim>) => { await sim.settle(); return JSON.stringify(sim.getState()); };
  return { r, mk, runTo, hash, cpu, save, NanGuard };
};

// V13: save → load → identical hash. A continued run and a run resumed from the save must be the same world,
// which only holds if GPU fields, CPU sim state, rng streams and the in-flight window readback all roundtrip.
test('save → load → run M ticks gives the same hash as running on without saving', async ({ page }) => {
  test.setTimeout(240_000);
  const res = await page.evaluate(async (setupSrc) => {
    const { mk, runTo, hash, cpu, save } = await ((0, eval)(setupSrc) as typeof SETUP)();
    const A = mk([3, 1.5], [-1, 0.5]);
    A.sim.runTicks(40); // window end: counters readback issued, not landed
    const p = A.sim as unknown as { pending: unknown; pendingResult: unknown };
    const inFlightAt40 = p.pending !== null && p.pendingResult === null;
    const s40 = await save.captureWorld(A.refs);
    const h40 = await hash(A.f);
    await runTo(A.sim, 70); // mid-window save
    const s70 = await save.captureWorld(A.refs);
    const h70 = await hash(A.f);
    await runTo(A.sim, 200);
    const hA = await hash(A.f), cA = await cpu(A.sim);

    // B: a different world that already ran, so nothing can leak through from construction
    const B = mk([-2, 1], [1, -1]);
    await runTo(B.sim, 45);
    await save.restoreWorld(B.refs, s40.blob);
    const hB40 = await hash(B.f); // immediately after load: uploads flushed
    const clockAfterLoad = B.clock.geoTime;
    await runTo(B.sim, 200);
    const hB1 = await hash(B.f), cB1 = await cpu(B.sim);
    const r70 = await save.restoreWorld(B.refs, s70.blob);
    const hB70 = await hash(B.f);
    await runTo(B.sim, 200);
    const hB2 = await hash(B.f), cB2 = await cpu(B.sim);

    // negative control: dropping the in-flight readback on load must change the outcome, else the
    // test above could not detect a broken pending-readback roundtrip
    await save.restoreWorld(B.refs, s40.blob);
    const st = B.sim.getState();
    B.sim.loadState({ ...st, pendingCounters: null });
    await runTo(B.sim, 200);
    const hDrop = await hash(B.f);
    return { inFlightAt40, h40, hB40, h70, hB70, hA, hB1, hB2, hDrop, same1: cA === cB1, same2: cA === cB2,
      warnings: r70.warnings, clockAfterLoad, size: s70.blob.size };
  }, SETUP.toString());
  expect(res.inFlightAt40).toBe(true);
  expect(res.hB40).toBe(res.h40);
  expect(res.hB70).toBe(res.h70);
  expect(res.hB1).toBe(res.hA);
  expect(res.hB2).toBe(res.hA);
  expect(res.same1).toBe(true);
  expect(res.same2).toBe(true);
  expect(res.hDrop).not.toBe(res.hA);
  expect(new Set([res.h40, res.h70, res.hA]).size).toBe(3); // the world evolves, so equal hashes mean something
  expect(res.warnings).toEqual([]);
  expect(res.clockAfterLoad).toBe(40 * 0.05);
  expect(res.size).toBeLessThan(8 * 1024 * 1024); // gzip keeps a ~40 MB world small
});

// V13: rejected loads leave the running world untouched. Unknown fields reject; missing fields (an older
// save, before a pass registered its field) zero-fill with a warning and still load.
test('version mismatch and unknown field reject without touching the world; missing field zero-fills', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async (setupSrc) => {
    const { r, mk, runTo, hash, save } = await ((0, eval)(setupSrc) as typeof SETUP)();
    const A = mk([3, 1.5], [-1, 0.5]);
    await runTo(A.sim, 60);
    const snap = await save.captureWorld(A.refs);
    await runTo(A.sim, 90);
    const h90 = await hash(A.f);
    const err = async (p: Promise<unknown>) => { try { await p; return null; } catch (e) { return (e as Error).message; } };

    const bytes = new Uint8Array(await snap.blob.arrayBuffer());
    const badVer = bytes.slice(); new DataView(badVer.buffer).setUint16(4, save.TERRA_VERSION + 1, true);
    const eVer = await err(save.restoreWorld(A.refs, new Blob([badVer])));
    const afterVer = { hash: await hash(A.f), tick: A.sim.tick };

    // re-encode the same save with an extra field / without 'veg'
    const d = await save.decodeTerra(snap.blob);
    const fieldsOf = (names: string[]) => d.header.fields.filter((e) => names.includes(e.name))
      .map((e) => ({ name: e.name, type: e.type, count: e.count, data: d.payload.slice(e.byteOffset, e.byteOffset + e.byteLength) }));
    const all = d.header.fields.map((e) => e.name);
    const { fields: _f, rawBytes: _r, ...hdr } = d.header;
    const withBogus = await save.encodeTerra({ seed: d.seed, geoTime: d.geoTime, header: hdr },
      [...fieldsOf(all), { name: 'bogus', type: 'float', count: 4, data: new ArrayBuffer(16) }]);
    const eUnknown = await err(save.restoreWorld(A.refs, withBogus));
    const afterUnknown = { hash: await hash(A.f), tick: A.sim.tick };

    const noVeg = await save.encodeTerra({ seed: d.seed, geoTime: d.geoTime, header: hdr }, fieldsOf(all.filter((n) => n !== 'veg')));
    const res = await save.restoreWorld(A.refs, noVeg);
    const vegZero = new Float32Array(await A.f.read(r, 'veg')).every((v) => v === 0);
    const vegWasNonZero = new Float32Array(d.payload, d.header.fields.find((e) => e.name === 'veg')!.byteOffset, 65536).some((v) => v !== 0);
    return { eVer, eUnknown, h90, afterVer, afterUnknown, warnings: res.warnings, tick: A.sim.tick, vegZero, vegWasNonZero };
  }, SETUP.toString());
  expect(res.eVer).toMatch(/version/);
  expect(res.afterVer).toEqual({ hash: res.h90, tick: 90 });
  expect(res.eUnknown).toMatch(/bogus/);
  expect(res.afterUnknown).toEqual({ hash: res.h90, tick: 90 });
  expect(res.warnings.join(' ')).toMatch(/veg/);
  expect(res.tick).toBe(60);
  expect(res.vegZero).toBe(true);
  expect(res.vegWasNonZero).toBe(true); // else zero-fill would be indistinguishable from loading
});

// V7: a NaN anywhere in a float field is found by the guard and the world rolls back to the last verified
// snapshot, which is finite and exactly the snapshot's state. A clean world must not trip it (a false
// positive would roll back forever).
test('NaN injected into water trips the guard and rolls back to a finite snapshot', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async (setupSrc) => {
    const { r, mk, runTo, hash, save, NanGuard } = await ((0, eval)(setupSrc) as typeof SETUP)();
    const A = mk([3, 1.5], [-1, 0.5]);
    const guard = new NanGuard(A.f, save.SAVE_EXCLUDE);
    const notices: string[] = [];
    let paused = false;
    const mgr = new save.SaveManager({ ...A.refs, guard }, null, { windowTicks: 40, notice: (m) => notices.push(m), pause: () => { paused = true; } });
    await runTo(A.sim, 40);
    mgr.frame(); // first frame: rolling snapshot (memory only, no IDB here)
    await mgr.idle();
    const snapTick = mgr.status().lastGoodTick;
    const hSnap = await hash(A.f);
    await runTo(A.sim, 70);
    const cleanOk = await mgr.check();

    const water = new Float32Array(await A.f.read(r, 'water'));
    water[12345] = NaN;
    (A.f.cpuArray('water') as Float32Array).set(water);
    A.f.markDirty('water');
    await runTo(A.sim, 75);
    const bitsBefore = await guard.check(r);
    const ok = await mgr.check();
    const bitsAfter = await guard.check(r);
    const w2 = new Float32Array(await A.f.read(r, 'water'));
    return {
      snapTick, cleanOk, flagged: guard.decode(bitsBefore), ok, bitsAfter, finite: w2.every(Number.isFinite),
      tick: A.sim.tick, hashSame: (await hash(A.f)) === hSnap, log: mgr.log, notices, paused, checked: guard.fieldNames(),
    };
  }, SETUP.toString());
  expect(res.snapTick).toBe(40);
  expect(res.cleanOk).toBe(true);
  expect(res.flagged).toContain('water');
  expect(res.checked).not.toContain('waterTmp');
  expect(res.ok).toBe(false);
  expect(res.bitsAfter).toBe(0);
  expect(res.finite).toBe(true);
  expect(res.tick).toBe(40);
  expect(res.hashSame).toBe(true);
  expect(res.log).toHaveLength(1);
  expect(res.log[0]!.fields).toContain('water');
  expect(res.log[0]!.restoredTick).toBe(40);
  expect(res.paused).toBe(false);
  expect(res.notices.join(' ')).toMatch(/rolled back/);
});

// I.idb: manual slots and the autosave ring persist in IndexedDB and load back to the same world.
test('IndexedDB slots: save, list, load latest', async ({ page }) => {
  test.setTimeout(180_000);
  const res = await page.evaluate(async (setupSrc) => {
    const { mk, runTo, hash, save, NanGuard } = await ((0, eval)(setupSrc) as typeof SETUP)();
    await new Promise((ok) => { const d = indexedDB.deleteDatabase(save.IDB_NAME); d.onsuccess = d.onerror = d.onblocked = () => ok(null); });
    const A = mk([3, 1.5], [-1, 0.5]);
    const store = await save.SaveStore.open();
    const mgr = new save.SaveManager({ ...A.refs, guard: new NanGuard(A.f, save.SAVE_EXCLUDE) }, store, { windowTicks: 40 });
    await runTo(A.sim, 50);
    const meta = await mgr.saveSlot();
    const h50 = await hash(A.f);
    await runTo(A.sim, 90);
    await mgr.loadSlot('latest');
    const list = await mgr.listSlots();
    return { key: meta.key, list: list.map((m) => [m.key, m.kind, m.tick]), tick: A.sim.tick, same: (await hash(A.f)) === h50 };
  }, SETUP.toString());
  expect(res.key).toBe('slot-0');
  expect(res.list).toEqual([['slot-0', 'manual', 50]]);
  expect(res.tick).toBe(50);
  expect(res.same).toBe(true);
});
