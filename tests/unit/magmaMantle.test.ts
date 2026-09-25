import { describe, expect, it } from 'vitest';
import { PCG32 } from '../../src/core/rng';
import {
  PlumeSet, plumesFromWorld, plumeStrength, plumeDist2, initMantleTemps, mantleRef, MAX_PLUMES, MIN_PLUMES, PLUME,
  MX, MZ, MY, MANTLE, type Plume,
} from '../../src/sim/mantleModel';
import {
  MAGMA, arcWindow, arcNormK, emaStep, ARC_E_SCALE, ARC_W_SCALE, ARC_K_SCALE, chamberDepth, encLavaY, lavaTempC, lavaSil, magmaBudgetCpu, chamberMassPerColumn,
} from '../../src/sim/magmaModel';
import { generateWorld } from '../../src/sim/worldgen';
import { NCOL, NVOX, Mat, FLAG_HOTSPOT, packVoxel, voxMat, voxFlags } from '../../src/sim/layout';

const seedPlumes = (n: number): Plume[] => Array.from({ length: n }, (_, i) => ({ x: 30 * i, z: 50, r: 10, age: 0, life: 200 }));

describe('PlumeSet', () => {
  // V2/V13: plume evolution is sim state; same seed + same step sequence must give identical plumes, and a
  // save/restore mid-run must continue bit-identically.
  it('is deterministic and state round-trips', () => {
    const a = new PlumeSet(new PCG32(9, 5), seedPlumes(3));
    const b = new PlumeSet(new PCG32(9, 5), seedPlumes(3));
    for (let i = 0; i < 500; i++) { a.step(0.2); b.step(0.2); }
    expect(a.getState()).toEqual(b.getState());
    const c = new PlumeSet(new PCG32(1, 1), []);
    c.setState(JSON.parse(JSON.stringify(a.getState())));
    for (let i = 0; i < 500; i++) { a.step(0.2); c.step(0.2); }
    expect(c.getState()).toEqual(a.getState());
  });

  // Longevity (V8 "≥1 eruption per 50 My"): at least one plume always exists, never more than the table holds,
  // strengths fade in/out so the heat source never pops.
  it('keeps plume count in [MIN, MAX] and strength in [0,1] over 20 Gy', () => {
    const s = new PlumeSet(new PCG32(3, 7), seedPlumes(2), { spawnPerMy: 0.05 });
    let maxN = 0;
    for (let i = 0; i < 20000; i++) {
      s.step(1);
      expect(s.plumes.length).toBeGreaterThanOrEqual(MIN_PLUMES);
      expect(s.plumes.length).toBeLessThanOrEqual(MAX_PLUMES);
      maxN = Math.max(maxN, s.plumes.length);
      for (const p of s.plumes) { const k = plumeStrength(p); expect(k).toBeGreaterThanOrEqual(0); expect(k).toBeLessThanOrEqual(1); }
    }
    expect(maxN).toBeGreaterThan(2); // births happen
  });

  // Plumes must wander far slower than plates move (1-4 cells/My), or chains would not be straight.
  it('wanders slowly (random walk with the configured diffusivity)', () => {
    let sum2 = 0;
    const N = 200, T = 100;
    for (let k = 0; k < N; k++) {
      const p: Plume = { x: 128, z: 128, r: 10, age: 50, life: 1e9 };
      const s = new PlumeSet(new PCG32(k, 3), [p], { spawnPerMy: 0 });
      for (let i = 0; i < T; i++) s.step(1);
      const q = s.plumes[0]!;
      sum2 += plumeDist2(128, 128, q);
    }
    const rms = Math.sqrt(sum2 / N);
    const expected = PLUME.wander * Math.sqrt(2 * T);
    expect(rms).toBeGreaterThan(expected * 0.7);
    expect(rms).toBeLessThan(expected * 1.3);
    expect(rms / T).toBeLessThan(0.05); // ≪ 1 cell/My
  });

  it('evolve: false freezes plumes (tests)', () => {
    const s = new PlumeSet(new PCG32(1, 1), seedPlumes(1), { evolve: false });
    for (let i = 0; i < 100; i++) s.step(1);
    expect(s.plumes[0]!.x).toBe(0);
    expect(s.plumes[0]!.life).toBe(200);
  });
});

describe('plumesFromWorld', () => {
  // Worldgen hotspot chambers must sit on a plume, or they would be fossil chambers with no feed.
  it('puts a plume under every worldgen hotspot chamber', () => {
    for (const seed of [1, 42]) {
      const w = generateWorld(seed);
      const hot: number[] = [];
      for (let i = 0; i < NVOX; i++) if (voxMat(w.vox[i]!) === Mat.MAGMA && voxFlags(w.vox[i]!) & FLAG_HOTSPOT) hot.push(i % NCOL);
      expect(hot.length).toBeGreaterThan(0);
      const ps = plumesFromWorld(w, new PCG32(seed, 2));
      expect(ps.length).toBeLessThanOrEqual(MAX_PLUMES);
      for (const c of hot) {
        const x = c % 256, z = Math.floor(c / 256);
        expect(Math.min(...ps.map((p) => plumeDist2(x, z, p) / (p.r * p.r)))).toBeLessThan(1);
      }
      for (const p of ps) expect(plumeStrength(p)).toBe(1); // mature at t = 0
    }
  });
});

describe('mantle init', () => {
  it('reference profile hot at depth, conduits hot, torus-wrapped distances', () => {
    expect(mantleRef(MY - 1)).toBe(MANTLE.tTop);
    expect(mantleRef(0)).toBe(MANTLE.tBot);
    const t = initMantleTemps([{ x: 1, z: 1, r: 10, age: 50, life: 100 }]);
    const at = (mx: number, mz: number, my: number) => t[mx + mz * MX + my * MX * MZ]!;
    // plume at (1,1): cells on both sides of both seams are heated equally-ish
    expect(at(0, 0, MY - 1)).toBeGreaterThan(MANTLE.tTop + 200);
    expect(at(MX - 1, MZ - 1, MY - 1)).toBeGreaterThan(MANTLE.tTop + 100);
    expect(at(32, 32, MY - 1)).toBe(MANTLE.tTop);
    expect(plumeDist2(255, 255, { x: 1, z: 1, r: 1, age: 0, life: 1 })).toBe(8);
  });
});

describe('magma calibration', () => {
  // The arc returns a fixed share of the subducted slab mass to the crust (continental regrowth, V3 long-term
  // balance) regardless of how the weight is spread: in steady state Σ K·weight over a period = arcFrac·slabMass·events.
  it('arc normalization returns arcFrac × slab mass per subduction event', () => {
    const events = 37, weightPerPeriod = 1234.5; // arbitrary geometry
    let e = 0, w = 0;
    for (let k = 0; k < 400; k++) { e = emaStep(e, events * ARC_E_SCALE); w = emaStep(w, Math.round(weightPerPeriod * ARC_W_SCALE)); }
    const K = arcNormK(e, w) / ARC_K_SCALE;
    const emitted = K * weightPerPeriod;
    expect(emitted / (MAGMA.arcFrac * MAGMA.slabMass * events)).toBeCloseTo(1, 2);
    expect(arcNormK(e, 0)).toBe(0); // no arc columns: nothing emitted, no division by zero
  });

  it('arc window: nothing at the trench, full 4-7 cells inland, nothing beyond 9', () => {
    for (const d of [0, 1, 2]) expect(arcWindow(d)).toBe(0);
    for (const d of [4, 5, 6, 7]) expect(arcWindow(d)).toBe(1);
    for (const d of [9, 12, 16]) expect(arcWindow(d)).toBe(0);
  });

  it('chambers sit at ~1/3 crust depth, bounded', () => {
    expect(chamberDepth(9)).toBe(3);
    expect(chamberDepth(40)).toBe(MAGMA.depthMax);
    expect(chamberDepth(3)).toBe(MAGMA.depthMin);
  });

  it('lava.y packing round-trips', () => {
    const y = encLavaY(1187.25, 200);
    expect(lavaTempC(y)).toBe(1187.25);
    expect(lavaSil(y)).toBe(200);
  });

  // The budget helper is what the GPU conservation tests trust; it must count MAGMA as magma, not crust.
  it('budget counts crust, chambers, pending, lava, reservoir separately', () => {
    const vox = new Uint32Array(NVOX);
    vox[5] = packVoxel(Mat.BASALT, 200, 0, 0);
    vox[6] = packVoxel(Mat.MAGMA, 255, 0, 0);
    vox[7] = packVoxel(Mat.PERIDOTITE, 255, 0, 0);
    vox[6 + NCOL] = packVoxel(Mat.MAGMA, 255, 0, 0);
    const magCol = new Uint32Array(NCOL * 2); magCol[1] = 30;
    const lava = new Uint32Array(NCOL * 2); lava[2] = 17;
    const b = magmaBudgetCpu(vox, 1000, magCol, lava);
    expect(b).toEqual({ crust: 200, reservoir: 1000, chambers: 510, pending: 30, lava: 17, total: 1757 });
    expect(chamberMassPerColumn(vox)[6]).toBe(510);
  });
});
