// Ocean leveling: open ocean equilibrates almost instantly on geologic timescales, but the local pipe model
// needs thousands of substeps to level a basin, and tectonic column shifts re-roughen it every run (B13).
// Per tick:
//   1 reduce  : fixed-point Σ level and count over open-ocean columns; deepest column (absorber) by atomicMax
//   2 apply   : each open-ocean column moves LEVEL_RATE of the way to the mean, in whole quanta Q = 2^-17
//               (exact float adds for depths < 128, fast-math-proof); Σ quanta accumulated
//   3 balance : the absorber takes back the quantised residual, so Σ water is exactly unchanged (V4)
//   4 commit  : store the mean for next tick's ocean test (separate pass: no read/write race, V2)
import type * as THREE from 'three/webgpu';
import { Fn, If, int, uint, float, instanceIndex, Return, atomicAdd, atomicLoad, atomicStore, atomicMax, select, max } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL } from './layout';
import { iMax, iMin } from './tslLayout';

/** Seafloor this far below sea level counts as open ocean (inland lakes above it keep their own level). */
export const OPEN_DEPTH = 3;
/** Relaxation fraction per tick. */
export const LEVEL_RATE = 0.5;
/**
 * Groundwater seepage: standing water (lakes, > SEEP_MIN_DEPTH) whose surface is more than SEEP_ABOVE above sea
 * level loses SEEP_FRAC per tick to the ocean pool (half-life ≈ 2 My). Uplift lifts whole basins with their lakes;
 * without an outlet only evaporation drained them, and highland lakes piled up on rising ranges.
 */
export const SEEP_FRAC = 0.017;
export const SEEP_ABOVE = 2;
export const SEEP_MIN_DEPTH = 0.3;
const FIX = 1024;        // level quantum for the mean (1/1024 voxel), summed exactly as hi/lo int32 pairs
const OFFSET = 64;       // levels measured from y = 64 → |q| ≤ 2^16
const Q = 2 ** -17;      // water quantum for exact adds
const MAXQ = 2 ** 22;    // |Δ| ≤ 32 voxels per column per tick in quanta

// oceanSum slots
const S_HI = 0, S_N = 1, S_PREV = 2, S_DQ = 3, S_ABS = 4, S_DQLO = 5, S_LO = 6;
/** Water drained from subducted/overridden columns, in 2^-10 voxel units, spread over open ocean each tick. */
export const OCEAN_POOL = 7;

export function registerOceanFields(f: GpuFields): void {
  f.add('oceanSum', 'int', 8, { atomic: true });
}

export class OceanLevel {
  private seep: THREE.ComputeNode;
  private clear: THREE.ComputeNode;
  private reduce: THREE.ComputeNode;
  private apply: THREE.ComputeNode;
  private balance: THREE.ComputeNode;

  constructor(fields: GpuFields) {
    const surfY = fields.cur('surfY');
    const water = fields.cur('water');
    const acc = fields.cur<'int'>('oceanSum');

    this.clear = Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      for (const k of [S_HI, S_LO, S_N, S_DQ, S_DQLO, S_ABS]) atomicStore(acc.element(k), int(0));
    })().compute(1);

    // ocean = seafloor well below the previous mean sea level (bathymetry, not current depth: fresh ridge
    // gaps start dry and overlap columns start over-full, both must level); first tick: deep water only
    // slot S_PREV holds (mean - OFFSET)·FIX; 0 before the first tick
    const hasPrev = () => atomicLoad(acc.element(S_PREV)).notEqual(int(0));
    const isOcean = (floorY: THREE.Node<'float'>, w: THREE.Node<'float'>) => {
      const prev = float(atomicLoad(acc.element(S_PREV))).div(FIX).add(OFFSET);
      return select(hasPrev(), floorY.lessThan(prev.sub(OPEN_DEPTH)), w.greaterThan(OPEN_DEPTH));
    };
    const meanOf = (n: THREE.Node<'int'>) =>
      float(atomicLoad(acc.element(S_HI))).mul(1024).add(float(atomicLoad(acc.element(S_LO)))).div(float(n).mul(FIX)).add(OFFSET);

    // seepage in whole 2^-10 units into the pool (spread over open ocean by apply below): Σ water unchanged (V4)
    this.seep = Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      If(hasPrev().not(), () => { Return(); });
      const w = water.element(c).toVar();
      const s = surfY.element(c).toVar();
      const sea = float(atomicLoad(acc.element(S_PREV))).div(FIX).add(OFFSET);
      If(w.greaterThan(SEEP_MIN_DEPTH).and(s.add(w).greaterThan(sea.add(SEEP_ABOVE))).and(isOcean(s, w).not()), () => {
        const units = int(w.mul(SEEP_FRAC * 1024).floor()).toVar();
        If(units.greaterThan(int(0)), () => {
          water.element(c).assign(w.sub(float(units).div(1024)));
          atomicAdd(acc.element(OCEAN_POOL), units);
        });
      });
    })().compute(NCOL);

    this.reduce = Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const w = water.element(c).toVar();
      const s = surfY.element(c).toVar();
      If(isOcean(s, w), () => {
        // exact Σ level: q = hi·1024 + lo, both partial sums fit int32 (a single int32 sum would overflow)
        const q = int(s.add(w).sub(OFFSET).mul(FIX).round()).toVar();
        atomicAdd(acc.element(S_HI), q.shiftRight(int(10)));
        atomicAdd(acc.element(S_LO), q.bitAnd(int(1023)));
        atomicAdd(acc.element(S_N), int(1));
        // deepest ocean column absorbs the residual: key = depth·64 in high bits, column index low 16 bits
        const key = int(max(w, 0).min(500).mul(64)).shiftLeft(int(16)).bitOr(int(c as unknown as THREE.Node<'int'>));
        atomicMax(acc.element(S_ABS), key);
      });
    })().compute(NCOL);

    this.apply = Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const n = atomicLoad(acc.element(S_N));
      If(n.lessThan(int(1)), () => { Return(); });
      const mean = meanOf(n);
      const w = water.element(c).toVar();
      const s = surfY.element(c).toVar();
      If(isOcean(s, w), () => {
        // + an even share of the pool (whole 2^-10 units; the remainder waits for the next tick)
        const share = float(atomicLoad(acc.element(OCEAN_POOL)).div(n)).div(1024);
        const want = mean.sub(s.add(w)).mul(LEVEL_RATE).add(share);
        // whole quanta, never below zero water
        const dq = iMax(iMin(iMax(int(want.div(Q).round()), int(-MAXQ)), int(MAXQ)), int(w.div(Q).floor()).negate()).toVar();
        water.element(c).assign(w.add(float(dq).mul(Q)));
        // Σ dq can exceed int32 (2^22 per column × 2^16 columns): split as hi·4096 + lo
        atomicAdd(acc.element(S_DQ), dq.shiftRight(int(12)));
        atomicAdd(acc.element(S_DQLO), dq.bitAnd(int(4095)));
      });
    })().compute(NCOL);

    this.balance = Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      const n = atomicLoad(acc.element(S_N));
      If(n.greaterThan(int(0)), () => {
        const col = uint(atomicLoad(acc.element(S_ABS)).bitAnd(int(0xffff)));
        // Σdq should equal what the pool paid out (n·⌊pool/n⌋ units × 128 quanta); the rest is rounding residual
        const paid = float(atomicLoad(acc.element(OCEAN_POOL)).div(n).mul(n)).mul(128);
        const r = float(atomicLoad(acc.element(S_DQ))).mul(4096).add(float(atomicLoad(acc.element(S_DQLO)))).sub(paid);
        water.element(col).assign(water.element(col).sub(r.mul(Q)).max(0));
        const mean = meanOf(n);
        atomicStore(acc.element(S_PREV), int(mean.sub(OFFSET).mul(FIX).round()));
        // pool paid out n·share units (added to the columns via dq); keep the remainder
        const pool = atomicLoad(acc.element(OCEAN_POOL));
        atomicStore(acc.element(OCEAN_POOL), pool.sub(pool.div(n).mul(n)));
      });
    })().compute(1);
  }

  step(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.seep);
    renderer.compute(this.clear);
    renderer.compute(this.reduce);
    renderer.compute(this.apply);
    renderer.compute(this.balance);
  }
}
