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
const FIX = 1024;        // level quantum for the mean (1/1024 voxel), summed exactly as hi/lo int32 pairs
const OFFSET = 64;       // levels measured from y = 64 → |q| ≤ 2^16
const Q = 2 ** -17;      // water quantum for exact adds
const MAXQ = 2 ** 22;    // |Δ| ≤ 32 voxels per column per tick in quanta

// oceanSum slots
const S_HI = 0, S_N = 1, S_PREV = 2, S_DQ = 3, S_ABS = 4, S_DQLO = 5, S_LO = 6;

export function registerOceanFields(f: GpuFields): void {
  f.add('oceanSum', 'int', 8, { atomic: true });
}

export class OceanLevel {
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
        const key = int(max(w, 0).min(500).mul(64)).shiftLeft(uint(16)).bitOr(int(c as unknown as THREE.Node<'int'>));
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
        const want = mean.sub(s.add(w)).mul(LEVEL_RATE);
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
        const r = float(atomicLoad(acc.element(S_DQ))).mul(4096).add(float(atomicLoad(acc.element(S_DQLO))));
        water.element(col).assign(water.element(col).sub(r.mul(Q)).max(0));
        const mean = meanOf(n);
        atomicStore(acc.element(S_PREV), int(mean.sub(OFFSET).mul(FIX).round()));
      });
    })().compute(1);
  }

  step(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.clear);
    renderer.compute(this.reduce);
    renderer.compute(this.apply);
    renderer.compute(this.balance);
  }
}
