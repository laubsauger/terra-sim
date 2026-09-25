// §T.26 slow rock cycle, per voxel, in place, every DIAG_EVERY ticks:
//   aging: log-scale age code advances stochastically (deterministic hash) so expected age tracks real time
//   diagenesis: buried loose SEDIMENT lithifies by environment → LIMESTONE (warm shallow sea),
//               SHALE (deep water), SANDSTONE (land / cold shallows)
//   metamorphism: depth below surface ≈ pressure/temperature → SHALE→SCHIST→GNEISS, SANDSTONE→GNEISS
// Material changes keep fill and flags, so crust mass is untouched (V3). Ids append-only (V21).
import type * as THREE from 'three/webgpu';
import { Fn, If, float, uint, uniform, instanceIndex, Return, select, exp2 } from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NCOL, NVOX, Mat } from './layout';
import { tMat, tFill, tAge, tFlags, tPack } from './tslLayout';

export const DIAG_EVERY = 20;          // ticks (1 My at dtGeo 0.05)
export const LITHIFY_DEPTH = 3;        // layers of cover before loose sediment turns to rock
export const SCHIST_DEPTH = 26;
export const GNEISS_DEPTH = 40;

/** 32-bit integer hash (PCG-style) for deterministic per-voxel randomness. */
const hash = (v: THREE.Node<'uint'>) => {
  const s = v.mul(uint(747796405)).add(uint(2891336453));
  const w = s.shiftRight(s.shiftRight(uint(28)).add(uint(4))).bitXor(s).mul(uint(277803737));
  return w.shiftRight(uint(22)).bitXor(w);
};

export class Diagenesis {
  private k: [THREE.ComputeNode, THREE.ComputeNode];
  private tick = uniform(0, 'uint');
  private elapsed = uniform(0); // My since last run

  constructor(private fields: GpuFields) {
    const [a, b] = fields.pair<'uint'>('vox');
    this.k = [this.build(a), this.build(b)];
  }

  private build(vox: StorageNode<'uint'>): THREE.ComputeNode {
    const surfY = this.fields.cur('surfY');
    const water = this.fields.cur('water');
    const temp = this.fields.cur('surfTemp');
    const { tick, elapsed } = this;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NVOX)), () => { Return(); });
      const v = vox.element(i).toVar();
      const m = tMat(v).toVar();
      If(m.equal(uint(Mat.AIR)).or(m.equal(uint(Mat.PERIDOTITE))).or(m.equal(uint(Mat.MAGMA))), () => { Return(); });
      const c = i.bitAnd(uint(NCOL - 1));
      const y = float(i.shiftRight(uint(Math.log2(NCOL))));
      const depth = surfY.element(c).sub(y).sub(1); // full layers of cover above this voxel
      const age = tAge(v).toVar();
      // aging: E[code step] matches elapsed / (age(c+1) - age(c)), age(c) = 2^(c/16) - 1
      const span = exp2(float(age.add(1)).div(16)).sub(exp2(float(age).div(16)));
      const p = elapsed.div(span);
      const r = float(hash(i.bitXor(tick.mul(uint(0x9e3779b9))))).div(4294967296.0);
      If(r.lessThan(p).and(age.lessThan(uint(255))), () => { age.addAssign(1); });
      // diagenesis of buried loose sediment
      If(m.equal(uint(Mat.SEDIMENT)).and(depth.greaterThanEqual(LITHIFY_DEPTH)), () => {
        const w = water.element(c), t = temp.element(c);
        const shallowWarm = w.greaterThan(1).and(w.lessThan(10)).and(t.greaterThan(18));
        m.assign(select(shallowWarm, uint(Mat.LIMESTONE), select(w.greaterThan(10), uint(Mat.SHALE), uint(Mat.SANDSTONE))));
      });
      // burial metamorphism
      If(depth.greaterThanEqual(GNEISS_DEPTH).and(m.equal(uint(Mat.SCHIST)).or(m.equal(uint(Mat.SANDSTONE)))), () => { m.assign(uint(Mat.GNEISS)); });
      If(depth.greaterThanEqual(SCHIST_DEPTH).and(m.equal(uint(Mat.SHALE))), () => { m.assign(uint(Mat.SCHIST)); });
      vox.element(i).assign(tPack(m, tFill(v), age, tFlags(v)));
    })().compute(NVOX);
  }

  /** Run if due. In place on the current vox buffer (one thread per voxel, no neighbours → no races). */
  step(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number): boolean {
    if (tick % DIAG_EVERY !== 0) return false;
    this.tick.value = tick >>> 0;
    this.elapsed.value = DIAG_EVERY * dtGeo;
    renderer.compute(this.k[this.fields.parity('vox') as 0 | 1]);
    return true;
  }
}
