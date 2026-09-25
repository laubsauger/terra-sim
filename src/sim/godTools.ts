// God tools (§T.48). All edits go through the event path (V16) and move mass through the budgets:
//   uplift / subsidence: add / remove crustal root layers at the base (reservoir pays / receives), isostasy
//                        then raises or lowers the surface naturally
//   meteor:              excavate a bowl (mass to reservoir) and lay an ejecta rim of SEDIMENT (from reservoir)
//   storm:               condense vapor over an area into rain (water budget internal move)
// Kernels are per column, in place on the current vox buffer (no neighbour reads → race-free).
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, int, uint, uniform, instanceIndex, Return, sqrt, max, atomicAdd } from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NCOL, NX, NZ, NY, Mat, FLAG_CONTINENTAL, packVoxel } from './layout';
import { tColXZ, tVoxIdx, tMat, tFill, tPack, tAge, tFlags, uMin, iMin, iMax } from './tslLayout';
import { CTR_RESERVOIR } from './fields';

const wrapD = (a: THREE.Node<'float'>, n: number) => a.add(n / 2).mod(n).sub(n / 2);

export class GodTools {
  private cx = uniform(0); private cz = uniform(0); private radius = uniform(8); private strength = uniform(1);
  private brush: [THREE.ComputeNode, THREE.ComputeNode];
  private crater: [THREE.ComputeNode, THREE.ComputeNode];
  private storm: THREE.ComputeNode;

  constructor(private fields: GpuFields) {
    const [a, b] = fields.pair<'uint'>('vox');
    this.brush = [this.buildBrush(a), this.buildBrush(b)];
    this.crater = [this.buildCrater(a), this.buildCrater(b)];
    this.storm = this.buildStorm();
  }

  /** Falloff 0..1 at this column (smooth dome inside radius, torus-aware distance). */
  private falloff(x: THREE.Node<'int'>, z: THREE.Node<'int'>) {
    const dx = wrapD(float(x).sub(this.cx), NX), dz = wrapD(float(z).sub(this.cz), NZ);
    const d = sqrt(dx.mul(dx).add(dz.mul(dz))).div(this.radius);
    return max(float(0), float(1).sub(d.mul(d)));
  }

  private buildBrush(vox: StorageNode<'uint'>): THREE.ComputeNode {
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const ctr = this.fields.cur<'int'>('counters');
    const GNEISS = packVoxel(Mat.GNEISS, 255, 0, FLAG_CONTINENTAL);
    const PERI = packVoxel(Mat.PERIDOTITE, 255, 0, 0);
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const layers = int(this.falloff(x, z).mul(this.strength).round()).toVar(); // + uplift, − subsidence
      If(layers.equal(int(0)), () => { Return(); });
      const base = int(colInfo.element(c).x.bitAnd(uint(0xff)));
      If(layers.greaterThan(int(0)), () => {
        const n = iMin(layers, base.sub(1)).toVar(); // root grows down, stays above y=0
        Loop({ start: int(1), end: n.add(1), condition: '<' }, ({ i }) => { vox.element(tVoxIdx(x, base.sub(i), z)).assign(uint(GNEISS)); });
        atomicAdd(ctr.element(CTR_RESERVOIR), n.mul(-255));
      }).Else(() => {
        // remove only full, plain crust layers from the base, keeping ≥ 4 layers of crust
        const thick = int(colInfo.element(c).y.div(uint(255)));
        const want = iMin(layers.negate(), iMax(int(0), thick.sub(4))).toVar();
        const done = int(0).toVar();
        Loop({ start: int(0), end: want, condition: '<' }, ({ i }) => {
          const v = vox.element(tVoxIdx(x, base.add(i), z));
          const m = tMat(v);
          If(done.equal(i).and(tFill(v).equal(uint(255))).and(m.notEqual(uint(Mat.MAGMA))).and(m.notEqual(uint(Mat.AIR))).and(m.notEqual(uint(Mat.PERIDOTITE))), () => {
            vox.element(tVoxIdx(x, base.add(i), z)).assign(uint(PERI));
            done.addAssign(1);
          });
        });
        atomicAdd(ctr.element(CTR_RESERVOIR), done.mul(255));
      });
    })().compute(NCOL);
  }

  private buildCrater(vox: StorageNode<'uint'>): THREE.ComputeNode {
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const ctr = this.fields.cur<'int'>('counters');
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const dx = wrapD(float(x).sub(this.cx), NX), dz = wrapD(float(z).sub(this.cz), NZ);
      const d = sqrt(dx.mul(dx).add(dz.mul(dz))).div(this.radius);
      If(d.greaterThan(1.6), () => { Return(); });
      const top = int(colInfo.element(c).x.shiftRight(uint(8)).bitAnd(uint(0xff))).toVar();
      const base = int(colInfo.element(c).x.bitAnd(uint(0xff)));
      If(d.lessThan(1), () => {
        // bowl: dig (1 - d²)·depth layers of crust from the top, never below base + 2
        const depth = int(float(1).sub(d.mul(d)).mul(this.strength).mul(this.radius).mul(0.5).round());
        const dig = iMin(depth, iMax(int(0), top.sub(base).sub(2))).toVar();
        const removed = int(0).toVar();
        Loop({ start: int(0), end: dig, condition: '<' }, ({ i }) => {
          const vi = tVoxIdx(x, top.sub(i), z);
          const v = vox.element(vi);
          const m = tMat(v);
          If(m.notEqual(uint(Mat.PERIDOTITE)).and(m.notEqual(uint(Mat.MAGMA))), () => {
            removed.addAssign(int(tFill(v)));
            vox.element(vi).assign(uint(0));
          });
        });
        atomicAdd(ctr.element(CTR_RESERVOIR), removed);
      }).Else(() => {
        // ejecta rim: up to 3 layers of loose sediment, drawn from the reservoir
        const rim = float(1).sub(d.sub(1).div(0.6)).clamp(0, 1);
        const add = uMin(uint(rim.mul(this.strength).mul(3).round()), uint(NY - 2).sub(uMin(uint(top), uint(NY - 2)))).toVar();
        // fill the partial top voxel first so only the top stays partial
        const tv = vox.element(tVoxIdx(x, top, z)).toVar();
        const gap = uint(255).sub(tFill(tv));
        vox.element(tVoxIdx(x, top, z)).assign(tPack(tMat(tv), uint(255), tAge(tv), tFlags(tv)));
        Loop({ start: int(1), end: int(add).add(1), condition: '<' }, ({ i }) => {
          vox.element(tVoxIdx(x, top.add(i), z)).assign(uint(packVoxel(Mat.SEDIMENT, 255, 0, 0)));
        });
        If(add.greaterThan(uint(0)).or(gap.greaterThan(uint(0))), () => { atomicAdd(ctr.element(CTR_RESERVOIR), int(add.mul(255).add(gap)).negate()); });
      });
    })().compute(NCOL);
  }

  private buildStorm(): THREE.ComputeNode {
    const vapor = this.fields.cur('vapor');
    const water = this.fields.cur('water');
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const f = this.falloff(x, z).mul(this.strength).clamp(0, 1);
      If(f.greaterThan(0), () => {
        const a = vapor.element(c).mul(f).toVar();
        vapor.element(c).subAssign(a);
        water.element(c).addAssign(a);
      });
    })().compute(NCOL);
  }

  private set(x: number, z: number, radius: number, strength: number) {
    this.cx.value = x; this.cz.value = z; this.radius.value = Math.max(1, radius); this.strength.value = strength;
  }

  /** Caller must run derive afterwards (colInfo/surfY stale). */
  uplift(r: THREE.WebGPURenderer, x: number, z: number, radius: number, layers: number): void {
    this.set(x, z, radius, layers);
    r.compute(this.brush[this.fields.parity('vox') as 0 | 1]);
  }
  meteor(r: THREE.WebGPURenderer, x: number, z: number, radius: number, strength = 1): void {
    this.set(x, z, radius, strength);
    r.compute(this.crater[this.fields.parity('vox') as 0 | 1]);
  }
  rainStorm(r: THREE.WebGPURenderer, x: number, z: number, radius: number, strength = 0.8): void {
    this.set(x, z, radius, strength);
    r.compute(this.storm);
  }
}
