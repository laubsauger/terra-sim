// Lower-crustal flow: thick continental roots spread into thinner neighbours, one layer per face per run.
// Into continental neighbours it turns collision fronts into plateaus (B5, V24); into oceanic neighbours it
// stretches passive margins into thin continental wedges instead of cliffs that slump into the sea (B9).
// Race-free: an "out" kernel decides per-face transfers from pre-pass data, an "apply" kernel moves whole
// root layers (GNEISS ↔ PERIDOTITE at the crust base). Mass moves in full layers, exactly (V3).
import type * as THREE from 'three/webgpu';
import { Fn, If, Loop, int, uint, instanceIndex, Return, select } from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NCOL, NY, Mat, FLAG_CONTINENTAL, packVoxel } from './layout';
import { tColIdx, tColXZ, tMat, tFill, tVoxIdx, uMin } from './tslLayout';
import { COL_CONTINENTAL } from './derive';

/**
 * Thickness difference (layers) that drives one layer of flow across a face. Isostasy turns a thickness
 * gradient of FLOW_DIFF into ≈ 0.5·FLOW_DIFF layers of surface slope per cell, kept below the thermal
 * talus (2.5) so stretched margins stop slumping (B9).
 */
export const FLOW_DIFF = 4;
/** Crust never drains below this thickness into continental neighbours (collision plateaus). */
export const FLOW_MIN_THICK = 30;
/** Only continental crust thicker than this stretches into oceanic neighbours. */
export const FLOW_MIN_THICK_MARGIN = 24;
/**
 * Oceanic receivers only accept margin crust while thinner than this. Without the cap every new margin column
 * grew into a sender and continents pancaked over the oceans to ~24 layers, which isostasy floats right at sea
 * level: the whole world flattened to just below the water surface (B15).
 */
export const MARGIN_RECEIVER_MAX = 14;
/** Receiver never grows past this (layers). */
export const FLOW_MAX_THICK = 100;

const DIRS: readonly [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

export class CrustFlow {
  private out: [THREE.ComputeNode, THREE.ComputeNode];
  private apply: [THREE.ComputeNode, THREE.ComputeNode];

  constructor(private fields: GpuFields) {
    const [a, b] = fields.pair<'uint'>('vox');
    this.out = [this.buildOut(a), this.buildOut(b)];
    this.apply = [this.buildApply(a), this.buildApply(b)];
  }

  /** How many bottom crust layers (≤4) are full, plain crust voxels that may be given away. */
  private static givable(vox: StorageNode<'uint'>, x: THREE.Node<'int'>, z: THREE.Node<'int'>, base: THREE.Node<'uint'>) {
    const n = uint(0).toVar();
    const ok = uint(1).toVar();
    for (let i = 0; i < 4; i++) {
      const v = vox.element(tVoxIdx(x, int(base).add(i), z));
      const m = tMat(v);
      const plain = tFill(v).equal(uint(255)).and(m.notEqual(uint(Mat.MAGMA))).and(m.notEqual(uint(Mat.PERIDOTITE))).and(m.notEqual(uint(Mat.AIR)));
      ok.assign(select(plain, ok, uint(0)));
      n.addAssign(ok);
    }
    return n;
  }

  private buildOut(vox: StorageNode<'uint'>): THREE.ComputeNode {
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const flowOut = this.fields.cur<'uint'>('talusOut'); // erosion scratch, free during tectonics
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const ci = colInfo.element(c).toVar();
      const bits = uint(0).toVar();
      const cont = ci.x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL)).equal(uint(1));
      const base = ci.x.bitAnd(uint(0xff));
      const thick = ci.y.div(uint(255)).toVar();
      If(cont, () => {
        // one layer per qualifying face, never more than the verified plain bottom layers or the thickness floor
        const spare = select(thick.greaterThan(uint(FLOW_MIN_THICK_MARGIN)), thick.sub(uint(FLOW_MIN_THICK_MARGIN)), uint(0));
        const budget = uMin(CrustFlow.givable(vox, x, z, base), spare).toVar();
        for (let d = 0; d < 4; d++) {
          const [dx, dz] = DIRS[d]!;
          const n = colInfo.element(tColIdx(x.add(dx), z.add(dz))).toVar();
          const ncont = n.x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL)).equal(uint(1));
          const nthick = n.y.div(uint(255));
          const nbase = n.x.bitAnd(uint(0xff));
          // oceanic receivers become transitional continental crust, but only from thick margins
          const minT = select(ncont, uint(FLOW_MIN_THICK), uint(FLOW_MIN_THICK_MARGIN));
          const recvOk = ncont.or(nthick.lessThan(uint(MARGIN_RECEIVER_MAX)));
          If(budget.greaterThan(uint(0)).and(recvOk).and(thick.greaterThan(minT)).and(thick.greaterThan(nthick.add(uint(FLOW_DIFF))))
            .and(nthick.lessThan(uint(FLOW_MAX_THICK))).and(nbase.greaterThan(uint(4))).and(nbase.lessThan(uint(NY))), () => {
            // nbase = NY means the neighbour has no crust at all: never a receiver (B11, root would land at the ceiling)
            bits.assign(bits.bitOr(uint(1 << d)));
            budget.subAssign(1);
          });
        }
      });
      flowOut.element(c).assign(bits);
    })().compute(NCOL);
  }

  private buildApply(vox: StorageNode<'uint'>): THREE.ComputeNode {
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const flowOut = this.fields.cur<'uint'>('talusOut');
    const GNEISS = packVoxel(Mat.GNEISS, 255, 0, FLAG_CONTINENTAL);
    const PERI = packVoxel(Mat.PERIDOTITE, 255, 0, 0);
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const mine = flowOut.element(c).toVar();
      const loss = uint(0).toVar();
      Loop(4, ({ i }) => { loss.addAssign(mine.shiftRight(uint(i)).bitAnd(uint(1))); });
      // a neighbour sending toward me: its face bit pointing back at me
      const gain = uint(0).toVar();
      for (let d = 0; d < 4; d++) {
        const [dx, dz] = DIRS[d]!;
        const back = d ^ 1; // opposite face index
        const n = flowOut.element(tColIdx(x.add(dx), z.add(dz)));
        gain.addAssign(n.shiftRight(uint(back)).bitAnd(uint(1)));
      }
      const base = int(colInfo.element(c).x.bitAnd(uint(0xff)));
      // remove `loss` bottom layers, then add `gain` layers below: net shift of the crust base
      If(loss.greaterThan(gain), () => {
        Loop({ start: int(0), end: int(loss.sub(gain)), condition: '<' }, ({ i }) => {
          vox.element(tVoxIdx(x, base.add(i), z)).assign(uint(PERI));
        });
      }).ElseIf(gain.greaterThan(loss), () => {
        Loop({ start: int(1), end: int(gain.sub(loss)).add(1), condition: '<' }, ({ i }) => {
          vox.element(tVoxIdx(x, base.sub(i), z)).assign(uint(GNEISS));
        });
      });
    })().compute(NCOL);
  }

  run(renderer: THREE.WebGPURenderer): void {
    const p = this.fields.parity('vox') as 0 | 1;
    renderer.compute(this.out[p]);
    renderer.compute(this.apply[p]);
  }
}
