// TSL helpers shared by magma.ts and lava.ts: deterministic dither, lava / pending packing, column top edits.
import type * as THREE from 'three/webgpu';
import { float, int, uint, uvec2, floor, max, clamp, round, If, Loop, Break } from 'three/tsl';
import { Mat, NY } from './layout';
import { tMat, tFill, tAge, tFlags, tPack, tVoxIdx, uMin } from './tslLayout';

type U = THREE.Node<'uint'>;
type I = THREE.Node<'int'>;
type F = THREE.Node<'float'>;
type Vox = THREE.StorageBufferNode<'uint'>;

export const isCrust = (m: U) => m.notEqual(uint(Mat.AIR)).and(m.notEqual(uint(Mat.PERIDOTITE))).and(m.notEqual(uint(Mat.MAGMA)));

/** Integer hash of (column, tick, salt) → [0,1). Deterministic (V2): no state, same on every run. */
export function tHash01(a: U, tick: U, salt: number): F {
  const h = a.mul(uint(747796405)).add(tick.mul(uint(2891336453))).add(uint(Math.imul(salt, 0x9e3779b9) >>> 0)).toVar();
  h.assign(h.bitXor(h.shiftRight(uint(16))).mul(uint(0x7feb352d)));
  h.assign(h.bitXor(h.shiftRight(uint(15))).mul(uint(0x846ca68b)));
  h.assign(h.bitXor(h.shiftRight(uint(16))));
  return float(h.shiftRight(uint(8))).mul(1 / 16777216);
}

/** float ≥ 0 → integer units, dithered so small rates still act on average, capped. */
export const tQuant = (v: F, dither: F, cap: U) => uMin(uint(floor(max(v, 0).add(dither))), cap);

// lava.y = temp °C × 16 (bits 0-15) | silica code (16-23) | channel memory 0..255 (24-31). lava.x = mass, fill units.
export const tLavaTemp = (y: U) => float(y.bitAnd(uint(0xffff))).div(16);
export const tLavaSil = (y: U) => y.shiftRight(uint(16)).bitAnd(uint(0xff));
export const tLavaMem = (y: U) => y.shiftRight(uint(24));
/** temp + silica bits only (memory bits 0). */
export const tLavaY = (tC: F, sil: U) => uint(clamp(round(tC.mul(16)), 0, 65535)).bitOr(sil.bitAnd(uint(0xff)).shiftLeft(uint(16)));
/** Lava thickness in voxel layers (render: surface = surfY + this). */
export const tLavaDepth = (lava: THREE.Node<'uvec2'>) => float(lava.x).div(255);

/** Mass-weighted mix of (m0, y0) with an inflow (m1, t1, s1); keeps y0's channel memory. */
export function tLavaMix(m0: U, y0: U, m1: U, t1: F, s1: U): THREE.Node<'uvec2'> {
  const mt = m0.add(m1);
  const w = float(m1).div(max(float(mt), 1));
  const t = tLavaTemp(y0).add(t1.sub(tLavaTemp(y0)).mul(w));  // callers pass vars (no voxel dependence)
  const s = float(tLavaSil(y0)).add(float(s1).sub(float(tLavaSil(y0))).mul(w));
  return uvec2(mt, tLavaY(t, uint(round(s))).bitOr(y0.bitAnd(uint(0xff000000))));
}

// volcano.w (f32, exact integers < 2^24) = crater carve this tick (units·2 + felsic, bits 0-10)
//   | silica code of the pending melt batch (bits 11-18) | pending batch is hotspot-derived (bit 19).
export const tVwCarve = (w: F) => uint(w).bitAnd(uint(2047));
export const tVwSil = (w: F) => uint(w).shiftRight(uint(11)).bitAnd(uint(0xff));
export const tVwHot = (w: F) => uint(w).shiftRight(uint(19)).bitAnd(uint(1));
export const tVwPack = (carve: U, sil: U, hot: U) => float(carve.bitAnd(uint(2047)).bitOr(sil.bitAnd(uint(0xff)).shiftLeft(uint(11))).bitOr(hot.shiftLeft(uint(19))));

/** Scan down for the top solid voxel: y (-1 if empty) and its packed value. */
export function tScanTop(vox: Vox, x: I, z: I) {
  const topY = int(-1).toVar();
  const topV = uint(0).toVar();
  Loop({ start: int(NY - 1), end: int(0), condition: '>=' }, ({ i }) => {
    const v = vox.element(tVoxIdx(x, i, z));
    If(tMat(v).notEqual(uint(Mat.AIR)), () => { topY.assign(i); topV.assign(v); Break(); });
  });
  return { topY, topV };
}

/**
 * Add n fill units of rock on a column: top up a partial crust top voxel (keeps its material, erosion.ts
 * convention: only the top voxel may be partial), then stack up to maxNew voxels of `mat` (≤ NY-2).
 * topY/topV are updated in place. Returns units placed.
 */
export function tAddRock(vox: Vox, x: I, z: I, topY: I, topV: U, nIn: U, mat: U, flags: U, maxNew: number): U {
  // snapshot inputs: TSL inlines expressions, and they may depend on the voxels edited below
  const n = uint(nIn).toVar();
  const rem = n.toVar();
  If(rem.greaterThan(uint(0)).and(topY.greaterThanEqual(int(0))).and(isCrust(tMat(topV))).and(tFill(topV).lessThan(uint(255))), () => {
    const t = uMin(rem, uint(255).sub(tFill(topV))).toVar();
    topV.assign(tPack(tMat(topV), tFill(topV).add(t), tAge(topV), tFlags(topV)));
    vox.element(tVoxIdx(x, topY, z)).assign(topV);
    rem.subAssign(t);
  });
  Loop(maxNew, () => {
    If(rem.greaterThan(uint(0)).and(topY.add(int(1)).lessThanEqual(int(NY - 2))), () => {
      topY.addAssign(1);
      const put = uMin(rem, uint(255));
      topV.assign(tPack(mat, put, uint(0), flags));
      vox.element(tVoxIdx(x, topY, z)).assign(topV);
      rem.subAssign(put);
    });
  });
  return n.sub(rem);
}

/** Remove up to n (≤ 3·255) fill units of crust from the column top, never at or below yMin. Returns units removed. */
export function tRemoveTop(vox: Vox, x: I, z: I, topY: I, nIn: U, yMinIn: I): U {
  const n = uint(nIn).toVar();
  const yMin = int(yMinIn).toVar();
  const rem = n.toVar();
  const y = topY.toVar();
  Loop(4, () => {
    If(rem.greaterThan(uint(0)).and(y.greaterThan(yMin)).and(y.greaterThanEqual(int(1))), () => {
      const idx = tVoxIdx(x, y, z).toVar();
      const v = vox.element(idx).toVar();
      If(isCrust(tMat(v)).not(), () => { y.assign(int(-1)); }).Else(() => {
        const f = tFill(v).toVar();
        const t = uMin(rem, f);
        const nf = f.sub(t).toVar();
        rem.subAssign(t);
        If(nf.equal(uint(0)), () => { vox.element(idx).assign(uint(0)); y.subAssign(int(1)); })
          .Else(() => { vox.element(idx).assign(tPack(tMat(v), nf, tAge(v), tFlags(v))); });
      });
    });
  });
  return n.sub(rem);
}
