// TSL helpers shared by magma.ts and lava.ts: deterministic dither, lava (mass, temp|silica) packing.
import type * as THREE from 'three/webgpu';
import { float, uint, uvec2, floor, max, clamp, round } from 'three/tsl';
import { Mat } from './layout';
import { uMin } from './tslLayout';

type U = THREE.Node<'uint'>;
type F = THREE.Node<'float'>;

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

// lava.y = temp °C × 16 (bits 0-15) | silica code 0..255 (bits 16-23). lava.x = mass, fill units.
export const tLavaTemp = (y: U) => float(y.bitAnd(uint(0xffff))).div(16);
export const tLavaSil = (y: U) => y.shiftRight(uint(16)).bitAnd(uint(0xff));
export const tLavaY = (tC: F, sil: U) => uint(clamp(round(tC.mul(16)), 0, 65535)).bitOr(sil.bitAnd(uint(0xff)).shiftLeft(uint(16)));
/** Lava thickness in voxel layers (render: surface = surfY + this). */
export const tLavaDepth = (lava: THREE.Node<'uvec2'>) => float(lava.x).div(255);

/** Mass-weighted mix of (m0, y0) with an inflow (m1, t1, s1). Returns the new packed lava value. */
export function tLavaMix(m0: U, y0: U, m1: U, t1: F, s1: U): THREE.Node<'uvec2'> {
  const mt = m0.add(m1);
  const w = float(m1).div(max(float(mt), 1));
  const t = tLavaTemp(y0).add(t1.sub(tLavaTemp(y0)).mul(w));
  const s = float(tLavaSil(y0)).add(float(s1).sub(float(tLavaSil(y0))).mul(w));
  return uvec2(mt, tLavaY(t, uint(round(s))));
}
