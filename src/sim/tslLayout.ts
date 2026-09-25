// TSL mirrors of layout.ts helpers. Plain builders (inline into kernels), not Fn() calls.
import { uint, int, select } from 'three/tsl';
import type * as THREE from 'three/webgpu';
import { NX, NZ, NCOL } from './layout';

type U = THREE.Node<'uint'>;
type I = THREE.Node<'int'>;

export const tMat = (v: U) => v.bitAnd(uint(0xff));
export const tFill = (v: U) => v.shiftRight(uint(8)).bitAnd(uint(0xff));
export const tAge = (v: U) => v.shiftRight(uint(16)).bitAnd(uint(0xff));
export const tFlags = (v: U) => v.shiftRight(uint(24)).bitAnd(uint(0xff));

export const tPack = (mat: U, fill: U, age: U, flags: U) =>
  mat.bitAnd(uint(0xff))
    .bitOr(fill.bitAnd(uint(0xff)).shiftLeft(uint(8)))
    .bitOr(age.bitAnd(uint(0xff)).shiftLeft(uint(16)))
    .bitOr(flags.bitAnd(uint(0xff)).shiftLeft(uint(24)));

/** Torus wrap (V1): int & (N-1) handles negatives since N is a power of two. */
export const tWrapX = (x: I) => x.bitAnd(int(NX - 1));
export const tWrapZ = (z: I) => z.bitAnd(int(NZ - 1));
export const tColIdx = (x: I, z: I) => uint(tWrapX(x).add(tWrapZ(z).mul(int(NX))));
export const tVoxIdx = (x: I, y: I, z: I) => tColIdx(x, z).add(uint(y).mul(uint(NCOL)));

/** Column coords from a column-sized dispatch index. */
export const tColXZ = (i: U) => ({ x: int(i.bitAnd(uint(NX - 1))), z: int(i.shiftRight(uint(Math.log2(NX)))) });

// @types/three only types min/max for float vectors; integer versions via select.
export const uMin = (a: U, b: U) => select(a.lessThan(b), a, b);
export const uMax = (a: U, b: U) => select(a.greaterThan(b), a, b);
export const iMin = (a: I, b: I) => select(a.lessThan(b), a, b);
export const iMax = (a: I, b: I) => select(a.greaterThan(b), a, b);
