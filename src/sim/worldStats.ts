// Per-window world statistics into integer counters (fixed-point, V2): emergent sea level, land fraction,
// water budget terms. Read back with the window snapshot like everything else.
import type * as THREE from 'three/webgpu';
import { Fn, If, int, uint, instanceIndex, Return, atomicAdd } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL } from './layout';
import { CTR_WORLD } from './fields';

/** Columns deeper than this count as open ocean for the sea-level estimate. */
export const OCEAN_DEPTH = 3;

export function createWorldStats(fields: GpuFields): THREE.ComputeNode {
  const surfY = fields.cur('surfY');
  const water = fields.cur('water');
  const vapor = fields.cur('vapor');
  const ice = fields.cur('ice');
  const ctr = fields.cur<'int'>('counters');
  return Fn(() => {
    const c = instanceIndex;
    If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const w = water.element(c).toVar();
    If(w.greaterThan(OCEAN_DEPTH), () => {
      atomicAdd(ctr.element(CTR_WORLD), int(surfY.element(c).add(w).mul(16).round()));
      atomicAdd(ctr.element(CTR_WORLD + 1), int(1));
    });
    If(w.lessThan(0.5), () => { atomicAdd(ctr.element(CTR_WORLD + 2), int(1)); });
    atomicAdd(ctr.element(CTR_WORLD + 3), int(w.mul(64).round()));
    atomicAdd(ctr.element(CTR_WORLD + 4), int(vapor.element(c).add(ice.element(c)).mul(64).round()));
  })().compute(NCOL);
}

export interface WorldStats { seaLevel: number; landFrac: number; water: number; vaporIce: number }

export function parseWorldStats(buf: ArrayBuffer): WorldStats {
  const a = new Int32Array(buf);
  const n = a[CTR_WORLD + 1]!;
  return {
    seaLevel: n > 0 ? a[CTR_WORLD]! / 16 / n : NaN,
    landFrac: a[CTR_WORLD + 2]! / NCOL,
    water: a[CTR_WORLD + 3]! / 64,
    vaporIce: a[CTR_WORLD + 4]! / 64,
  };
}
