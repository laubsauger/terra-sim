// Single source of truth for voxel/column grid -> render world mapping, plus shared TSL samplers
// over the column fields. Every render part (terrain, sides, water, later probe/overlays) goes
// through here so geometry lines up exactly at the cut block edges.
import type * as THREE from 'three/webgpu';
import { uniform, float, int, uint, floor, fract, clamp, max, ceil, step, If } from 'three/tsl';
import { NX, NY, CELL, BLOCK_SIZE, VOXEL_H } from '../sim/layout';
import { tColIdx, tVoxIdx } from '../sim/tslLayout';
import type { GpuFields, StorageNode } from '../core/gpu';

type F = THREE.Node<'float'>;
type I = THREE.Node<'int'>;
type U = THREE.Node<'uint'>;

export const HALF = BLOCK_SIZE / 2;
/** Voxel y rendered at world y=0. Grid centre, so the block sits around the origin. */
export const Y_BASE_OFFSET = NY / 2;
export const VERT_EX_DEFAULT = 1.5;

/** Vertical exaggeration, shared by every render part. Change via setVertEx only. */
export const vertEx = uniform(VERT_EX_DEFAULT);
export function setVertEx(v: number): void { vertEx.value = v; }

// ---- CPU mapping (probe, camera framing, tests) ----
/** Voxel/column index -> world x/z of its centre. */
export const cellToWorld = (i: number) => (i + 0.5) * CELL - HALF;
/** Continuous voxel y (layer k spans [k, k+1)) -> world y. */
export const voxelToWorldY = (y: number) => (y - Y_BASE_OFFSET) * VOXEL_H * vertEx.value;
export const worldToVoxelY = (yw: number) => yw / (VOXEL_H * vertEx.value) + Y_BASE_OFFSET;
/** World x/z -> continuous cell coordinate where column c's centre is at c. */
export const worldToCell = (w: number) => (w + HALF) / CELL - 0.5;

// ---- TSL mapping ----
export const tWorldY = (vy: F): F => vy.sub(Y_BASE_OFFSET).mul(VOXEL_H).mul(vertEx);
export const tVoxelY = (yw: F): F => yw.div(vertEx.mul(VOXEL_H)).add(Y_BASE_OFFSET);
export const tWorldToCell = (w: F): F => w.add(HALF).div(CELL).sub(0.5);
/** Column index containing world x/z, clamped to the block (for reading the cut face's own column). */
export const tWorldToColumn = (w: F): I => int(clamp(floor(w.add(HALF).div(CELL)), 0, NX - 1));

/** One of the four columns around a continuous cell coordinate, with its bilinear weight. */
export interface Corner { x: I; z: I; w: F; surf: F; water: F }

/**
 * Samplers over the column fields. Reads wrap on X,Z (V1) so interpolation across the cut edge
 * uses the columns on the far side: the data stays seamless even though the block is cut.
 */
export function columnSampler(fields: GpuFields) {
  const surf = fields.cur('surfY');
  const water = fields.cur('water');
  const at = (buf: StorageNode, x: I, z: I): F => buf.element(tColIdx(x, z)) as unknown as F;

  function corners(u: F, v: F): Corner[] {
    const u0 = floor(u).toVar(), v0 = floor(v).toVar();
    const fu = fract(u).toVar(), fv = fract(v).toVar();
    const xi = int(u0).toVar(), zi = int(v0).toVar();
    const out: Corner[] = [];
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      const x = xi.add(dx), z = zi.add(dz);
      const wx = dx ? fu : float(1).sub(fu);
      const wz = dz ? fv : float(1).sub(fv);
      out.push({ x, z, w: wx.mul(wz), surf: at(surf, x, z).toVar(), water: at(water, x, z).toVar() });
    }
    return out;
  }

  const sum = (c: Corner[], f: (k: Corner) => F): F => c.slice(1).reduce((a, k) => a.add(f(k)), f(c[0]!));

  /** Bilinear surface height (voxel-y units) — identical formula in terrain mesh and side silhouette. */
  const height = (c: Corner[]): F => sum(c, (k) => k.w.mul(k.surf));

  /**
   * Water level from wet corners only, so a coast does not drag the sheet down onto dry land;
   * the flat level meets rising terrain and the depth test draws the shoreline.
   * `wet` is the bilinear fraction of wet columns (0 = no water anywhere near).
   */
  function level(c: Corner[]): { level: F; wet: F } {
    const wetW = (k: Corner) => k.w.mul(step(0.01, k.water));
    const wet = sum(c, wetW).toVar();
    const lv = sum(c, (k) => wetW(k).mul(k.surf.add(k.water))).div(max(wet, 1e-4));
    return { level: wet.greaterThan(1e-4).select(lv, height(c).sub(2)), wet };
  }

  return { corners, height, level, surfAt: (x: I, z: I) => at(surf, x, z), waterAt: (x: I, z: I) => at(water, x, z) };
}

/**
 * Voxel read that follows the 'vox' ping-pong parity without rebuilding materials:
 * both buffers are bound, and a uniform refreshed on every render (onRenderUpdate) picks one.
 * Needs a Fn() stack (uses If). One real load per call; the branch is uniform so it is coherent.
 */
export function voxReader(fields: GpuFields) {
  const [a, b] = fields.pair<'uint'>('vox');
  const parity = uniform(0, 'uint').onRenderUpdate(() => fields.parity('vox'));
  return (x: I, y: I, z: I): U => {
    const idx = tVoxIdx(x, int(clamp(float(y), 0, NY - 1)), z);
    const v = uint(0).toVar();
    If(parity.equal(uint(0)), () => { v.assign(a.element(idx)); }).Else(() => { v.assign(b.element(idx)); });
    return v;
  };
}

/** Index of the top solid voxel of a column from its surfY (fill 255 gives surfY = k+1, still layer k). */
export const tTopVoxel = (surfY: F): I => int(clamp(ceil(surfY).sub(1), 0, NY - 1));
