// Single source of truth for voxel/column grid -> render world mapping, plus shared TSL samplers
// over the column fields. Every render part (terrain, sides, water, later probe/overlays) goes
// through here so geometry lines up exactly at the cut block edges.
import type * as THREE from 'three/webgpu';
import {
  uniform, float, int, uint, floor, fract, clamp, max, min, ceil, step, mix, If, Fn, Return, vec4, instanceIndex, instancedArray,
  positionWorld, cameraPosition,
} from 'three/tsl';
import { NX, NY, NCOL, CELL, BLOCK_SIZE, VOXEL_H } from '../sim/layout';
import { tColIdx, tVoxIdx, tColXZ } from '../sim/tslLayout';
import type { GpuFields, StorageNode } from '../core/gpu';

type F = THREE.Node<'float'>;
type I = THREE.Node<'int'>;
type U = THREE.Node<'uint'>;

export const HALF = BLOCK_SIZE / 2;
/** Voxel y rendered at world y=0. Grid centre, so the block sits around the origin. */
export const Y_BASE_OFFSET = NY / 2;
export const VERT_EX_DEFAULT = 1.0;
/**
 * Render-only crop: the slice is drawn from this voxel layer up (the lower asthenosphere is cut
 * away) so crust and water dominate the cut faces. Sides, plinth seat and camera keep-out use it.
 */
export const Y_RENDER_BOTTOM = 22;

/** Vertical exaggeration, shared by every render part. Change via setVertEx only. */
export const vertEx = uniform(VERT_EX_DEFAULT);
export function setVertEx(v: number): void { vertEx.value = v; }

/**
 * Ambience clock for waves, caustics and day/night (V17: independent of the geo clock). The GPU
 * only sees it wrapped (V11). Scroll speeds in shaders are multiples of 1/AMB_PERIOD texture
 * periods per second, so the wrap is seamless.
 */
export const AMB_PERIOD = 3600;
export const ambTime = uniform(0);
export function setAmbTime(t: number): void { ambTime.value = ((t % AMB_PERIOD) + AMB_PERIOD) % AMB_PERIOD; }

/** World-space unit vector camera → fragment (TSL's positionWorldDirection is the position's own direction). */
export const viewDirWorld = positionWorld.sub(cameraPosition).normalize() as THREE.Node<'vec3'>;

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
export interface Corner {
  x: I; z: I; w: F;
  /** Render height: surfY smoothed 3×3 (see renderColumns). */
  surf: F;
  /** Raw surfY (voxel indexing: top solid layer). */
  raw: F;
  water: F;
  /** biome id + veg fraction, or -1 without biome fields (see renderColumns). */
  bio: F;
}

/**
 * Render-only column summary (vec4 per column), recomputed by updateRenderColumns() before a frame:
 * x = surfY smoothed with a 1-2-1 3×3 kernel (wrapping, V1): softens single-cell steps in the
 *     terrain mesh, the cut-face silhouette and the water depth together, so they stay consistent;
 * y = raw surfY, z = water depth (raw level = y + z, so seas stay flat),
 * w = biome id + veg (0..0.999) from the sim 'biome'/'veg' fields, or -1 when they are absent.
 * Allocated once per GpuFields (V10); reads sim fields only (V15).
 */
const renderCols = new WeakMap<GpuFields, { buf: StorageNode<'vec4'>; kernel: THREE.ComputeNode }>();
function renderColumns(fields: GpuFields) {
  let rc = renderCols.get(fields);
  if (rc) return rc;
  activeFields.add(fields);
  const buf = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  buf.setName('renderCols');
  const surf = fields.cur('surfY');
  const water = fields.cur('water');
  const hasBio = fields.names().includes('biome') && fields.names().includes('veg');
  const kernel = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(instanceIndex);
    const acc = float(0).toVar();
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const w = (dx === 0 ? 2 : 1) * (dz === 0 ? 2 : 1) / 16;
      acc.addAssign((surf.element(tColIdx(x.add(dx), z.add(dz))) as unknown as F).mul(w));
    }
    const raw = (surf.element(instanceIndex) as unknown as F).toVar();
    const wd = (water.element(instanceIndex) as unknown as F).toVar();
    const bio = hasBio
      ? float(fields.cur<'uint'>('biome').element(instanceIndex)).add(min(max(fields.cur('veg').element(instanceIndex) as unknown as F, 0), 0.999))
      : float(-1);
    buf.element(instanceIndex).assign(vec4(acc, raw, wd, bio));
  })().compute(NCOL);
  rc = { buf, kernel };
  renderCols.set(fields, rc);
  return rc;
}

/**
 * Render-only heat summary (vec4 per column), refreshed with the render columns:
 * x = crust age (My, current 'crustAge' parity; 1e3 when absent), y = lava thickness (voxel layers),
 * z = lava temperature °C, w = 'ice' (water-eq snow/ice, voxel layers; 0 when absent).
 * A separate kernel keeps each under the storage-buffer limit (V23).
 */
const renderHeats = new WeakMap<GpuFields, { buf: StorageNode<'vec4'>; kernel: THREE.ComputeNode }>();
function renderHeat(fields: GpuFields) {
  let rh = renderHeats.get(fields);
  if (rh) return rh;
  const has = (n: string) => fields.names().includes(n);
  const buf = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  buf.setName('renderHeat');
  const agePair = has('crustAge') ? fields.pair('crustAge') : null;
  const ageParity = uniform(0, 'uint').onRenderUpdate(() => (agePair ? fields.parity('crustAge') : 0));
  const lava = has('lava') ? fields.cur<'uvec2'>('lava') : null;
  const ice = has('ice') ? fields.cur('ice') : null;
  const kernel = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const age = float(1e3).toVar();
    if (agePair) {
      If(ageParity.equal(uint(0)), () => { age.assign(agePair[0].element(instanceIndex) as unknown as F); })
        .Else(() => { age.assign(agePair[1].element(instanceIndex) as unknown as F); });
    }
    const lv = lava ? lava.element(instanceIndex).toVar() : null;
    const depth = lv ? float(lv.x).div(255) : float(0);
    const temp = lv ? float(lv.y.bitAnd(uint(0xffff))).div(16) : float(0);
    const snow = ice ? (ice.element(instanceIndex) as unknown as F) : float(0);
    buf.element(instanceIndex).assign(vec4(age, depth, temp, snow));
  })().compute(NCOL);
  rh = { buf, kernel };
  renderHeats.set(fields, rh);
  return rh;
}

/** Bilinear read of the heat summary at a continuous cell coordinate (wraps, V1). */
export function heatSampler(fields: GpuFields) {
  const { buf } = renderHeat(fields);
  return (u: F, v: F): THREE.Node<'vec4'> => {
    const u0 = floor(u), v0 = floor(v), fu = fract(u), fv = fract(v);
    const xi = int(u0), zi = int(v0);
    const e = (dx: number, dz: number) => buf.element(tColIdx(xi.add(dx), zi.add(dz))) as unknown as THREE.Node<'vec4'>;
    return mix(mix(e(0, 0), e(1, 0), fu), mix(e(0, 1), e(1, 1), fu), fv);
  };
}

/** Refresh the render column + heat summaries from the current sim fields. Call once per frame before rendering. */
export function updateRenderColumns(renderer: THREE.WebGPURenderer, fields: GpuFields): void {
  renderer.compute(renderColumns(fields).kernel);
  renderer.compute(renderHeat(fields).kernel);
}

/** Field sets that have render samplers; the stage refreshes them before every frame. */
const activeFields = new Set<GpuFields>();
/** Refresh every field set any render material samples (stage.ts calls this each frame). */
export function updateAllRenderColumns(renderer: THREE.WebGPURenderer): void {
  for (const f of activeFields) updateRenderColumns(renderer, f);
}

/**
 * Samplers over the render column summary. Reads wrap on X,Z (V1) so interpolation across the cut
 * edge uses the columns on the far side: the data stays seamless even though the block is cut.
 */
export function columnSampler(fields: GpuFields) {
  const { buf } = renderColumns(fields);
  const at = (x: I, z: I) => buf.element(tColIdx(x, z)) as unknown as THREE.Node<'vec4'>;

  function corners(u: F, v: F): Corner[] {
    const u0 = floor(u).toVar(), v0 = floor(v).toVar();
    const fu = fract(u).toVar(), fv = fract(v).toVar();
    const xi = int(u0).toVar(), zi = int(v0).toVar();
    const out: Corner[] = [];
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      const x = xi.add(dx), z = zi.add(dz);
      const wx = dx ? fu : float(1).sub(fu);
      const wz = dz ? fv : float(1).sub(fv);
      const e = at(x, z).toVar();
      out.push({ x, z, w: wx.mul(wz), surf: e.x, raw: e.y, water: e.z, bio: e.w });
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
    const lv = sum(c, (k) => wetW(k).mul(k.raw.add(k.water))).div(max(wet, 1e-4));
    // branch-free (see BRANCH-FREE NOTE below)
    return { level: mix(height(c).sub(2), lv, step(1e-4, wet)), wet };
  }

  return {
    corners, height, level,
    /** Raw surfY of a column (for tTopVoxel). */
    surfAt: (x: I, z: I) => at(x, z).y,
    waterAt: (x: I, z: I) => at(x, z).z,
    /** biome id + veg, or -1 without biome fields. */
    bioAt: (x: I, z: I) => at(x, z).w,
  };
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

/*
 * BRANCH-FREE NOTE: render fragment graphs avoid select()/If outside their own Fn stacks. TSL caches
 * a shared node (toVar or common subexpression) where it is first built; when that is inside a
 * select branch, other branches read it unassigned and a Discard inside it only runs in that branch.
 * So blends use mix() with 0/1 masks (step, float(bool)).
 */

/** Index of the top solid voxel of a column from its surfY (fill 255 gives surfY = k+1, still layer k). */
export const tTopVoxel = (surfY: F): I => int(clamp(ceil(surfY).sub(1), 0, NY - 1));
