// Single source of truth for voxel/column grid -> render world mapping, plus shared TSL samplers
// over the column fields. Every render part (terrain, sides, water, later probe/overlays) goes
// through here so geometry lines up exactly at the cut block edges.
import * as THREE from 'three/webgpu';
const THREE_Vector2 = THREE.Vector2;
import {
  uniform, uniformArray, float, int, uint, floor, fract, clamp, max, min, ceil, step, smoothstep, mix, If, Fn, Return, vec2, vec4, instanceIndex, instancedArray,
  exp,
  positionWorld, cameraPosition,
} from 'three/tsl';
import { NX, NY, NCOL, CELL, BLOCK_SIZE, VOXEL_H, Mat } from '../sim/layout';
import { tColIdx, tVoxIdx, tColXZ, uMin, tMat } from '../sim/tslLayout';
import { ACT_NEW as ACT_NEW_BIT } from '../sim/tectonics';
import { MAX_PLATES as MAX_PLATES_R } from '../sim/worldData';
import { createBiomeNodes, createPaletteNodes, BIOME_SLOTS } from './palette';
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
  /** Render height: display surfY, smoothed 3×3 (5×5 around spikes), eased in time. */
  surf: F;
  /** True surfY of the column (voxel indexing: top solid layer). */
  raw: F;
  /** Display water depth (eased); level = rawEased + water. */
  water: F;
  /** Display (eased) raw surfY, for the water level. */
  rawEased: F;
  /** Display veg 0..1, or -1 without biome fields. */
  veg: F;
}

// ============================================================================================
// Display state (render-only, V15): the sim's column fields are advected with the plate jumps
// and eased in time so the world morphs continuously instead of pulsing on every sim update.
//   cols   = (hs: display surfY, smoothed 3×3 / adaptive 5×5 around spikes; rawEased; water; veg | -1)
//   heat   = (crust age My, lava layers, lava °C, snow cover 0..1 from 'ice' + biome snow)
//   bioG   = (biome ground rgb, arid)          bioV = (biome vegetation rgb, forest)  — colours, not ids
//   volc   = (vent activity 0..1 from 'volcano'.x, lava channel memory 0..1 from lava.y>>24, 0, 0)
//   motion = (plate sub-cell offset x, z in cells (blended across plate boundaries), shift, 0)
//            shift = rawEased − raw: vertical display lag of the column (faces slide with it)
// Per frame: (1) if the sim ran tectonics since last frame, gather every display buffer from the
// 'tecAct' source column (fresh ridge crust snaps); (2) ease toward the current sim values with
// α = 1 − exp(−dt/τ); (3) refresh plate offsets. All buffers allocated once per GpuFields (V10).
// ============================================================================================

/** Display easing time constant (real seconds). */
export const DISPLAY_TAU = 0.45;

/** What the display needs from the sim to follow plate motion (Sim.tectonics provides both). */
export interface RenderMotion {
  /** Increments on every GPU tectonics run ('tecAct' is valid for the latest run). */
  readonly runId: number;
  /** (x, z) sub-cell offset per plate, cells in −1..1, MAX_PLATES pairs. */
  plateOffsets(out: Float32Array): void;
  /** Unwrapped distance moved per plate (cells, MAX_PLATES pairs); eased for display. */
  plateTravel?(out: Float64Array): void;
}

type V4 = THREE.Node<'vec4'>;
interface Display {
  cols: StorageNode<'vec4'>; heat: StorageNode<'vec4'>; bioG: StorageNode<'vec4'>; bioV: StorageNode<'vec4'>;
  volc: StorageNode<'vec4'>; top: StorageNode<'vec4'>; bed: StorageNode<'vec4'>;
  motion: StorageNode<'vec4'>;
  kernels: THREE.ComputeNode[];
  alpha: THREE.UniformNode<'float', number>;
  /** Motion easing weight this frame (per-column offset glide, MOTION_TAU). */
  alphaM: THREE.UniformNode<'float', number>;
  /** 0 no remap, 1 by 'tecAct' (one tectonics run since the last frame), 2 by per-plate shift (several). */
  remapOn: THREE.UniformNode<'uint', number>;
  /** Whole cells each plate moved since the last frame (remap mode 2), and the running (travel − accum). */
  plateShift: THREE.Vector2[];
  intPrev: Float64Array;
  offsets: THREE.Vector2[];
  offsetArr: Float32Array;
  travel: Float64Array;   // sim plate travel (cells)
  travelEased: Float64Array; // shown plate travel, glides over the sim step interval
  travelPrev: Float64Array;
  travelFrom: Float64Array; // shown travel when the latest step arrived
  clock: number; lastStep: number; stepGap: number; // real seconds
  src: RenderMotion | null;
  lastRun: number;
  snap: boolean;
  initialised: boolean;
}
const displays = new WeakMap<GpuFields, Display>();
/**
 * Real seconds over which a column's drawn plate offset follows a sudden change of its target: a split,
 * merge or majority clean moves columns to a plate with another sub-cell offset (up to a cell apart), and a
 * resync after a stall. Short, so it adds no visible lag to normal motion (all columns of a plate lag alike).
 */
const MOTION_TAU = 0.15;
/** Field sets that have render samplers; the stage refreshes them before every frame. */
const activeFields = new Set<GpuFields>();

function display(fields: GpuFields): Display {
  let d = displays.get(fields);
  if (d) return d;
  activeFields.add(fields);
  const has = (n: string) => fields.names().includes(n);
  const mk = (name: string) => { const b = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>; b.setName(name); return b; };
  const cols = mk('dispCols'), heat = mk('dispHeat'), bioG = mk('dispBioG'), bioV = mk('dispBioV'), motion = mk('dispMotion');
  const volc = mk('dispVolc'), sVolc = mk('dispVolcPrev');
  const sCols = mk('dispColsPrev'), sHeat = mk('dispHeatPrev'), sBioG = mk('dispBioGPrev'), sBioV = mk('dispBioVPrev');
  const alpha = uniform(1), alphaM = uniform(1);
  const remapOn = uniform(0, 'uint');
  const plateShift = [...Array(MAX_PLATES_R)].map(() => new THREE_Vector2());
  const shiftU = uniformArray(plateShift, 'vec2');
  const act = has('tecAct') ? fields.cur<'uint'>('tecAct') : null;
  const pidPair = has('plateId') ? fields.pair<'uint'>('plateId') : null;
  const pidParity = uniform(0, 'uint').onRenderUpdate(() => (pidPair ? fields.parity('plateId') : 0));
  const i = instanceIndex;
  const guard = () => If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
  /** Plate id of column j on the current parity (needs a Fn stack). */
  const pidAt = (j: U): U => {
    const idx = j.toVar(); // var before the If: see voxReader
    const pid = uint(0).toVar();
    if (pidPair) If(pidParity.equal(uint(0)), () => { pid.assign(pidPair[0].element(idx)); }).Else(() => { pid.assign(pidPair[1].element(idx)); });
    return pid;
  };
  const wrapD = (d: F): F => d.sub(float(NX).mul(floor(d.div(NX).add(0.5)))) as F;
  /**
   * Where column i's content was last frame: mode 1 from 'tecAct' (exact, flags fresh crust), mode 2 (several
   * runs this frame; 'tecAct' maps only the latest) from the column's plate's total whole-cell shift.
   * Returns (source index, fresh 0|1, displacement dx, dz in cells). Needs a Fn stack.
   */
  const source = () => {
    const srcI = i.toVar(), fresh = float(0).toVar(), k = vec2(0).toVar();
    const { x, z } = tColXZ(i);
    If(remapOn.equal(uint(1)), () => {
      if (!act) return;
      const t = act.element(i).toVar();
      srcI.assign(t.bitAnd(uint(0xffff)));
      fresh.assign(float(t.bitAnd(uint(ACT_NEW_BIT)).notEqual(uint(0))));
      k.assign(vec2(wrapD(float(x).sub(float(srcI.mod(uint(NX))))), wrapD(float(z).sub(float(srcI.div(uint(NX)))))));
    }).ElseIf(remapOn.equal(uint(2)), () => {
      const K = (shiftU.element(uMin(pidAt(i), uint(MAX_PLATES_R - 1))) as unknown as THREE.Node<'vec2'>).toVar();
      srcI.assign(tColIdx(x.sub(int(K.x)), z.sub(int(K.y))));
      k.assign(K);
    });
    return { srcI, fresh, k };
  };

  // (1) remap: scratch[d] = disp[src(d)]; fresh crust (opened at a spreading seam) eases from what was drawn
  // at that spot before, instead of snapping to the sim value (no one-frame pop along the ridges)
  const remap = (a: StorageNode<'vec4'>, b: StorageNode<'vec4'>, sa: StorageNode<'vec4'>, sb: StorageNode<'vec4'>) => Fn(() => {
    guard();
    const { srcI, fresh } = source();
    const va = (a.element(srcI) as unknown as V4).toVar(), vb = (b.element(srcI) as unknown as V4).toVar();
    If(fresh.greaterThan(0.5), () => { va.assign(a.element(i) as unknown as V4); vb.assign(b.element(i) as unknown as V4); });
    sa.element(i).assign(va); sb.element(i).assign(vb);
  })().compute(NCOL);

  const ease = (prev: V4, target: V4): V4 =>
    mix(prev, target, alpha) as unknown as V4;

  // (2a) cols from surfY / water / veg
  const surf = fields.cur('surfY'), water = fields.cur('water');
  const veg = has('veg') ? fields.cur('veg') : null;
  const kCols = Fn(() => {
    guard();
    const { x, z } = tColXZ(i);
    // 3×3 (1-2-1) and 5×5 (1-4-6-4-1) binomial means; the wide one takes over where the column
    // sticks out of its neighbourhood (single-column needles, pinnacles), planar slopes keep 3×3.
    const s3 = float(0).toVar(), s5 = float(0).toVar();
    const b5 = [1, 4, 6, 4, 1];
    for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
      const v = (surf.element(tColIdx(x.add(dx), z.add(dz))) as unknown as F).toVar();
      s5.addAssign(v.mul(b5[dx + 2]! * b5[dz + 2]! / 256));
      if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) s3.addAssign(v.mul((dx === 0 ? 2 : 1) * (dz === 0 ? 2 : 1) / 16));
    }
    const raw = (surf.element(i) as unknown as F).toVar();
    const hs = mix(s3, s5, smoothstep(1.5, 6.0, raw.sub(s5).abs()));
    const vg = veg ? min(max(veg.element(i) as unknown as F, 0), 1) : float(-1);
    cols.element(i).assign(ease(sCols.element(i) as unknown as V4, vec4(hs, raw, water.element(i) as unknown as F, vg)));
  })().compute(NCOL);

  // (2b) heat from crustAge (ping-pong) / lava / ice + biome snow
  const agePair = has('crustAge') ? fields.pair('crustAge') : null;
  const ageParity = uniform(0, 'uint').onRenderUpdate(() => (agePair ? fields.parity('crustAge') : 0));
  const lava = has('lava') ? fields.cur<'uvec2'>('lava') : null;
  const ice = has('ice') ? fields.cur('ice') : null;
  const biome = has('biome') ? fields.cur<'uint'>('biome') : null;
  const bioN = createBiomeNodes();
  const kHeat = Fn(() => {
    guard();
    const age = float(1e3).toVar();
    if (agePair) {
      If(ageParity.equal(uint(0)), () => { age.assign(agePair[0].element(i) as unknown as F); })
        .Else(() => { age.assign(agePair[1].element(i) as unknown as F); });
    }
    const lv = lava ? lava.element(i).toVar() : null;
    const depth = lv ? float(lv.x).div(255) : float(0);
    const temp = lv ? float(lv.y.bitAnd(uint(0xffff))).div(16) : float(0);
    const bId = biome ? int(uMin(biome.element(i) as unknown as U, uint(BIOME_SLOTS - 1))) : int(0);
    const snowBio = biome ? bioN.flags(bId).z : float(0);
    const snow = max(ice ? smoothstep(0.02, 0.3, ice.element(i) as unknown as F) : float(0), snowBio);
    heat.element(i).assign(ease(sHeat.element(i) as unknown as V4, vec4(min(age, 1e3), depth, temp, snow)));
  })().compute(NCOL);

  // (2c) biome colours (blend colours in time, never ids)
  const kBio = Fn(() => {
    guard();
    if (!biome) {
      bioG.element(i).assign(vec4(0.4, 0.35, 0.25, 0)); bioV.element(i).assign(vec4(0.3, 0.5, 0.2, 0));
      return;
    }
    const bId = int(uMin(biome.element(i) as unknown as U, uint(BIOME_SLOTS - 1))).toVar();
    const f = bioN.flags(bId);
    bioG.element(i).assign(ease(sBioG.element(i) as unknown as V4, vec4(bioN.ground(bId), f.x)));
    bioV.element(i).assign(ease(sBioV.element(i) as unknown as V4, vec4(bioN.veg(bId), f.y)));
  })().compute(NCOL);

  // (2e) terrain albedo inputs: colour of the top voxel's material, and of the bedrock under loose
  // sediment (steep faces) with its roughness. Erosion, deposition and lava freezing change the top
  // material within a tick; eased like the rest, the terrain shade blends over DISPLAY_TAU, never flips.
  const top = mk('dispTop'), bed = mk('dispBed'), sTop = mk('dispTopPrev'), sBed = mk('dispBedPrev');
  const voxR = has('vox') ? voxReader(fields) : null;
  const palN = createPaletteNodes();
  const kTop = Fn(() => {
    guard();
    if (!voxR) { top.element(i).assign(vec4(0.3, 0.3, 0.3, 0)); bed.element(i).assign(vec4(0.3, 0.3, 0.3, 0.8)); return; }
    const { x, z } = tColXZ(i);
    const ty = tTopVoxel(surf.element(i) as unknown as F).toVar();
    const m = tMat(voxR(x, ty, z)).toVar();
    const mb = tMat(voxR(x, ty.sub(2), z));
    const bm = m.add(mb.sub(m).mul(uint(m.equal(uint(Mat.SEDIMENT))))); // branch-free select
    top.element(i).assign(ease(sTop.element(i) as unknown as V4, vec4(palN.color(m), 0)));
    bed.element(i).assign(ease(sBed.element(i) as unknown as V4, vec4(palN.color(bm), palN.rough(bm))));
  })().compute(NCOL);

  // (2d) volcano activity + lava channel memory; z, w = spreading-seam glow (see seamGlow) from crust age.
  // Core: a thin crack about a cell wide on the seam axis. Crust ages come quantised by the tectonics
  // steps (the youngest seam cells are 0, 0.25 or 0.5 My depending on the step phase), so the core is
  // measured against the local youngest age over 5×5, e^(−(age − min)/0.12 My) over a 3×3 binomial, and
  // gated by how young that minimum is (active seams only). Halo: e^(−age/1 My) over a 5×5 binomial.
  // Blurred first, the seam is a soft curved line instead of a per-cell staircase.
  const volcF = has('volcano') ? fields.cur<'vec4'>('volcano') : null;
  const kVolc = Fn(() => {
    guard();
    const act0 = volcF ? min(max((volcF.element(i) as unknown as V4).x, 0), 1) : float(0);
    const mem = lava ? float(lava.element(i).y.shiftRight(uint(24))).div(255) : float(0);
    const core = float(0).toVar(), halo = float(0).toVar();
    if (agePair) {
      const { x, z } = tColXZ(i);
      const b5 = [1, 4, 6, 4, 1];
      const seam = (buf: StorageNode<'float'>) => {
        const ages: F[] = [];
        for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) ages.push(max(buf.element(tColIdx(x.add(dx), z.add(dz))) as unknown as F, 0).toVar());
        const m = ages.reduce((acc, a) => min(acc, a) as F).toVar();
        let hs = float(0) as F, cs = float(0) as F;
        for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
          const a = ages[(dz + 2) * 5 + dx + 2]!;
          hs = hs.add(exp(a.div(-1.0)).mul(b5[dx + 2]! * b5[dz + 2]! / 256));
          if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) cs = cs.add(exp(a.sub(m).div(-0.12)).mul((dx === 0 ? 2 : 1) * (dz === 0 ? 2 : 1) / 16));
        }
        halo.assign(hs); core.assign(cs.mul(smoothstep(1.2, 0.5, m)));
      };
      If(ageParity.equal(uint(0)), () => { seam(agePair[0] as unknown as StorageNode<'float'>); })
        .Else(() => { seam(agePair[1] as unknown as StorageNode<'float'>); });
    }
    volc.element(i).assign(ease(sVolc.element(i) as unknown as V4, vec4(act0, mem, core, halo)));
  })().compute(NCOL);
  const remapVolc = Fn(() => {
    guard();
    const { srcI } = source();
    sVolc.element(i).assign(volc.element(srcI));
  })().compute(NCOL);

  // (3) motion: plate offset blended over the 3×3 neighbourhood (no cracks at plate boundaries), eased per
  // column over MOTION_TAU: a column that changes plate (split / merge / clean) slides to its new plate's
  // offset instead of jumping. The eased offset rides the remap like the other display buffers, less the
  // cells the content moved, so a whole-cell plate shift still cancels exactly.
  const offsets = [...Array(MAX_PLATES_R)].map(() => new THREE_Vector2());
  const offU = uniformArray(offsets, 'vec2');
  const sMot = mk('dispMotionPrev');
  const remapMot = Fn(() => {
    guard();
    const { srcI, fresh, k } = source();
    sMot.element(i).assign(vec4((motion.element(srcI) as unknown as V4).xy.sub(k), 0, fresh));
  })().compute(NCOL);
  const kMotion = Fn(() => {
    guard();
    const { x, z } = tColXZ(i);
    const o = vec2(0).toVar();
    if (pidPair) {
      for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
        const pid = pidAt(tColIdx(x.add(dx), z.add(dz)));
        const w = (dx === 0 ? 2 : 1) * (dz === 0 ? 2 : 1) / 16;
        o.addAssign((offU.element(uMin(pid, uint(MAX_PLATES_R - 1))) as unknown as THREE.Node<'vec2'>).mul(w));
      }
    }
    const prev = (sMot.element(i) as unknown as V4).toVar();
    const oe = mix(prev.xy, o, max(alphaM, prev.w));
    const c = (cols.element(i) as unknown as V4);
    motion.element(i).assign(vec4(oe, c.y.sub(surf.element(i) as unknown as F), 0));
  })().compute(NCOL);

  d = {
    cols, heat, bioG, bioV, volc, top, bed, motion, alpha, alphaM, remapOn, plateShift, intPrev: new Float64Array(MAX_PLATES_R * 2), offsets, offsetArr: new Float32Array(MAX_PLATES_R * 2),
    travel: new Float64Array(MAX_PLATES_R * 2), travelEased: new Float64Array(MAX_PLATES_R * 2),
    travelPrev: new Float64Array(MAX_PLATES_R * 2), travelFrom: new Float64Array(MAX_PLATES_R * 2), clock: 0, lastStep: 0, stepGap: 1 / 60,
    kernels: [remap(cols, heat, sCols, sHeat), remap(bioG, bioV, sBioG, sBioV), remap(top, bed, sTop, sBed), remapVolc, remapMot,
      kCols, kHeat, kBio, kTop, kVolc, kMotion],
    src: null, lastRun: -1, snap: true, initialised: false,
  };
  displays.set(fields, d);
  return d;
}

/** Let the display follow plate motion (sub-cell advection + remap on tectonics runs). */
export function setRenderMotion(fields: GpuFields, m: RenderMotion | null): void {
  const d = display(fields);
  d.src = m;
  d.lastRun = m ? m.runId : -1;
}

/** Next update takes the sim state as is (after load, world reset, teleports). */
export function snapRenderColumns(fields: GpuFields): void { display(fields).snap = true; }

/**
 * Refresh the display state from the current sim fields. dt = real seconds since the last frame
 * (eases with DISPLAY_TAU). The stage loop is the one owner of the per-frame update.
 * Without dt this only initialises a display that was never updated and is otherwise a no-op:
 * other modules may call it to make sure the display exists, but must not advance it (a second
 * call per frame used to snap the display to the raw sim state every few frames → flicker of
 * terrain, shadows and cut-face layers). Use snapRenderColumns() + dt for an explicit snap.
 */
export function updateRenderColumns(renderer: THREE.WebGPURenderer, fields: GpuFields, dt?: number): void {
  const d = display(fields);
  if (dt === undefined) {
    if (d.initialised) return;
    d.snap = true;
  }
  d.initialised = true;
  const run = d.src ? d.src.runId : -1;
  const runs = d.src ? run - d.lastRun : 0;
  // several runs since the last frame (high speed): 'tecAct' maps only the latest → remap by plate shift
  d.remapOn.value = runs === 0 ? 0 : runs > 1 && d.src?.plateTravel ? 2 : 1;
  d.lastRun = run;
  const alpha = d.snap || dt === undefined ? 1 : 1 - Math.exp(-Math.max(dt, 0) / DISPLAY_TAU);
  d.alpha.value = alpha;
  d.alphaM.value = alpha === 1 ? 1 : 1 - Math.exp(-Math.max(dt ?? 0, 0) / MOTION_TAU);
  d.snap = false;
  if (d.src) {
    d.src.plateOffsets(d.offsetArr);
    // Plates advance in steps (one per tectonics run, often only every few frames). The display glides
    // linearly from where it is to the new travel over the measured time between steps (constant speed,
    // lag ≈ one step, ≤ ~0.5 cell, so plate boundaries do not tear) and draws at accum + (shown − travel):
    // a whole-cell shift still cancels exactly against the remap. At high speed the sim runs in bursts (one
    // stats window, then a readback wait); the glide spreads those too. > 6 cells behind (load, teleport) → jump.
    const tr = d.travel, te = d.travelEased;
    d.src.plateTravel?.(tr);
    // whole cells each plate moved since the last frame: travel − accum counts the integer shifts
    for (let k = 0; k < tr.length; k++) {
      const n = tr[k]! - d.offsetArr[k]!;
      d.plateShift[k >> 1]![(k & 1) ? 'y' : 'x'] = alpha === 1 ? 0 : Math.round(n - d.intPrev[k]!);
      d.intPrev[k] = n;
    }
    const now = (d.clock += Math.max(dt ?? 0, 0));
    let moved = false;
    for (let k = 0; k < tr.length; k++) if (tr[k] !== d.travelPrev[k]) { moved = true; break; }
    if (moved) {
      d.stepGap += (Math.min(now - d.lastStep, 0.5) - d.stepGap) * 0.3;
      d.lastStep = now - Math.max(dt ?? 0, 0); // the step landed during this frame: glide from the frame start
      d.travelPrev.set(tr);
      d.travelFrom.set(te);
    }
    // glide 20 % longer than the gap: still moving when the next step lands (no stop frame), lag stays bounded
    const gap = Math.min(0.6, Math.max(1 / 60, d.stepGap * 1.2));
    const u = alpha === 1 ? 1 : Math.min(1, (now - d.lastStep) / gap);
    for (let k = 0; k < tr.length; k++) {
      te[k] = Math.abs(tr[k]! - d.travelFrom[k]!) > 6 ? tr[k]! : d.travelFrom[k]! + (tr[k]! - d.travelFrom[k]!) * u;
    }
    for (let k = 0; k < d.offsets.length; k++) {
      d.offsets[k]!.set(d.offsetArr[k * 2]! + te[k * 2]! - tr[k * 2]!, d.offsetArr[k * 2 + 1]! + te[k * 2 + 1]! - tr[k * 2 + 1]!);
    }
  }
  for (const k of d.kernels) renderer.compute(k);
}

/** Refresh every field set any render material samples (stage.ts calls this each frame with its dt). */
export function updateAllRenderColumns(renderer: THREE.WebGPURenderer, dt?: number): void {
  for (const f of activeFields) updateRenderColumns(renderer, f, dt);
}

/** Displayed (eased) plate offsets in cells, [x0, z0, x1, z1, …], for tests and tools. */
export function displayPlateOffsets(fields: GpuFields): number[] {
  return display(fields).offsets.flatMap((v) => [v.x, v.y]);
}

/** Display buffers, for tests and tools (read-only use). */
export function displayBuffers(fields: GpuFields) {
  const d = display(fields);
  return { cols: d.cols, heat: d.heat, bioG: d.bioG, bioV: d.bioV, volc: d.volc, top: d.top, bed: d.bed, motion: d.motion };
}

/** Bilinear read of a display buffer at a continuous cell coordinate (wraps, V1). */
function bilinear(buf: StorageNode<'vec4'>, u: F, v: F): V4 {
  const u0 = floor(u), v0 = floor(v), fu = fract(u), fv = fract(v);
  const xi = int(u0), zi = int(v0);
  const e = (dx: number, dz: number) => buf.element(tColIdx(xi.add(dx), zi.add(dz))) as unknown as V4;
  return mix(mix(e(0, 0), e(1, 0), fu), mix(e(0, 1), e(1, 1), fu), fv) as unknown as V4;
}

/**
 * Sub-cell plate advection (display): content of a column on plate p is drawn at grid + offset(p),
 * so every lookup at a rendered cell coordinate samples the data at (u, v) − offset. Use this for
 * any custom column read (probe, overlays): `const [su, sv] = advect(fields)(u, v)`.
 */
export function advect(fields: GpuFields) {
  const { motion } = display(fields);
  return (u: F, v: F): [F, F] => {
    const o = bilinear(motion, u, v);
    return [u.sub(o.x), v.sub(o.y)];
  };
}

/**
 * Spreading-seam glow from the blurred display seam (volcanoSampler .zw), shared by terrain (rifts on
 * land, seams on the seafloor) and water (lets that light through). Narrow core (a soft ramp on the
 * blurred seam: tightens toward the axis, stays a smooth curve) scaled by `pulse` (slow patchy
 * intensity, see terrain.ts seamPulse), faint wide halo. Returns (core, halo) 0..1.
 */
export function seamGlow(seam: THREE.Node<'vec2'>, pulse: F): THREE.Node<'vec2'> {
  // ≈ 1 cell FWHM: ages grow ~0.25 My per cell off the axis, so a cell away the blurred core is ~0.3
  const core = smoothstep(0.3, 0.55, seam.x);
  return vec2(core.mul(pulse), seam.y.mul(0.05).mul(pulse.mul(0.5).add(0.5))) as unknown as THREE.Node<'vec2'>;
}

/** Bilinear read of the heat display (age, lava layers, lava °C, snow) at a rendered cell coordinate. */
export function heatSampler(fields: GpuFields) {
  const { heat } = display(fields);
  const adv = advect(fields);
  return (u: F, v: F): V4 => { const [su, sv] = adv(u, v); return bilinear(heat, su, sv); };
}

/** Bilinear (vent activity, lava channel memory, seam core, seam halo) at a rendered cell coordinate. */
export function volcanoSampler(fields: GpuFields) {
  const { volc } = display(fields);
  const adv = advect(fields);
  return (u: F, v: F): V4 => { const [su, sv] = adv(u, v); return bilinear(volc, su, sv); };
}

/** Bilinear biome colours at a rendered cell coordinate: ground (rgb, arid), vegetation (rgb, forest). */
export function biomeSampler(fields: GpuFields) {
  const { bioG, bioV } = display(fields);
  const adv = advect(fields);
  return (u: F, v: F): { ground: V4; veg: V4 } => { const [su, sv] = adv(u, v); return { ground: bilinear(bioG, su, sv), veg: bilinear(bioV, su, sv) }; };
}

/**
 * Samplers over the display columns. Reads wrap on X,Z (V1) so interpolation across the cut edge
 * uses the columns on the far side: the data stays seamless even though the block is cut.
 * corners() applies the plate advection, so everything built on it moves continuously.
 */
export function columnSampler(fields: GpuFields) {
  const { cols, motion } = display(fields);
  const adv = advect(fields);
  const at = (x: I, z: I) => cols.element(tColIdx(x, z)) as unknown as V4;
  const shiftAt = (x: I, z: I) => (motion.element(tColIdx(x, z)) as unknown as V4).z;

  function corners(uR: F, vR: F): Corner[] {
    const [u, v] = adv(uR, vR);
    const u0 = floor(u).toVar(), v0 = floor(v).toVar();
    const fu = fract(u).toVar(), fv = fract(v).toVar();
    const xi = int(u0).toVar(), zi = int(v0).toVar();
    const out: Corner[] = [];
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      const x = xi.add(dx), z = zi.add(dz);
      const wx = dx ? fu : float(1).sub(fu);
      const wz = dz ? fv : float(1).sub(fv);
      const e = at(x, z).toVar();
      out.push({ x, z, w: wx.mul(wz), surf: e.x, rawEased: e.y, raw: e.y.sub(shiftAt(x, z)), water: e.z, veg: e.w });
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
   * Rain films (< ~0.05 layer, they come and go every tick) are wet ground, not a water sheet, and
   * the level is depth-weighted: a thin film perched on a slope next to open water cannot lift the
   * sheet up the mountain (translucent curtains). Weights are continuous in depth → no seams.
   */
  function level(c: Corner[]): { level: F; wet: F; dry: F } {
    const wetW = (k: Corner) => k.w.mul(smoothstep(0.05, 0.2, k.water));
    const wet = sum(c, wetW).toVar();
    const lvW = (k: Corner) => wetW(k).mul(min(k.water, 2));
    const lvSum = sum(c, lvW).toVar();
    const lv = sum(c, (k) => lvW(k).mul(k.rawEased.add(k.water))).div(max(lvSum, 1e-8));
    // True coastline: bilinear share of corner columns that are dry land (water < 0.05). Zero over
    // submerged crests however shallow, so shore effects never fire there.
    const dry = sum(c, (k) => k.w.mul(float(1).sub(step(0.05, k.water))));
    // branch-free (see BRANCH-FREE NOTE below)
    return { level: mix(height(c).sub(2), lv, step(1e-7, lvSum)), wet, dry };
  }

  return {
    corners, height, level,
    /** True surfY of a grid column (for tTopVoxel); no advection (grid access). */
    surfAt: (x: I, z: I) => at(x, z).y.sub(shiftAt(x, z)),
    /** Display water depth of a grid column. */
    waterAt: (x: I, z: I) => at(x, z).z,
    /** Vertical display lag of a grid column in voxel layers (rawEased − raw). */
    shiftAt,
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
    // The index goes into a var before the If: built inside one branch, its (cached) subexpressions
    // would be unassigned in the other branch (see BRANCH-FREE NOTE) → wrong voxels on one parity.
    const idx = tVoxIdx(x, int(clamp(float(y), 0, NY - 1)), z).toVar();
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
