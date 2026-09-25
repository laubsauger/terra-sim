// §T.22/§T.23 magma + lava: CPU half. Calibration constants, integer-budget bookkeeping, CPU helpers for tests.
//
// Mass (V3): M_total = crust (Σ fill of crust voxels) + counters[CTR_RESERVOIR] + magma + lava (+ sedSusp).
//   magma = Σ fill of MAGMA voxels (chambers) + Σ magCol.y (pending, ascending melt)
//   lava  = Σ lava.x
// All of it is integer fill units (255 = one full voxel). Every transfer is an exact integer move:
//   reservoir → magCol.y            melt generation (atomicAdd reservoir −n)
//   magCol.y → MAGMA voxel          chamber growth, 255 at a time (column inflates by one layer)
//   MAGMA voxel → lava.x            eruption, 255 at a time (column above the chamber drops one layer)
//   MAGMA voxel → pluton voxel      chamber crystallization (magma → crust)
//   lava.x → crust voxel            solidification (top-up + new BASALT/ANDESITE voxels)
//   reservoir → lava.x              minor ridge fissure lava
//   lost chambers → reservoir       tectonics consumed a column holding magma (afterTectonics fix-up)
//
// Units: time My, rates fill units per My per column, temperatures °C, distances cells.
import { NCOL, NY, Mat } from './layout';

export const MAGMA = {
  // hotspot melt: rate = hotGain · max(0, (heatFlow − qHot) / qHotSpan) · volcanism
  hotGain: 900,
  qHot: 110,         // mW/m²; background mantle gives ~65, a mature plume centre ~225
  qHotSpan: 100,
  // arc melt: a fixed share of the subducted slab mass returns as arc magma (continental growth, V3 balance).
  // Normalized globally (see arcNormK): Σ arc melt = arcFrac · slabMass · subduction events, distributed over
  // the arc-window weights, so output does not depend on trench geometry. Calibrated on generated worlds so
  // volcanic outflow ≈ net tectonic inflow into the reservoir over hundreds of My (see report in magma.spec).
  arcFrac: 0.065,    // share of (events · slabMass) returned through arcs
  slabMass: 9 * 255, // nominal oceanic column mass per subduction event
  arcNear: 3,        // arc window, cells inland of the trench (same plate)
  arcFar: 8,
  arcMaxDist: 16,    // propagation limit of the trench-distance field
  arcActCap: 32,     // cap on trench activity in the distribution weight (fixed-point overflow guard)
  tauTrench: 6,      // My, subduction event-rate EMA at trench columns (arc.z)
  tauRecent: 1,      // My, recency of the last event (arc.w); a column is a trench while arc.w > trenchRecent
  trenchRecent: 0.3,
  normEma: 8,        // periods (tectonics runs) of EMA smoothing in the arc normalization
  // ridge: minor fissure lava on fresh crust, rate = ridgeGain · max(0, 1 − crustAge / ridgeAge)
  ridgeGain: 4,
  ridgeAge: 1,       // My
  // chambers
  meltAccMax: 255 * 4,   // pending melt cap per column; beyond it melting pauses
  segMin: 32,            // pending ≥ this may segregate early as a partial MAGMA voxel …
  segTau: 3,             // … with mean wait segTau My (full 255 always segregates), so no melt stays in limbo
  chamberMax: 12,        // MAGMA voxels per column; a full chamber must erupt
  depthMin: 2,           // chamber roof depth below surface: clamp(thickness / 3, depthMin, depthMax)
  depthMax: 10,
  lidPerVoxel: 4,        // overpressure: erupts only with n ≥ 1 + lid / lidPerVoxel chamber voxels
  eruptRate: 3,          // per My (× volcanism) once overpressured
  freezeTau: 10,         // My, mean residence of a chamber voxel before crystallizing (P = n·dt/τ per step)
  reservoirMin: 255 * 256, // melting pauses below this reservoir level (snapshot, deterministic)
  silAndesite: 128,      // silica code ≥ this → ANDESITE lava / GRANITE pluton (FLAG_CONTINENTAL)
} as const;

export const LAVA = {
  tBasalt: 1200,     // eruption °C, silica 0
  tAndesite: 1000,   // eruption °C, silica 255
  tSolid: 750,       // below: freezes completely
  tAmb: 15,
  kAir: 0.6,         // 1/My cooling in air for a 1-layer flow (thicker cools slower: / (1 + 0.5·layers))
  kWater: 25,        // 1/My quench under water
  wetDepth: 0.05,    // water depth (voxel-y) that counts as submerged
  mobility: 1,       // fraction of head excess moved per substep at full fluidity (× 0.2 per neighbour)
  yield: 0.08,       // layers of head difference below which lava does not move (Bingham)
  freezeRate: 0.25,  // 1/My, fraction of hot lava crusting over into rock per My
  thin: 24,          // units; thinner films freeze at once
  substeps: 2,       // flow substeps per tick
  maxNewVoxels: 4,   // rock voxels stacked per column per tick
} as const;

/** magmaCtr slots. ERUPTIONS..SOLIDIFIED are cumulative stats; clearStats() zeroes them. */
export const MCTR = {
  GATHER_OLD: 0, GATHER_NEW: 1, RES_SNAP: 2,
  ERUPTIONS: 3,  // eruption events (one MAGMA voxel each)
  DRAWN: 4,      // units drawn from the reservoir (melt + ridge lava + inject)
  RETURNED: 5,   // units of magma lost to subduction, credited back to the reservoir
  FROZEN: 6,     // units of chamber magma crystallized into plutons
  SOLIDIFIED: 7, // units of lava turned into rock
  // arc normalization state (persistent; save with the buffer, V13)
  EVENTS: 8,     // subduction events since the last normalization (gather)
  ARCW: 9,       // Σ arc weights × ARC_W_SCALE since the last normalization (melt kernel)
  E_EMA: 10,     // events per period × ARC_E_SCALE, EMA
  W_EMA: 11,     // weight per period × ARC_W_SCALE, EMA
  ARCK: 12,      // melt units per (weight · step) × ARC_K_SCALE
  SIZE: 16,
} as const;
export const ARC_W_SCALE = 16;
export const ARC_E_SCALE = 256;
export const ARC_K_SCALE = 1024;

/**
 * Arc normalization (CPU mirror of the GPU period kernel): per-step arc melt of a column = K · weight, with
 * K chosen so that over a period Σ columns Σ steps K·weight = arcFrac · slabMass · events.
 * eEma, wEma in fixed point (ARC_E_SCALE, ARC_W_SCALE). Returns K × ARC_K_SCALE, truncated.
 */
export function arcNormK(eEma: number, wEma: number): number {
  if (wEma <= 0) return 0;
  return Math.trunc(((MAGMA.arcFrac * MAGMA.slabMass * (eEma / ARC_E_SCALE)) / (wEma / ARC_W_SCALE)) * ARC_K_SCALE);
}
/** EMA update in fixed point (integer, as on the GPU). */
export const emaStep = (ema: number, sample: number) => ema + Math.trunc((sample - ema) / MAGMA.normEma);

export interface MagmaStats { eruptions: number; drawn: number; returned: number; frozen: number; solidified: number }

export function parseMagmaStats(buf: ArrayBuffer): MagmaStats {
  const a = new Int32Array(buf);
  return { eruptions: a[MCTR.ERUPTIONS]!, drawn: a[MCTR.DRAWN]!, returned: a[MCTR.RETURNED]!, frozen: a[MCTR.FROZEN]!, solidified: a[MCTR.SOLIDIFIED]! };
}

const smoothstep = (e0: number, e1: number, x: number) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

/** Arc window weight at trench distance d (cells); same formula as the GPU kernel. */
export function arcWindow(d: number): number {
  return smoothstep(MAGMA.arcNear - 1, MAGMA.arcNear + 0.5, d) * (1 - smoothstep(MAGMA.arcFar - 0.5, MAGMA.arcFar + 1, d));
}
/** Chamber roof depth below the top voxel for a crust of (top − base) layers. */
export function chamberDepth(thickness: number): number {
  return Math.min(MAGMA.depthMax, Math.max(MAGMA.depthMin, Math.floor(thickness / 3)));
}

// ---- lava.y packing: temp °C × 16 in bits 0-15, silica code 0..255 in bits 16-23 ----
export const encLavaY = (tC: number, sil: number) => (Math.min(65535, Math.max(0, Math.round(tC * 16))) | ((sil & 0xff) << 16)) >>> 0;
export const lavaTempC = (y: number) => (y & 0xffff) / 16;
export const lavaSil = (y: number) => (y >>> 16) & 0xff;

export interface MagmaBudget { crust: number; reservoir: number; chambers: number; pending: number; lava: number; total: number }

/**
 * Exact V3 budget from readbacks (tests, stats tooling). Crust and chambers come from the voxels themselves,
 * not from GPU caches, so a cache/voxel mismatch shows up as drift.
 */
export function magmaBudgetCpu(vox: Uint32Array, reservoir: number, magCol: Uint32Array, lava: Uint32Array): MagmaBudget {
  let crust = 0, chambers = 0, pending = 0, lv = 0;
  for (let i = 0; i < vox.length; i++) {
    const v = vox[i]!, m = v & 0xff, f = (v >>> 8) & 0xff;
    if (m === Mat.MAGMA) chambers += f;
    else if (m !== Mat.AIR && m !== Mat.PERIDOTITE) crust += f;
  }
  for (let c = 0; c < NCOL; c++) { pending += magCol[c * 2 + 1]!; lv += lava[c * 2]!; }
  return { crust, reservoir, chambers, pending, lava: lv, total: crust + reservoir + chambers + pending + lv };
}

/** Per-column Σ MAGMA fill (to check the GPU magCol.x cache against the voxels). */
export function chamberMassPerColumn(vox: Uint32Array): Uint32Array {
  const out = new Uint32Array(NCOL);
  for (let y = 0; y < NY; y++) for (let c = 0; c < NCOL; c++) {
    const v = vox[c + y * NCOL]!;
    if ((v & 0xff) === Mat.MAGMA) out[c] = out[c]! + ((v >>> 8) & 0xff);
  }
  return out;
}
