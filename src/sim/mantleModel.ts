// §T.20 mantle: CPU half. Grid layout, calibration, plume population (seeded PCG32, deterministic V2).
//
// Frames: the voxel/column grid is the mantle (hotspot) reference frame. Plates translate across it, so a
// plume that stays put under a moving plate leaves a chain of volcanoes carried away with the plate.
//
// Mantle grid: MX × MZ × MY = 64 × 64 × 32 cells below the voxel grid. One mantle cell spans 4×4 columns.
// Index: mx + mz·MX + my·MX·MZ (y-major like vox). my = MY-1 is the top layer (just under voxel y = 0).
// Units: temperature °C (plausible, not real), velocity mantle cells per My, time My.
import { PCG32, type RngState } from '../core/rng';
import { NX, NZ, NCOL, NY, Mat, FLAG_HOTSPOT, voxMat, voxFlags } from './layout';
import type { WorldData } from './worldData';

export const MX = 64;
export const MZ = 64;
export const MY = 32;
export const MCELLS = MX * MZ * MY;
/** Columns per mantle cell along X and Z. */
export const M_SCALE = NX / MX;

/** Crust temperature grid: half-res voxel grid (§C). Index cx + cz·CX + cy·CX·CZ. */
export const CX = NX / 2;
export const CZ = NZ / 2;
export const CY = NY / 2;
export const CT_CELLS = CX * CZ * CY;

export const MANTLE = {
  tTop: 1300,        // °C at the top of the mantle grid (asthenosphere)
  tBot: 2500,        // °C at the bottom boundary (held fixed: basal heating)
  plumeExcess: 400,  // °C plume conduit excess over the reference profile (× mantleHeat)
  plumeTau: 0.5,     // My, conduit relaxation time toward plume temperature
  plumeRise: 0.8,    // mantle cells / My upward velocity on the conduit axis (visual flow)
  slabCold: 450,     // °C below reference at freshly subducted slab top cells
  slabK: 0.08,       // fraction toward slab temperature per subducted column per update
  buoyancy: 0.0025,  // mantle cells / My per °C anomaly
  rollAmp: 0.25,     // mantle cells / My, procedural convection roll amplitude
  rollOmega: 0.004,  // rad / My, roll pattern drift
  kappa: 0.25,       // cells² / My thermal diffusivity (explicit; κ·dt ≤ 1/6 needs dt ≤ 0.66 My)
  bgTau: 150,        // My, weak relaxation to the reference profile (keeps the field bounded)
  every: 4,          // ticks between mantle updates (time-sliced, V19); 2 substeps per update
  // column heat flow from mantle top temperature: q = q0·mantleHeat + qPerC·(Ttop - tTop), mW/m²
  q0: 65,
  qPerC: 0.5,
  qMin: 20,
} as const;

export const CRUST_T = {
  tSurf: 15,       // °C surface (constant until climate provides surfTemp)
  tMagma: 1150,    // °C in cells holding MAGMA
  kCond: 4.3,      // (mW/m²) per (°C / voxel layer): q = 65 → 15 °C/layer → ~600 °C at 40-layer Moho
  tMax: 1300,      // geotherm cap (asthenosphere)
  relax: 0.08,     // fraction toward the column geotherm per iteration
  kappa: 0.12,     // explicit diffusion weight per iteration (≤ 1/6)
  every: 4,        // ticks between updates, offset by 2 from the mantle (spread load, V19)
} as const;

/** Reference (adiabat-ish) temperature of mantle layer my. */
export function mantleRef(my: number): number {
  return MANTLE.tTop + (MANTLE.tBot - MANTLE.tTop) * (MY - 1 - my) / (MY - 1);
}

// ---- plumes ----------------------------------------------------------------------------------

export const MAX_PLUMES = 8;
export const MIN_PLUMES = 1;

export interface Plume {
  x: number;    // column units [0, NX), mantle frame
  z: number;
  r: number;    // conduit radius, columns
  age: number;  // My since birth
  life: number; // My remaining
}

export const PLUME = {
  radius: [8, 12] as const, // columns; the hot (melting) footprint is ~0.6 r
  life: [80, 400] as const,   // My
  ramp: 10,                   // My fade-in / fade-out
  wander: 0.08,               // columns / sqrt(My) random walk (≪ plate speed 1-4 cells/My)
  spawnPerMy: 1 / 120,        // birth probability per My while below MAX_PLUMES
  extraAtStart: 1,            // seeded plumes added to those found in the world
} as const;

export interface PlumeOptions { wander?: number; spawnPerMy?: number; evolve?: boolean }

/** Strength 0..1: fades in after birth and out before death, so the heat source never pops. */
export function plumeStrength(p: Plume): number {
  return Math.max(0, Math.min(1, p.age / PLUME.ramp, p.life / PLUME.ramp));
}

const wrapN = (v: number, n: number) => ((v % n) + n) % n;

/**
 * Plume population. Deterministic from its PCG32 stream and the sequence of step(dt) calls (V2);
 * getState/setState round-trip for saves (V13).
 */
export class PlumeSet {
  plumes: Plume[];
  private rng: PCG32;
  readonly opts: Required<PlumeOptions>;

  constructor(rng: PCG32, initial: Plume[], opts: PlumeOptions = {}) {
    this.rng = rng;
    this.plumes = initial.slice(0, MAX_PLUMES).map((p) => ({ ...p }));
    this.opts = { wander: opts.wander ?? PLUME.wander, spawnPerMy: opts.spawnPerMy ?? PLUME.spawnPerMy, evolve: opts.evolve ?? true };
  }

  private gauss(): number {
    const u1 = Math.max(1e-12, this.rng.nextFloat()), u2 = this.rng.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  spawn(): Plume {
    const r = this.rng;
    return { x: r.range(0, NX), z: r.range(0, NZ), r: r.range(PLUME.radius[0], PLUME.radius[1]), age: 0, life: r.range(PLUME.life[0], PLUME.life[1]) };
  }

  step(dt: number): void {
    for (const p of this.plumes) { p.age += dt; if (this.opts.evolve) p.life -= dt; }
    if (!this.opts.evolve) return;
    const s = this.opts.wander * Math.sqrt(dt);
    for (const p of this.plumes) {
      p.x = wrapN(p.x + this.gauss() * s, NX);
      p.z = wrapN(p.z + this.gauss() * s, NZ);
    }
    // never let the last plume die (keeps at least one hotspot, V8 "≥1 eruption per 50 My")
    this.plumes = this.plumes.filter((p) => p.life > 0);
    while (this.plumes.length < MIN_PLUMES) this.plumes.push(this.spawn());
    if (this.plumes.length < MAX_PLUMES && this.rng.nextFloat() < this.opts.spawnPerMy * dt) this.plumes.push(this.spawn());
  }

  getState(): { plumes: Plume[]; rng: RngState } { return { plumes: this.plumes.map((p) => ({ ...p })), rng: this.rng.getState() }; }
  setState(s: { plumes: Plume[]; rng: RngState }): void { this.plumes = s.plumes.map((p) => ({ ...p })); this.rng = PCG32.fromState(s.rng); }
}

/**
 * Plumes under worldgen's FLAG_HOTSPOT magma chambers (so initial chambers keep being fed), plus
 * PLUME.extraAtStart seeded ones. Greedy clustering with torus distance (V1).
 */
export function plumesFromWorld(w: WorldData, rng: PCG32): Plume[] {
  const cols: number[] = [];
  for (let c = 0; c < NCOL; c++) {
    for (let y = 0; y < NY; y++) {
      const v = w.vox[c + y * NCOL]!;
      if (voxMat(v) === Mat.MAGMA && (voxFlags(v) & FLAG_HOTSPOT) !== 0) { cols.push(c); break; }
    }
  }
  const out: Plume[] = [];
  const used = new Uint8Array(cols.length);
  const d = (a: number, b: number, n: number) => { const t = Math.abs(a - b) % n; return Math.min(t, n - t); };
  for (let i = 0; i < cols.length && out.length < MAX_PLUMES; i++) {
    if (used[i]) continue;
    const x0 = cols[i]! % NX, z0 = Math.floor(cols[i]! / NX);
    // circular mean of the cluster on the torus
    let sx = 0, cx = 0, sz = 0, cz = 0;
    for (let j = i; j < cols.length; j++) {
      const x = cols[j]! % NX, z = Math.floor(cols[j]! / NX);
      if (used[j] || d(x, x0, NX) > 12 || d(z, z0, NZ) > 12) continue;
      used[j] = 1;
      const ax = (2 * Math.PI * x) / NX, az = (2 * Math.PI * z) / NZ;
      sx += Math.sin(ax); cx += Math.cos(ax); sz += Math.sin(az); cz += Math.cos(az);
    }
    const x = wrapN((Math.atan2(sx, cx) * NX) / (2 * Math.PI), NX);
    const z = wrapN((Math.atan2(sz, cz) * NZ) / (2 * Math.PI), NZ);
    // already mature: full strength from tick 0
    out.push({ x, z, r: 10, age: PLUME.ramp, life: rng.range(PLUME.life[0], PLUME.life[1]) });
  }
  const set = new PlumeSet(rng, [], {});
  for (let k = 0; k < PLUME.extraAtStart && out.length < MAX_PLUMES; k++) {
    const p = set.spawn();
    p.age = PLUME.ramp;
    out.push(p);
  }
  return out;
}

/** Wrapped horizontal distance² (columns) between a point and a plume axis (V1). */
export function plumeDist2(x: number, z: number, p: Plume): number {
  let dx = Math.abs(x - p.x) % NX; dx = Math.min(dx, NX - dx);
  let dz = Math.abs(z - p.z) % NZ; dz = Math.min(dz, NZ - dz);
  return dx * dx + dz * dz;
}

/** Compact bump 1 → 0 at t = 1 (same as the GPU kernel). */
export const bump = (t2: number) => { const a = 1 - t2; return a <= 0 ? 0 : a * a; };

/** Initial mantle temperatures: reference profile + fully developed plume conduits. */
export function initMantleTemps(plumes: Plume[], mantleHeat = 1): Float32Array {
  const t = new Float32Array(MCELLS);
  for (let my = 0; my < MY; my++) for (let mz = 0; mz < MZ; mz++) for (let mx = 0; mx < MX; mx++) {
    let v = mantleRef(my);
    const px = (mx + 0.5) * M_SCALE, pz = (mz + 0.5) * M_SCALE;
    let hot = 0;
    for (const p of plumes) hot = Math.max(hot, plumeStrength(p) * bump(plumeDist2(px, pz, p) / (p.r * p.r)));
    if (my > 0) v += MANTLE.plumeExcess * mantleHeat * hot;
    t[mx + mz * MX + my * MX * MZ] = v;
  }
  return t;
}
