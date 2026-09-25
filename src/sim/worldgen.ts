// §T8 worldgen: deterministic world from a seed (V2), plate count within [3, 12] (V6).
//
// Units
// - Horizontal: 1 cell ≈ 20 km. Vertical: 1 voxel layer ≈ 250 m (Y_SEA_NOMINAL - Y_OCEAN_FLOOR
//   = 16 layers ≈ 4 km abyssal depth); vertical exaggeration is a render concern.
// - Plate.vel: cells per My. |vel| ∈ [PLATE_SPEED_MIN, PLATE_SPEED_MAX] = 1..4 cells/My
//   (≈ 2..8 cm/yr), so a plate crosses the 256-cell block in ~100 My.
// - crustAge, voxel age codes: My (ageToCode).
// - Crust mass (mantleReservoir, crustMass): integer fill units, Σ fill over crust voxels
//   (non-AIR, non-PERIDOTITE, non-MAGMA), same unit as derive colInfo.y and GPU counters (V3).
//
// Structure: all noise is evaluated per column (2D), producing integer layer boundaries;
// voxels are then written layer by layer (y-major storage → contiguous writes). No 3D noise.
// Only IEEE-exact math (+ - * / sqrt floor round) in the generator itself for cross-engine
// determinism; ageToCode (layout.ts) uses Math.log2.

import { PCG32 } from '../core/rng';
import { PeriodicNoise2D } from '../core/noise';
import {
  NX, NZ, NY, NCOL, Mat, FLAG_CONTINENTAL, FLAG_HOTSPOT, ageToCode, voxMat, voxFill,
  Y_OCEAN_FLOOR, Y_SEA_NOMINAL, Y_CONT_SURFACE,
} from './layout';
import { emptyWorld, type WorldData } from './worldData';

export interface WorldgenOptions {
  /** Initial plate count, default 7, rounded and clamped to [3, 12] (V6). */
  plates?: number;
}

export const DEFAULT_PLATES = 7;
export const MIN_PLATES = 3;
export const MAX_INIT_PLATES = 12;
/** Plate speed range, cells per My (1 cell ≈ 20 km → 2..8 cm/yr). */
export const PLATE_SPEED_MIN = 1;
export const PLATE_SPEED_MAX = 4;
/** Target continental area fraction range (exact per seed via mask quantile). */
export const CONT_FRAC_MIN = 0.3;
export const CONT_FRAC_MAX = 0.4;
/**
 * Initial seafloor age proxy: age = distance to nearest plate boundary / this rate.
 * Slow half-spreading (0.3 cells/My ≈ 0.6 cm/yr) so initial oceans span ~0..150 My
 * (boundary distances reach ~40-55 cells with 7 plates); the sim rewrites ages once real
 * spreading runs.
 */
export const INIT_HALF_SPREAD = 0.3;
export const MAX_OCEAN_AGE = 180;
/**
 * Initial mantleReservoir as a fraction of crust mass in voxels. Gives uplift / ridges /
 * volcanism (V16, T19) a buffer to draw from before subduction starts refilling it.
 */
export const MANTLE_RESERVOIR_FRAC = 0.05;

/** Crust mass in full-voxel equivalents: Σ fill/255 over non-AIR, non-PERIDOTITE voxels. */
export function crustMass(vox: Uint32Array): number {
  let fills = 0; // integer sum, exact up to 2^53
  for (let i = 0; i < vox.length; i++) {
    const v = vox[i]!;
    const m = voxMat(v);
    if (m !== Mat.AIR && m !== Mat.PERIDOTITE && m !== Mat.MAGMA) fills += voxFill(v);
  }
  return fills;
}

const smooth01 = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
/** Compact bump: 1 at t=0, 0 for |t| ≥ 1 (exp-free for determinism). */
const bump = (t: number) => {
  const a = 1 - t * t;
  return a <= 0 ? 0 : a * a;
};
const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);
/** Wrapped torus delta in [-0.5, 0.5]. */
const wrapD = (d: number) => d - Math.round(d);

/** Surface at the continental coastline: just below nominal sea so margins make shallow seas. */
const COAST_Y = Y_SEA_NOMINAL - 0.5;
const SQRT2 = 1.4142135623730951;

/**
 * out[c] = chamfer (1, √2) distance in cells from column c to the nearest coastline
 * (boundary between isCont 0/1), torus-wrapped (V1). Coast-adjacent columns get 0.5.
 * Two forward/backward sweeps so distances propagate across the wrap edges.
 */
function coastDistance(isCont: Uint8Array, out: Float32Array): void {
  const mx = NX - 1;
  const mz = NZ - 1;
  for (let z = 0; z < NZ; z++) {
    for (let x = 0; x < NX; x++) {
      const c = x + z * NX;
      const k = isCont[c]!;
      const edge =
        isCont[((x + 1) & mx) + z * NX] !== k || isCont[((x - 1) & mx) + z * NX] !== k ||
        isCont[x + ((z + 1) & mz) * NX] !== k || isCont[x + ((z - 1) & mz) * NX] !== k;
      out[c] = edge ? 0.5 : 1e6;
    }
  }
  const relax = (n: number, wgt: number, d: number) => {
    const dn = out[n]! + wgt;
    return dn < d ? dn : d;
  };
  for (let it = 0; it < 2; it++) {
    for (let z = 0; z < NZ; z++) {
      const zm = ((z - 1) & mz) * NX;
      const z0 = z * NX;
      for (let x = 0; x < NX; x++) {
        const c = x + z0;
        const xm = (x - 1) & mx;
        const xp = (x + 1) & mx;
        let d = out[c]!;
        d = relax(xm + z0, 1, d);
        d = relax(xm + zm, SQRT2, d);
        d = relax(x + zm, 1, d);
        d = relax(xp + zm, SQRT2, d);
        out[c] = d;
      }
    }
    for (let z = NZ - 1; z >= 0; z--) {
      const zp = ((z + 1) & mz) * NX;
      const z0 = z * NX;
      for (let x = NX - 1; x >= 0; x--) {
        const c = x + z0;
        const xm = (x - 1) & mx;
        const xp = (x + 1) & mx;
        let d = out[c]!;
        d = relax(xp + z0, 1, d);
        d = relax(xp + zp, SQRT2, d);
        d = relax(x + zp, 1, d);
        d = relax(xm + zp, SQRT2, d);
        out[c] = d;
      }
    }
  }
}

// Noise streams: one PCG32 stream per purpose so adding a field never reshuffles others.
const STREAM_PLATES = 1;
const STREAM_NOISE = 100;

export function generateWorld(seed: number, opts: WorldgenOptions = {}): WorldData {
  const req = opts.plates ?? DEFAULT_PLATES;
  if (!Number.isFinite(req)) throw new RangeError(`generateWorld: plates must be finite, got ${req}`);
  const nPlates = clamp(Math.round(req), MIN_PLATES, MAX_INIT_PLATES);

  const w = emptyWorld(seed);
  const rng = new PCG32(seed, STREAM_PLATES);
  let ns = STREAM_NOISE;
  const mkNoise = () => new PeriodicNoise2D(new PCG32(seed, ns++));
  const nPlateWarp = mkNoise();
  const nMask = mkNoise();
  const nHills = mkNoise();
  const nMtn = mkNoise();
  const nBelt = mkNoise();
  const nCrust = mkNoise();
  const nFold = mkNoise();
  const nOcean = mkNoise();

  // ---- plates: seeds with min torus separation, additive weights for size variety ----
  const cx = new Float64Array(nPlates);
  const cz = new Float64Array(nPlates);
  const pw = new Float64Array(nPlates);
  let dmin = 0.55 / Math.sqrt(nPlates);
  for (let i = 0, tries = 0; i < nPlates; ) {
    const x = rng.nextFloat();
    const z = rng.nextFloat();
    let ok = true;
    for (let j = 0; j < i && ok; j++) {
      const dx = wrapD(x - cx[j]!);
      const dz = wrapD(z - cz[j]!);
      if (dx * dx + dz * dz < dmin * dmin) ok = false;
    }
    if (ok) {
      cx[i] = x;
      cz[i] = z;
      i++;
      tries = 0;
    } else if (++tries > 200) {
      dmin *= 0.9;
      tries = 0;
    }
  }
  for (let i = 0; i < nPlates; i++) pw[i] = rng.range(0, 0.25 * dmin);
  for (let i = 0; i < nPlates; i++) {
    const p = w.plates[i]!;
    // uniform direction by disc rejection (no trig)
    let dx = 0;
    let dz = 0;
    let r2 = 0;
    do {
      dx = rng.range(-1, 1);
      dz = rng.range(-1, 1);
      r2 = dx * dx + dz * dz;
    } while (r2 > 1 || r2 < 1e-4);
    const inv = rng.range(PLATE_SPEED_MIN, PLATE_SPEED_MAX) / Math.sqrt(r2);
    p.vel = [dx * inv, dz * inv];
    p.accum = [0, 0];
  }
  // Convergence per plate pair: closing speed along the seed-to-seed direction → [0, 1].
  const conv = new Float64Array(nPlates * nPlates);
  for (let a = 0; a < nPlates; a++) {
    for (let b = 0; b < nPlates; b++) {
      if (a === b) continue;
      const nx = wrapD(cx[b]! - cx[a]!);
      const nz = wrapD(cz[b]! - cz[a]!);
      const len = Math.sqrt(nx * nx + nz * nz) || 1;
      const va = w.plates[a]!.vel;
      const vb = w.plates[b]!.vel;
      const closing = -((vb[0] - va[0]) * nx + (vb[1] - va[1]) * nz) / len; // cells/My
      conv[a * nPlates + b] = smooth01(closing / 3);
    }
  }

  // ---- pass 1 per column: plates, boundary distance, continent mask ----
  const plateId = w.plateId;
  const bdCells = new Float32Array(NCOL); // distance proxy to nearest plate boundary, cells
  const plate2 = new Uint8Array(NCOL); // second-nearest plate
  const mask = new Float32Array(NCOL);
  const tmp = new Float64Array(2);
  for (let z = 0; z < NZ; z++) {
    const v = (z + 0.5) / NZ;
    for (let x = 0; x < NX; x++) {
      const u = (x + 0.5) / NX;
      const c = x + z * NX;
      nPlateWarp.warp(u, v, 4, 4, 4, 0.5, 0.09, tmp);
      const wu = tmp[0]!;
      const wv = tmp[1]!;
      let d1 = 1e9;
      let d2 = 1e9;
      let i1 = 0;
      let i2 = 0;
      for (let i = 0; i < nPlates; i++) {
        const dx = wrapD(wu - cx[i]!);
        const dz = wrapD(wv - cz[i]!);
        const d = Math.sqrt(dx * dx + dz * dz) - pw[i]!;
        if (d < d1) {
          d2 = d1;
          i2 = i1;
          d1 = d;
          i1 = i;
        } else if (d < d2) {
          d2 = d;
          i2 = i;
        }
      }
      plateId[c] = i1;
      plate2[c] = i2;
      bdCells[c] = (d2 - d1) * 0.5 * NX;
      nMask.warp(u, v, 4, 4, 3, 0.5, 0.05, tmp);
      mask[c] = nMask.fbm(tmp[0]!, tmp[1]!, 2, 2, 6, 0.5);
    }
  }

  // continental fraction exact per seed: threshold at quantile
  const contFrac = rng.range(CONT_FRAC_MIN, CONT_FRAC_MAX);
  const sorted = Float32Array.from(mask).sort();
  const thr = sorted[Math.floor((1 - contFrac) * NCOL)]!;
  const isCont = new Uint8Array(NCOL);
  for (let c = 0; c < NCOL; c++) isCont[c] = mask[c]! >= thr ? 1 : 0;
  // distance to coastline (cells) drives shelves and coastal plains: slopes independent of mask gradient
  const coast = new Float32Array(NCOL);
  coastDistance(isCont, coast);

  // ---- pass 2 per column: surface, crust stack, ages ----
  const ty = new Int16Array(NCOL); // top solid voxel y
  const topFill = new Uint8Array(NCOL);
  const t0 = new Int16Array(NCOL); // y < t0 → PERIDOTITE
  const t1 = new Int16Array(NCOL); // y < t1 → GABBRO
  const t2 = new Int16Array(NCOL); // y < t2 → GNEISS
  const t3 = new Int16Array(NCOL); // y < t3 → midMat (GRANITE | BASALT)
  const t4 = new Int16Array(NCOL); // y < t4 → banded strata; else SEDIMENT
  const midMat = new Uint8Array(NCOL);
  const colFlags = new Uint8Array(NCOL);
  const cBase = new Uint8Array(NCOL); // age code of crystalline crust
  const cCov0 = new Uint8Array(NCOL); // age code at base of strata
  const cCov1 = new Uint8Array(NCOL); // age code at top of strata
  const cSed = new Uint8Array(NCOL); // age code of loose sediment
  const foldOff = new Float32Array(NCOL); // strata vertical offset (folding + uplift)
  const oceanAge = new Float32Array(NCOL); // My, oceanic columns only
  const crustAge = w.crustAge;
  const water = w.water;
  const codeSed = ageToCode(1);

  for (let z = 0; z < NZ; z++) {
    const v = (z + 0.5) / NZ;
    for (let x = 0; x < NX; x++) {
      const u = (x + 0.5) / NX;
      const c = x + z * NX;
      const cont = isCont[c] === 1;
      const dc = coast[c]!;
      const hillsN = nHills.fbm(u, v, 8, 8, 5, 0.5);
      const crustN = nCrust.fbm(u, v, 6, 6, 3, 0.5);
      let S: number; // float surface height, voxel-y units
      let B: number; // crust base
      if (cont) {
        const r = smooth01(dc / 20); // coast → interior over ~20 cells
        const rm = smooth01(dc / 8); // mountains may sit close to coasts (Andes-style)
        // mountains: convergent plate boundaries + a few noise belts
        const bd = bdCells[c]!;
        const beltPlate = conv[plateId[c]! * nPlates + plate2[c]!]! * bump(bd / 18);
        const bn = nBelt.noise(u * 3, v * 3, 3, 3);
        const beltNoise = bump(bn / 0.16) * smooth01(nBelt.noise(u * 2 + 0.5, v * 2 + 0.5, 2, 2) * 4);
        const belt = beltPlate > beltNoise * 0.7 ? beltPlate : beltNoise * 0.7;
        const mtn = belt > 0.01 ? belt * rm * (5 + 17 * nMtn.ridged(u, v, 12, 12, 5, 0.5)) : 0;
        S = COAST_Y + (Y_CONT_SURFACE - COAST_Y) * r + hillsN * (1.5 + 2.5 * r) + mtn;
        const tFull = 38 + 3.5 * crustN + 0.45 * mtn; // ~34..44, mountain roots deeper
        const T = 26 + (tFull - 26) * r; // thinned passive margin
        B = S - T;
        const cratonN = nCrust.fbm(u + 0.41, v + 0.83, 3, 3, 3, 0.5);
        const baseAge = 500 + 2500 * smooth01(0.5 + 0.9 * cratonN); // 500..3000 My
        crustAge[c] = baseAge;
        const loose = clamp((81 - S) * 0.5, 0, 2.5) * (1 - smooth01(mtn / 3)) + 1.5 * (1 - r);
        const cover = Math.max(0, 12 + 3 * crustN - 0.7 * mtn) + 3 * (1 - r);
        const plutonN = nFold.noise(u * 10 + 0.5, v * 10 + 0.5, 10, 10);
        const coverTop = S - loose;
        const graniteTop = coverTop - cover + 1.5 * plutonN;
        const gneissTop = B + 0.25 * T + 2 * crustN;
        const gabbroTop = B + 2 + 1.5 * nFold.noise(u * 7, v * 7, 7, 7);
        t1[c] = Math.round(gabbroTop);
        t2[c] = Math.round(gneissTop);
        t3[c] = Math.round(graniteTop);
        t4[c] = Math.round(coverTop);
        midMat[c] = Mat.GRANITE;
        colFlags[c] = FLAG_CONTINENTAL;
        cBase[c] = ageToCode(baseAge);
        cCov0[c] = ageToCode(Math.min(540, 0.4 * baseAge));
        cCov1[c] = ageToCode(10);
        cSed[c] = codeSed;
        // gently folded banding; tighter + carried up in mountain belts
        foldOff[c] = (3 + 0.4 * mtn) * nFold.fbm(u, v, 5, 5, 3, 0.5) + 0.8 * mtn;
      } else {
        const q = smooth01((dc - 5) / 28); // shallow shelf ~5 cells, slope to abyss over ~28
        const shelf = COAST_Y - 0.2 - 2.3 * smooth01(dc / 6); // 1-3 layers below nominal sea
        const bd = bdCells[c]!;
        const age = Math.min(MAX_OCEAN_AGE, bd / INIT_HALF_SPREAD);
        oceanAge[c] = age;
        const relief = 1.5 * nOcean.fbm(u, v, 12, 12, 4, 0.55);
        // thermal subsidence ~ sqrt(age); ridge crest bump at boundaries
        const abyss = Y_OCEAN_FLOOR + 5 - 1.0 * Math.sqrt(age) + 2 * bump(bd / 6) + relief;
        S = shelf + (abyss - shelf) * q + hillsN * 1.5 * (1 - q);
        const T = 18 + (9 + 1.5 * crustN - 18) * q; // ~8..11 at abyss, thicker at margins
        B = S - T;
        crustAge[c] = age;
        const sed = Math.min(4, 0.015 * age + 3 * (1 - q));
        t1[c] = Math.round(B + 0.55 * T); // GABBRO
        t2[c] = t1[c]!; // no gneiss
        t3[c] = Math.round(S - sed); // BASALT
        t4[c] = t3[c]!; // no banded strata
        midMat[c] = Mat.BASALT;
        colFlags[c] = 0;
        const code = ageToCode(age);
        cBase[c] = code;
        cCov0[c] = code;
        cCov1[c] = code;
        cSed[c] = ageToCode(age * 0.5);
      }
      S = clamp(S, 8, NY - 2);
      const top = Math.ceil(S) - 1;
      let f = Math.round((S - top) * 255);
      if (f < 1) f = 1;
      if (f > 255) f = 255;
      ty[c] = top;
      topFill[c] = f;
      // order + clamp boundaries: 1 ≤ t0 ≤ t1 ≤ t2 ≤ t3 ≤ t4 ≤ top + 1 (y=0 always mantle)
      const lim = top + 1;
      const b0 = clamp(Math.round(B), 1, lim);
      t0[c] = b0;
      t1[c] = clamp(t1[c]!, b0, lim);
      t2[c] = clamp(t2[c]!, t1[c]!, lim);
      t3[c] = clamp(t3[c]!, t2[c]!, lim);
      t4[c] = clamp(t4[c]!, t3[c]!, lim);
      const surf = top + f / 255;
      water[c] = Math.max(0, Y_SEA_NOMINAL - surf);
    }
  }

  // ---- banded strata lookup: variable band thickness 1.5..4 layers, no repeats ----
  const BAND_RES = 4; // lookup cells per layer
  const BAND_Y0 = -64; // lookup covers y + fold offset in [BAND_Y0, NY + 64)
  const bandLen = (NY + 128) * BAND_RES;
  const bandMat = new Uint8Array(bandLen);
  {
    const strata = [Mat.SANDSTONE, Mat.SHALE, Mat.LIMESTONE];
    let prev = -1;
    let i = 0;
    while (i < bandLen) {
      let m = rng.int(3);
      if (m === prev) m = (m + 1 + rng.int(2)) % 3;
      prev = m;
      const len = Math.round(rng.range(1.5, 4) * BAND_RES);
      for (let k = 0; k < len && i < bandLen; k++) bandMat[i++] = strata[m]!;
    }
  }

  // ---- write voxels layer by layer (y-major: contiguous) ----
  const vox = w.vox;
  const PERI = Mat.PERIDOTITE;
  for (let y = 0; y < NY; y++) {
    const base = y * NCOL;
    for (let c = 0; c < NCOL; c++) {
      const top = ty[c]!;
      if (y > top) continue; // AIR = 0 already
      let mat: number;
      let age: number;
      let flags = colFlags[c]!;
      if (y < t0[c]!) {
        mat = PERI;
        age = 0;
        flags = 0;
      } else if (y < t1[c]!) {
        mat = Mat.GABBRO;
        age = cBase[c]!;
      } else if (y < t2[c]!) {
        mat = Mat.GNEISS;
        age = cBase[c]!;
      } else if (y < t3[c]!) {
        mat = midMat[c]!;
        age = cBase[c]!;
      } else if (y < t4[c]!) {
        let bi = Math.floor((y - foldOff[c]! - BAND_Y0) * BAND_RES);
        bi = bi < 0 ? 0 : bi >= bandLen ? bandLen - 1 : bi;
        mat = bandMat[bi]!;
        const lo = t3[c]!;
        const span = t4[c]! - lo;
        const a0 = cCov0[c]!;
        age = Math.round(a0 + ((cCov1[c]! - a0) * (y - lo + 0.5)) / span);
      } else {
        mat = Mat.SEDIMENT;
        age = cSed[c]!;
      }
      const fill = y === top ? topFill[c]! : 255;
      vox[base + c] = (mat | (fill << 8) | (age << 16) | (flags << 24)) >>> 0;
    }
  }

  // ---- 1-2 hotspot magma chambers (solid lid of ≥ 3 layers kept) ----
  const nHot = 1 + rng.int(2);
  for (let h = 0; h < nHot; h++) {
    const hx = rng.int(NX);
    const hz = rng.int(NZ);
    const hc = hx + hz * NX;
    const topH = ty[hc]!;
    const cy = clamp(t0[hc]! + 3, 3, topH - 6);
    const RX = 4;
    const RY = 2.5;
    for (let dz = -RX; dz <= RX; dz++) {
      for (let dx = -RX; dx <= RX; dx++) {
        const cc = ((hx + dx) & (NX - 1)) + ((hz + dz) & (NZ - 1)) * NX;
        const rr = (dx * dx + dz * dz) / (RX * RX);
        if (rr > 1) continue;
        const hy = RY * Math.sqrt(1 - rr);
        const yTop = Math.min(Math.floor(cy + hy), ty[cc]! - 3);
        for (let y = Math.max(1, Math.ceil(cy - hy)); y <= yTop; y++) {
          vox[cc + y * NCOL] = (Mat.MAGMA | (255 << 8) | (FLAG_HOTSPOT << 24)) >>> 0;
        }
      }
    }
  }

  // ---- plate table ----
  const area = new Float64Array(nPlates);
  const contArea = new Float64Array(nPlates);
  const maxOcean = new Float64Array(nPlates);
  for (let c = 0; c < NCOL; c++) {
    const p = plateId[c]!;
    area[p] = area[p]! + 1;
    if (colFlags[c]! & FLAG_CONTINENTAL) contArea[p] = contArea[p]! + 1;
    else if (oceanAge[c]! > maxOcean[p]!) maxOcean[p] = oceanAge[c]!;
  }
  let alive = 0;
  for (let i = 0; i < nPlates; i++) {
    const p = w.plates[i]!;
    p.alive = area[i]! > 0;
    if (!p.alive) continue;
    alive++;
    p.continental = contArea[i]! * 2 > area[i]!;
    // My since creation: continental plates old; oceanic ≈ oldest seafloor they carry
    p.age = p.continental ? rng.range(300, 1200) : Math.max(10, maxOcean[i]!);
  }
  if (alive < MIN_PLATES) throw new Error(`generateWorld: only ${alive} plates own columns (seed ${seed})`);

  w.mantleReservoir = Math.round(MANTLE_RESERVOIR_FRAC * crustMass(vox));
  return w;
}
