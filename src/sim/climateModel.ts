// §C pass 7 (climate) model: constants, analytic wind/temperature, CPU reference step.
// Pure CPU (no three import) so vitest can run it; climate.ts mirrors it op-for-op in TSL.
//
// Units
// - water, vapor, ice, precip: water-equivalent column depth in voxel-y units (same as 'water'),
//   so W_total = Σ water + Σ vapor + Σ ice (V4). precip is the amount that fell this tick.
// - surfTemp: °C-like. wind: cells per geo tick.
// - Rates are per geo tick. Climate is quasi-steady relative to geology; dtGeo is fixed (V12),
//   so each tick is one climate relaxation step and dtGeo does not scale the rates.
//
// Latitude on the torus (V1): z = 0 is the "equator" (warm belt), z = NZ/2 the "pole" (cold belt).
// Real-world-like latitude φ = π·d/NZ ∈ [0, π/2] with d = distance in cells to z = 0 (wrapped).
// Both halves of the Z axis act as mirrored hemispheres.
import { NX, NZ, NCOL, Y_SEA_NOMINAL, colIdx } from './layout';

export const CLIMATE = {
  // temperature
  T_BASE: 15,        // °C at sea level, mid latitude (12 made ~a third of land snow/tundra)
  T_AMP: 17,         // equator +T_AMP, pole -T_AMP around T_BASE
  LAPSE: 0.6,        // °C per voxel above sea level (stylized, ≈2.4 °C/km at 250 m/voxel)
  ICE_AGE_DT: 8,     // °C cooling at iceAge = 1, iceAgeStrength = 1
  // wind (cells/tick). Outflow fraction ≤ U0 + 2·V0 + 4·VAPOR_DIFF < 1 keeps vapor ≥ 0.
  U0: 0.35,          // zonal: u = -U0·cos(4φ): trade easterlies, mid-lat westerlies, polar easterlies
  V0: 0.08,          // meridional: v = -V0·sin(6φ) (poleward +): Hadley/Ferrel/polar cells
  VAPOR_DIFF: 0.05,  // eddy diffusion per face per tick (poleward/inland moisture transport)
  // moisture
  OCEAN_DEPTH: 1,    // water depth at which a column evaporates like open ocean
  EVAP_K: 0.005,     // evaporation per tick over warm open ocean at zero humidity
  EVAP_T0: -15,      // °C where evaporation stops
  EVAP_TSPAN: 35,    // evaporation factor = clamp((T - T0)/span, 0, EVAP_FMAX)
  EVAP_FMAX: 1.2,
  CAP0: 0.04,        // vapor capacity at 0 °C
  CAP_K: 0.06,       // capacity = CAP0·exp(CAP_K·T) (Clausius-Clapeyron-like)
  RAIN_K: 0.006,     // background precip fraction per tick at rh = 1 (∝ rh)
  SAT_K: 0.3,        // fraction of supersaturated vapor that rains out per tick
  ORO_K: 0.05,       // extra precip fraction per voxel/tick of forced ascent (wind·∇h)
  SNOW_T0: -1,       // °C: all snow at or below
  SNOW_T1: 1,        // °C: all rain at or above
  // ice
  MELT_K: 0.004,     // ice melt per °C above 0 per tick
  BASAL_K: 0.0002,   // basal melt fraction per tick (geothermal; bounds ice caps)
  SEA_ICE_MAX: 0.3,  // ice over ocean above this calves back into water
  CALVE_K: 0.05,     // calving fraction of the excess per tick
  GLACIER_K: 0.02,   // ice flux fraction per voxel of ice-surface height drop
  GLACIER_MAX: 0.2,  // max fraction of donor ice through one face per tick (4·0.2 < 1 keeps ice ≥ 0)
} as const;

export interface ClimateUniforms {
  seaLevel: number;       // voxel y
  iceAge: number;         // 0..1, driven by the event scheduler
  iceAgeStrength: number; // param
  rainfall: number;       // param, scales evaporation (hydrologic cycle strength)
}

export const DEFAULT_CLIMATE_UNIFORMS: ClimateUniforms = { seaLevel: Y_SEA_NOMINAL, iceAge: 0, iceAgeStrength: 1, rainfall: 1 };

const TAU = Math.PI * 2;

/** Latitude angle φ ∈ [0, π/2] and hemisphere sign (poleward = sign·(+z)) at z position p (cells, any real). */
export function latitude(p: number): { phi: number; hemi: number } {
  const q = ((p % NZ) + NZ) % NZ;
  const d = Math.min(q, NZ - q);
  return { phi: (d * Math.PI) / NZ, hemi: q < NZ / 2 ? 1 : -1 };
}

/** Zonal wind (along +X) at row z. */
export function windU(z: number): number {
  return -CLIMATE.U0 * Math.cos(4 * latitude(z).phi);
}

/** Meridional wind (along +Z) at z position p. */
export function windV(p: number): number {
  const { phi, hemi } = latitude(p);
  return -CLIMATE.V0 * Math.sin(6 * phi) * hemi;
}

/** Z face k (between rows k-1 and k), k wrapped to [0,NZ): position k - 0.5, wrapped into [0,NZ). */
export const faceZPos = (k: number) => (k === 0 ? NZ - 0.5 : k - 0.5);

export function seaLevelTemp(z: number, u: ClimateUniforms): number {
  return CLIMATE.T_BASE + CLIMATE.T_AMP * Math.cos((TAU * z) / NZ) - CLIMATE.ICE_AGE_DT * u.iceAge * u.iceAgeStrength;
}

/** Lapse by ground height above sea level; oceans and lakes sit at their floor's clamp (alt ≥ 0), so the
 *  temperature does not depend on how the hydro pass happens to pile water up. */
export function surfaceTemp(z: number, surfY: number, u: ClimateUniforms): number {
  return seaLevelTemp(z, u) - CLIMATE.LAPSE * Math.max(0, surfY - u.seaLevel);
}

export const vaporCapacity = (t: number) => CLIMATE.CAP0 * Math.exp(CLIMATE.CAP_K * t);

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

export interface ClimateState {
  surfY: Float32Array;
  water: Float32Array;
  vapor: Float32Array;
  ice: Float32Array;
  surfTemp: Float32Array;
  precip: Float32Array;
}

export function makeClimateState(): ClimateState {
  return {
    surfY: new Float32Array(NCOL), water: new Float32Array(NCOL), vapor: new Float32Array(NCOL),
    ice: new Float32Array(NCOL), surfTemp: new Float32Array(NCOL), precip: new Float32Array(NCOL),
  };
}

/** Orography seen by the wind: ocean is flat at sea level. */
const oroH = (surfY: number, seaLevel: number) => Math.max(surfY, seaLevel);

/** Ice-surface height drives glacier flow. */
const iceH = (surfY: number, ice: number) => surfY + ice;

/**
 * Net ice flow A→B for one face. Antisymmetric by construction: f(B,A) = -f(A,B) bit-exactly
 * (a-b = -(b-a) in IEEE), so the two threads sharing a face move identical amounts (V4).
 */
export function glacierFlux(hA: number, iA: number, hB: number, iB: number): number {
  const dh = hA - hB;
  return dh > 0
    ? Math.min(CLIMATE.GLACIER_K * dh, CLIMATE.GLACIER_MAX) * iA
    : -(Math.min(CLIMATE.GLACIER_K * -dh, CLIMATE.GLACIER_MAX) * iB);
}

/**
 * Vapor flux through a face (+ direction, from cell a to cell b): donor-cell upwind advection with
 * Courant number c plus eddy diffusion. Both threads sharing the face evaluate it with identical
 * inputs, so what leaves one column enters the other exactly (V4).
 */
export const vaporFlux = (c: number, a: number, b: number) => (c > 0 ? c * a : c * b) + CLIMATE.VAPOR_DIFF * (a - b);

/** Scratch for the two-kernel split (column physics → vaporSrc/iceSrc, then transport). GPU packs it as one vec2. */
export interface ClimateScratch { vaporSrc: Float32Array; iceSrc: Float32Array }
export const makeClimateScratch = (): ClimateScratch => ({ vaporSrc: new Float32Array(NCOL), iceSrc: new Float32Array(NCOL) });

const qf = new Float32Array(1), qu32 = new Uint32Array(qf.buffer);
/**
 * f32 ulp of the binade of `mag` (power of two, ≥ 2^-126), mag = |w| + |Δw|. w + (multiple of it) is
 * exact whenever the result does not climb above w's binade: always the case that matters, a small
 * Δw on deep water. Otherwise (thin water, big Δw) the rounding is ≤ ½ ulp of the result and unbiased.
 * Integer bit ops, so GPU fast-math cannot fold it.
 */
export function f32Quantum(mag: number): number {
  qf[0] = mag;
  qu32[0] = (Math.max((qu32[0]! >>> 23) & 0xff, 24) - 23) << 23;
  return qf[0]!;
}

/** Kernel 1 (per column, own-index writes only): temperature, evaporation, precip, snow, melt. */
export function climateColumnCpu(s: ClimateState, k: ClimateScratch, u: ClimateUniforms): void {
  const C = CLIMATE;
  for (let z = 0; z < NZ; z++) {
    const uz = windU(z), vz = windV(z);
    for (let x = 0; x < NX; x++) {
      const i = colIdx(x, z);
      const sy = s.surfY[i]!, w = s.water[i]!, q = s.vapor[i]!, ice = s.ice[i]!;
      const t = surfaceTemp(z, sy, u);
      const cap = vaporCapacity(t);
      // evaporation, bounded by available water and by humidity deficit
      // (max(w,0): hydro may leave float-dust negatives, which must not evaporate)
      const wp = Math.max(w, 0);
      const wet = Math.min(wp / C.OCEAN_DEPTH, 1);
      const fT = clamp((t - C.EVAP_T0) / C.EVAP_TSPAN, 0, C.EVAP_FMAX);
      const e = Math.min(u.rainfall * C.EVAP_K * fT * wet * Math.max(1 - q / cap, 0), wp * 0.5);
      const q1 = q + e;
      // forced ascent: wind · ∇h (central difference, wrapped)
      const gx = (oroH(s.surfY[colIdx(x + 1, z)]!, u.seaLevel) - oroH(s.surfY[colIdx(x - 1, z)]!, u.seaLevel)) * 0.5;
      const gz = (oroH(s.surfY[colIdx(x, z + 1)]!, u.seaLevel) - oroH(s.surfY[colIdx(x, z - 1)]!, u.seaLevel)) * 0.5;
      const lift = Math.max(uz * gx + vz * gz, 0);
      const rh = Math.min(q1 / cap, 1);
      const p = Math.min(C.RAIN_K * rh * q1 + C.SAT_K * Math.max(q1 - cap, 0) + C.ORO_K * lift * q1, q1);
      const snow = p * clamp((C.SNOW_T1 - t) / (C.SNOW_T1 - C.SNOW_T0), 0, 1);
      const rain = p - snow;
      const i1 = ice + snow;
      const calve = w > C.OCEAN_DEPTH ? C.CALVE_K * Math.max(i1 - C.SEA_ICE_MAX, 0) : 0;
      const melt = Math.min(C.MELT_K * Math.max(t, 0) + C.BASAL_K * i1 + calve, i1);
      // Deep water has a coarse f32 ulp: a small net change would round away (e.g. slow evaporation
      // from a 16-voxel ocean) and create/destroy water systematically. Water takes only the part of
      // Δw on its own ulp grid (exact add); the remainder stays in the fine-grained vapor (V4).
      const dw = rain + melt - e;
      const qu = f32Quantum(Math.abs(w) + Math.abs(dw));
      const dwq = Math.round(dw / qu) * qu;
      s.water[i] = w + dwq;
      k.vaporSrc[i] = q1 - p + (dw - dwq);
      k.iceSrc[i] = i1 - melt;
      s.surfTemp[i] = t;
      s.precip[i] = p;
    }
  }
}

/** Kernel 2 (gather, own-index writes only): donor-cell vapor advection + eddy diffusion + glacier flow. Exactly conservative. */
export function climateTransportCpu(s: ClimateState, k: ClimateScratch): void {
  const q = k.vaporSrc, ic = k.iceSrc, sy = s.surfY;
  for (let z = 0; z < NZ; z++) {
    const cx = windU(z);
    const cDown = windV(faceZPos(z)), cUp = windV(faceZPos((z + 1) & (NZ - 1)));
    for (let x = 0; x < NX; x++) {
      const i = colIdx(x, z), iL = colIdx(x - 1, z), iR = colIdx(x + 1, z), iD = colIdx(x, z - 1), iU = colIdx(x, z + 1);
      const fl = vaporFlux(cx, q[iL]!, q[i]!), fr = vaporFlux(cx, q[i]!, q[iR]!);
      const fd = vaporFlux(cDown, q[iD]!, q[i]!), fu = vaporFlux(cUp, q[i]!, q[iU]!);
      s.vapor[i] = q[i]! + fl - fr + fd - fu;
      const h = iceH(sy[i]!, ic[i]!);
      const out =
        glacierFlux(h, ic[i]!, iceH(sy[iL]!, ic[iL]!), ic[iL]!) + glacierFlux(h, ic[i]!, iceH(sy[iR]!, ic[iR]!), ic[iR]!) +
        glacierFlux(h, ic[i]!, iceH(sy[iD]!, ic[iD]!), ic[iD]!) + glacierFlux(h, ic[i]!, iceH(sy[iU]!, ic[iU]!), ic[iU]!);
      s.ice[i] = ic[i]! - out;
    }
  }
}

export function climateStepCpu(s: ClimateState, k: ClimateScratch, u: ClimateUniforms): void {
  climateColumnCpu(s, k, u);
  climateTransportCpu(s, k);
}

/** W_total contribution of the climate-coupled fields (V4), summed in f64. */
export function waterTotal(s: Pick<ClimateState, 'water' | 'vapor' | 'ice'>): number {
  let t = 0;
  for (let i = 0; i < NCOL; i++) t += s.water[i]! + s.vapor[i]! + s.ice[i]!;
  return t;
}
