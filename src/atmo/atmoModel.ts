// §T.41/§T.42 atmosphere FX model: constants and pure CPU helpers shared by the TSL kernels and tests.
// No three import, so vitest could use it. Everything here is render-only: FX read sim fields and
// never write them (V15). Motion runs on ambTime (V17), never on the geo clock.
//
// World mapping (space.ts): the block spans x,z ∈ [-HALF, HALF], one column = CELL world units.
// Wind (climateModel.ts) is in cells per geo tick; FX turn it into world units per second with
// WIND_SCALE, and strengthen it with height (windProfile): surface bands below, a mid-latitude
// westerly jet at the top of the troposphere slab.
import { NZ, CELL, BLOCK_SIZE } from '../sim/layout';
import { CLIMATE, windU, windV } from '../sim/climateModel';

/** Must equal space.ts AMB_PERIOD (ambTime wraps there, V11). Duplicated to keep this file three-free. */
export const AMB_PERIOD = 3600;
export const HALF = BLOCK_SIZE / 2;

export const ATMO = {
  // ---- wind on the ambience clock ----
  /** Visual cells/s per sim cell/tick at the surface: max zonal wind 0.35 → ~3 cells/s. */
  WIND_SCALE: 9,
  /** Height (world, above sea level) over which the wind profile ramps to full strength. */
  WIND_H: 0.9,
  /** Surface → top multiplier of the band wind, and the jet (world/s, +x, peaks at mid-latitudes). */
  WIND_TOP: 2.2,
  JET: 0.07,
  /** Meridional turning of the band drift (fraction of |u|): diagonal, never grid-aligned. */
  CURL: 0.55,
  /** Height fraction (of WIND_H) the cumulus layer steers with. */
  CLOUD_HF: 0.4,

  // ---- climate summary (coarse grid over the columns) ----
  /** Summary grid: SUM × SUM cells of SUM_STEP² columns. */
  SUM: 64,
  /** Relative humidity (vapor / capacity(surfTemp)) where clouds start / are full. Ocean median ≈ 0.84. */
  RH0: 0.74,
  RH1: 0.98,
  /** precip per tick where clouds start / storm cells start / storm is full (95-99.5 % of columns). */
  PRECIP_CLOUD: 3.5e-4,
  STORM0: 1.4e-3,
  STORM1: 4.0e-3,
  /** Rain / snow streaks fall under columns with precip above this (≈ top 10 % of columns). */
  PRECIP_RAIN: 8e-4,

  // ---- weather map (2D coverage / type / base, drives the cloud volume) ----
  WEATHER_RES: 128,
  /** Gaussian blur of the sim cover/precip field, in summary cells (4 columns): σ ≈ 9 columns. */
  BLUR_SIGMA: 2.2,
  /** Coverage noise tile (world; the domain warp uses 2×) and warp strength (fraction of the tile). */
  COV_TILE: 2.4,
  WARP: 0.2,
  /** All drifting noise tiles divide this (world), so band drift over one ambTime period wraps seamlessly. */
  DRIFT_TILE: 4.8,
  /** Towers lean downwind: the height-dependent part of the wind is applied over this many seconds. */
  TILT_S: 5,
  /** Global coverage scale: noise survives where it exceeds 1 - blurredCover·COVERAGE (≤ ~35 % sky). */
  COVERAGE: 0.75,
  /** Coverage noise evolves through the 3D noise's third axis: cycles per ambTime period. */
  EVOLVE: 5,
  /** Cloud base above the tallest ground nearby (voxels) and absolute floor above sea (voxels). */
  BASE_CLEAR: 9,
  BASE_ABOVE_SEA: 22,
  /** Lightning / test probe grid over the weather map. */
  CELLS: 32,
  /** Wet-haze curtain sprites per side. */
  CURTAIN_GRID: 20,

  // ---- cloud volume (raymarched) ----
  /** Density volume VX × VY × VX over the block × slab; the light volume is half resolution. */
  VX: 128, VY: 64,
  /** Slab: bottom at sea + SLAB_BASE voxels, SLAB_H world units tall. */
  SLAB_BASE: 16,
  SLAB_H: 0.85,
  /** Extinction per world unit at density 1. */
  SIGMA: 38,
  /** Detail noise tile (world); the base noise tile is 4× so wrapped offsets tile both. */
  DETAIL_TILE: 0.3,
  /** Raymarch iteration budget and the in-cloud step (world units). */
  STEPS_HIGH: 56,
  STEPS_LOW: 36,
  STEP_IN_HIGH: 0.02,
  STEP_IN_LOW: 0.03,
  /** Raymarch resolution (fraction of the drawing buffer) and history blend (weight of the new frame). */
  RES_HIGH: 0.5,
  RES_LOW: 0.25,
  TEMPORAL: 0.2,
  /** World units: clouds fade to nothing within this distance of the camera. */
  NEAR_FADE0: 0.25,
  NEAR_FADE1: 1.1,
  /** Cirrus: sparse high streaks, share of the sky and peak density. */
  CIRRUS_COVER: 0.28,
  CIRRUS_DENS: 0.16,

  // ---- cloud shadow texture ----
  SHADOW_RES: 128,
  /** Max darkening of direct sun under the thickest cloud (1 = black). */
  SHADOW_MAX: 0.62,

  // ---- plumes ----
  PARTICLES_HIGH: 6144,
  PARTICLES_LOW: 2048,
  VENTS_MAX: 256,
  /** Lava hotter than this (°C) at a column counts as an active vent; full strength at VENT_T1. */
  VENT_T0: 850,
  VENT_T1: 1150,
  /** Water depth (voxel-y) that makes a hot vent a steam vent (lava.ts wetDepth = 0.05). */
  STEAM_WET: 0.05,
  /** Particles per second per full-strength vent; at most VENTS_EMIT vents' worth in total. */
  VENT_RATE: 34,
  VENTS_EMIT: 14,
  /** Share of a vent's spawns that are lava-fountain spray / ballistic bombs (× vent heat, × heat² for bombs). */
  FOUNTAIN_SHARE: 0.22,
  BOMB_SHARE: 0.08,
  /** Gravity for ejecta (world/s², diorama scale: a fast fountain reaches ~0.15 world). */
  GRAVITY: 1.1,
  /** Pyroclastic density currents: mean episodes per second at a full-heat vent. */
  PDC_RATE: 1 / 30,
  /** Volcanic lightning inside big ash columns: flashes/s at a full-heat vent at night (day × 0.15). */
  VOLC_FLASH_RATE: 0.4,
  BURSTS_MAX: 12,
  /** Big eruption plumes rendered in the cloud volume (column + umbrella). */
  PLUMES: 6,
  /** Volume plumes: umbrella drift length (× the wind-based length), ease-in / fade-out rates (1/s). */
  PLUME_DRIFT: 1.7, PLUME_RISE: 0.35, PLUME_DECAY: 0.1,
  /** Steam (degassing wisps, sea blasts): particle lifetime multiplier, so plumes trail off downwind. */
  STEAM_LIFE: 1.6,
  // hydrothermal vents on spreading ridges (young crust)
  /** crustAge (My) below which a column counts as a fresh rift / ridge axis. */
  HYDRO_AGE: 1,
  /** Fraction of summary cells (4×4 columns) with young crust that host a vent (deterministic hash). */
  HYDRO_FRAC: 0.3,
  HYDRO_MAX: 64,
  /** Particles per second per hydrothermal vent, and the cap over all of them (small budget). */
  HYDRO_RATE: 5,
  HYDRO_CAP: 110,

  // ---- rain / snow ----
  RAIN_HIGH: 4096,
  RAIN_LOW: 1536,
  /** Surface temperature below which precipitation falls as snow (climateModel SNOW_T0..T1 midpoint). */
  SNOW_T: 0,
  RAIN_SPEED: 0.8,   // world units / s
  SNOW_SPEED: 0.12,

  // ---- lightning ----
  STORM_LIGHTNING: 0.55,
  /** Mean flashes per second per fully stormy cloudlet. */
  FLASH_RATE: 0.06,
} as const;

/** Burst kinds (event-driven emitters); order is the GPU code. */
export const BURST = { ASH: 0, DUST: 1, HAZE: 2, STEAM: 3, FOUNTAIN: 4, BOMB: 5, PDC: 6 } as const;
/** Particle kinds: ash/steam/fountain spray/bombs/tephra jets from vents, pyroclastic currents, dust/haze from impacts and flood basalts, bubble/smoker/turbid clouds underwater, floating pumice. */
export const PK = { ASH: 0, STEAM: 1, DUST: 2, HAZE: 3, BUBBLE: 4, SMOKER: 5, TURBID: 6, FOUNTAIN: 7, BOMB: 8, PDC: 9, PUMICE: 10, TEPHRA: 11, SLICK: 12 } as const;
/** Underwater kinds (drawn before the water sheet): BUBBLE..TURBID. */
export const PK_UNDER = [PK.BUBBLE, PK.TURBID] as const;

/** Vent phase classes in the vent list (magma.ts PHASE for eruption episodes; 0 = plain hot lava pool). */
export const VENT_PHASE = { POOL: 0, BUILD: 1, ACTIVE: 2, WANING: 3 } as const;
/** Water depth classes at a vent (voxel-y): deep = underwater only, medium = steam, shallow = Surtseyan. */
export const VENT_DEEP = 4, VENT_SHALLOW = 1;
/** Decode a vent list entry's w (see plumes.ts vent kernel). */
export function decodeVent(w: number): { heat: number; depth: number; phase: number; coast: number } {
  const code = Math.floor(w / 2);
  return { heat: w - code * 2, depth: (code % 64) / 4, phase: Math.floor(code / 64) % 4, coast: Math.floor(code / 256) };
}

/** World z → continuous cell coordinate (cell c centre at c), as space.ts worldToCell. */
export const zToCell = (z: number) => (z + HALF) / CELL - 0.5;

/** Latitude φ ∈ [0, π/2] at world z (climateModel.latitude). */
const phiAt = (z: number) => { const q = ((zToCell(z) % NZ) + NZ) % NZ; return (Math.min(q, NZ - q) * Math.PI) / NZ; };

/**
 * Wind (world units / s) at world z and height fraction hf ∈ [0,1] above sea level (of WIND_H):
 * the sim's analytic bands, stronger aloft, plus a westerly jet at mid-latitudes near the top.
 * Mirrored in TSL by atmoTsl.tWind.
 */
export function windProfile(z: number, hf: number): { u: number; v: number } {
  const c = zToCell(z);
  const k = ATMO.WIND_SCALE * CELL * (1 + (ATMO.WIND_TOP - 1) * hf);
  const jet = ATMO.JET * Math.sin(2 * phiAt(z)) ** 2 * hf * hf;
  return { u: windU(c) * k + jet, v: windV(((c % NZ) + NZ) % NZ) * k };
}

/** Wind bands of the analytic model across the torus: centred on cells 0 (trades), 64 (westerlies), 128 (polar), 192 (westerlies). */
export const BANDS = 4;
/**
 * Band-average steering wind (world/s, x and z) per band, quantised to DRIFT_TILE per ambTime period.
 * Clouds translate rigidly inside a band (no shear stretching them into streaks) and cross-fade where
 * two bands meet, like convergence zones. The meridional part is the model's windV plus a turning
 * component (CURL × |u|, alternating per band: trades toward the equator, westerlies poleward), so the
 * drift is diagonal and never runs along the block's grid axes.
 */
export function bandSpeeds(): { u: number; v: number }[] {
  const q = ATMO.DRIFT_TILE / AMB_PERIOD;
  return Array.from({ length: BANDS }, (_, i) => {
    let su = 0, sv = 0;
    const n = 32;
    for (let k = 0; k < n; k++) {
      const cell = i * (NZ / BANDS) - NZ / (2 * BANDS) + ((k + 0.5) / n) * (NZ / BANDS);
      const w = windProfile((cell + 0.5) * CELL - HALF, ATMO.CLOUD_HF);
      su += w.u; sv += w.v;
    }
    su /= n; sv /= n;
    const turn = ATMO.CURL * Math.abs(su) * (i % 2 === 0 ? 1 : -1);
    return { u: Math.round(su / q) * q, v: Math.round((sv + turn) / q) * q };
  });
}

/** Cirrus drift (world/s along +x): uniform so the streaks never shear apart; quantised so the noise tile wraps seamlessly with ambTime. */
export function cirrusSpeed(): number {
  const q = ATMO.DRIFT_TILE / AMB_PERIOD;
  return Math.round((ATMO.JET * 0.9 + ATMO.WIND_SCALE * CELL * CLIMATE.U0) / q) * q;
}

/** Saturation vapor capacity (same as climateModel.vaporCapacity). */
export const capacity = (t: number) => CLIMATE.CAP0 * Math.exp(CLIMATE.CAP_K * t);

const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** Cloud cover 0..1 of one column from vapor, precip and surfTemp (mirrors the summary kernel). */
export function columnCover(vapor: number, precip: number, temp: number): number {
  const rh = vapor / capacity(temp);
  return Math.min(1, smooth(ATMO.RH0, ATMO.RH1, rh) * 0.8 + smooth(ATMO.PRECIP_CLOUD, ATMO.STORM0, precip) * 0.7);
}

/** Storm strength 0..1 from precip. */
export const columnStorm = (precip: number) => smooth(ATMO.STORM0, ATMO.STORM1, precip);
