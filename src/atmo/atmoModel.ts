// §T.41/§T.42 atmosphere FX model: constants and pure CPU helpers shared by the TSL kernels and tests.
// No three import, so vitest could use it. Everything here is render-only: FX read sim fields and
// never write them (V15). Motion runs on ambTime (V17), never on the geo clock.
//
// World mapping (space.ts): the block spans x,z ∈ [-HALF, HALF], one column = CELL world units.
// Wind (climateModel.ts) is in cells per geo tick; FX turn it into world units per second with
// WIND_SCALE, and strengthen it with height (windProfile): surface bands below, and aloft a
// westerly regime (thermal-wind shear, return flow of the meridional cells, a mid-latitude jet) at
// the top of the troposphere slab, so high cloud drifts across the low layer.
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
  /**
   * Upper-level regime (above hf ALOFT_HF0, full at hf 1): the surface bands weaken (ALOFT_DAMP of
   * them is lost at the top), the meridional cells reverse (return flow, ALOFT_RETURN) and a westerly
   * thermal-wind shear (ALOFT_SHEAR world/s) is added at every latitude. Below ALOFT_HF0 the profile
   * is the plain surface bands (plumes, rain, low clouds unchanged); aloft the flow turns westerly,
   * so high cloud moves faster and in a different direction than the low layer (tropics: opposite).
   */
  ALOFT_HF0: 0.5,
  ALOFT_DAMP: 0.8,
  ALOFT_RETURN: 2,
  ALOFT_SHEAR: 0.07,
  /** Height fraction the cirrus layer steers with, and its turning (opposite sign to CURL's per band). */
  CIRRUS_HF: 1,
  CIRRUS_CURL: 0.3,

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
  /** Upper bound of that lean (world): the westerlies aloft would otherwise topple tropical towers. */
  TILT_MAX: 0.12,
  /** Global coverage scale: noise survives where it exceeds 1 - blurredCover·COVERAGE (≤ ~35 % sky). */
  COVERAGE: 0.75,
  /** Extra coverage threshold under full storm precip (may exceed 1: storm cores stay filled, only rims fray). */
  STORM_FILL: 0.35,
  /**
   * Coverage noise evolves through the 3D noise's third axis, in cycles per ambTime period (whole numbers:
   * seamless when ambTime wraps). EVOLVE: the large-scale pattern (regional cloudiness drifts slowly; even,
   * the domain warp runs at half rate). LIFE: the cell-scale octave, so single clouds form, grow and
   * dissolve over ~20-40 s while the region's cloud amount stays steady (cirrus morphs at LIFE / 2).
   */
  EVOLVE: 6,
  LIFE: 30,
  /** Cloud base above the tallest ground nearby (voxels) and absolute floor above sea (voxels). */
  BASE_CLEAR: 9,
  BASE_ABOVE_SEA: 22,

  // ---- terrain coupling (weather map; reads surfY / water directly, so it follows new relief at once) ----
  /** Upwind look-back distances (world) along the low-level drift for slope and rain-shadow tests. */
  OROG_UP: [0.09, 0.2, 0.34],
  /** Terrain rise (voxels) over OROG_UP[0] for full windward lift; upwind excess height (voxels) for a full rain shadow. */
  OROG_RISE: 6,
  LEE_H0: 2, LEE_H1: 10,
  /** Height above sea (voxels) where peaks start to wear cap clouds / wear them fully, and their prominence over the ~12-column mean ground. */
  CAP_H0: 18, CAP_H1: 34, CAP_PROM0: 1, CAP_PROM1: 6,
  /** Relative humidity where lifted air starts / fully condenses (orographic + cap cloud). */
  OROG_RH0: 0.4, OROG_RH1: 0.8,
  /** Cover added by windward lift, peak caps, warm humid land convection, cold-ocean stratus; lee cut; desert damping. */
  OROG_GAIN: 0.75, CAP_GAIN: 0.6, CONV_GAIN: 0.45, STRAT_GAIN: 0.35, LEE_CUT: 0.85, DESERT_CUT: 0.9,
  /** Convection: warm land (°C) from / full, humid (rh) from / full. Cold ocean stratus: sea colder than (°C) from / full. */
  CONV_T0: 12, CONV_T1: 24, CONV_RH0: 0.5, CONV_RH1: 0.85,
  STRAT_T0: 8, STRAT_T1: 2,
  /** Hugging cloud base on lifted flanks / caps: voxels above the local mean ground (caps swallow the summit). */
  HUG_CLEAR: 1, CAP_SINK: 4,
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
  CIRRUS_COVER: 0.34,
  CIRRUS_DENS: 0.5,
  /** Cirrus noise tile along the upper wind (world): long streaks. */
  CIRRUS_TILE: 4.8,
  /** Share of the march's detail erosion cirrus gets (wispy edges, but the veil survives), and its step length (× in-cloud step). */
  CIRRUS_ERO: 0.35,
  CIRRUS_STEP: 2.5,
  /** Cirrus share of the cloud shadow (a thin ice veil barely dims the sun). */
  CIRRUS_SHADOW: 0.15,

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
  FOUNTAIN_SHARE: 0.26,
  BOMB_SHARE: 0.14,
  /** Gravity for ejecta (world/s², diorama scale: a fast fountain reaches ~0.15 world). */
  GRAVITY: 1.1,
  /**
   * Lava bombs fly in slow motion (slower launch, lower gravity, more drag): heavy arcs that hang ~1 s and
   * come down on the flanks a few cells out, instead of streaking off.
   */
  BOMB_GRAVITY: 0.5, BOMB_DRAG: 0.45,
  /** Explosive openings (natural BUILD → ACTIVE): at most one blast per BLAST_GAP s; vents within BLAST_NEAR (world) are the same vent. */
  BLAST_GAP: 3, BLAST_NEAR: 0.1,
  /** Pyroclastic density currents: mean episodes per second at a full-heat vent. */
  PDC_RATE: 1 / 30,
  /** Volcanic lightning inside big ash columns: flashes/s at a full-heat vent at night (day × 0.15). */
  VOLC_FLASH_RATE: 0.4,
  BURSTS_MAX: 12,
  /** Big eruption plumes rendered in the cloud volume (column + umbrella). */
  PLUMES: 6,
  /** Volume plumes: umbrella drift length (× the wind-based length), ease-in / fade-out rates (1/s). */
  PLUME_DRIFT: 1.7, PLUME_RISE: 0.35, PLUME_DECAY: 0.1,
  /** Longest umbrella trail (world): the aloft wind is fast, and an endless canopy reads as detached cloud. */
  PLUME_LEN_MAX: 0.9,
  /**
   * Plume particles steer with the wind at height fraction ≤ PLUME_HF (below the aloft regime) and drift at
   * most PLUME_DRIFT_MAX world/s (× 2.5 for strong eruption bursts): a small plume stays within ~0.3 world of
   * its vent over its life, a big column trails further but stays attached to its umbrella.
   */
  PLUME_HF: 0.45, PLUME_DRIFT_MAX: 0.035,
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
export const BURST = { ASH: 0, DUST: 1, HAZE: 2, STEAM: 3, FOUNTAIN: 4, BOMB: 5, PDC: 6, TEPHRA: 7 } as const;
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
  const a = smooth(ATMO.ALOFT_HF0, 1, hf); // upper-level regime (0 below ALOFT_HF0)
  return {
    u: windU(c) * k * (1 - ATMO.ALOFT_DAMP * a) + jet + ATMO.ALOFT_SHEAR * a,
    v: windV(((c % NZ) + NZ) % NZ) * k * (1 - ATMO.ALOFT_RETURN * a),
  };
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
  return bandMean(ATMO.CLOUD_HF, ATMO.CURL).map((w) => ({ u: Math.round(w.u / q) * q, v: Math.round(w.v / q) * q }));
}

/** Band-average wind at height fraction hf, turned by curl × |u| (alternating sign per band). */
function bandMean(hf: number, curl: number): { u: number; v: number }[] {
  return Array.from({ length: BANDS }, (_, i) => {
    let su = 0, sv = 0;
    const n = 32;
    for (let k = 0; k < n; k++) {
      const cell = i * (NZ / BANDS) - NZ / (2 * BANDS) + ((k + 0.5) / n) * (NZ / BANDS);
      const w = windProfile((cell + 0.5) * CELL - HALF, hf);
      su += w.u; sv += w.v;
    }
    su /= n; sv /= n;
    return { u: su, v: sv + curl * Math.abs(su) * (i % 2 === 0 ? 1 : -1) };
  });
}

/**
 * Cirrus drift per band (world/s, x and z) at the top of the profile (CIRRUS_HF), turned the other way
 * from the low layer: the streaks translate rigidly along their own wind (no shear inside a band) and
 * are stretched along it. The speed is quantised so one ambTime period moves the streak noise by a
 * whole number of tiles along the wind (CIRRUS_TILE): seamless when ambTime wraps.
 */
export function cirrusBands(): { u: number; v: number }[] {
  const q = ATMO.CIRRUS_TILE / AMB_PERIOD;
  return bandMean(ATMO.CIRRUS_HF, -ATMO.CIRRUS_CURL).map((w) => {
    const s = Math.hypot(w.u, w.v), sq = Math.max(1, Math.round(s / q)) * q;
    return { u: (w.u / s) * sq, v: (w.v / s) * sq };
  });
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
