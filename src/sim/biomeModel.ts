// §C pass 10 (derived) biome + vegetation model: enum, Whittaker-like classification, CPU reference.
// Pure CPU (no three import); biome.ts mirrors it in TSL.
import { NCOL } from './layout';

// Biome ids. APPEND-ONLY: persisted in saves and read by render/life.
export const Biome = {
  OCEAN: 0,
  ICE: 1,
  TUNDRA: 2,
  TAIGA: 3,
  TEMPERATE_FOREST: 4,
  GRASSLAND: 5,
  DESERT: 6,
  SAVANNA: 7,
  RAINFOREST: 8,
  BEACH: 9,
  ALPINE: 10,
} as const;
export type BiomeId = (typeof Biome)[keyof typeof Biome];
export const BIOME_COUNT = 11;

/** Vegetation cover each biome relaxes toward, indexed by biome id. */
export const BIOME_VEG_TARGET: readonly number[] = [0, 0, 0.2, 0.7, 0.85, 0.5, 0.05, 0.4, 1, 0.05, 0.15];

export const BIOME = {
  AVG_ALPHA: 0.02,     // EMA weight per tick for long-term temp/precip (≈50-tick memory)
  VEG_RATE: 0.01,      // veg relaxation fraction per tick toward biome target
  P_REF: 0.0005,       // precip per tick that counts as moisture index 1 (wet tropics)
  ICE_MIN: 0.2,        // ice (water-eq voxels) that makes a column ICE
  OCEAN_MIN: 1,        // water depth that makes a column OCEAN (incl. deep lakes)
  BEACH_ALT: 1,        // voxels above sea level
  ALPINE_ALT: 20,      // voxels above sea level
  // Whittaker bands: T in °C (long-term avg), m = precipAvg / P_REF
  T_TUNDRA: -5, T_BOREAL: 3, T_TROPIC: 20,
  M_TAIGA: 0.25,
  M_TEMP_DESERT: 0.12, M_TEMP_FOREST: 0.4,
  M_HOT_DESERT: 0.2, M_RAINFOREST: 0.7,
} as const;

/**
 * Erosion coupling (T31): the hydro/erosion pass multiplies erodibility by this.
 * Dense cover (veg = 1) cuts erosion by 70 %.
 */
export const VEG_ERODIBILITY_K = 0.7;
export const vegErodibilityFactor = (veg: number) => 1 - VEG_ERODIBILITY_K * veg;

export function classifyBiome(tAvg: number, pAvg: number, alt: number, water: number, ice: number): BiomeId {
  const B = BIOME;
  if (ice > B.ICE_MIN) return Biome.ICE;
  if (water > B.OCEAN_MIN) return Biome.OCEAN;
  const m = pAvg / B.P_REF;
  if (alt < B.BEACH_ALT && tAvg > 0) return Biome.BEACH;
  if (tAvg < B.T_BOREAL && alt > B.ALPINE_ALT) return Biome.ALPINE;
  if (tAvg < B.T_TUNDRA) return Biome.TUNDRA;
  if (tAvg < B.T_BOREAL) return m > B.M_TAIGA ? Biome.TAIGA : Biome.TUNDRA;
  if (tAvg < B.T_TROPIC) return m < B.M_TEMP_DESERT ? Biome.DESERT : m < B.M_TEMP_FOREST ? Biome.GRASSLAND : Biome.TEMPERATE_FOREST;
  return m < B.M_HOT_DESERT ? Biome.DESERT : m < B.M_RAINFOREST ? Biome.SAVANNA : Biome.RAINFOREST;
}

export interface BiomeState {
  surfTemp: Float32Array; precip: Float32Array; surfY: Float32Array; water: Float32Array; ice: Float32Array;
  climAvg: Float32Array; // interleaved vec2 (tAvg, pAvg)
  veg: Float32Array;
  biome: Uint32Array;
}

/** CPU reference of the biome kernel. */
export function biomeStepCpu(s: BiomeState, seaLevel: number): void {
  const a = BIOME.AVG_ALPHA;
  for (let i = 0; i < NCOL; i++) {
    const tAvg = s.climAvg[2 * i]! + (s.surfTemp[i]! - s.climAvg[2 * i]!) * a;
    const pAvg = s.climAvg[2 * i + 1]! + (s.precip[i]! - s.climAvg[2 * i + 1]!) * a;
    s.climAvg[2 * i] = tAvg;
    s.climAvg[2 * i + 1] = pAvg;
    const b = classifyBiome(tAvg, pAvg, s.surfY[i]! - seaLevel, s.water[i]!, s.ice[i]!);
    s.biome[i] = b;
    s.veg[i] = s.veg[i]! + (BIOME_VEG_TARGET[b]! - s.veg[i]!) * BIOME.VEG_RATE;
  }
}
