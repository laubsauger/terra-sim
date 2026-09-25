// §C pass 10 (derived) biome + vegetation model: enum, Whittaker-like classification, CPU reference.
// Pure CPU (no three import); biome.ts mirrors it in TSL.
import { NCOL, NX, NZ } from './layout';
import { CLIMATE } from './climateModel';

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
  STEPPE: 11,      // cool semi-arid grassland
  SHRUBLAND: 12,   // warm semi-arid / mediterranean scrub
  COLD_DESERT: 13, // dry and cold (above the treeline temperature)
} as const;
export type BiomeId = (typeof Biome)[keyof typeof Biome];
export const BIOME_COUNT = 14;

/**
 * Nominal vegetation cover per biome id (hand-painted worlds, ICE/OCEAN/BEACH, and the cold side of
 * the treeline). The sim's target above the treeline is the continuous climateVeg(t, m) instead, so
 * veg has no cliffs at Whittaker borders; the table values sit inside each biome's climateVeg range.
 */
export const BIOME_VEG_TARGET: readonly number[] = [0, 0, 0.2, 0.7, 0.85, 0.5, 0.05, 0.4, 1, 0.05, 0.3, 0.3, 0.3, 0.08];

export const BIOME = {
  AVG_ALPHA: 0.02,     // EMA weight per tick for long-term temp/precip (≈50-tick memory)
  VEG_RATE: 0.01,      // veg relaxation fraction per tick toward its target
  P_REF: 0.0005,       // precip per tick that counts as moisture index 1 (wet tropics)
  ICE_MIN: 0.2,        // ice (water-eq voxels) that makes a column ICE
  OCEAN_MIN: 1,        // water depth that makes a column OCEAN (incl. deep lakes)
  BEACH_ALT: 1,        // voxels above sea level, and a 4-neighbour must be OCEAN-deep water
  // Treeline temperature (°C, long-term avg) rises with ground height: polar treeline in lowlands,
  // alpine treeline on high ground: T_tree = mix(TREE_T_LOW, TREE_T_HIGH, smoothstep(TREE_ALT0, TREE_ALT1, alt)).
  TREE_T_LOW: -8, TREE_T_HIGH: 2, TREE_ALT0: 4, TREE_ALT1: 16,
  TREE_BAND: 1.5,      // ±°C around T_tree over which veg blends cold ↔ warm biome target
  ROCK_T: 10,          // °C below T_tree where cold veg has faded to bare rock / barren ground
  ALPINE_ALT: 6,       // below T_tree: ALPINE if higher than this and the cold is due to altitude, else TUNDRA
  // Whittaker bands above the treeline. T in °C (long-term avg), m = precipAvg / P_REF
  T_BOREAL: 3, T_WARM: 12, T_TROPIC: 20,
  M_DRY: 0.05,         // < : (cold) desert, temperate bands
  M_TAIGA: 0.2,        // boreal: steppe below, taiga above
  M_SEMI: 0.2,         // temperate: steppe (cool) | shrubland (warm) below, grassland above
  M_FOREST: 0.5,       // temperate: forest above
  M_TROP_DRY: 0.06, M_TROP_SEMI: 0.2, M_RAINFOREST: 0.7, // tropics: desert | shrubland | savanna | rainforest
} as const;

export const smoothstep = (a: number, b: number, x: number) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

/** Treeline temperature at a ground height above sea level. */
export const treelineTemp = (alt: number) =>
  BIOME.TREE_T_LOW + (BIOME.TREE_T_HIGH - BIOME.TREE_T_LOW) * smoothstep(BIOME.TREE_ALT0, BIOME.TREE_ALT1, alt);

/** Whittaker-like class for a climate warm enough for trees (t ≥ treeline). */
function warmBiome(t: number, m: number): BiomeId {
  const B = BIOME;
  if (t < B.T_BOREAL) return m < B.M_DRY ? Biome.COLD_DESERT : m < B.M_TAIGA ? Biome.STEPPE : Biome.TAIGA;
  if (t < B.T_WARM) return m < B.M_DRY ? Biome.COLD_DESERT : m < B.M_SEMI ? Biome.STEPPE : m < B.M_FOREST ? Biome.GRASSLAND : Biome.TEMPERATE_FOREST;
  if (t < B.T_TROPIC) return m < B.M_DRY ? Biome.DESERT : m < B.M_SEMI ? Biome.SHRUBLAND : m < B.M_FOREST ? Biome.GRASSLAND : Biome.TEMPERATE_FOREST;
  return m < B.M_TROP_DRY ? Biome.DESERT : m < B.M_TROP_SEMI ? Biome.SHRUBLAND : m < B.M_RAINFOREST ? Biome.SAVANNA : Biome.RAINFOREST;
}

/**
 * Continuous vegetation target above the treeline: moisture sets cover (desert 0.05 → steppe/shrub
 * ≈0.2 → grassland ≈0.45 → forest ≥0.8), cool boreal climates are sparser.
 */
export function climateVeg(t: number, m: number): number {
  const moist = 0.05 + 0.25 * smoothstep(0.03, 0.2, m) + 0.3 * smoothstep(0.2, 0.5, m) + 0.4 * smoothstep(0.45, 0.9, m);
  return moist * (0.75 + 0.25 * smoothstep(BIOME.T_BOREAL - 3, BIOME.T_WARM, t));
}

/**
 * Biome id and vegetation target. `coastal`: a 4-neighbour holds OCEAN-deep water (default true
 * for callers without neighbourhood info). alt = surfY - seaLevel.
 */
export function classifyBiomeVeg(tAvg: number, pAvg: number, alt: number, water: number, ice: number, coastal = true): { biome: BiomeId; veg: number } {
  const B = BIOME, V = BIOME_VEG_TARGET;
  if (ice > B.ICE_MIN) return { biome: Biome.ICE, veg: V[Biome.ICE]! };
  if (water > B.OCEAN_MIN) return { biome: Biome.OCEAN, veg: V[Biome.OCEAN]! };
  if (alt < B.BEACH_ALT && tAvg > 0 && coastal) return { biome: Biome.BEACH, veg: V[Biome.BEACH]! };
  const m = pAvg / B.P_REF;
  const tt = treelineTemp(alt);
  // cold because of altitude (lowland at this latitude would be above the treeline) → ALPINE
  const cold = alt > B.ALPINE_ALT && tAvg + CLIMATE.LAPSE * alt >= tt ? Biome.ALPINE : Biome.TUNDRA;
  const tw = Math.max(tAvg, tt);
  const warm = warmBiome(tw, m);
  // cold side fades to bare rock further below the line and to barren ground when dry (polar desert)
  const coldVeg = V[cold]! * smoothstep(tt - B.ROCK_T, tt - B.TREE_BAND, tAvg) * smoothstep(0, 0.15, m);
  const veg = coldVeg + (climateVeg(tw, m) - coldVeg) * smoothstep(tt - B.TREE_BAND, tt + B.TREE_BAND, tAvg);
  return { biome: tAvg < tt ? cold : warm, veg };
}

export function classifyBiome(tAvg: number, pAvg: number, alt: number, water: number, ice: number, coastal = true): BiomeId {
  return classifyBiomeVeg(tAvg, pAvg, alt, water, ice, coastal).biome;
}

/**
 * Erosion coupling (T31): the hydro/erosion pass multiplies erodibility by this.
 * Dense cover (veg = 1) cuts erosion by 70 %.
 */
export const VEG_ERODIBILITY_K = 0.7;
export const vegErodibilityFactor = (veg: number) => 1 - VEG_ERODIBILITY_K * veg;

export interface BiomeState {
  surfTemp: Float32Array; precip: Float32Array; surfY: Float32Array; water: Float32Array; ice: Float32Array;
  climAvg: Float32Array; // interleaved vec2 (tAvg, pAvg)
  veg: Float32Array;
  biome: Uint32Array;
}

/** CPU reference of the biome kernel. */
export function biomeStepCpu(s: BiomeState, seaLevel: number): void {
  const a = BIOME.AVG_ALPHA;
  const deep = (x: number, z: number) => s.water[(x & (NX - 1)) + (z & (NZ - 1)) * NX]! > BIOME.OCEAN_MIN;
  for (let i = 0; i < NCOL; i++) {
    const tAvg = s.climAvg[2 * i]! + (s.surfTemp[i]! - s.climAvg[2 * i]!) * a;
    const pAvg = s.climAvg[2 * i + 1]! + (s.precip[i]! - s.climAvg[2 * i + 1]!) * a;
    s.climAvg[2 * i] = tAvg;
    s.climAvg[2 * i + 1] = pAvg;
    const x = i & (NX - 1), z = i >> Math.log2(NX);
    const coastal = deep(x + 1, z) || deep(x - 1, z) || deep(x, z + 1) || deep(x, z - 1);
    const c = classifyBiomeVeg(tAvg, pAvg, s.surfY[i]! - seaLevel, s.water[i]!, s.ice[i]!, coastal);
    s.biome[i] = c.biome;
    s.veg[i] = s.veg[i]! + (c.veg - s.veg[i]!) * BIOME.VEG_RATE;
  }
}
