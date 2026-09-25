// §T.45 tiny life: pure CPU model shared by the GPU flora kernel (flora.ts), the CPU creatures
// (creatures.ts) and tests. No three import. Life only reads sim fields (V15).
import { Biome, BIOME_COUNT } from '../sim/biomeModel';
import { NX, NZ, BLOCK_SIZE, CELL } from '../sim/layout';

// ---- flora ----

/**
 * Biome ids life handles beyond the ones biomeModel.ts exports today. Ids are final (climate agent);
 * switch to Biome.* once they land there. LIFE_BIOMES bounds every per-biome table here.
 */
export const LBiome = { ...Biome, STEPPE: 11, SHRUBLAND: 12, COLD_DESERT: 13 } as const;
export const LIFE_BIOMES = Math.max(BIOME_COUNT, 14);

/** Plant species. 0 = empty slot. Species picks mesh kind, colour and size. Not persisted, so ids may change freely. */
export const Sp = {
  NONE: 0,
  BROADLEAF: 1,
  BROADLEAF_RAIN: 2,
  PINE: 3,
  CACTUS: 4,
  SHRUB_DRY: 5,
  GRASS: 6,
  GRASS_DRY: 7,
  FLOWER: 8,
  ACACIA: 9,
  SHRUB_TUNDRA: 10, // also the low shrub at the treeline
  FERN: 11,
  GRASS_ALPINE: 12, // alpine meadow / treeline tufts
  SHRUB_SAGE: 13,   // shrubland: dense rounded olive/sage scrub
  CYPRESS: 14,      // shrubland: columnar cypress
  GRASS_STEPPE: 15,
  SHRUB_STEPPE: 16,
  HARDY_TREE: 17,   // steppe: occasional small tough tree
  CUSHION: 18,      // cold desert: sparse cushion shrubs
  // fine ground cover (fine slot set)
  F_GRASS: 19, F_GRASS_TALL: 20, F_SEED: 21, F_CLOVER: 22, F_FLOWER: 23, F_STRAW: 24, FEATHER: 25, REED: 26,
  // small set extras
  TUSSOCK: 27, SAGEBRUSH: 28, BUSH_GREEN: 29, BOULDER: 30, STONE: 31,
  // large set extras
  HEDGE: 32,
} as const;
export const SPECIES_COUNT = 33;

/** Mesh kinds (one indirect draw each). KIND_SET: 0 = large slot set, 1 = small, 2 = fine ground cover. */
export const Kind = {
  TREE: 0, PINE: 1, CACTUS: 2, ACACIA: 3, SHRUB: 4, GRASS: 5, FLOWER: 6, CYPRESS: 7,
  ROCK: 8, TUSSOCK: 9, HEDGE: 10, F_SHORT: 11, F_TALL: 12, F_SEED: 13, F_CLOVER: 14, F_FLOWER: 15,
} as const;
export const KIND_COUNT = 16;
export const KIND_NAMES = ['tree', 'pine', 'cactus', 'acacia', 'shrub', 'grass', 'flower', 'cypress',
  'rock', 'tussock', 'hedge', 'fshort', 'ftall', 'fseed', 'fclover', 'fflower'] as const;
export const KIND_SET: readonly number[] = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 2, 2, 2, 2, 2];

/**
 * Per species: mesh kind, world height of the unit mesh at scale 1, foliage + trunk colour (sRGB),
 * wind flex (sway multiplier), static windswept lean.
 */
const SPECIES_TABLE: readonly [number, number, number, number, number, number][] = [
  [-1, 0, 0xff00ff, 0xff00ff, 0, 0],
  [Kind.TREE, 0.018, 0x6cb743, 0x7a5236, 0.6, 0],     // BROADLEAF
  [Kind.TREE, 0.022, 0x2e7a3c, 0x5a3d27, 0.5, 0],     // BROADLEAF_RAIN
  [Kind.PINE, 0.024, 0x2f6b52, 0x654630, 0.5, 0],     // PINE
  [Kind.CACTUS, 0.016, 0x74a95e, 0x5f8f4e, 0.1, 0],   // CACTUS
  [Kind.SHRUB, 0.006, 0x8f9a58, 0x7a6243, 0.6, 0],    // SHRUB_DRY
  [Kind.GRASS, 0.0065, 0x7cc64a, 0x5c9a3a, 1.2, 0],    // GRASS
  [Kind.GRASS, 0.006, 0xd5b95c, 0xa08840, 1.2, 0],    // GRASS_DRY
  [Kind.FLOWER, 0.005, 0x76b84a, 0x4f9a3c, 1.0, 0],   // FLOWER
  [Kind.ACACIA, 0.016, 0x93a848, 0x6e5038, 0.4, 0],   // ACACIA
  [Kind.SHRUB, 0.006, 0x8c7046, 0x6f5a3a, 0.6, 0],    // SHRUB_TUNDRA
  [Kind.GRASS, 0.006, 0x3f9642, 0x2f7434, 1.0, 0],    // FERN
  [Kind.GRASS, 0.0045, 0x8fbf5a, 0x6c9a44, 1.2, 0],    // GRASS_ALPINE
  [Kind.SHRUB, 0.009, 0x6a8a48, 0x5d5a3c, 0.5, 0],     // SHRUB_SAGE
  [Kind.CYPRESS, 0.022, 0x3d6b45, 0x5a4330, 0.5, 0],   // CYPRESS
  [Kind.GRASS, 0.0065, 0xb8b46e, 0x8e8a4e, 1.4, 0],    // GRASS_STEPPE
  [Kind.SHRUB, 0.006, 0x9aa27e, 0x6e6a4c, 0.6, 0],    // SHRUB_STEPPE
  [Kind.TREE, 0.013, 0x7d9a4a, 0x6a4a32, 0.5, 0.55],   // HARDY_TREE (windswept)
  [Kind.SHRUB, 0.0045, 0xa3a06c, 0x7a7050, 0.3, 0],    // CUSHION
  [Kind.F_SHORT, 0.0065, 0x7cc94e, 0x5aa83c, 1.3, 0],  // F_GRASS
  [Kind.F_TALL, 0.009, 0x8fd35c, 0x6aae44, 1.6, 0],    // F_GRASS_TALL
  [Kind.F_SEED, 0.01, 0xd6d27e, 0x8fb04e, 1.6, 0],   // F_SEED (seed heads)
  [Kind.F_CLOVER, 0.004, 0x4f9e3e, 0x3f8a34, 0.4, 0], // F_CLOVER
  [Kind.F_FLOWER, 0.0065, 0x5a9e3a, 0x4a8a30, 1.2, 0], // F_FLOWER (petals by patch palette)
  [Kind.F_SHORT, 0.0065, 0xd8c27a, 0xae9a58, 1.4, 0],  // F_STRAW
  [Kind.F_TALL, 0.011, 0xe6e0bc, 0xb8b488, 2.6, 0],   // FEATHER (feather grass, strong sway)
  [Kind.F_TALL, 0.013, 0x7a9e48, 0x62843a, 0.9, 0],   // REED
  [Kind.TUSSOCK, 0.0075, 0xc8b46a, 0x9a8a50, 1.1, 0],   // TUSSOCK (straw/khaki/silver-green by slot)
  [Kind.SHRUB, 0.006, 0xb4bca4, 0x6e6e5a, 0.5, 0],    // SAGEBRUSH
  [Kind.SHRUB, 0.007, 0x5a9a44, 0x3f6a30, 0.5, 0],    // BUSH_GREEN
  [Kind.ROCK, 0.006, 0x8c9096, 0xb8b060, 0, 0],       // BOULDER (trunk colour = lichen)
  [Kind.ROCK, 0.003, 0x8a8c8e, 0xa8a060, 0, 0],       // STONE
  [Kind.HEDGE, 0.008, 0x3f7a36, 0x3a2e22, 0.3, 0],    // HEDGE
];
export const SPECIES_KIND: readonly number[] = SPECIES_TABLE.map((r) => r[0]);
export const SPECIES_SIZE: readonly number[] = SPECIES_TABLE.map((r) => r[1]);
export const SPECIES_COLOR: readonly number[] = SPECIES_TABLE.map((r) => r[2]);
export const SPECIES_TRUNK: readonly number[] = SPECIES_TABLE.map((r) => r[3]);
export const SPECIES_FLEX: readonly number[] = SPECIES_TABLE.map((r) => r[4]);
export const SPECIES_LEAN: readonly number[] = SPECIES_TABLE.map((r) => r[5]);
/**
 * Per mesh kind, in unit-height mesh units: base contact radius (a plant on a slope sinks by
 * slope × this so its downhill edge touches the ground) and contact-shadow blob radius x/z + opacity
 * (0 = no blob; grass tufts and fine ground cover have none: too many, the root darkening grounds them).
 */
export const KIND_FOOT: readonly number[] = [0.08, 0.06, 0.12, 0.05, 0.45, 0.1, 0.1, 0.05, 0.6, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1];
export const KIND_BLOB: readonly (readonly [number, number, number])[] = [
  [0.55, 0.55, 0.72], [0.5, 0.5, 0.72], [0.42, 0.42, 0.7], [0.62, 0.62, 0.68], [0.8, 0.8, 0.85], [0, 0, 0], [0.48, 0.48, 0.28], [0.4, 0.4, 0.72],
  [1.0, 0.85, 0.9], [0.75, 0.75, 0.7], [1.6, 0.7, 0.8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
];
/** Extra sink (voxels) of a plant base on a slope: gradient (voxels/column) × footprint (columns). */
export const baseSink = (slope: number, species: number, targetScale: number) =>
  slope * (SPECIES_SIZE[species]! * KIND_FOOT[SPECIES_KIND[species]!]! * targetScale * 1.2) / CELL;

/** Wildflower palettes (sRGB), one per flower patch; 3 colours each. */
export const PETAL_PALETTES: readonly (readonly [number, number, number])[] = [
  [0xff7aa2, 0xffc2d6, 0xf6f2ff], // meadow pinks
  [0xffd23f, 0xffe98a, 0xfff6d0], // buttercups
  [0x6f8dff, 0xb28dff, 0xe8e0ff], // cornflower / lavender
  [0xff5a3c, 0xff9a3c, 0xfff0e0], // poppies
];
/** Tussock tones per slot: straw, khaki, silver-green. */
export const TUSSOCK_TONES: readonly number[] = [0xcdb56d, 0xa89d5e, 0x9fb08a];

/**
 * Three jittered slot grids over the block. Large: trees, pines, cacti, acacias, cypress, hedges.
 * Small: shrubs, tufts, flowers, rocks, tussocks. Fine: ground-cover grass, clover, wildflowers,
 * reeds; these are compacted every frame with camera culling and distance thinning (flora.ts).
 * A slot hosts at most one plant, so instance counts are bounded by the slot counts.
 */
export const GRID_L = 256;
export const GRID_S = 320;
export const GRID_F = 448;
export const NSLOT_L = GRID_L * GRID_L;
export const NSLOT_S = GRID_S * GRID_S;
export const NSLOT_F = GRID_F * GRID_F;
export const NSLOT_LS = NSLOT_L + NSLOT_S;
export const NSLOT = NSLOT_LS + NSLOT_F;
export const SET_GRID: readonly number[] = [GRID_L, GRID_S, GRID_F];
export const SET_BASE: readonly number[] = [0, NSLOT_L, NSLOT_LS];
export const KIND_CAP: readonly number[] = KIND_SET.map((s) => SET_GRID[s]! ** 2);
export const KIND_BASE: readonly number[] = KIND_CAP.map((_, k) => KIND_CAP.slice(0, k).reduce((a, b) => a + b, 0));
export const LIST_SIZE = KIND_CAP.reduce((a, b) => a + b, 0);
/** Slot centre jitter as a fraction of the slot spacing. */
export const JITTER = 0.9;
/** Plants keep this far (world units) from the cut faces so crowns do not overhang them. */
export const EDGE_MARGIN: readonly number[] = [0.012, 0.008, 0.006];

export const FLORA = {
  /** Max water depth (voxels) at any of the 4 bilinear corner columns; wetter → no plant. */
  WET_MAX: 0.05,
  /** Max terrain gradient |∇surfY| in voxel layers per column for trees; steeper → no plant. */
  SLOPE_MAX: 1.1,
  /** Per slot set [large, small, fine]: grass holds on steeper hillsides than trees. */
  SLOPE_MAX_SET: [1.1, 1.4, 1.8] as readonly number[],
  /** Voxels a plant base sinks below the terrain surface, per slot set [large, small, fine]. */
  SINK: 0.35,
  SINK_SET: [0.15, 0.1, 0.06] as readonly number[],
  /** Real seconds between refreshes, and the grow/shrink ramp length. */
  REFRESH_S: 2,
  GROW_S: 2.5,
  /** Fraction of slots usable in the low quality tier. */
  LOW_QUALITY: 0.5,
  /** Below this displayed scale a slot may switch species (shrink first, then regrow). */
  SWITCH_MIN: 0.02,
  /** Reeds grow where the corner water is between WET_MAX and this (voxels), in biomes with ground cover. */
  REED_MAX: 0.8,
  REED_P: 0.75,
  /** Fine ground cover distance thinning: all kept within LOD_D0, down to LOD_MIN at LOD_D1 (world units). */
  LOD_D0: 2.2, LOD_D1: 7.5, LOD_MIN: 0.18,
} as const;

/**
 * Patch noise (deterministic, world xz): value noise, 2 octaves. Clumps meadows into dense drifts
 * and bare spots; flowers have their own patch field and palette per patch cell.
 */
export const PATCH = {
  CELL_A: 0.55, CELL_B: 0.19, CELL_FLOWER: 0.33, CELL_PALETTE: 0.45,
  /** density multiplier = M_MIN + M_SPAN · smoothstep(N0, N1, n)² */
  M_MIN: 0.15, M_SPAN: 4.6, N0: 0.38, N1: 0.85,
  /** Large-set trees follow the patches only partly (copses and clearings, not holes in forests). */
  LARGE_MIX: 0.6,
  /** flower share multiplier = F_MIN + F_SPAN · smoothstep(F0, F1, nf) */
  F_MIN: 0.05, F_SPAN: 7, F0: 0.55, F1: 0.88,
} as const;

/**
 * Altitude zonation inside a biome (voxels above the emergent sea level). The climate pass sets
 * biome + veg (incl. its own temperature treeline → ALPINE); this only shapes the transition:
 * forest thins and stunts from TREE_A0 to TREE_A1, broadleaf gives way to pine from PINE_A0 to
 * PINE_A1, low shrubs and tufts fill in where trees fade, and every plant fades out from BARE_A0 to
 * BARE_A1 (bare rock and snow). All ramps are smoothsteps.
 */
export const TREELINE = {
  TREE_A0: 10, TREE_A1: 22,
  PINE_A0: 5, PINE_A1: 15,
  /** Tree scale at the treeline, relative to lowland. */
  STUNT: 0.45,
  /** Extra small-slot occupancy per unit veg where trees have faded (forest biomes). */
  LINE_BOOST: 1.2,
  /** Share of small slots that turn into treeline shrubs/tufts where trees have faded. */
  LINE_SHARE: 0.8,
  BARE_A0: 26, BARE_A1: 34,
} as const;
/** Biomes whose trees follow the altitude treeline (1) or not (0). */
export const TREELINE_BIOME: readonly number[] = Array.from({ length: LIFE_BIOMES }, (_, b) =>
  b === LBiome.TAIGA || b === LBiome.TEMPERATE_FOREST || b === LBiome.RAINFOREST ? 1 : 0);

export const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/**
 * Per (biome, slot set): occupancy per unit veg (× patch multiplier), up to 4 species with
 * cumulative thresholds t1 < t2 < t3 (r < t1 → a, < t2 → b, < t3 → c, else d), and a flower
 * species with a base share that the flower patches modulate.
 */
export interface FloraRule { density: number; sp: [number, number, number, number]; t: [number, number, number]; flower: number; pF: number }
const R = (density: number, mix: [number, number][], flower: [number, number] = [0, 0]): FloraRule => {
  const tot = mix.reduce((a, m) => a + m[1], 0) || 1;
  const sp: [number, number, number, number] = [0, 0, 0, 0];
  const t: [number, number, number] = [1, 1, 1];
  let acc = 0;
  for (let i = 0; i < 4; i++) {
    const m = mix[Math.min(i, mix.length - 1)] ?? [0, 0];
    sp[i] = m[0];
    if (i < 3) { acc += i < mix.length ? m[1] / tot : 0; t[i] = i < mix.length - 1 ? acc : 1; }
  }
  return { density, sp, t, flower: flower[0], pF: flower[1] };
};
const NONE = R(0, [[Sp.NONE, 1]]);

/** FLORA_RULES[biome] = [large, small, fine]. Nothing on ocean, ice, beach; alpine is meadow only. */
export const FLORA_RULES: readonly (readonly [FloraRule, FloraRule, FloraRule])[] = (() => {
  const t: [FloraRule, FloraRule, FloraRule][] = Array.from({ length: LIFE_BIOMES }, () => [NONE, NONE, NONE]);
  const B = LBiome;
  t[B.TUNDRA] = [NONE, R(1.4, [[Sp.SHRUB_TUNDRA, 0.9], [Sp.STONE, 0.1]], [Sp.FLOWER, 0.12]), R(0.8, [[Sp.F_STRAW, 0.5], [Sp.F_CLOVER, 0.5]])];
  t[B.TAIGA] = [R(1.3, [[Sp.PINE, 1]]), R(0.35, [[Sp.SHRUB_TUNDRA, 0.6], [Sp.FERN, 0.4]]), R(0.3, [[Sp.F_GRASS, 0.6], [Sp.F_CLOVER, 0.4]])];
  t[B.TEMPERATE_FOREST] = [
    R(0.9, [[Sp.BROADLEAF, 0.88], [Sp.PINE, 0.12]]),
    R(0.55, [[Sp.GRASS, 0.7], [Sp.BUSH_GREEN, 0.18], [Sp.BOULDER, 0.12]], [Sp.FLOWER, 0.15]),
    R(0.35, [[Sp.F_GRASS, 0.6], [Sp.F_CLOVER, 0.4]], [Sp.F_FLOWER, 0.05]),
  ];
  t[B.GRASSLAND] = [
    R(0.03, [[Sp.BROADLEAF, 0.7], [Sp.HEDGE, 0.3]]),
    R(0.5, [[Sp.GRASS, 0.8], [Sp.BUSH_GREEN, 0.1], [Sp.BOULDER, 0.1]], [Sp.FLOWER, 0.12]),
    R(0.95, [[Sp.F_GRASS, 0.45], [Sp.F_GRASS_TALL, 0.25], [Sp.F_SEED, 0.12], [Sp.F_CLOVER, 0.18]], [Sp.F_FLOWER, 0.07]),
  ];
  t[B.DESERT] = [R(2.6, [[Sp.CACTUS, 1]]), R(3.5, [[Sp.SHRUB_DRY, 0.85], [Sp.STONE, 0.15]]), NONE];
  t[B.SAVANNA] = [
    R(0.3, [[Sp.ACACIA, 1]]),
    R(2.2, [[Sp.GRASS_DRY, 0.85], [Sp.SHRUB_DRY, 0.1], [Sp.STONE, 0.05]]),
    R(0.9, [[Sp.F_STRAW, 0.6], [Sp.F_SEED, 0.4]]),
  ];
  t[B.RAINFOREST] = [R(1.05, [[Sp.BROADLEAF_RAIN, 1]]), R(0.7, [[Sp.FERN, 1]], [Sp.FLOWER, 0.08]), R(0.4, [[Sp.F_GRASS_TALL, 0.5], [Sp.F_CLOVER, 0.5]])];
  t[B.ALPINE] = [NONE, R(3.0, [[Sp.GRASS_ALPINE, 0.85], [Sp.STONE, 0.15]], [Sp.FLOWER, 0.22]), R(1.0, [[Sp.F_GRASS, 0.6], [Sp.F_CLOVER, 0.4]], [Sp.F_FLOWER, 0.1])];
  t[B.STEPPE] = [
    R(0.05, [[Sp.HARDY_TREE, 1]]),
    R(1.4, [[Sp.TUSSOCK, 0.62], [Sp.SAGEBRUSH, 0.24], [Sp.STONE, 0.14]]),
    R(1.3, [[Sp.F_STRAW, 0.45], [Sp.FEATHER, 0.4], [Sp.F_SEED, 0.15]], [Sp.F_FLOWER, 0.02]),
  ];
  t[B.SHRUBLAND] = [R(0.12, [[Sp.CYPRESS, 1]]), R(2.6, [[Sp.SHRUB_SAGE, 0.75], [Sp.GRASS_DRY, 0.15], [Sp.STONE, 0.1]]), R(0.4, [[Sp.F_STRAW, 1]])];
  t[B.COLD_DESERT] = [NONE, R(1.5, [[Sp.CUSHION, 0.8], [Sp.STONE, 0.2]]), NONE];
  return t;
})();

// ---- hashing (bit-identical in TSL, flora.ts) ----

/** lowbias32 integer hash. */
export function hashU(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}
/** Hash channels per slot. */
export const H = { OCC: 0, SPECIES: 1, SCALE: 2, JX: 3, JZ: 4, YAW: 5, QUALITY: 6, TINT: 7, SQUASH: 8, PETAL: 9, ALT: 10, ALT2: 11, FLOWER: 12, LOGN: 13, TILT: 14, LOD: 15 } as const;
/** Key of (slot-local index, set, channel): local < 2^20, channel < 16, set < 16. */
export const slotKey = (local: number, set: number, ch: number) => (local * 16 + ch + set * 0x1000000) >>> 0;
export const slotRand = (local: number, set: number, ch: number) => (hashU(slotKey(local, set, ch)) >>> 8) / 16777216;

/** Slot index → (set, local, grid size). */
export function slotInfo(slot: number): { set: number; local: number; grid: number } {
  const set = slot < NSLOT_L ? 0 : slot < NSLOT_LS ? 1 : 2;
  return { set, local: slot - SET_BASE[set]!, grid: SET_GRID[set]! };
}

// ---- patch noise (bit-identical lattice hash in TSL, flora.ts) ----

const LAT_OFF = 4096;
/** Lattice value 0..1 at integer cell (ix, iz) with salt. */
export function latticeRand(ix: number, iz: number, salt: number): number {
  const k = (Math.imul((ix + LAT_OFF) >>> 0, 73856093) ^ Math.imul((iz + LAT_OFF) >>> 0, 19349663) ^ salt) >>> 0;
  return (hashU(k) >>> 8) / 16777216;
}
/** Smooth value noise 0..1 at world x,z on a lattice of `cell` world units. */
export function valueNoise(x: number, z: number, cell: number, salt: number): number {
  const u = x / cell, v = z / cell;
  const i = Math.floor(u), j = Math.floor(v);
  let fu = u - i, fv = v - j;
  fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv);
  const a = latticeRand(i, j, salt), b = latticeRand(i + 1, j, salt), c = latticeRand(i, j + 1, salt), d = latticeRand(i + 1, j + 1, salt);
  return (a + (b - a) * fu) + ((c + (d - c) * fu) - (a + (b - a) * fu)) * fv;
}
export const SALT = { PATCH_A: 101, PATCH_B: 202, FLOWER: 303, PALETTE: 404 } as const;
/** Meadow patch field 0..1 (dense clumps high, bare spots low). */
export const patchNoise = (x: number, z: number) =>
  0.65 * valueNoise(x, z, PATCH.CELL_A, SALT.PATCH_A) + 0.35 * valueNoise(x, z, PATCH.CELL_B, SALT.PATCH_B);
/** Density multiplier from the patch field. */
export const patchMul = (n: number) => { const t = smoothstep(PATCH.N0, PATCH.N1, n); return PATCH.M_MIN + PATCH.M_SPAN * t * t; };
export const flowerMul = (x: number, z: number) =>
  PATCH.F_MIN + PATCH.F_SPAN * smoothstep(PATCH.F0, PATCH.F1, valueNoise(x, z, PATCH.CELL_FLOWER, SALT.FLOWER));

/** World x,z of a slot (deterministic jittered grid). */
export function slotXZ(slot: number): [number, number] {
  const { set, local, grid } = slotInfo(slot);
  const i = local % grid, j = Math.floor(local / grid);
  const sp = BLOCK_SIZE / grid;
  const x = (i + 0.5 + (slotRand(local, set, H.JX) - 0.5) * JITTER) * sp - BLOCK_SIZE / 2;
  const z = (j + 0.5 + (slotRand(local, set, H.JZ) - 0.5) * JITTER) * sp - BLOCK_SIZE / 2;
  return [x, z];
}

// ---- CPU column map (readback of sim fields) ----

export interface ColumnMap {
  surfY: Float32Array; water: Float32Array; biome: Uint32Array; veg?: Float32Array;
  /** Render height per column (flora renderH readback: the terrain as drawn); defaults to surfY. */
  surfR?: Float32Array;
}

const wrap = (i: number, n: number) => ((i % n) + n) % n;
const col = (x: number, z: number) => wrap(x, NX) + wrap(z, NZ) * NX;
/** World x/z → continuous cell coordinate (column c centre at c). */
export const toCell = (w: number) => (w + BLOCK_SIZE / 2) / CELL - 0.5;
/** Column containing world x,z (clamped to the block). */
export function columnAt(x: number, z: number): number {
  const cx = Math.min(NX - 1, Math.max(0, Math.floor((x + BLOCK_SIZE / 2) / CELL)));
  const cz = Math.min(NZ - 1, Math.max(0, Math.floor((z + BLOCK_SIZE / 2) / CELL)));
  return cx + cz * NX;
}

/**
 * Render height (voxel units) at world x,z as the terrain mesh draws it: vertices at column centres,
 * each quad split along the (u0+1, v0)–(u0, v0+1) diagonal (terrain.ts createColumnGrid), so the
 * surface is linear per triangle, not bilinear. Wraps like the terrain sampler (V1).
 */
export function heightAt(m: ColumnMap, x: number, z: number): number {
  const u = toCell(x), v = toCell(z);
  const u0 = Math.floor(u), v0 = Math.floor(v), fu = u - u0, fv = v - v0;
  const s = m.surfR ?? m.surfY;
  const a = s[col(u0, v0)]!, b = s[col(u0 + 1, v0)]!, c = s[col(u0, v0 + 1)]!, d = s[col(u0 + 1, v0 + 1)]!;
  return fu + fv <= 1 ? a + fu * (b - a) + fv * (c - a) : d + (1 - fu) * (c - d) + (1 - fv) * (b - d);
}
/** Max water depth over the 4 bilinear corner columns. */
export function cornerWater(m: ColumnMap, x: number, z: number): number {
  const u0 = Math.floor(toCell(x)), v0 = Math.floor(toCell(z));
  const w = m.water;
  return Math.max(w[col(u0, v0)]!, w[col(u0 + 1, v0)]!, w[col(u0, v0 + 1)]!, w[col(u0 + 1, v0 + 1)]!);
}
/** Central-difference gradient magnitude of surfY at the column containing x,z (voxels per column). */
export function slopeAt(m: ColumnMap, x: number, z: number): number {
  const c = columnAt(x, z);
  const cx = c % NX, cz = Math.floor(c / NX);
  const gx = (m.surfY[col(cx + 1, cz)]! - m.surfY[col(cx - 1, cz)]!) * 0.5;
  const gz = (m.surfY[col(cx, cz + 1)]! - m.surfY[col(cx, cz - 1)]!) * 0.5;
  return Math.hypot(gx, gz);
}
/** Water depth in the 3×3 columns around the one containing x,z (max), for "never in water" checks. */
export function waterNear(m: ColumnMap, x: number, z: number): number {
  const c = columnAt(x, z);
  const cx = c % NX, cz = Math.floor(c / NX);
  let w = 0;
  for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) w = Math.max(w, m.water[col(cx + dx, cz + dz)]!);
  return w;
}

/**
 * CPU reference of the flora decision for one slot (what the GPU kernel targets, before the
 * grow/shrink ramp). Returns species (0 = none) and target scale.
 */
export function floraDecide(m: ColumnMap & { veg: Float32Array }, slot: number, quality = 1, seaLevel = 76): { species: number; scale: number } {
  const { set, local } = slotInfo(slot);
  const [x, z] = slotXZ(slot);
  const none = { species: 0, scale: 0 };
  if (Math.abs(x) > BLOCK_SIZE / 2 - EDGE_MARGIN[set]! || Math.abs(z) > BLOCK_SIZE / 2 - EDGE_MARGIN[set]!) return none;
  const c = columnAt(x, z);
  const b = m.biome[c]! < LIFE_BIOMES ? m.biome[c]! : 0;
  const rule = FLORA_RULES[b]![set]!;
  const veg = m.veg[c]!;
  const T = TREELINE;
  const alt = m.surfY[c]! - seaLevel;
  const above = smoothstep(T.TREE_A0, T.TREE_A1, alt); // 0 lowland → 1 above the treeline
  const bare = 1 - smoothstep(T.BARE_A0, T.BARE_A1, alt);
  const line = TREELINE_BIOME[b]! * above;
  const pm = patchMul(patchNoise(x, z));
  const patch = set === 0 ? 1 + (pm - 1) * PATCH.LARGE_MIX : pm;
  const p = (set === 0 ? veg * rule.density * (1 - line)
    : set === 1 ? veg * (rule.density + line * T.LINE_BOOST) : veg * rule.density) * bare * patch;
  if (slotRand(local, set, H.OCC) >= p) return none;
  if (slotRand(local, set, H.QUALITY) >= quality) return none;
  // thresholds as f32, like the GPU compares them (water values are f32 already)
  const wet = cornerWater(m, x, z), wetMax = Math.fround(FLORA.WET_MAX), reedMax = Math.fround(FLORA.REED_MAX);
  const reed = set === 2 && wet > wetMax && wet <= reedMax && slotRand(local, set, H.ALT) < FLORA.REED_P;
  if ((wet > wetMax && !reed) || slopeAt(m, x, z) > FLORA.SLOPE_MAX_SET[set]!) return none;
  const r = slotRand(local, set, H.SPECIES);
  let species = r < rule.t[0] ? rule.sp[0] : r < rule.t[1] ? rule.sp[1] : r < rule.t[2] ? rule.sp[2] : rule.sp[3];
  if (rule.flower && slotRand(local, set, H.FLOWER) < rule.pF * flowerMul(x, z)) species = rule.flower;
  let scale = (0.75 + 0.5 * slotRand(local, set, H.SCALE)) * (0.6 + 0.4 * veg);
  if (set === 0) {
    const broad = species === Sp.BROADLEAF || species === Sp.BROADLEAF_RAIN;
    if (broad && slotRand(local, set, H.ALT) < smoothstep(T.PINE_A0, T.PINE_A1, alt)) species = Sp.PINE;
    scale *= 1 - T.STUNT * line;
  } else if (set === 1 && slotRand(local, set, H.ALT) < line * T.LINE_SHARE) {
    species = slotRand(local, set, H.ALT2) < 0.45 ? Sp.SHRUB_TUNDRA : Sp.GRASS_ALPINE;
  }
  if (reed) species = Sp.REED;
  return { species, scale };
}

// ---- creatures ----

export const CREATURES = {
  BIRD_FLOCKS: [2, 1] as const,      // [high, low] quality; fewer on mostly-ocean worlds (flockCount)
  BIRD_MIN: 6, BIRD_MAX: 10,
  BIRD_SIZE: 0.014,
  /** Herds: at most [high, low], one per HERD_COLS grazing columns, only on grazing patches of at
   * least HERD_MIN_AREA connected columns (tiny islands get none). */
  HERDS_MAX: [4, 2] as const,
  HERD_COLS: 2500,
  HERD_MIN_AREA: 700,
  HERD_MIN: 2, HERD_MAX: 4,
  /** Herd spawn radius, min spacing between members, min spacing between herd centres (world). */
  HERD_RADIUS: 0.12, HERD_SPACING: 0.035, HERD_APART: 0.5,
  CRITTER_SIZE: 0.013,
  FISH_SCHOOLS: [5, 3] as const,
  FISH_MIN: 9, FISH_MAX: 15,
  /** Birds keep at least this far (world) above terrain and water. */
  BIRD_CLEARANCE: 0.12,
  BIRD_CEIL: 1.35,
  /** Birds stay this far inside the block faces. */
  BIRD_MARGIN: 0.05,
  CRITTER_MARGIN: 0.05,
  /** Critter cells must be this dry (max water of 3×3 columns). */
  CRITTER_WET: 0.02,
  CRITTER_SLOPE: 1.6,
  /** Fish water depth band (voxels). */
  FISH_DEPTH_MIN: 1, FISH_DEPTH_MAX: 6,
  /** Fish swim at least this far (voxels) below the water level, and dither out between FADE_A0..A1. */
  FISH_SURFACE_MIN: 0.6,
  FISH_FADE_A0: 2.8, FISH_FADE_A1: 4,
  /** Real seconds between column-map readbacks. */
  MAP_S: 2,
} as const;

export const GRAZING = new Set<number>([LBiome.GRASSLAND, LBiome.SAVANNA, LBiome.STEPPE]);

/**
 * Connected grazing patches (4-neighbour, no wrap: the cut block shows them apart): label per
 * column (-1 = not grazing land) and patch sizes in columns. Grazing land = grazing biome, dry.
 */
export function grazingPatches(m: ColumnMap): { label: Int32Array; size: number[] } {
  const label = new Int32Array(NX * NZ).fill(-1);
  const size: number[] = [];
  const ok = (c: number) => GRAZING.has(m.biome[c]!) && m.water[c]! <= CREATURES.CRITTER_WET;
  const stack: number[] = [];
  for (let c0 = 0; c0 < NX * NZ; c0++) {
    if (label[c0] !== -1 || !ok(c0)) continue;
    const id = size.length;
    let n = 0;
    label[c0] = id; stack.push(c0);
    while (stack.length) {
      const c = stack.pop()!;
      n++;
      const x = c % NX, z = (c / NX) | 0;
      const nb = [x > 0 ? c - 1 : -1, x < NX - 1 ? c + 1 : -1, z > 0 ? c - NX : -1, z < NZ - 1 ? c + NX : -1];
      for (const q of nb) if (q >= 0 && label[q] === -1 && ok(q)) { label[q] = id; stack.push(q); }
    }
    size.push(n);
  }
  return { label, size };
}

/** Bird flocks for a world: none on (almost) all-ocean worlds, fewer when land is scarce. */
export function flockCount(m: ColumnMap, high: boolean): number {
  let land = 0;
  for (let c = 0; c < m.water.length; c++) if (m.water[c]! < 0.5) land++;
  const frac = land / m.water.length, n = CREATURES.BIRD_FLOCKS[high ? 0 : 1];
  return frac < 0.03 ? 0 : frac < 0.12 ? Math.min(1, n) : n;
}
