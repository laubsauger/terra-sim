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
} as const;
export const SPECIES_COUNT = 19;

/** Mesh kinds (one indirect draw each). KIND_SET: 0 = large slot set, 1 = small. */
export const Kind = { TREE: 0, PINE: 1, CACTUS: 2, ACACIA: 3, SHRUB: 4, GRASS: 5, FLOWER: 6, CYPRESS: 7 } as const;
export const KIND_COUNT = 8;
export const KIND_NAMES = ['tree', 'pine', 'cactus', 'acacia', 'shrub', 'grass', 'flower', 'cypress'] as const;
export const KIND_SET: readonly number[] = [0, 0, 0, 0, 1, 1, 1, 0];

// Per species: mesh kind, world height of the unit-height mesh at scale 1, foliage + trunk colour (sRGB).
const SPECIES_TABLE: readonly [number, number, number, number][] = [
  [-1, 0, 0xff00ff, 0xff00ff],
  [Kind.TREE, 0.062, 0x6cb743, 0x7a5236],     // BROADLEAF
  [Kind.TREE, 0.078, 0x2e7a3c, 0x5a3d27],     // BROADLEAF_RAIN
  [Kind.PINE, 0.078, 0x2f6b52, 0x654630],     // PINE
  [Kind.CACTUS, 0.052, 0x74a95e, 0x5f8f4e],   // CACTUS
  [Kind.SHRUB, 0.012, 0x8f9a58, 0x7a6243],    // SHRUB_DRY
  [Kind.GRASS, 0.015, 0x7cc64a, 0x5c9a3a],    // GRASS
  [Kind.GRASS, 0.017, 0xd5b95c, 0xa08840],    // GRASS_DRY
  [Kind.FLOWER, 0.015, 0x76b84a, 0x4f9a3c],   // FLOWER
  [Kind.ACACIA, 0.058, 0x93a848, 0x6e5038],   // ACACIA
  [Kind.SHRUB, 0.011, 0x8c7046, 0x6f5a3a],    // SHRUB_TUNDRA
  [Kind.GRASS, 0.018, 0x3f9642, 0x2f7434],    // FERN
  [Kind.GRASS, 0.012, 0x8fbf5a, 0x6c9a44],    // GRASS_ALPINE
  [Kind.SHRUB, 0.02, 0x6a8a48, 0x5d5a3c],     // SHRUB_SAGE
  [Kind.CYPRESS, 0.07, 0x3d6b45, 0x5a4330],   // CYPRESS
  [Kind.GRASS, 0.014, 0xb8b46e, 0x8e8a4e],    // GRASS_STEPPE
  [Kind.SHRUB, 0.011, 0x9aa27e, 0x6e6a4c],    // SHRUB_STEPPE
  [Kind.TREE, 0.04, 0x7d9a4a, 0x6a4a32],      // HARDY_TREE
  [Kind.SHRUB, 0.008, 0xa3a06c, 0x7a7050],    // CUSHION
];
export const SPECIES_KIND: readonly number[] = SPECIES_TABLE.map((r) => r[0]);
export const SPECIES_SIZE: readonly number[] = SPECIES_TABLE.map((r) => r[1]);
export const SPECIES_COLOR: readonly number[] = SPECIES_TABLE.map((r) => r[2]);
export const SPECIES_TRUNK: readonly number[] = SPECIES_TABLE.map((r) => r[3]);
/** Flower petal colours (sRGB hex), picked per slot by hash. */
export const PETALS: readonly number[] = [0xff6f91, 0xffd23f, 0xf6f2ff, 0xb28dff, 0xff9a3c];

/**
 * Two jittered slot grids over the block. Large: trees, pines, cacti, acacias. Small: shrubs,
 * grass tufts, flowers. A slot hosts at most one plant, so the drawn instance count is bounded by
 * the slot count (worst case 62.5k, all land fully vegetated; typical worlds draw ~15-30k).
 */
export const GRID_L = 160;
export const GRID_S = 192;
export const NSLOT_L = GRID_L * GRID_L;
export const NSLOT_S = GRID_S * GRID_S;
export const NSLOT = NSLOT_L + NSLOT_S;
export const KIND_CAP: readonly number[] = KIND_SET.map((s) => (s === 0 ? NSLOT_L : NSLOT_S));
export const KIND_BASE: readonly number[] = KIND_CAP.map((_, k) => KIND_CAP.slice(0, k).reduce((a, b) => a + b, 0));
export const LIST_SIZE = KIND_CAP.reduce((a, b) => a + b, 0);
/** Slot centre jitter as a fraction of the slot spacing. */
export const JITTER = 0.9;
/** Plants keep this far (world units) from the cut faces so crowns do not overhang them. */
export const EDGE_MARGIN: readonly number[] = [0.03, 0.012];

export const FLORA = {
  /** Max water depth (voxels) at any of the 4 bilinear corner columns; wetter → no plant. */
  WET_MAX: 0.05,
  /** Max terrain gradient |∇surfY| in voxel layers per column; steeper → no plant. */
  SLOPE_MAX: 1.1,
  /** Voxels a plant base sinks below the sampled surface (hides the mesh/triangle mismatch). */
  SINK: 0.35,
  /** Real seconds between refreshes, and the grow/shrink ramp length. */
  REFRESH_S: 2,
  GROW_S: 2.5,
  /** Fraction of slots usable in the low quality tier. */
  LOW_QUALITY: 0.5,
  /** Below this displayed scale a slot may switch species (shrink first, then regrow). */
  SWITCH_MIN: 0.02,
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

/** Per (biome, slot set): probability per unit veg that a slot is occupied, species a/b, P(b). */
export interface FloraRule { density: number; a: number; b: number; pB: number }
const R = (density: number, a: number, b = a, pB = 0): FloraRule => ({ density, a, b, pB });
const NONE = R(0, Sp.NONE);

/**
 * FLORA_RULES[biome] = [large set rule, small set rule]. Nothing on ocean, ice, beach; alpine gets
 * meadow tufts and flowers only (up to the bare zone).
 */
export const FLORA_RULES: readonly (readonly [FloraRule, FloraRule])[] = (() => {
  const t: [FloraRule, FloraRule][] = Array.from({ length: LIFE_BIOMES }, () => [NONE, NONE]);
  const B = LBiome;
  t[B.TUNDRA] = [NONE, R(1.4, Sp.SHRUB_TUNDRA, Sp.FLOWER, 0.12)];
  t[B.TAIGA] = [R(1.3, Sp.PINE), R(0.35, Sp.SHRUB_TUNDRA, Sp.FERN, 0.4)];
  t[B.TEMPERATE_FOREST] = [R(0.9, Sp.BROADLEAF, Sp.PINE, 0.12), R(0.55, Sp.GRASS, Sp.FLOWER, 0.15)];
  t[B.GRASSLAND] = [R(0.07, Sp.BROADLEAF), R(1.7, Sp.GRASS, Sp.FLOWER, 0.16)];
  t[B.DESERT] = [R(2.6, Sp.CACTUS), R(3.5, Sp.SHRUB_DRY)];
  t[B.SAVANNA] = [R(0.3, Sp.ACACIA), R(2.2, Sp.GRASS_DRY, Sp.SHRUB_DRY, 0.1)];
  t[B.RAINFOREST] = [R(1.05, Sp.BROADLEAF_RAIN), R(0.7, Sp.FERN, Sp.FLOWER, 0.08)];
  t[B.ALPINE] = [NONE, R(3.0, Sp.GRASS_ALPINE, Sp.FLOWER, 0.22)];
  t[B.STEPPE] = [R(0.05, Sp.HARDY_TREE), R(1.8, Sp.GRASS_STEPPE, Sp.SHRUB_STEPPE, 0.25)];
  t[B.SHRUBLAND] = [R(0.12, Sp.CYPRESS), R(2.0, Sp.SHRUB_SAGE, Sp.GRASS_DRY, 0.2)];
  t[B.COLD_DESERT] = [NONE, R(1.5, Sp.CUSHION)];
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
export const H = { OCC: 0, SPECIES: 1, SCALE: 2, JX: 3, JZ: 4, YAW: 5, QUALITY: 6, TINT: 7, SQUASH: 8, PETAL: 9, ALT: 10, ALT2: 11 } as const;
/** Key of (slot-local index, set, channel): local < 2^16, channel < 16. */
export const slotKey = (local: number, set: number, ch: number) => (local * 16 + ch + set * 0x1000000) >>> 0;
export const slotRand = (local: number, set: number, ch: number) => (hashU(slotKey(local, set, ch)) >>> 8) / 16777216;

/** Slot index → (set, local, grid size). */
export function slotInfo(slot: number): { set: number; local: number; grid: number } {
  return slot < NSLOT_L ? { set: 0, local: slot, grid: GRID_L } : { set: 1, local: slot - NSLOT_L, grid: GRID_S };
}

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
  /** Render height (smoothSurf(surfY)); defaults to surfY. */
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
 * Mirror of the render column smoothing (space.ts renderColumns .x): 1-2-1 3×3 kernel over surfY,
 * wrapping (V1). The terrain mesh is displaced by this, so creatures stand on it.
 */
export function smoothSurf(surfY: Float32Array): Float32Array {
  const out = new Float32Array(surfY.length);
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    let a = 0;
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) a += surfY[col(x + dx, z + dz)]! * ((dx === 0 ? 2 : 1) * (dz === 0 ? 2 : 1) / 16);
    out[x + z * NX] = a;
  }
  return out;
}

/** Bilinear render height (voxel units) at world x,z; wraps like the terrain sampler (V1). */
export function heightAt(m: ColumnMap, x: number, z: number): number {
  const u = toCell(x), v = toCell(z);
  const u0 = Math.floor(u), v0 = Math.floor(v), fu = u - u0, fv = v - v0;
  const s = m.surfR ?? m.surfY;
  return (s[col(u0, v0)]! * (1 - fu) + s[col(u0 + 1, v0)]! * fu) * (1 - fv)
    + (s[col(u0, v0 + 1)]! * (1 - fu) + s[col(u0 + 1, v0 + 1)]! * fu) * fv;
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
  const p = set === 0
    ? veg * rule.density * (1 - line) * bare
    : veg * (rule.density + line * T.LINE_BOOST) * bare;
  if (slotRand(local, set, H.OCC) >= p) return none;
  if (slotRand(local, set, H.QUALITY) >= quality) return none;
  if (cornerWater(m, x, z) > FLORA.WET_MAX || slopeAt(m, x, z) > FLORA.SLOPE_MAX) return none;
  let species = slotRand(local, set, H.SPECIES) < rule.pB ? rule.b : rule.a;
  let scale = (0.75 + 0.5 * slotRand(local, set, H.SCALE)) * (0.6 + 0.4 * veg);
  if (set === 0) {
    const broad = species === Sp.BROADLEAF || species === Sp.BROADLEAF_RAIN;
    if (broad && slotRand(local, set, H.ALT) < smoothstep(T.PINE_A0, T.PINE_A1, alt)) species = Sp.PINE;
    scale *= 1 - T.STUNT * line;
  } else if (slotRand(local, set, H.ALT) < line * T.LINE_SHARE) {
    species = slotRand(local, set, H.ALT2) < 0.45 ? Sp.SHRUB_TUNDRA : Sp.GRASS_ALPINE;
  }
  return { species, scale };
}

// ---- creatures ----

export const CREATURES = {
  BIRD_FLOCKS: [3, 2] as const,      // [high, low] quality
  BIRD_MIN: 8, BIRD_MAX: 15,
  CRITTERS: [40, 20] as const,
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
  /** Real seconds between column-map readbacks. */
  MAP_S: 2,
} as const;

export const GRAZING = new Set<number>([LBiome.GRASSLAND, LBiome.SAVANNA, LBiome.STEPPE]);
