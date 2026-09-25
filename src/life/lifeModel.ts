// §T.45 tiny life: pure CPU model shared by the GPU flora kernel (flora.ts), the CPU creatures
// (creatures.ts) and tests. No three import. Life only reads sim fields (V15).
import { Biome, BIOME_COUNT } from '../sim/biomeModel';
import { NX, NZ, BLOCK_SIZE, CELL } from '../sim/layout';

// ---- flora ----

/** Plant species. 0 = empty slot. Species picks mesh kind, colour and size. */
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
  SHRUB_TUNDRA: 10,
  FERN: 11,
} as const;
export const SPECIES_COUNT = 12;

/** Mesh kinds (one indirect draw each). Kinds 0-3 live on the large slot set, 4-6 on the small one. */
export const Kind = { TREE: 0, PINE: 1, CACTUS: 2, ACACIA: 3, SHRUB: 4, GRASS: 5, FLOWER: 6 } as const;
export const KIND_COUNT = 7;
export const KIND_NAMES = ['tree', 'pine', 'cactus', 'acacia', 'shrub', 'grass', 'flower'] as const;
export const KIND_SET: readonly number[] = [0, 0, 0, 0, 1, 1, 1];

/** Species → mesh kind. */
export const SPECIES_KIND: readonly number[] = [
  -1, Kind.TREE, Kind.TREE, Kind.PINE, Kind.CACTUS, Kind.SHRUB, Kind.GRASS, Kind.GRASS, Kind.FLOWER, Kind.ACACIA, Kind.SHRUB, Kind.GRASS,
];
/** World height of the unit-height mesh at scale 1. */
export const SPECIES_SIZE: readonly number[] = [0, 0.062, 0.078, 0.078, 0.052, 0.012, 0.015, 0.017, 0.015, 0.058, 0.011, 0.018];
/** Foliage / main colour (sRGB hex). */
export const SPECIES_COLOR: readonly number[] = [
  0xff00ff, 0x6cb743, 0x2e7a3c, 0x2f6b52, 0x74a95e, 0x8f9a58, 0x7cc64a, 0xd5b95c, 0x76b84a, 0x93a848, 0x9a6a48, 0x3f9642,
];
/** Trunk / secondary colour (sRGB hex). */
export const SPECIES_TRUNK: readonly number[] = [
  0xff00ff, 0x7a5236, 0x5a3d27, 0x654630, 0x5f8f4e, 0x7a6243, 0x5c9a3a, 0xa08840, 0x4f9a3c, 0x6e5038, 0x6f5a3a, 0x2f7434,
];
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

/** Per (biome, slot set): probability per unit veg that a slot is occupied, species a/b, P(b). */
export interface FloraRule { density: number; a: number; b: number; pB: number }
const R = (density: number, a: number, b = a, pB = 0): FloraRule => ({ density, a, b, pB });
const NONE = R(0, Sp.NONE);

/** FLORA_RULES[biome] = [large set rule, small set rule]. Nothing on ocean, ice, alpine, beach. */
export const FLORA_RULES: readonly (readonly [FloraRule, FloraRule])[] = (() => {
  const t: [FloraRule, FloraRule][] = Array.from({ length: BIOME_COUNT }, () => [NONE, NONE]);
  t[Biome.TUNDRA] = [NONE, R(1.4, Sp.SHRUB_TUNDRA, Sp.FLOWER, 0.12)];
  t[Biome.TAIGA] = [R(1.3, Sp.PINE), R(0.35, Sp.SHRUB_TUNDRA, Sp.FERN, 0.4)];
  t[Biome.TEMPERATE_FOREST] = [R(0.9, Sp.BROADLEAF, Sp.PINE, 0.12), R(0.55, Sp.GRASS, Sp.FLOWER, 0.15)];
  t[Biome.GRASSLAND] = [R(0.07, Sp.BROADLEAF), R(1.7, Sp.GRASS, Sp.FLOWER, 0.16)];
  t[Biome.DESERT] = [R(2.6, Sp.CACTUS), R(3.5, Sp.SHRUB_DRY)];
  t[Biome.SAVANNA] = [R(0.3, Sp.ACACIA), R(2.2, Sp.GRASS_DRY, Sp.SHRUB_DRY, 0.1)];
  t[Biome.RAINFOREST] = [R(1.05, Sp.BROADLEAF_RAIN), R(0.7, Sp.FERN, Sp.FLOWER, 0.08)];
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
export const H = { OCC: 0, SPECIES: 1, SCALE: 2, JX: 3, JZ: 4, YAW: 5, QUALITY: 6, TINT: 7, SQUASH: 8, PETAL: 9 } as const;
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
export function floraDecide(m: ColumnMap & { veg: Float32Array }, slot: number, quality = 1): { species: number; scale: number } {
  const { set, local } = slotInfo(slot);
  const [x, z] = slotXZ(slot);
  const none = { species: 0, scale: 0 };
  if (Math.abs(x) > BLOCK_SIZE / 2 - EDGE_MARGIN[set]! || Math.abs(z) > BLOCK_SIZE / 2 - EDGE_MARGIN[set]!) return none;
  const c = columnAt(x, z);
  const rule = FLORA_RULES[m.biome[c]!]?.[set] ?? NONE;
  const veg = m.veg[c]!;
  if (slotRand(local, set, H.OCC) >= veg * rule.density) return none;
  if (slotRand(local, set, H.QUALITY) >= quality) return none;
  if (cornerWater(m, x, z) > FLORA.WET_MAX || slopeAt(m, x, z) > FLORA.SLOPE_MAX) return none;
  const species = slotRand(local, set, H.SPECIES) < rule.pB ? rule.b : rule.a;
  return { species, scale: (0.75 + 0.5 * slotRand(local, set, H.SCALE)) * (0.6 + 0.4 * veg) };
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

export const GRAZING = new Set<number>([Biome.GRASSLAND, Biome.SAVANNA]);
