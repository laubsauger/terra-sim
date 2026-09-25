// World grid layout, voxel packing, material table. CPU side; TSL mirrors live in tslLayout.ts.

export const NX = 256; // X, wraps
export const NZ = 256; // Z, wraps
export const NY = 128; // Y up, no wrap
export const NCOL = NX * NZ;
export const NVOX = NCOL * NY;

/** World-space block size (render units). X,Z span [-HALF, HALF]. */
export const BLOCK_SIZE = 4;
export const CELL = BLOCK_SIZE / NX;
/** Render height of one voxel layer before vertical exaggeration param. */
export const VOXEL_H = CELL * 0.6;

/** Nominal layer anchors (voxel y). Sea level itself emerges from the water budget. */
export const Y_MANTLE_TOP = 24; // below: pure PERIDOTITE asthenosphere
export const Y_OCEAN_FLOOR = 60;
export const Y_SEA_NOMINAL = 76;
export const Y_CONT_SURFACE = 84;

// Materials. APPEND-ONLY: ids are persisted in saves (V21).
export const Mat = {
  AIR: 0,
  BASALT: 1,
  GABBRO: 2,
  GRANITE: 3,
  ANDESITE: 4,
  SEDIMENT: 5,
  SANDSTONE: 6,
  SHALE: 7,
  LIMESTONE: 8,
  SCHIST: 9,
  GNEISS: 10,
  MAGMA: 11,
  PERIDOTITE: 12,
} as const;
export type MatId = (typeof Mat)[keyof typeof Mat];
export const MAT_COUNT = 13;

/** Density g/cm³ (drives subduction choice and isostasy). */
export const MAT_DENSITY: readonly number[] = [0, 3.0, 3.0, 2.7, 2.8, 2.1, 2.3, 2.4, 2.6, 2.8, 2.8, 2.6, 3.3];
/** Erodibility multiplier (loose sediment high, crystalline low). */
export const MAT_ERODIBILITY: readonly number[] = [0, 0.4, 0.3, 0.25, 0.4, 2.0, 0.9, 1.1, 0.7, 0.35, 0.25, 0, 0.2];

// Voxel u32 pack: mat:8 | fill:8 | age:8 | flags:8
export const FLAG_CONTINENTAL = 1 << 0;
export const FLAG_HOTSPOT = 1 << 1;
export const FLAG_FRACTURED = 1 << 2;

export function packVoxel(mat: number, fill: number, age: number, flags: number): number {
  return ((mat & 0xff) | ((fill & 0xff) << 8) | ((age & 0xff) << 16) | ((flags & 0xff) << 24)) >>> 0;
}
export const voxMat = (v: number) => v & 0xff;
export const voxFill = (v: number) => (v >>> 8) & 0xff;
export const voxAge = (v: number) => (v >>> 16) & 0xff;
export const voxFlags = (v: number) => (v >>> 24) & 0xff;

/** Age code: log-scale, 0 = new, 255 ≈ 4 Gy. age My = 2^(code/16) - 1. */
export function ageToCode(my: number): number {
  return Math.max(0, Math.min(255, Math.round(16 * Math.log2(1 + Math.max(0, my)))));
}
export function codeToAge(code: number): number {
  return 2 ** (code / 16) - 1;
}

// Indexing. Torus wrap on X,Z (V1); NX,NZ powers of two so & works for negatives too.
export const wrapX = (x: number) => x & (NX - 1);
export const wrapZ = (z: number) => z & (NZ - 1);
export const colIdx = (x: number, z: number) => wrapX(x) + wrapZ(z) * NX;
/** y-major: one horizontal layer is contiguous (coalesced plate shifts, per-layer passes). */
export const voxIdx = (x: number, y: number, z: number) => colIdx(x, z) + y * NCOL;
