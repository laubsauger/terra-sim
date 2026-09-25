// Synthetic CPU-filled world for render tests: sine hills + a mountain on the land half, ocean on
// the other half, folded continental strata, oceanic basalt, a magma pocket straddling the z seam.
// Fills the fields directly (no worldgen / derive pass needed).
import { NX, NY, NZ, NCOL, Mat, packVoxel, voxIdx, colIdx, FLAG_CONTINENTAL, Y_SEA_NOMINAL } from '../../../src/sim/layout';
import type { GpuFields } from '../../../src/core/gpu';

const TAU = Math.PI * 2;
export const SEA = Y_SEA_NOMINAL;
/** Column x of the land crest and the deep ocean (all z). */
export const LAND_X = 64;
export const OCEAN_X = 192;

const wrapD = (a: number, b: number, n: number) => { const d = Math.abs(a - b) % n; return Math.min(d, n - d); };

export function heightAt(x: number, z: number): number {
  const base = 76 + 17 * Math.sin((TAU * x) / NX);
  const hills = 6 * Math.sin((TAU * 3 * x) / NX + 1.3) * Math.cos((TAU * 2 * z) / NZ)
    + 2.2 * Math.sin((TAU * 7 * x) / NX + (TAU * 5 * z) / NZ);
  const dx = wrapD(x, 70, NX), dz = wrapD(z, 150, NZ);
  const mountain = 24 * Math.exp(-(dx * dx + dz * dz) / (2 * 16 * 16));
  const land = Math.max(0, Math.sin((TAU * x) / NX)); // hills and peaks only on the land half
  return base + (hills + mountain) * (0.35 + 0.65 * land);
}

/** Material of a solid voxel at (x,y,z) in a column whose surface is at h. */
export function matAt(x: number, y: number, z: number, h: number): number {
  const fold = 5 * Math.sin((TAU * 2 * x) / NX + (TAU * z) / NZ) + 2 * Math.sin((TAU * 5 * z) / NZ);
  const moho = 24 + 2 * Math.sin((TAU * 3 * x) / NX);
  if (y < moho) return Mat.PERIDOTITE;
  // magma pocket centred on the z seam (z=0 ≡ z=NZ), visible on both ±Z faces
  const mx = (x - 96) / 12, my = (y - 40) / 6, mz = wrapD(z, 0, NZ) / 9;
  if (mx * mx + my * my + mz * mz < 1) return Mat.MAGMA;
  const continental = 76 + 17 * Math.sin((TAU * x) / NX) > 70;
  if (!continental) {
    if (y < 36) return Mat.GABBRO;
    return y >= h - 2.5 ? Mat.SEDIMENT : Mat.BASALT;
  }
  if (y >= h - 1.5) return Mat.SEDIMENT;
  const s = y - fold; // folded stack
  if (s < 30) return Mat.GABBRO;
  if (s < 44) return Mat.GRANITE;
  if (s < 50) return Mat.GNEISS;
  if (s < 56) return Mat.SCHIST;
  if (s < 62) return Mat.SHALE;
  if (s < 68) return Mat.SANDSTONE;
  if (s < 74) return Mat.LIMESTONE;
  if (s < 79) return Mat.SHALE;
  if (s < 86) return Mat.SANDSTONE;
  if (s < 92) return Mat.LIMESTONE;
  return Mat.ANDESITE;
}

export interface SynthInfo { surf: Float32Array; water: Float32Array }

/**
 * Real world goes into vox buffer 1 with parity 1; buffer 0 is filled with solid MAGMA as a decoy,
 * so a renderer that ignores the ping-pong parity shows an orange block instead of strata.
 */
export function fillSynthWorld(fields: GpuFields): SynthInfo {
  const vox = fields.cpuArray('vox', 1) as Uint32Array;
  const decoy = fields.cpuArray('vox', 0) as Uint32Array;
  decoy.fill(packVoxel(Mat.MAGMA, 255, 0, 0));
  vox.fill(0);
  const surf = fields.cpuArray('surfY') as Float32Array;
  const water = fields.cpuArray('water') as Float32Array;
  for (let z = 0; z < NZ; z++) {
    for (let x = 0; x < NX; x++) {
      const h = Math.min(NY - 1, heightAt(x, z));
      const top = Math.floor(h);
      const frac = h - top;
      const continental = 76 + 17 * Math.sin((TAU * x) / NX) > 70;
      const flags = continental ? FLAG_CONTINENTAL : 0;
      let topSolid = -1, topFill = 0;
      for (let y = 0; y <= top && y < NY; y++) {
        const fill = y < top ? 255 : Math.round(frac * 255);
        if (fill === 0) continue;
        vox[voxIdx(x, y, z)] = packVoxel(matAt(x, y, z, h), fill, 40, flags);
        topSolid = y; topFill = fill;
      }
      // derive pass contract: surfY = top solid y + fill/255
      const s = topSolid + topFill / 255;
      const c = colIdx(x, z);
      surf[c] = s;
      water[c] = s < SEA ? SEA - s : 0;
    }
  }
  fields.setParity('vox', 1);
  fields.markDirty('vox');
  fields.markDirty('surfY');
  fields.markDirty('water');
  return { surf: surf.slice(0, NCOL), water: water.slice(0, NCOL) };
}
