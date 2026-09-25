// Canonical GPU field list. Names are the contract between sim passes, render and save.
import type { GpuFields } from '../core/gpu';
import { NCOL, NVOX } from './layout';
import { MAX_PLATES } from './worldData';
import type { WorldData } from './worldData';

export function registerSimFields(f: GpuFields): void {
  f.add('vox', 'uint', NVOX, { pingPong: true }); // packed voxels
  f.add('plateId', 'uint', NCOL, { pingPong: true });
  f.add('surfY', 'float', NCOL);    // derived: top solid y + fill/255, voxel-y units
  f.add('water', 'float', NCOL);    // water depth above surfY, voxel-y units
  f.add('crustAge', 'float', NCOL, { pingPong: true }); // My
  // derived column summary (derive pass)
  // x = base:8 | top:8 | flags:8 (bit0 continental), y = crust mass: Σ fill of crust voxels
  // (not AIR/PERIDOTITE/MAGMA), fill units. Packed to stay under the storage-buffer limit (V23).
  f.add('colInfo', 'uvec2', NCOL);
  // tectonics
  f.add('tecAct', 'uint', NCOL);    // per-column action from decide kernel
  f.add('waterTmp', 'float', NCOL); // scratch for carried water
  // integer counters (V2, V3), see CTR_* below
  f.add('counters', 'int', CTR_SIZE, { atomic: true });
}

/** counters[CTR_RESERVOIR] = mantle reservoir, fill units. */
export const CTR_RESERVOIR = 0;
/** Per-plate counters at CTR_PLATE + p*4 + {0 area, 1 subducted cols, 2 new crust cols}; cleared after each readback. */
export const CTR_PLATE = 16;
export const CTR_SIZE = CTR_PLATE + MAX_PLATES * 4;

export function uploadWorld(f: GpuFields, w: WorldData): void {
  for (const name of ['vox', 'plateId', 'crustAge'] as const) {
    f.setParity(name, 0);
    (f.cpuArray(name, 0) as Uint32Array | Float32Array).set(w[name]);
  }
  (f.cpuArray('water') as Float32Array).set(w.water);
  (f.cpuArray('counters') as Int32Array)[CTR_RESERVOIR] = Math.round(w.mantleReservoir);
  for (const n of ['vox', 'plateId', 'crustAge', 'water', 'counters']) f.markDirty(n);
}
