// Canonical GPU field list. Names are the contract between sim passes, render and save.
import type { GpuFields } from '../core/gpu';
import { NCOL, NVOX } from './layout';
import { MAX_PLATES } from './worldData';
import { registerHydroFields } from './hydro';
import { registerClimateFields } from './climate';
import { registerMantleFields } from './mantle';
import { registerMagmaFields } from './magma';
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
  f.add('waterTmp', 'vec2', NCOL);  // scratch for carried (water, sedSusp)
  // integer counters (V2, V3), see CTR_* below
  f.add('counters', 'int', CTR_SIZE, { atomic: true });
  registerHydroFields(f);
  registerClimateFields(f);
  registerMantleFields(f);
  registerMagmaFields(f);
  f.add('probe', 'uint', PROBE_SIZE); // inspect probe gather target (probe.ts), scratch
}

/** probe buffer: [0..NY) column voxels, then PROBE_COL.. packed column fields (floats as bits). */
export const PROBE_COL = 128;
export const PROBE_SIZE = PROBE_COL + 16;

/** counters[CTR_RESERVOIR] = mantle reservoir, fill units. */
export const CTR_RESERVOIR = 0;
/**
 * Everything from CTR_PLATE on is per stats window and cleared after each readback.
 * CTR_PLATE + p*4 + {0 area·runs, 1 subducted cols, 2 new crust cols} — tectonics decide kernel.
 * CTR_AREA + p — exact plate area (lifecycle stats kernel).
 * CTR_CENT + p*4 + {Σcos x, Σsin x, Σcos z, Σsin z} ×256 — torus centroid.
 * CTR_CONTACT + p*16 + q — boundary cells where plate p (continental) touches q (continental).
 * CTR_WORLD + {0 Σ ocean level ×16, 1 ocean cols, 2 land cols, 3 Σ water ×64, 4 Σ vapor+ice ×64} — worldStats.ts.
 */
export const CTR_PLATE = 16;
export const CTR_AREA = CTR_PLATE + MAX_PLATES * 4;
export const CTR_CENT = CTR_AREA + MAX_PLATES;
export const CTR_CONTACT = CTR_CENT + MAX_PLATES * 4;
export const CTR_WORLD = CTR_CONTACT + MAX_PLATES * MAX_PLATES;
export const CTR_SIZE = CTR_WORLD + 8;

export function uploadWorld(f: GpuFields, w: WorldData): void {
  for (const name of ['vox', 'plateId', 'crustAge'] as const) {
    f.setParity(name, 0);
    (f.cpuArray(name, 0) as Uint32Array | Float32Array).set(w[name]);
  }
  (f.cpuArray('water') as Float32Array).set(w.water);
  (f.cpuArray('counters') as Int32Array)[CTR_RESERVOIR] = Math.round(w.mantleReservoir);
  for (const n of ['vox', 'plateId', 'crustAge', 'water', 'counters']) f.markDirty(n);
  // climate fields start at zero (vapor/ice/veg spin up in ~100 ticks)
}
