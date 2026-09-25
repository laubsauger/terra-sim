// Canonical GPU field list. Names are the contract between sim passes, render and save.
import type { GpuFields } from '../core/gpu';
import { NCOL, NVOX } from './layout';
import type { WorldData } from './worldData';

export function registerSimFields(f: GpuFields): void {
  f.add('vox', 'uint', NVOX, { pingPong: true }); // packed voxels
  f.add('plateId', 'uint', NCOL, { pingPong: true });
  f.add('surfY', 'float', NCOL);    // derived: top solid y + fill/255, voxel-y units
  f.add('water', 'float', NCOL);    // water depth above surfY, voxel-y units
  f.add('crustAge', 'float', NCOL, { pingPong: true }); // My
}

export function uploadWorld(f: GpuFields, w: WorldData): void {
  for (const name of ['vox', 'plateId', 'crustAge'] as const) {
    f.setParity(name, 0);
    (f.cpuArray(name, 0) as Uint32Array | Float32Array).set(w[name]);
  }
  (f.cpuArray('water') as Float32Array).set(w.water);
  for (const n of ['vox', 'plateId', 'crustAge', 'water']) f.markDirty(n);
}
