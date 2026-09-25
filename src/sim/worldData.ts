// CPU-side world snapshot: produced by worldgen, uploaded to GPU fields, read back for saves.
import { NCOL, NVOX } from './layout';

export const MAX_PLATES = 16;

export interface Plate {
  id: number;            // index into plate table, 0..MAX_PLATES-1
  alive: boolean;
  vel: [number, number]; // cells per My on X,Z
  accum: [number, number]; // sub-cell offset carried between ticks, cells
  continental: boolean;  // majority continental crust (display/stats only; per-column type lives in voxels)
  age: number;           // My since creation
}

export interface WorldData {
  seed: number;
  vox: Uint32Array;       // NVOX, packed voxels, layout.ts voxIdx
  plateId: Uint32Array;   // NCOL
  crustAge: Float32Array; // NCOL, My, age of surface crust column
  water: Float32Array;    // NCOL, water depth in voxel-y units above solid surface
  plates: Plate[];        // length MAX_PLATES, unused entries alive=false
  mantleReservoir: number; // crust mass units not in voxels (V3 budget)
}

export function emptyWorld(seed: number): WorldData {
  const plates: Plate[] = [];
  for (let i = 0; i < MAX_PLATES; i++) plates.push({ id: i, alive: false, vel: [0, 0], accum: [0, 0], continental: false, age: 0 });
  return {
    seed,
    vox: new Uint32Array(NVOX),
    plateId: new Uint32Array(NCOL),
    crustAge: new Float32Array(NCOL),
    water: new Float32Array(NCOL),
    plates,
    mantleReservoir: 0,
  };
}
