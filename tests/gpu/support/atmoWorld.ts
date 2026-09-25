// Hand-set climate / lava / crust-age fields for atmosphere tests. The atmosphere only reads them (V15),
// so they need not come from the sim: a real worldgen + derive terrain with painted weather.
import { NX, NZ, NCOL, colIdx } from '../../../src/sim/layout';
import { encLavaY } from '../../../src/sim/magmaModel';
import { capacity } from '../../../src/atmo/atmoModel';
import type { GpuFields } from '../../../src/core/gpu';

/** Column boxes (inclusive x0..x1, z0..z1) of the painted weather layout. */
export interface Box { x0: number; x1: number; z0: number; z1: number }
export const WET: Box = { x0: 24, x1: 104, z0: 40, z1: 120 };     // saturated air: clouds
export const STORM: Box = { x0: 56, x1: 88, z0: 64, z1: 96 };      // inside WET: heavy precip
export const SNOWY: Box = { x0: 150, x1: 200, z0: 150, z1: 200 };  // saturated, precip, below freezing
export const inBox = (b: Box, x: number, z: number) => x >= b.x0 && x <= b.x1 && z >= b.z0 && z <= b.z1;

export interface Paint {
  /** Column of a hot lava pool (radius 2 columns), or null; `erupt` marks its centre as an active vent. */
  lava?: { x: number; z: number; erupt?: boolean } | null;
  /** Rows z of a young-crust rift line (crustAge 0.2 My) spanning x0..x1. */
  rift?: { z: number; x0: number; x1: number } | null;
}

/**
 * Weather layout: dry everywhere (rh 0.2, no precip, 20 °C) except WET (rh 1.02) with STORM (precip
 * 5e-3) inside it, and SNOWY (−8 °C, saturated, precip 2e-3).
 */
export function paintWeather(fields: GpuFields, p: Paint = {}): void {
  const vapor = new Float32Array(NCOL), precip = new Float32Array(NCOL), temp = new Float32Array(NCOL);
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const c = colIdx(x, z);
    let t = 20, rh = 0.2, pr = 0;
    if (inBox(WET, x, z)) { rh = 1.02; pr = 2e-4; }
    if (inBox(STORM, x, z)) pr = 5e-3;
    if (inBox(SNOWY, x, z)) { t = -8; rh = 1.02; pr = 2e-3; }
    temp[c] = t; vapor[c] = rh * capacity(t); precip[c] = pr;
  }
  (fields.cpuArray('vapor') as Float32Array).set(vapor);
  (fields.cpuArray('precip') as Float32Array).set(precip);
  (fields.cpuArray('surfTemp') as Float32Array).set(temp);
  const lava = new Uint32Array(NCOL * 2);
  if (p.lava) {
    for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
      if (dx * dx + dz * dz > 5) continue;
      const c = colIdx(p.lava.x + dx, p.lava.z + dz);
      lava[c * 2] = 255 * 2; lava[c * 2 + 1] = encLavaY(1180, 0);
    }
  }
  (fields.cpuArray('lava') as Uint32Array).set(lava);
  // magma.ts eruption state: x activity, y phase (2 = active)
  const volc = new Float32Array(NCOL * 4);
  if (p.lava && p.lava.erupt !== false) { const c = colIdx(p.lava.x, p.lava.z); volc[c * 4] = 1; volc[c * 4 + 1] = 2; volc[c * 4 + 2] = 1; }
  (fields.cpuArray('volcano') as Float32Array).set(volc);
  fields.markDirty('volcano');
  for (const w of [0, 1] as const) {
    const age = fields.cpuArray('crustAge', w) as Float32Array;
    age.fill(80);
    if (p.rift) for (let x = p.rift.x0; x <= p.rift.x1; x++) for (let dz = -1; dz <= 1; dz++) age[colIdx(x, p.rift.z + dz)] = 0.2;
  }
  for (const n of ['vapor', 'precip', 'surfTemp', 'lava', 'crustAge']) fields.markDirty(n);
}
