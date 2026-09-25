// Synthetic worlds + GPU rig for hydro/erosion tests and debugging (no worldgen dependency).
import type * as THREE from 'three/webgpu';
import { GpuFields } from '../core/gpu';
import { Params } from '../core/params';
import { NX, NZ, NY, NCOL, NVOX, Mat, Y_MANTLE_TOP, packVoxel, voxIdx } from './layout';
import { registerSimFields, uploadWorld } from './fields';
import { emptyWorld } from './worldData';
import { createDerivePass } from './derive';
import { createHydroPass } from './hydro';
import { createErosionPass } from './erosion';

/** Solid column up to surface height H (voxel-y): full voxels below floor(H), partial top voxel. */
export function voxFromHeights(height: (x: number, z: number) => number, mat: (x: number, z: number) => number = () => Mat.BASALT): Uint32Array {
  const vox = new Uint32Array(NVOX);
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const H = Math.max(1, Math.min(NY - 0.01, height(x, z)));
    let top = Math.floor(H);
    let fill = Math.round((H - top) * 255);
    if (fill === 0) { top -= 1; fill = 255; }
    const m = mat(x, z);
    for (let y = 0; y <= top; y++) {
      vox[voxIdx(x, y, z)] = packVoxel(y < Y_MANTLE_TOP ? Mat.PERIDOTITE : m, y === top ? fill : 255, 0, 0);
    }
  }
  return vox;
}

/** Σ fill of crust voxels (derive's crust mass definition: not AIR/PERIDOTITE/MAGMA), f64. */
export function crustMassCpu(vox: Uint32Array): number {
  let s = 0;
  for (let i = 0; i < vox.length; i++) {
    const v = vox[i]!, m = v & 0xff;
    if (m !== Mat.AIR && m !== Mat.PERIDOTITE && m !== Mat.MAGMA) s += (v >>> 8) & 0xff;
  }
  return s;
}

export function sum64(a: ArrayLike<number>): number { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]!; return s; }

export async function makeHydroRig(renderer: THREE.WebGPURenderer, vox: Uint32Array, water?: Float32Array) {
  const f = new GpuFields();
  registerSimFields(f);
  f.freeze();
  const w = emptyWorld(1);
  w.vox.set(vox);
  if (water) w.water.set(water);
  uploadWorld(f, w);
  const params = new Params();
  const derive = createDerivePass(f);
  const hydro = createHydroPass(f, params);
  const erosion = createErosionPass(f, params);
  derive.run(renderer);
  const readF = async (name: string) => new Float32Array(await f.read(renderer, name));
  const readU = async (name: string) => new Uint32Array(await f.read(renderer, name));
  return {
    f, params, derive, hydro, erosion, readF, readU,
    /** one geo tick of pass 8 as the orchestrator runs it */
    tick(substeps?: number) { hydro.step(renderer, substeps); erosion.step(renderer); derive.run(renderer); },
    async mass() { return crustMassCpu(await readU('vox')) + sum64(await readF('sedSusp')); },
  };
}

export const idx = (x: number, z: number) => ((x % NX) + NX) % NX + (((z % NZ) + NZ) % NZ) * NX;
export { NX, NZ, NY, NCOL };
