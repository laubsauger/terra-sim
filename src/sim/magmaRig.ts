// GPU rig for magma / lava / mantle tests and debugging: passes 2-6 + 10 in orchestrator order, budget readback.
import type * as THREE from 'three/webgpu';
import { GpuFields } from '../core/gpu';
import { Params } from '../core/params';
import { NX, NZ, NY, NCOL, Mat, Y_MANTLE_TOP, packVoxel, voxIdx, colIdx } from './layout';
import { registerSimFields, uploadWorld, CTR_RESERVOIR } from './fields';
import { createDerivePass } from './derive';
import { Tectonics } from './tectonics';
import { emptyWorld, type WorldData } from './worldData';
import { MantlePass, type MantleOptions } from './mantle';
import { MagmaPass } from './magma';
import { LavaPass } from './lava';
import { magmaBudgetCpu, chamberMassPerColumn, parseMagmaStats } from './magmaModel';

export interface MagmaRigOptions { mantle?: MantleOptions; isoEvery?: number; dtGeo?: number }

export async function makeMagmaRig(renderer: THREE.WebGPURenderer, world: WorldData, opts: MagmaRigOptions = {}) {
  const f = new GpuFields();
  registerSimFields(f);
  f.freeze();
  uploadWorld(f, world);
  const params = new Params();
  const derive = createDerivePass(f);
  const tec = new Tectonics(f, world.plates);
  const mantle = new MantlePass(f, params, world, opts.mantle);
  const magma = new MagmaPass(f, params);
  const lava = new LavaPass(f);
  const dt = opts.dtGeo ?? 0.05;
  const isoEvery = opts.isoEvery ?? 1_000_000;
  magma.init(renderer);
  derive.run(renderer);
  mantle.init(renderer);
  let t = 0;
  const readU = async (n: string) => new Uint32Array(await f.read(renderer, n));
  const readF = async (n: string) => new Float32Array(await f.read(renderer, n));
  const rig = {
    f, params, derive, tec, mantle, magma, lava, readU, readF,
    get tick() { return t; },
    /** One geo tick, orchestrator order: 3-4 tectonics (+ derive, afterTectonics), 2 mantle, 5 magma, 6 lava, 10 derive. */
    step(n = 1, only?: { tectonics?: boolean; mantle?: boolean; magma?: boolean; lava?: boolean }) {
      for (let k = 0; k < n; k++) {
        t++;
        if (only?.tectonics !== false && tec.tick(renderer, t, dt, { speedMul: 1, isoEvery })) { derive.run(renderer); magma.afterTectonics(renderer); }
        if (only?.mantle !== false) mantle.step(renderer, t, dt);
        if (only?.magma !== false) magma.step(renderer, t, dt);
        if (only?.lava !== false) lava.step(renderer, t, dt);
        derive.run(renderer);
      }
    },
    async budget() {
      const vox = await readU('vox');
      const res = new Int32Array(await f.read(renderer, 'counters'))[CTR_RESERVOIR]!;
      const magCol = await readU('magCol');
      const b = magmaBudgetCpu(vox, res, magCol, await readU('lava'));
      // GPU chamber cache must match the voxels column by column
      const cpu = chamberMassPerColumn(vox);
      let cacheMismatch = 0;
      for (let c = 0; c < NCOL; c++) if (cpu[c] !== magCol[c * 2]) cacheMismatch++;
      return { ...b, cacheMismatch, vox, magCol };
    },
    async stats() { return parseMagmaStats(await f.read(renderer, 'magmaCtr')); },
  };
  return rig;
}

/** Dry world with a wide hill (surface 60 → 100) centred at (128,128), BASALT over PERIDOTITE. Lava tests. */
export function hillWorld(): WorldData {
  const w = emptyWorld(3);
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const dx = x - 128, dz = z - 128;
    const r2 = (dx * dx + dz * dz) / (70 * 70);
    const H = 60 + 40 * Math.max(0, 1 - r2);
    let top = Math.floor(H);
    let fill = Math.round((H - top) * 255);
    if (fill === 0) { top -= 1; fill = 255; }
    for (let y = 0; y <= top && y < NY; y++) {
      w.vox[voxIdx(x, y, z)] = packVoxel(y < Y_MANTLE_TOP ? Mat.PERIDOTITE : Mat.BASALT, y === top ? fill : 255, 0, 0);
    }
    w.plateId[colIdx(x, z)] = 0;
  }
  w.plates[0]!.alive = true;
  w.crustAge.fill(100); // not fresh ridge crust (no fissure lava)
  w.mantleReservoir = 10_000_000;
  return w;
}

export { NX, NZ, NY, NCOL };
