// §T.23 2D lava flow, cooling, solidification into voxels (§C pass 6). Calibration in magmaModel.ts (LAVA).
//
// lava (uvec2 per column): x = mass in fill units (thickness = x/255 voxel layers on top of surfY),
//   y = temperature °C·16 (bits 0-15) | silica code (bits 16-23). Integer mass → exact V3 accounting.
// Flow (LAVA.substeps per tick, two kernels each, like the thermal-erosion gather/apply):
//   out:   head H = surfY + x/255; toward each lower neighbour send 255·0.2·mobility(T, silica)·(ΔH − yield)
//          units (dithered to integers), scaled so Σ out ≤ own mass. Writes only its own lavaOut.
//   apply: x ← x − Σ own out + Σ neighbours' out toward me; temperature/silica mixed by mass.
//   Every unit sent is received by exactly one neighbour from the same stored integer → exact conservation.
// Cool (once per tick): Newtonian cooling toward tAmb (thick flows slower, water quenches fast). A fraction
//   crusts over each My; below tSolid or when thin, everything freezes. Frozen mass tops up a partial crust top
//   voxel (erosion.ts convention: only the top voxel may be partial), then stacks new BASALT / ANDESITE voxels
//   (ANDESITE carries FLAG_CONTINENTAL: arcs grow continents). Cones build where eruptions repeat.
// Race-free: one thread per column, writes only its own column (vox in place in the current buffer).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, Return, float, int, uint, uvec2, uvec4, max, floor, exp, select, smoothstep,
  instanceIndex, uniform, atomicAdd,
} from 'three/tsl';
import { ReaderKernel, type GpuFields, type StorageNode } from '../core/gpu';
import { NCOL, NY, Mat, FLAG_CONTINENTAL } from './layout';
import { tMat, tFill, tAge, tFlags, tPack, tVoxIdx, tColIdx, tColXZ, uMin } from './tslLayout';
import { CTR_RESERVOIR } from './fields';
import { LAVA, MAGMA, MCTR } from './magmaModel';
import { isCrust, tHash01, tQuant, tLavaTemp, tLavaSil, tLavaY, tLavaMix } from './magmaTsl';

type U = THREE.Node<'uint'>;

const lo16 = (v: U) => v.bitAnd(uint(0xffff));
const hi16 = (v: U) => v.shiftRight(uint(16));

export class LavaPass {
  readonly uniforms = {
    dt: uniform(0.05),
    tick: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    // inject (god tool / tests): one column, units drawn from the reservoir
    injCol: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    injUnits: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    injTemp: uniform(LAVA.tBasalt),
    injSil: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
  };
  private outK: THREE.ComputeNode;
  private applyK: THREE.ComputeNode;
  private coolK: ReaderKernel<'uint'>;
  private injectK: THREE.ComputeNode;

  constructor(private fields: GpuFields) {
    this.outK = this.buildOut();
    this.applyK = this.buildApply();
    this.coolK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildCool(vox));
    this.injectK = this.buildInject();
  }

  /** §C pass 6, every tick. Needs surfY fresh up to this tick's magma edits (±1 layer at vents is tolerated). */
  step(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number): void {
    this.uniforms.dt.value = dtGeo;
    for (let s = 0; s < LAVA.substeps; s++) {
      this.uniforms.tick.value = (tick * LAVA.substeps + s) >>> 0;
      renderer.compute(this.outK);
      renderer.compute(this.applyK);
    }
    this.uniforms.tick.value = tick >>> 0;
    this.coolK.run(renderer);
  }

  /** Put `units` of lava on column c (drawn from the mantle reservoir, V16). God tools, tests. */
  inject(renderer: THREE.WebGPURenderer, c: number, units: number, tempC: number = LAVA.tBasalt, sil = 0): void {
    const u = this.uniforms;
    u.injCol.value = c >>> 0; u.injUnits.value = units >>> 0; u.injTemp.value = tempC; u.injSil.value = sil & 0xff;
    renderer.compute(this.injectK);
  }

  private buildOut(): THREE.ComputeNode {
    // storage buffers: lava, surfY, lavaOut = 3
    const lava = this.fields.cur<'uvec2'>('lava');
    const surfY = this.fields.cur('surfY');
    const lavaOut = this.fields.cur<'uvec4'>('lavaOut');
    const { tick } = this.uniforms;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const L = lava.element(i).toVar();
      const m = L.x.toVar();
      If(m.equal(uint(0)), () => { lavaOut.element(i).assign(uvec4(0, 0, L.y, 0)); Return(); });
      const { x, z } = tColXZ(i);
      const h = surfY.element(i).add(float(m).div(255)).toVar();
      const fluid = smoothstep(LAVA.tSolid, LAVA.tAndesite, tLavaTemp(L.y));
      const mob = float(LAVA.mobility).mul(fluid).mul(float(1).sub(float(tLavaSil(L.y)).mul(0.6 / 255))).toVar();
      const nbs = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const;
      const q = nbs.map(([dx, dz], k) => {
        const n = tColIdx(x.add(int(dx)), z.add(int(dz)));
        const hn = surfY.element(n).add(float(lava.element(n).x).div(255));
        const ex = max(h.sub(hn).sub(LAVA.yield), 0);
        return tQuant(ex.mul(255 * 0.2).mul(mob), tHash01(i.mul(uint(4)).add(uint(k)), tick, 21), uint(0xffff)).toVar();
      });
      const sum = q[0]!.add(q[1]!).add(q[2]!).add(q[3]!).toVar();
      If(sum.greaterThan(m), () => {
        const sc = float(m).div(float(sum));
        for (const qk of q) qk.assign(uint(floor(float(qk).mul(sc))));
        // float rounding may leave Σ one or two above m: take the excess back, first come first served
        const ex = q[0]!.add(q[1]!).add(q[2]!).add(q[3]!).toVar();
        const over = select(ex.greaterThan(m), ex.sub(m), uint(0)).toVar();
        for (const qk of q) { const t = uMin(qk, over); qk.subAssign(t); over.subAssign(t); }
      });
      lavaOut.element(i).assign(uvec4(q[0]!.bitOr(q[1]!.shiftLeft(uint(16))), q[2]!.bitOr(q[3]!.shiftLeft(uint(16))), L.y, 0));
    })().compute(NCOL);
  }

  private buildApply(): THREE.ComputeNode {
    // storage buffers: lava, lavaOut = 2
    const lava = this.fields.cur<'uvec2'>('lava');
    const lavaOut = this.fields.cur<'uvec4'>('lavaOut');
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const o = lavaOut.element(i).toVar();
      const L = lava.element(i).toVar();
      const out = lo16(o.x).add(hi16(o.x)).add(lo16(o.y)).add(hi16(o.y));
      const m = L.x.sub(out).toVar();
      const tw = float(m).mul(tLavaTemp(L.y)).toVar();
      const sw = float(m).mul(float(tLavaSil(L.y))).toVar();
      const tot = m.toVar();
      // +x neighbour sends on its −x half, −x neighbour on +x, +z on −z, −z on +z
      const ins: [number, number, 'x' | 'y', boolean][] = [[1, 0, 'x', true], [-1, 0, 'x', false], [0, 1, 'y', true], [0, -1, 'y', false]];
      for (const [dx, dz, comp, hi] of ins) {
        const on = lavaOut.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).toVar();
        const v = on[comp];
        const q = (hi ? hi16(v) : lo16(v)).toVar();
        If(q.greaterThan(uint(0)), () => {
          tot.addAssign(q);
          tw.addAssign(float(q).mul(tLavaTemp(on.z)));
          sw.addAssign(float(q).mul(float(tLavaSil(on.z))));
        });
      }
      const inv = float(1).div(max(float(tot), 1));
      lava.element(i).assign(select(tot.equal(uint(0)), uvec2(0, 0),
        uvec2(tot, tLavaY(tw.mul(inv), uint(sw.mul(inv).add(0.5))))));
    })().compute(NCOL);
  }

  private buildCool(vox: StorageNode<'uint'>): THREE.ComputeNode {
    // storage buffers: vox, lava, water, magmaCtr = 4
    const lava = this.fields.cur<'uvec2'>('lava');
    const water = this.fields.cur('water');
    const mctr = this.fields.cur<'int'>('magmaCtr');
    const { dt, tick } = this.uniforms;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const L = lava.element(i).toVar();
      const m = L.x.toVar();
      If(m.equal(uint(0)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const sil = tLavaSil(L.y).toVar();
      const wet = water.element(i).greaterThan(LAVA.wetDepth);
      const k = select(wet, float(LAVA.kWater), float(LAVA.kAir).div(float(1).add(float(m).mul(0.5 / 255))));
      const T1 = float(LAVA.tAmb).add(tLavaTemp(L.y).sub(LAVA.tAmb).mul(exp(k.mul(dt).negate()))).toVar();
      const all = T1.lessThan(LAVA.tSolid).or(m.lessThanEqual(uint(LAVA.thin)));
      const frz = select(all, m, tQuant(float(m).mul(LAVA.freezeRate).mul(dt), tHash01(i, tick, 31), m)).toVar();
      If(frz.greaterThan(uint(0)), () => {
        const topY = int(-1).toVar();
        const topV = uint(0).toVar();
        Loop({ start: int(NY - 1), end: int(0), condition: '>=' }, ({ i: y }) => {
          const v = vox.element(tVoxIdx(x, y, z));
          If(tMat(v).notEqual(uint(Mat.AIR)), () => { topY.assign(y); topV.assign(v); Break(); });
        });
        const rem = frz.toVar();
        // top up a partial crust top voxel (keeps its material)
        If(topY.greaterThanEqual(int(0)).and(isCrust(tMat(topV))).and(tFill(topV).lessThan(uint(255))), () => {
          const t = uMin(rem, uint(255).sub(tFill(topV))).toVar();
          vox.element(tVoxIdx(x, topY, z)).assign(tPack(tMat(topV), tFill(topV).add(t), tAge(topV), tFlags(topV)));
          rem.subAssign(t);
        });
        const felsic = sil.greaterThanEqual(uint(MAGMA.silAndesite));
        const mat = select(felsic, uint(Mat.ANDESITE), uint(Mat.BASALT));
        const flg = select(felsic, uint(FLAG_CONTINENTAL), uint(0));
        Loop(LAVA.maxNewVoxels, () => {
          If(rem.greaterThan(uint(0)).and(topY.add(int(1)).lessThanEqual(int(NY - 2))), () => {
            topY.addAssign(1);
            const put = uMin(rem, uint(255));
            vox.element(tVoxIdx(x, topY, z)).assign(tPack(mat, put, uint(0), flg));
            rem.subAssign(put);
          });
        });
        const placed = frz.sub(rem);
        m.subAssign(placed);
        atomicAdd(mctr.element(MCTR.SOLIDIFIED), int(placed));
      });
      lava.element(i).assign(select(m.equal(uint(0)), uvec2(0, 0), uvec2(m, tLavaY(T1, sil))));
    })().compute(NCOL);
  }

  private buildInject(): THREE.ComputeNode {
    // storage buffers: counters, lava, magmaCtr = 3
    const ctr = this.fields.cur<'int'>('counters');
    const lava = this.fields.cur<'uvec2'>('lava');
    const mctr = this.fields.cur<'int'>('magmaCtr');
    const { injCol, injUnits, injTemp, injSil } = this.uniforms;
    return Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      const L = lava.element(injCol).toVar();
      lava.element(injCol).assign(tLavaMix(L.x, L.y, injUnits, injTemp, injSil));
      atomicAdd(ctr.element(CTR_RESERVOIR), int(0).sub(int(injUnits)));
      atomicAdd(mctr.element(MCTR.DRAWN), int(injUnits));
    })().compute(1);
  }
}

