// §T.23 2D lava flow, cooling, solidification into voxels (§C pass 6). Calibration in magmaModel.ts (LAVA).
//
// lava (uvec2 per column): x = mass in fill units (thickness = x/255 voxel layers on top of surfY),
//   y = temperature °C·16 (bits 0-15) | silica code (16-23) | channel memory 0..255 (24-31, kept when empty).
// Flow (LAVA.substeps per tick; gather/apply, deterministic — no random dither):
//   out:   head H = surfY + x/255. Steepest descent over the 8 neighbours (slope / distance, plus chanBias ×
//          channel memory, so new pulses follow the previous channel) receives main·fluidity of the mass, capped
//          at 0.4 of the head drop so the cells never overshoot; each lower axis neighbour gets a small viscous
//          spread (≤ 0.1 of its drop). Σ out ≤ own mass. Writes only its own lavaOut.
//   apply: x ← x − Σ own out + Σ neighbours' flow toward me; temperature/silica mixed by mass; channel memory
//          grows where lava is present and decays slowly elsewhere.
//   Every unit sent is received by exactly one neighbour from the same stored integer → exact conservation.
// Cool (once per tick, fused into the last apply substep): Newtonian cooling toward tAmb, much faster for thin films (dark crusted margins and
//   fronts) than for thick fed cores; water quenches. Freezing ∝ (1 − fluidity): hot channel cores stay liquid,
//   margins freeze into levees. Thin films and lava below tSolid freeze at once. The lava lake of an erupting
//   vent (volcano phase BUILD/ACTIVE/WANING) neither cools nor freezes. Frozen mass tops up a partial crust top
//   voxel, then stacks BASALT / ANDESITE (ANDESITE carries FLAG_CONTINENTAL: arcs grow continents).
// Race-free: one thread per column, writes only its own column (vox in place in the current buffer).
import * as THREE from 'three/webgpu';
import { Fn, If, Return, float, int, uint, uvec2, uvec4, max, min, floor, exp, select, smoothstep, instanceIndex, uniform, atomicAdd } from 'three/tsl';
import { ReaderKernel, type GpuFields, type StorageNode } from '../core/gpu';
import { NCOL, Mat, FLAG_CONTINENTAL } from './layout';
import { tColIdx, tColXZ, uMin } from './tslLayout';
import { CTR_RESERVOIR } from './fields';
import { LAVA, MAGMA, MCTR, PHASE } from './magmaModel';
import { tLavaTemp, tLavaSil, tLavaMem, tLavaY, tLavaMix, tScanTop, tAddRock } from './magmaTsl';

type U = THREE.Node<'uint'>;

const lo16 = (v: U) => v.bitAnd(uint(0xffff));
const hi16 = (v: U) => v.shiftRight(uint(16));
/** 8 neighbours; opposite direction of k is k ^ 1. */
const N8: readonly [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];

export class LavaPass {
  readonly uniforms = {
    dt: uniform(0.05),
    // inject (god tool / tests): one column, units drawn from the reservoir
    injCol: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    injUnits: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    injTemp: uniform(LAVA.tBasalt),
    injSil: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
  };
  private outK: THREE.ComputeNode;
  private applyK: THREE.ComputeNode;
  /** last substep: apply + cooling/solidification in one dispatch (dispatch overhead dominates here) */
  private applyCoolK: ReaderKernel<'uint'>;
  private injectK: THREE.ComputeNode;

  constructor(private fields: GpuFields) {
    this.outK = this.buildOut();
    this.applyK = this.buildApply(null);
    this.applyCoolK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildApply(vox));
    this.injectK = this.buildInject();
  }

  /** §C pass 6, every tick. Needs surfY fresh up to this tick's magma edits (±1 layer at vents is tolerated). */
  step(renderer: THREE.WebGPURenderer, _tick: number, dtGeo: number): void {
    this.uniforms.dt.value = dtGeo;
    for (let s = 0; s < LAVA.substeps; s++) {
      renderer.compute(this.outK);
      if (s < LAVA.substeps - 1) renderer.compute(this.applyK); else this.applyCoolK.run(renderer);
    }
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
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const L = lava.element(i).toVar();
      const m = L.x.toVar();
      If(m.equal(uint(0)), () => { lavaOut.element(i).assign(uvec4(0, 0, 0, L.y)); Return(); });
      const { x, z } = tColXZ(i);
      const h = surfY.element(i).add(float(m).div(255)).toVar();
      const fluid = smoothstep(LAVA.tSolid, LAVA.tAndesite, tLavaTemp(L.y));
      const mob = fluid.mul(float(1).sub(float(tLavaSil(L.y)).mul(0.6 / 255))).toVar();
      const drops = N8.map(([dx, dz]) => {
        const Ln = lava.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).toVar();
        const hn = surfY.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).add(float(Ln.x).div(255));
        return { drop: h.sub(hn).toVar(), mem: float(tLavaMem(Ln.y)).div(255) };
      });
      // steepest descent (+ channel memory) among lower neighbours
      const best = uint(8).toVar();
      const bestScore = float(0).toVar();
      const bestDrop = float(0).toVar();
      drops.forEach(({ drop, mem }, k) => {
        const score = drop.div(k < 4 ? 1 : Math.SQRT2).add(mem.mul(LAVA.chanBias));
        If(drop.greaterThan(LAVA.yield).and(score.greaterThan(bestScore)), () => { best.assign(k); bestScore.assign(score); bestDrop.assign(drop); });
      });
      const main = uint(0).toVar();
      If(best.lessThan(uint(8)), () => {
        main.assign(uint(floor(min(float(m).mul(mob).mul(LAVA.main), bestDrop.mul(255 * 0.4)))));
      });
      const sp = drops.slice(0, 4).map(({ drop }) => select(drop.greaterThan(LAVA.yield),
        uint(floor(min(float(m).mul(mob).mul(LAVA.spread), drop.mul(255 * 0.1)))), uint(0)).toVar());
      const sum = main.add(sp[0]!).add(sp[1]!).add(sp[2]!).add(sp[3]!).toVar();
      If(sum.greaterThan(m), () => {
        const sc = float(m).div(float(sum));
        main.assign(uint(floor(float(main).mul(sc))));
        for (const q of sp) q.assign(uint(floor(float(q).mul(sc))));
        const ex = main.add(sp[0]!).add(sp[1]!).add(sp[2]!).add(sp[3]!).toVar();
        const over = select(ex.greaterThan(m), ex.sub(m), uint(0)).toVar();
        for (const q of [main, ...sp]) { const t = uMin(q, over); q.subAssign(t); over.subAssign(t); }
      });
      const mainBits = select(main.greaterThan(uint(0)), main.bitOr(best.shiftLeft(uint(16))).bitOr(uint(1 << 19)), uint(0));
      lavaOut.element(i).assign(uvec4(mainBits, sp[0]!.bitOr(sp[1]!.shiftLeft(uint(16))), sp[2]!.bitOr(sp[3]!.shiftLeft(uint(16))), L.y));
    })().compute(NCOL);
  }

  private buildApply(vox: StorageNode<'uint'> | null): THREE.ComputeNode {
    // storage buffers: lava, lavaOut (+ with cooling: vox, water, magmaCtr, volcano) ≤ 6
    const lava = this.fields.cur<'uvec2'>('lava');
    const lavaOut = this.fields.cur<'uvec4'>('lavaOut');
    const cool = vox ? this.coolFn(vox) : null;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const o = lavaOut.element(i).toVar();
      const L = lava.element(i).toVar();
      const out = lo16(o.x).add(lo16(o.y)).add(hi16(o.y)).add(lo16(o.z)).add(hi16(o.z));
      const m = L.x.sub(out).toVar();
      const tw = float(m).mul(tLavaTemp(L.y)).toVar();
      const sw = float(m).mul(float(tLavaSil(L.y))).toVar();
      const tot = m.toVar();
      const inflow = uint(0).toVar();
      const recv = (q0: U, y: U) => {
        const q = uint(q0).toVar();
        If(q.greaterThan(uint(0)), () => {
          const qf = float(q).toVar();
          tot.addAssign(q); inflow.addAssign(q);
          tw.addAssign(qf.mul(tLavaTemp(y)));
          sw.addAssign(qf.mul(float(tLavaSil(y))));
        });
      };
      N8.forEach(([dx, dz], k) => {
        const on = lavaOut.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).toVar();
        // main flow of the neighbour at offset k points at me when its direction is the opposite (k ^ 1)
        const mainTo = on.x.shiftRight(uint(16)).bitAnd(uint(7));
        const has = on.x.shiftRight(uint(19)).bitAnd(uint(1)).equal(uint(1));
        recv(select(has.and(mainTo.equal(uint(k ^ 1))), lo16(on.x), uint(0)), on.w);
        if (k < 4) {
          // axis spread: +x neighbour (k=0) sends on its −x slot, −x on +x, +z on −z, −z on +z
          const q = k === 0 ? hi16(on.y) : k === 1 ? lo16(on.y) : k === 2 ? hi16(on.z) : lo16(on.z);
          recv(q, on.w);
        }
      });
      const mem = tLavaMem(L.y).toVar();
      mem.assign(select(tot.greaterThan(uint(LAVA.thin)).or(inflow.greaterThan(uint(0))),
        uMin(mem.add(uint(LAVA.memGain)), uint(255)), mem.sub(mem.add(uint(127)).div(uint(128)))));
      const inv = float(1).div(max(float(tot), 1));
      const Lnew = select(tot.equal(uint(0)), uvec2(0, mem.shiftLeft(uint(24))),
        uvec2(tot, tLavaY(tw.mul(inv), uint(sw.mul(inv).add(0.5))).bitOr(mem.shiftLeft(uint(24))))).toVar();
      if (cool) cool(i, x, z, Lnew);
      lava.element(i).assign(Lnew);
    })().compute(NCOL);
  }

  /** Cooling + solidification of one column's lava (in place on L), run inside the last apply substep. */
  private coolFn(vox: StorageNode<'uint'>) {
    const water = this.fields.cur('water');
    const mctr = this.fields.cur<'int'>('magmaCtr');
    const volcano = this.fields.cur<'vec4'>('volcano');
    const { dt } = this.uniforms;
    return (i: U, x: THREE.Node<'int'>, z: THREE.Node<'int'>, L: THREE.Node<'uvec2'>) => {
      const m = L.x.toVar();
      // an erupting vent keeps its lake hot and liquid
      const ph = uint(volcano.element(i).y);
      const erupting = ph.equal(uint(PHASE.BUILD)).or(ph.equal(uint(PHASE.ACTIVE))).or(ph.equal(uint(PHASE.WANING)));
      If(m.greaterThan(uint(0)).and(erupting.not()), () => {
        const sil = tLavaSil(L.y).toVar();
        const memBits = L.y.bitAnd(uint(0xff000000)).toVar();
        const wet = water.element(i).greaterThan(LAVA.wetDepth);
        const k = select(wet, float(LAVA.kWater), float(LAVA.kAir).div(float(1).add(float(m).mul(LAVA.thickK / 255))));
        const T1 = float(LAVA.tAmb).add(tLavaTemp(L.y).sub(LAVA.tAmb).mul(exp(k.mul(dt).negate()))).toVar();
        const fluid = smoothstep(LAVA.tSolid, LAVA.tAndesite, T1);
        const all = T1.lessThan(LAVA.tSolid).or(m.lessThanEqual(uint(LAVA.thin)));
        const frz = select(all, m, uMin(uint(floor(float(m).mul(LAVA.freezeRate).mul(dt).mul(float(1).sub(fluid)))), m)).toVar();
        If(frz.greaterThan(uint(0)), () => {
          const { topY, topV } = tScanTop(vox, x, z);
          const felsic = sil.greaterThanEqual(uint(MAGMA.silAndesite));
          const placed = tAddRock(vox, x, z, topY, topV, frz, select(felsic, uint(Mat.ANDESITE), uint(Mat.BASALT)),
            select(felsic, uint(FLAG_CONTINENTAL), uint(0)), LAVA.maxNewVoxels).toVar();
          m.subAssign(placed);
          atomicAdd(mctr.element(MCTR.SOLIDIFIED), int(placed));
        });
        L.assign(select(m.equal(uint(0)), uvec2(0, memBits), uvec2(m, tLavaY(T1, sil).bitOr(memBits))));
      });
    };
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
