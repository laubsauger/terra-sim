// §T.22 melt generation, ascent, chambers + eruption trigger (§C pass 5). CPU calibration in magmaModel.ts.
//
// Sources (fill units per column, × 'volcanism'):
//   hotspot  heatFlow above qHot (mantle plume conduits, fixed in the grid frame; plates drift over them).
//   arc      columns arcNear..arcFar cells inland of an active trench ON THE OVERRIDING PLATE. The arc field
//            (vec4: x = distance to the nearest trench of the same plate, y = that trench's event rate EMA,
//            z = subduction events at this column (EMA, τ = tauTrench), w = recency of the last event here)
//            rides with the plate (gathered in afterTectonics). Trench marks come from tecAct ACT_SUBDUCT (set on
//            the winner = overriding column); a column is a trench while w > trenchRecent (event within ~1 My), and
//            distances relax only between columns of the same plate, so the subducting plate never sees arc heat.
//            Output is normalized: Σ arc melt = arcFrac · slabMass · subduction events (EMA over normEma periods),
//            spread over the columns in proportion to weight = min(y, cap) · window(x). Ragged, areal convergence
//            zones therefore cannot starve (or flood) the arcs; the reservoir inflow ↔ outflow balance holds.
//   ridge    minor fissure lava on fresh crust (crustAge < ridgeAge), straight from the reservoir to the lava field.
// Ascent + chambers: melt drawn from the reservoir first collects in magCol.y (pending, ascending batch);
//   each full 255 units is emplaced as one MAGMA voxel; smaller batches (≥ segMin) segregate after a random
//   wait (mean segTau) as a partial-fill MAGMA voxel. A new chamber is inserted at the neutral-buoyancy
//   depth (chamberDepth(thickness) below the top, always above the lowest crust voxel so tectonics never drops
//   it), later voxels on top of the chamber. Insertion shifts the column above up by one (inflation).
// Eruption: n chamber voxels under a lid of L layers are overpressured when n ≥ 1 + L / lidPerVoxel; then
//   erupts with probability eruptRate·volcanism·dt (always when the chamber is full). The top chamber voxel
//   moves to the lava field and the column above drops one layer (caldera subsidence).
// Crystallization: chamber voxels freeze one at a time (mean residence freezeTau) into a pluton
//   (GRANITE + FLAG_CONTINENTAL if felsic, else GABBRO).
// Silica code (MAGMA voxel age byte, lava.y bits 16-23): arc share of the melt × 255, blended halfway to 255 in
//   continental columns. ≥ silAndesite → ANDESITE lava / GRANITE plutons carrying FLAG_CONTINENTAL, so arcs
//   grow continental crust from the reservoir; hotspot and ridge basalt stays oceanic.
//
// vox is edited in place (current buffer); one thread per column writes only its own column.
// Budget (V3): every transfer is an integer move between reservoir / magCol.y / MAGMA voxels / lava / crust.
// Tectonics destroys the chambers of subducted columns; afterTectonics gathers magCol along tecAct, and the
// Σ(before) − Σ(after) difference goes back to the reservoir (integer atomics, order independent, V2).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Return, float, int, uint, uvec2, vec4, max, min, clamp, round, select, smoothstep, instanceIndex,
  uniform, atomicAdd, atomicLoad, atomicStore,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { ReaderKernel } from '../core/gpu';
import type { Params } from '../core/params';
import { NCOL, NY, Mat, FLAG_CONTINENTAL, FLAG_HOTSPOT } from './layout';
import { tMat, tFill, tAge, tFlags, tPack, tVoxIdx, tColIdx, tColXZ, iMax, iMin, uMin } from './tslLayout';
import { CTR_RESERVOIR } from './fields';
import { ACT_NEW, ACT_SUBDUCT } from './tectonics';
import { MX, M_SCALE } from './mantleModel';
import { MAGMA, LAVA, MCTR, ARC_W_SCALE, ARC_E_SCALE, ARC_K_SCALE, parseMagmaStats, type MagmaStats } from './magmaModel';
import { isCrust, tHash01, tQuant, tLavaMix } from './magmaTsl';

type U = THREE.Node<'uint'>;
type F = THREE.Node<'float'>;

/** Distance value meaning "no trench in reach". */
export const ARC_FAR = 99;

/** Magma + lava fields. Call before fields.freeze(); needs registerMantleFields too (slabIn, heatFlow). */
export function registerMagmaFields(f: GpuFields): void {
  f.add('magCol', 'uvec2', NCOL);    // x = Σ MAGMA fill in the column (chamber), y = pending melt; fill units
  f.add('magColTmp', 'uvec2', NCOL); // scratch for the tectonics gather
  f.add('arc', 'vec4', NCOL);        // (trench dist cells, trench event rate, events EMA here, event recency here)
  f.add('arcTmp', 'vec4', NCOL);     // scratch (relax ping-pong, tectonics gather)
  f.add('lava', 'uvec2', NCOL);      // x = mass fill units (/255 = layers), y = temp °C·16 | silica << 16
  f.add('lavaOut', 'uvec4', NCOL);   // scratch: flow out per substep (+x | −x<<16, +z | −z<<16, sender lava.y, 0)
  f.add('magmaCtr', 'int', MCTR.SIZE, { atomic: true });
}

/** T of fresh lava from its silica code. */
export const tEruptTemp = (sil: U) => float(LAVA.tBasalt).add(float(LAVA.tAndesite - LAVA.tBasalt).mul(float(sil).div(255)));

export class MagmaPass {
  readonly uniforms = {
    dt: uniform(0.05),              // My per step
    tick: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    volcanism: uniform(1),
    decayTrench: uniform(1),        // per relax iteration
    decayRecent: uniform(1),
  };
  private snapK: THREE.ComputeNode;
  private relaxK: [[THREE.ComputeNode, THREE.ComputeNode], [THREE.ComputeNode, THREE.ComputeNode]]; // [dir][pidParity]
  private meltK: [[THREE.ComputeNode, THREE.ComputeNode], [THREE.ComputeNode, THREE.ComputeNode]]; // [voxPar][agePar]
  private initK: ReaderKernel<'uint'>;
  private gatherK: THREE.ComputeNode;
  private copyK: THREE.ComputeNode;
  private fixK: THREE.ComputeNode;
  private normK: THREE.ComputeNode;
  private clearK: THREE.ComputeNode;

  constructor(private fields: GpuFields, params: Params) {
    const u = this.uniforms;
    u.volcanism.value = params.get('volcanism') as number;
    params.onChange((k, v) => { if (k === 'volcanism') u.volcanism.value = v as number; });
    const ctr = fields.cur<'int'>('counters');
    const mctr = fields.cur<'int'>('magmaCtr');
    const arc = fields.cur<'vec4'>('arc');
    const arcTmp = fields.cur<'vec4'>('arcTmp');
    const magCol = fields.cur<'uvec2'>('magCol');
    const magColTmp = fields.cur<'uvec2'>('magColTmp');

    this.snapK = Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      atomicStore(mctr.element(MCTR.RES_SNAP), atomicLoad(ctr.element(CTR_RESERVOIR)));
      // saturate the arc-weight accumulator if normK has not run for a long time (tectonics paused): no int wrap
      If(atomicLoad(mctr.element(MCTR.ARCW)).greaterThan(int(1 << 30)), () => { atomicStore(mctr.element(MCTR.ARCW), int(1 << 30)); });
    })().compute(1);

    const [p0, p1] = fields.pair<'uint'>('plateId');
    this.relaxK = [[this.buildRelax(arc, arcTmp, p0), this.buildRelax(arc, arcTmp, p1)],
      [this.buildRelax(arcTmp, arc, p0), this.buildRelax(arcTmp, arc, p1)]];
    const [v0, v1] = fields.pair<'uint'>('vox');
    const [a0, a1] = fields.pair('crustAge');
    this.meltK = [[this.buildMelt(v0, a0), this.buildMelt(v0, a1)], [this.buildMelt(v1, a0), this.buildMelt(v1, a1)]];
    this.initK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildInit(vox));
    this.gatherK = this.buildGather();

    this.copyK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      magCol.element(i).assign(magColTmp.element(i));
      arc.element(i).assign(arcTmp.element(i));
    })().compute(NCOL);

    this.fixK = Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      const lost = atomicLoad(mctr.element(MCTR.GATHER_OLD)).sub(atomicLoad(mctr.element(MCTR.GATHER_NEW))).toVar();
      atomicAdd(ctr.element(CTR_RESERVOIR), lost);
      atomicAdd(mctr.element(MCTR.RETURNED), lost);
      atomicStore(mctr.element(MCTR.GATHER_OLD), int(0));
      atomicStore(mctr.element(MCTR.GATHER_NEW), int(0));
    })().compute(1);

    // arc normalization, once per tectonics run (mirrors magmaModel.arcNormK / emaStep)
    this.normK = Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      const e = atomicLoad(mctr.element(MCTR.EVENTS)).mul(int(ARC_E_SCALE));
      const w = atomicLoad(mctr.element(MCTR.ARCW));
      const eE = atomicLoad(mctr.element(MCTR.E_EMA)).toVar();
      const wE = atomicLoad(mctr.element(MCTR.W_EMA)).toVar();
      eE.addAssign(e.sub(eE).div(int(MAGMA.normEma)));
      wE.addAssign(w.sub(wE).div(int(MAGMA.normEma)));
      const k = select(wE.greaterThan(int(0)),
        float(eE).mul((MAGMA.arcFrac * MAGMA.slabMass * ARC_W_SCALE * ARC_K_SCALE) / ARC_E_SCALE).div(float(iMax(wE, int(1)))),
        float(0));
      atomicStore(mctr.element(MCTR.E_EMA), eE);
      atomicStore(mctr.element(MCTR.W_EMA), wE);
      atomicStore(mctr.element(MCTR.ARCK), int(min(k, 2e9)));
      atomicStore(mctr.element(MCTR.EVENTS), int(0));
      atomicStore(mctr.element(MCTR.ARCW), int(0));
    })().compute(1);

    this.clearK = Fn(() => {
      const i = instanceIndex.add(uint(MCTR.ERUPTIONS));
      If(i.greaterThan(uint(MCTR.SOLIDIFIED)), () => { Return(); });
      atomicStore(mctr.element(i), int(0));
    })().compute(MCTR.SOLIDIFIED - MCTR.ERUPTIONS + 1);
  }

  /** Build magCol from the voxels (worldgen chambers). MAGMA below a column's lowest crust voxel becomes mantle. */
  init(renderer: THREE.WebGPURenderer): void { this.initK.run(renderer); }

  /** Call right after a tectonics GPU run (tecAct is only fresh then): carry chambers + arc state with the plates. */
  afterTectonics(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.gatherK);
    renderer.compute(this.copyK);
    renderer.compute(this.fixK);
    renderer.compute(this.normK);
  }

  /** §C pass 5, every tick. */
  step(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number): void {
    const u = this.uniforms;
    const f = this.fields;
    u.dt.value = dtGeo;
    u.tick.value = tick >>> 0;
    u.decayTrench.value = Math.exp(-dtGeo / 2 / MAGMA.tauTrench);
    u.decayRecent.value = Math.exp(-dtGeo / 2 / MAGMA.tauRecent);
    renderer.compute(this.snapK);
    const pp = f.parity('plateId') as 0 | 1;
    renderer.compute(this.relaxK[0][pp]);
    renderer.compute(this.relaxK[1][pp]);
    renderer.compute(this.meltK[f.parity('vox') as 0 | 1][f.parity('crustAge') as 0 | 1]);
  }

  /** Zero cumulative stats (eruptions, drawn, returned, frozen, solidified); call right after a readback. */
  clearStats(renderer: THREE.WebGPURenderer): void { renderer.compute(this.clearK); }
  async readStats(renderer: THREE.WebGPURenderer): Promise<MagmaStats> { return parseMagmaStats(await this.fields.read(renderer, 'magmaCtr')); }

  // ---- kernels ----

  private buildRelax(src: StorageNode<'vec4'>, dst: StorageNode<'vec4'>, pid: StorageNode<'uint'>): THREE.ComputeNode {
    // storage buffers: src, dst, plateId = 3. Bellman-Ford step of "distance to the nearest active trench of my plate".
    const { decayTrench, decayRecent } = this.uniforms;
    const EPS = 0.05;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const s = src.element(i).toVar();
      const p = pid.element(i).toVar();
      const ev = s.z.mul(decayTrench).toVar();
      const rec = s.w.mul(decayRecent).toVar();
      const dist = float(ARC_FAR).toVar();
      const act = float(0).toVar();
      If(rec.greaterThan(MAGMA.trenchRecent), () => { dist.assign(0); act.assign(ev); }).Else(() => {
        for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const n = tColIdx(x.add(int(dx)), z.add(int(dz))).toVar();
          If(pid.element(n).equal(p), () => {
            const an = src.element(n).toVar();
            If(an.y.greaterThan(EPS).and(an.x.add(1).lessThan(dist)), () => { dist.assign(an.x.add(1)); act.assign(an.y); });
          });
        }
        If(dist.greaterThan(MAGMA.arcMaxDist), () => { dist.assign(ARC_FAR); act.assign(0); });
      });
      dst.element(i).assign(vec4(dist, act, ev, rec));
    })().compute(NCOL);
  }

  private buildGather(): THREE.ComputeNode {
    // storage buffers: tecAct, magCol, magColTmp, arc, arcTmp, magmaCtr, slabIn = 7 (V23)
    const f = this.fields;
    const act = f.cur<'uint'>('tecAct');
    const magCol = f.cur<'uvec2'>('magCol');
    const magColTmp = f.cur<'uvec2'>('magColTmp');
    const arc = f.cur<'vec4'>('arc');
    const arcTmp = f.cur<'vec4'>('arcTmp');
    const mctr = f.cur<'int'>('magmaCtr');
    const slabIn = f.cur<'int'>('slabIn');
    const LOG_M = Math.log2(M_SCALE);
    return Fn(() => {
      const d = instanceIndex;
      If(d.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const a = act.element(d).toVar();
      const src = a.bitAnd(uint(0xffff));
      const isNew = a.bitAnd(uint(ACT_NEW)).notEqual(uint(0)).toVar();
      const old = magCol.element(d).toVar();
      atomicAdd(mctr.element(MCTR.GATHER_OLD), int(old.x.add(old.y)));
      const nm = select(isNew, uvec2(0, 0), magCol.element(src)).toVar();
      magColTmp.element(d).assign(nm);
      atomicAdd(mctr.element(MCTR.GATHER_NEW), int(nm.x.add(nm.y)));
      const s = select(isNew, vec4(ARC_FAR, 0, 0, 0), arc.element(src)).toVar();
      If(a.bitAnd(uint(ACT_SUBDUCT)).notEqual(uint(0)), () => {
        s.z.addAssign(1);
        s.w.assign(1);
        atomicAdd(mctr.element(MCTR.EVENTS), int(1));
        const { x, z } = tColXZ(d);
        atomicAdd(slabIn.element(uint(x.shiftRight(int(LOG_M)).add(z.shiftRight(int(LOG_M)).mul(int(MX))))), int(1));
      });
      arcTmp.element(d).assign(s);
    })().compute(NCOL);
  }

  private buildInit(vox: StorageNode<'uint'>): THREE.ComputeNode {
    // storage buffers: vox, magCol = 2
    const magCol = this.fields.cur<'uvec2'>('magCol');
    const PERI = tPack(uint(Mat.PERIDOTITE), uint(255), uint(0), uint(0));
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const base = int(NY).toVar();
      Loop(NY, ({ i: y }) => {
        If(isCrust(tMat(vox.element(tVoxIdx(x, y, z)))), () => { base.assign(iMin(base, y)); });
      });
      const ch = uint(0).toVar();
      Loop(NY, ({ i: y }) => {
        const idx = tVoxIdx(x, y, z).toVar();
        const v = vox.element(idx).toVar();
        If(tMat(v).equal(uint(Mat.MAGMA)), () => {
          If(y.lessThan(base), () => { vox.element(idx).assign(PERI); }).Else(() => { ch.addAssign(tFill(v)); });
        });
      });
      magCol.element(i).assign(uvec2(ch, 0));
    })().compute(NCOL);
  }

  private buildMelt(vox: StorageNode<'uint'>, crustAge: StorageNode): THREE.ComputeNode {
    // storage buffers: vox, magCol, arc, heatFlow, crustAge, counters, magmaCtr, lava = 8 (V23)
    const f = this.fields;
    const magCol = f.cur<'uvec2'>('magCol');
    const arc = f.cur<'vec4'>('arc');
    const heatFlow = f.cur('heatFlow');
    const ctr = f.cur<'int'>('counters');
    const mctr = f.cur<'int'>('magmaCtr');
    const lava = f.cur<'uvec2'>('lava');
    const { dt, tick, volcanism: volc } = this.uniforms;
    const M = MAGMA;

    const draw = (n: U) => {
      atomicAdd(ctr.element(CTR_RESERVOIR), int(0).sub(int(n)));
      atomicAdd(mctr.element(MCTR.DRAWN), int(n));
    };
    const lavaAdd = (i: U, m: U, t: F, sil: U) => {
      const L = lava.element(i).toVar();
      lava.element(i).assign(tLavaMix(L.x, L.y, m, t, sil));
    };

    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const mc = magCol.element(i).toVar();
      const ch = mc.x.toVar();
      const acc = mc.y.toVar();
      const a = arc.element(i).toVar();

      const hotU = float(M.hotGain).mul(max(heatFlow.element(i).sub(M.qHot).div(M.qHotSpan), 0)).mul(volc).mul(dt).toVar();
      // arc: weight → normalized share of the global arc budget (K from normK, previous periods)
      const win = smoothstep(M.arcNear - 1, M.arcNear + 0.5, a.x).mul(float(1).sub(smoothstep(M.arcFar - 0.5, M.arcFar + 1, a.x)));
      const wgt = min(a.y, M.arcActCap).mul(win).toVar();
      const arcU = float(0).toVar();
      If(wgt.greaterThan(0), () => {
        atomicAdd(mctr.element(MCTR.ARCW), int(round(wgt.mul(ARC_W_SCALE))));
        arcU.assign(wgt.mul(float(atomicLoad(mctr.element(MCTR.ARCK)))).mul(volc.div(ARC_K_SCALE)));
      });
      const melt = hotU.add(arcU).toVar();

      // melt generation: reservoir → pending (gated on a deterministic reservoir snapshot)
      If(atomicLoad(mctr.element(MCTR.RES_SNAP)).greaterThan(int(M.reservoirMin)), () => {
        const n = tQuant(melt, tHash01(i, tick, 11), uint(M.meltAccMax).sub(acc)).toVar();
        If(n.greaterThan(uint(0)), () => { draw(n); acc.addAssign(n); });
        const fresh = max(float(1).sub(crustAge.element(i).div(M.ridgeAge)), 0);
        const rl = tQuant(float(M.ridgeGain).mul(fresh).mul(volc).mul(dt), tHash01(i, tick, 12), uint(1024)).toVar();
        If(rl.greaterThan(uint(0)), () => { draw(rl); lavaAdd(i, rl, float(LAVA.tBasalt), uint(0)); });
      });

      // segregation: a full voxel's worth always, a partial batch now and then (thin arcs would strand melt)
      const seg = acc.greaterThanEqual(uint(255)).or(acc.greaterThanEqual(uint(M.segMin)).and(tHash01(i, tick, 15).lessThan(dt.div(M.segTau)))).toVar();
      If(seg.or(ch.greaterThan(uint(0))), () => {
        // column scan: top solid, lowest crust, chamber extent, continental crust present
        const top = int(-1).toVar();
        const base = int(NY).toVar();
        const n = int(0).toVar();
        const chTop = int(-1).toVar();
        const chBot = int(-1).toVar();
        const cont = uint(0).toVar();
        Loop(NY, ({ i: y }) => {
          const v = vox.element(tVoxIdx(x, y, z)).toVar();
          const m = tMat(v).toVar();
          If(m.notEqual(uint(Mat.AIR)), () => { top.assign(y); });
          If(isCrust(m), () => { base.assign(iMin(base, y)); cont.assign(cont.bitOr(tFlags(v).bitAnd(uint(FLAG_CONTINENTAL)))); });
          If(m.equal(uint(Mat.MAGMA)), () => {
            n.addAssign(1); chTop.assign(y);
            If(chBot.lessThan(int(0)), () => { chBot.assign(y); });
          });
        });

        // chamber growth: one MAGMA voxel (≤ 255 units) per step, inserted (column above shifts up)
        If(seg.and(n.lessThan(int(M.chamberMax))).and(top.greaterThanEqual(int(0))).and(top.lessThanEqual(int(NY - 3))), () => {
          const depth = iMin(iMax(top.sub(base).div(int(3)), int(M.depthMin)), int(M.depthMax));
          const ins = select(n.greaterThan(int(0)), chTop.add(int(1)), top.sub(depth)).toVar();
          If(ins.greaterThan(base).and(ins.lessThanEqual(top.add(int(1)))), () => {
            const silArc = uint(clamp(arcU.div(max(melt, 1e-6)).mul(255), 0, 255));
            const sil = select(cont.notEqual(uint(0)), silArc.add(uint(255)).shiftRight(uint(1)), silArc);
            const hotFlag = select(hotU.greaterThan(arcU), uint(FLAG_HOTSPOT), uint(0));
            Loop({ start: top, end: ins, condition: '>=' }, ({ i: y }) => {
              vox.element(tVoxIdx(x, y.add(int(1)), z)).assign(vox.element(tVoxIdx(x, y, z)));
            });
            const put = uMin(acc, uint(255)).toVar();
            vox.element(tVoxIdx(x, ins, z)).assign(tPack(uint(Mat.MAGMA), put, sil, hotFlag));
            acc.subAssign(put);
            ch.addAssign(put);
            If(n.equal(int(0)), () => { chBot.assign(ins); });
            n.addAssign(1);
            top.assign(iMax(top.add(int(1)), ins));
            chTop.assign(ins);
          });
        });

        If(n.greaterThan(int(0)), () => {
          const lid = top.sub(chTop);
          const need = lid.div(int(M.lidPerVoxel)).add(int(1));
          const pErupt = float(M.eruptRate).mul(volc).mul(dt);
          const full = n.greaterThanEqual(int(M.chamberMax));
          If(full.or(n.greaterThanEqual(need).and(tHash01(i, tick, 13).lessThan(pErupt))), () => {
            // eruption: top chamber voxel → lava; column above drops one layer
            const v = vox.element(tVoxIdx(x, chTop, z)).toVar();
            const fm = tFill(v).toVar();
            const s = tAge(v).toVar();
            Loop({ start: chTop, end: top, condition: '<' }, ({ i: y }) => {
              vox.element(tVoxIdx(x, y, z)).assign(vox.element(tVoxIdx(x, y.add(int(1)), z)));
            });
            vox.element(tVoxIdx(x, top, z)).assign(uint(0));
            ch.subAssign(fm);
            lavaAdd(i, fm, tEruptTemp(s), s);
            atomicAdd(mctr.element(MCTR.ERUPTIONS), int(1));
          }).ElseIf(tHash01(i, tick, 14).lessThan(float(n).mul(dt).div(M.freezeTau)), () => {
            // chamber crystallizes from the bottom: magma → pluton (crust); much arc magma ends up intrusive
            const idx = tVoxIdx(x, chBot, z).toVar();
            const v = vox.element(idx).toVar();
            const felsic = tAge(v).greaterThanEqual(uint(M.silAndesite));
            vox.element(idx).assign(tPack(select(felsic, uint(Mat.GRANITE), uint(Mat.GABBRO)), tFill(v), uint(0),
              select(felsic, uint(FLAG_CONTINENTAL), uint(0))));
            ch.subAssign(tFill(v));
            atomicAdd(mctr.element(MCTR.FROZEN), int(tFill(v)));
          });
        });
      });
      magCol.element(i).assign(uvec2(ch, acc));
    })().compute(NCOL);
  }
}
