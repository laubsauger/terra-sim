// §T.22 melt generation, ascent, chambers, eruption episodes (§C pass 5). CPU calibration in magmaModel.ts.
//
// Sources (fill units per column, × 'volcanism'):
//   hotspot  heatFlow above qHot (mantle plume conduits, fixed in the grid frame; plates drift over them).
//   arc      columns arcNear..arcFar cells inland of an active trench ON THE OVERRIDING PLATE. The arc field
//            (vec4: x = distance to the nearest trench of the same plate, y = that trench's event rate EMA,
//            z = subduction events at this column (EMA, τ = tauTrench), w = recency of the last event here)
//            rides with the plate (gathered in afterTectonics). A column is a trench while w > trenchRecent;
//            distances relax only between columns of the same plate, so the subducting plate never sees arc heat.
//            Output is normalized: Σ arc melt = arcFrac · slabMass · subduction events (EMA over normEma periods),
//            spread in proportion to weight = min(y, cap) · window(x), independent of trench geometry.
//   ridge    minor fissure lava on fresh crust (crustAge < ridgeAge), straight from the reservoir to the lava field.
// Ascent + focusing: melt drawn from the reservoir collects in magCol.y (pending batch, with its silica code);
//   pending migrates one cell per tick toward the same-plate neighbour with the larger chamber (melt focusing),
//   so magma gathers under discrete volcanoes. A full 255 units (or, after a random wait, ≥ segMin) is emplaced
//   as one MAGMA voxel at the neutral-buoyancy depth, always above the lowest crust voxel; the column above
//   inflates by one layer.
// Eruption episodes (state in 'volcano', see VOLC/PHASE): only a vent — a local maximum of chamber mass whose
//   neighbours are not erupting — starts an episode when overpressured (n ≥ 1 + lid / lidPerVoxel):
//   BUILD (glow ramps) → ACTIVE (explosive opening carves the summit crater; steady effusion from the highest
//   charged chamber voxel into the vent's lava lake, draining the chamber over the phase; drained voxels stay as
//   mass-free MAGMA cavities, fill 0, so the lake overflows the rim instead of sinking) → WANING → POST (lake
//   freezes, cavities collapse = caldera subsidence, crater carved) → IDLE (repose). New melt refills cavities
//   before inserting voxels. Crater carving credits the reservoir; the next tick's focus kernel deposits spatter on
//   the 8 neighbours drawn from the reservoir (exact via the reservoir, V3).
// Crystallization: chamber voxels of non-erupting columns freeze one at a time (mean residence freezeTau) into
//   plutons (GRANITE + FLAG_CONTINENTAL if felsic, else GABBRO).
// Silica code (MAGMA voxel age byte, lava.y bits 16-23): arc share × 255, blended halfway to 255 in continental
//   columns. ≥ silAndesite → ANDESITE lava / GRANITE plutons with FLAG_CONTINENTAL (arcs grow continents);
//   hotspot and ridge basalt stays oceanic.
//
// vox is edited in place (current buffer); one thread per column writes only its own column. Neighbour reads
// only touch buffers the kernel does not write (magFocus, volcano.w, surfY), so every kernel is race-free.
// Tectonics destroys the chambers / lava of subducted columns; afterTectonics gathers magCol, volcano and lava
// along tecAct, and Σ(before) − Σ(after) goes back to the reservoir (integer atomics, order independent, V2).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Return, float, int, uint, uvec2, uvec4, vec4, max, min, clamp, floor, round, select, smoothstep,
  instanceIndex, uniform, atomicAdd, atomicLoad, atomicStore,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { ReaderKernel } from '../core/gpu';
import type { Params } from '../core/params';
import { NCOL, NY, Mat, FLAG_CONTINENTAL, FLAG_HOTSPOT } from './layout';
import { tMat, tFill, tAge, tFlags, tPack, tVoxIdx, tColIdx, tColXZ, iMax, iMin, uMin } from './tslLayout';
import { CTR_RESERVOIR } from './fields';
import { ACT_NEW, ACT_SUBDUCT } from './tectonics';
import { MX, M_SCALE } from './mantleModel';
import { MAGMA, LAVA, VOLC, PHASE, MCTR, ARC_W_SCALE, ARC_E_SCALE, ARC_K_SCALE, parseMagmaStats, type MagmaStats } from './magmaModel';
import {
  isCrust, tHash01, tQuant, tLavaMix, tVwCarve, tVwSil, tVwHot, tVwPack, tAddRock, tRemoveTop, tScanTop,
} from './magmaTsl';

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
  f.add('lava', 'uvec2', NCOL);      // x = mass fill units (/255 = layers), y = temp °C·16 | silica<<16 | channel memory<<24
  f.add('lavaOut', 'uvec4', NCOL);   // scratch: flow out per substep (see lava.ts); tectonics gather scratch for lava
  // volcano: x = activity 0..1 (glow / plume), y = phase (PHASE), z = phase timer My, w = packed ints (magmaTsl
  // tVw*): crater carve this tick (units·2 + felsic, rim spatter input) | pending batch silica | hotspot flag
  f.add('volcano', 'vec4', NCOL);
  f.add('volcanoTmp', 'vec4', NCOL);
  f.add('magFocus', 'uint', NCOL);   // scratch: pending send dir(3) | vent-eligible(1) | silica(8)<<4 | units<<12
  f.add('magmaCtr', 'int', MCTR.SIZE, { atomic: true });
}

/** T of fresh lava from its silica code. */
export const tEruptTemp = (sil: U) => float(LAVA.tBasalt).add(float(LAVA.tAndesite - LAVA.tBasalt).mul(float(sil).div(255)));

const AX: readonly [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const N8: readonly [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];
/** Pending may pile up at a vent to this before neighbours stop sending (keeps the 16-bit field safe). */
const PEND_FOCUS_MAX = MAGMA.meltAccMax * 8;

export class MagmaPass {
  readonly uniforms = {
    dt: uniform(0.05),              // My per step
    tick: uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>,
    volcanism: uniform(1),
    decayTrench: uniform(1),        // per relax iteration
    decayRecent: uniform(1),
    // sea level (voxel-y), set by the orchestrator with the climate's (saved there): submarine vents carve no crater
    seaLevel: uniform(76) as THREE.UniformNode<'float', number>,
  };
  private relaxK: [[THREE.ComputeNode, THREE.ComputeNode], [THREE.ComputeNode, THREE.ComputeNode]]; // [dir][pidParity]
  private focusK: [[THREE.ComputeNode, THREE.ComputeNode], [THREE.ComputeNode, THREE.ComputeNode]]; // [voxPar][pidPar]
  private meltK: THREE.ComputeNode;
  private chamberK: ReaderKernel<'uint'>;
  private initK: ReaderKernel<'uint'>;
  private gatherK: THREE.ComputeNode;
  private gather2K: THREE.ComputeNode;
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
    const volcano = fields.cur<'vec4'>('volcano');
    const volcanoTmp = fields.cur<'vec4'>('volcanoTmp');
    const lava = fields.cur<'uvec2'>('lava');
    const lavaOut = fields.cur<'uvec4'>('lavaOut');

    const [p0, p1] = fields.pair<'uint'>('plateId');
    // crustAge parity always equals plateId parity (tectonics swaps both together)
    const [a0, a1] = fields.pair('crustAge');
    this.relaxK = [[this.buildRelax(arc, arcTmp, p0, true, null), this.buildRelax(arc, arcTmp, p1, true, null)],
      [this.buildRelax(arcTmp, arc, p0, false, a0), this.buildRelax(arcTmp, arc, p1, false, a1)]];
    const [v0, v1] = fields.pair<'uint'>('vox');
    this.focusK = [[this.buildFocus(p0, v0), this.buildFocus(p1, v0)], [this.buildFocus(p0, v1), this.buildFocus(p1, v1)]];
    this.meltK = this.buildMelt();
    this.chamberK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildChamber(vox));
    this.initK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildInit(vox));
    this.gatherK = this.buildGather();
    this.gather2K = this.buildGather2();

    this.copyK = Fn(() => {
      // storage buffers: magCol, magColTmp, arc, arcTmp, volcano, volcanoTmp, lava, lavaOut = 8
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      magCol.element(i).assign(magColTmp.element(i));
      arc.element(i).assign(arcTmp.element(i));
      volcano.element(i).assign(volcanoTmp.element(i));
      lava.element(i).assign(lavaOut.element(i).xy);
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

  /** Call right after a tectonics GPU run (tecAct is only fresh then): carry chambers, vents, lava with the plates. */
  afterTectonics(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.gatherK);
    renderer.compute(this.gather2K);
    renderer.compute(this.copyK);
    renderer.compute(this.fixK);
    renderer.compute(this.normK);
  }

  /**
   * §C pass 5, every tick. Dispatch count matters more than GPU time here (≈ 0.05 ms CPU per dispatch), so the
   * arc relax pair runs every other tick (carrying the reservoir snapshot) and rim spatter rides in the focus
   * kernel of the following tick.
   */
  step(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number): void {
    const u = this.uniforms;
    const f = this.fields;
    u.dt.value = dtGeo;
    u.tick.value = tick >>> 0;
    const pp = f.parity('plateId') as 0 | 1;
    if (tick % 2 === 0) {
      u.decayTrench.value = Math.exp(-dtGeo / MAGMA.tauTrench); // two ticks per pair of iterations
      u.decayRecent.value = Math.exp(-dtGeo / MAGMA.tauRecent);
      renderer.compute(this.relaxK[0][pp]);
      renderer.compute(this.relaxK[1][pp]);
    }
    renderer.compute(this.focusK[f.parity('vox') as 0 | 1][pp]);
    if (f.parity('crustAge') !== pp) throw new Error('MagmaPass: plateId/crustAge parity diverged');
    renderer.compute(this.meltK);
    this.chamberK.run(renderer);
  }

  /** Zero cumulative stats (eruptions, drawn, returned, frozen, solidified); call right after a readback. */
  clearStats(renderer: THREE.WebGPURenderer): void { renderer.compute(this.clearK); }
  async readStats(renderer: THREE.WebGPURenderer): Promise<MagmaStats> { return parseMagmaStats(await this.fields.read(renderer, 'magmaCtr')); }

  // ---- kernels ----

  private buildRelax(src: StorageNode<'vec4'>, dst: StorageNode<'vec4'>, pid: StorageNode<'uint'>, snap: boolean,
    crustAge: StorageNode | null): THREE.ComputeNode {
    // storage buffers: src, dst, plateId, counters, magmaCtr (+ crustAge, lava) ≤ 7. Bellman-Ford step of "distance
    // to the nearest active trench of my plate". The first iteration's thread 0 also snapshots the reservoir for
    // the melt gate (nothing in that dispatch touches the reservoir → deterministic, V2); the second iteration
    // carries the minor ridge fissure lava (fresh crust, crustAge < ridgeAge), 2 ticks' worth.
    const { decayTrench, decayRecent, dt, tick, volcanism: volc } = this.uniforms;
    const lava = this.fields.cur<'uvec2'>('lava');
    const ctr = this.fields.cur<'int'>('counters');
    const mctr = this.fields.cur<'int'>('magmaCtr');
    const EPS = 0.05;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      if (snap) {
        If(i.equal(uint(0)), () => {
          atomicStore(mctr.element(MCTR.RES_SNAP), atomicLoad(ctr.element(CTR_RESERVOIR)));
          // saturate the arc-weight accumulator if normK has not run for a long time (tectonics paused): no int wrap
          If(atomicLoad(mctr.element(MCTR.ARCW)).greaterThan(int(1 << 30)), () => { atomicStore(mctr.element(MCTR.ARCW), int(1 << 30)); });
        });
      }
      const { x, z } = tColXZ(i);
      const s = src.element(i).toVar();
      const p = pid.element(i).toVar();
      const ev = s.z.mul(decayTrench).toVar();
      const rec = s.w.mul(decayRecent).toVar();
      const dist = float(ARC_FAR).toVar();
      const act = float(0).toVar();
      If(rec.greaterThan(MAGMA.trenchRecent), () => { dist.assign(0); act.assign(ev); }).Else(() => {
        for (const [dx, dz] of AX) {
          const n = tColIdx(x.add(int(dx)), z.add(int(dz))).toVar();
          If(pid.element(n).equal(p), () => {
            const an = src.element(n).toVar();
            If(an.y.greaterThan(EPS).and(an.x.add(1).lessThan(dist)), () => { dist.assign(an.x.add(1)); act.assign(an.y); });
          });
        }
        If(dist.greaterThan(MAGMA.arcMaxDist), () => { dist.assign(ARC_FAR); act.assign(0); });
      });
      dst.element(i).assign(vec4(dist, act, ev, rec));
      if (crustAge) {
        const fresh = max(float(1).sub(crustAge.element(i).div(MAGMA.ridgeAge)), 0);
        const rl = tQuant(float(MAGMA.ridgeGain * 2).mul(fresh).mul(volc).mul(dt), tHash01(i, tick, 12), uint(1024)).toVar();
        If(rl.greaterThan(uint(0)).and(atomicLoad(mctr.element(MCTR.RES_SNAP)).greaterThan(int(MAGMA.reservoirMin))), () => {
          atomicAdd(ctr.element(CTR_RESERVOIR), int(0).sub(int(rl)));
          atomicAdd(mctr.element(MCTR.DRAWN), int(rl));
          const L = lava.element(i).toVar();
          lava.element(i).assign(tLavaMix(L.x, L.y, rl, float(LAVA.tBasalt), uint(0)));
        });
      }
    })().compute(NCOL);
  }

  /** Melt focusing direction + vent eligibility, from read-only magCol / volcano. */
  /**
   * Melt focusing direction + vent eligibility (from read-only magCol / volcano), and rim spatter: columns next
   * to a vent that carved its crater last tick get 1/8 of the carved rock each, drawn from the reservoir.
   */
  private buildFocus(pid: StorageNode<'uint'>, vox: StorageNode<'uint'>): THREE.ComputeNode {
    // storage buffers: magCol, volcano, magFocus, plateId, vox, counters, magmaCtr = 7 (V23)
    const f = this.fields;
    const magCol = f.cur<'uvec2'>('magCol');
    const volcano = f.cur<'vec4'>('volcano');
    const focus = f.cur<'uint'>('magFocus');
    const ctr = f.cur<'int'>('counters');
    const mctr = f.cur<'int'>('magmaCtr');
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      // rim spatter (volcano.w of the neighbours = carve units·2 + felsic, written by last tick's chamber kernel)
      const dep = uint(0).toVar();
      const fel = uint(0).toVar();
      for (const [dx, dz] of N8) {
        const w = tVwCarve(volcano.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).w).toVar();
        dep.addAssign(w.shiftRight(uint(1)).div(uint(8)));
        fel.assign(fel.bitOr(w.bitAnd(uint(1))));
      }
      If(dep.greaterThan(uint(0)).and(atomicLoad(mctr.element(MCTR.RES_SNAP)).greaterThan(int(0))), () => {
        const { topY, topV } = tScanTop(vox, x, z);
        const f1 = fel.equal(uint(1));
        const put = tAddRock(vox, x, z, topY, topV, dep, select(f1, uint(Mat.ANDESITE), uint(Mat.BASALT)),
          select(f1, uint(FLAG_CONTINENTAL), uint(0)), 2).toVar();
        atomicAdd(ctr.element(CTR_RESERVOIR), int(0).sub(int(put)));
      });
      const mc = magCol.element(i).toVar();
      const m0 = mc.x.toVar();
      const p = mc.y.toVar();
      const pl = pid.element(i).toVar();
      const vw = volcano.element(i).toVar();
      const ph = uint(vw.y).toVar();
      const bestM = m0.toVar();
      const dir = uint(0).toVar();
      AX.forEach(([dx, dz], k) => {
        const n = tColIdx(x.add(int(dx)), z.add(int(dz))).toVar();
        const mn = magCol.element(n).toVar();
        If(pid.element(n).equal(pl).and(mn.x.greaterThan(bestM)).and(mn.y.lessThan(uint(PEND_FOCUS_MAX))), () => {
          bestM.assign(mn.x); dir.assign(uint(k + 1));
        });
      });
      const send = dir.notEqual(uint(0)).and(p.greaterThan(uint(0))).and(ph.equal(uint(PHASE.IDLE))).toVar();
      // vent: strict local maximum of chamber mass over (2r+1)² (ties → lower index) with no erupting neighbour
      const elig = m0.greaterThanEqual(uint(VOLC.minCharge * 255 - 254)).toVar();
      const R = VOLC.ventRadius;
      const ring: [number, number][] = [];
      for (let dz = -R; dz <= R; dz++) for (let dx = -R; dx <= R; dx++) if (dx || dz) ring.push([dx, dz]);
      for (const [dx, dz] of ring) {
        const n = tColIdx(x.add(int(dx)), z.add(int(dz))).toVar();
        const mn = magCol.element(n).x;
        const pn = uint(volcano.element(n).y);
        const beaten = mn.greaterThan(m0).or(mn.equal(m0).and(n.lessThan(i)));
        const busy = pn.notEqual(uint(PHASE.IDLE)).and(pn.notEqual(uint(PHASE.POST)));
        If(beaten.or(busy), () => { elig.assign(false); });
      }
      focus.element(i).assign(select(send, dir, uint(0)).bitOr(select(elig, uint(8), uint(0)))
        .bitOr(tVwSil(vw.w).shiftLeft(uint(4))).bitOr(select(send, p, uint(0)).shiftLeft(uint(12))));
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

  /** Vent state and surface lava ride with their column too (lava of subducted columns → reservoir via the sums). */
  private buildGather2(): THREE.ComputeNode {
    // storage buffers: tecAct, volcano, volcanoTmp, lava, lavaOut (scratch), magmaCtr = 6
    const f = this.fields;
    const act = f.cur<'uint'>('tecAct');
    const volcano = f.cur<'vec4'>('volcano');
    const volcanoTmp = f.cur<'vec4'>('volcanoTmp');
    const lava = f.cur<'uvec2'>('lava');
    const lavaOut = f.cur<'uvec4'>('lavaOut');
    const mctr = f.cur<'int'>('magmaCtr');
    return Fn(() => {
      const d = instanceIndex;
      If(d.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const a = act.element(d).toVar();
      const src = a.bitAnd(uint(0xffff));
      const isNew = a.bitAnd(uint(ACT_NEW)).notEqual(uint(0)).toVar();
      atomicAdd(mctr.element(MCTR.GATHER_OLD), int(lava.element(d).x));
      const L = select(isNew, uvec2(0, 0), lava.element(src)).toVar();
      atomicAdd(mctr.element(MCTR.GATHER_NEW), int(L.x));
      lavaOut.element(d).assign(uvec4(L, 0, 0));
      const v = select(isNew, vec4(0, 0, 0, 0), volcano.element(src)).toVar();
      volcanoTmp.element(d).assign(vec4(v.x, v.y, v.z, tVwPack(uint(0), tVwSil(v.w), tVwHot(v.w)))); // drop the carve event
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

  /** Melt generation (reservoir → pending) and focusing transfer; pending silica / hot flag live in volcano.w. */
  private buildMelt(): THREE.ComputeNode {
    // storage buffers: magCol, magFocus, arc, heatFlow, counters, magmaCtr, volcano = 7 (V23)
    const f = this.fields;
    const magCol = f.cur<'uvec2'>('magCol');
    const focus = f.cur<'uint'>('magFocus');
    const arc = f.cur<'vec4'>('arc');
    const heatFlow = f.cur('heatFlow');
    const ctr = f.cur<'int'>('counters');
    const mctr = f.cur<'int'>('magmaCtr');
    const volcano = f.cur<'vec4'>('volcano');
    const { dt, tick, volcanism: volc } = this.uniforms;
    const M = MAGMA;
    const draw = (n: U) => {
      atomicAdd(ctr.element(CTR_RESERVOIR), int(0).sub(int(n)));
      atomicAdd(mctr.element(MCTR.DRAWN), int(n));
    };
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const mc = magCol.element(i).toVar();
      const vs = volcano.element(i).toVar();
      const p = mc.y.toVar();
      const sw = float(p).mul(float(tVwSil(vs.w))).toVar(); // silica · mass, mixed by mass
      const hot = tVwHot(vs.w).toVar();
      // focusing: my batch leaves if I send; neighbours whose direction points at me arrive
      If(focus.element(i).bitAnd(uint(7)).notEqual(uint(0)), () => { p.assign(0); sw.assign(0); });
      AX.forEach(([dx, dz], k) => {
        const fn = focus.element(tColIdx(x.add(int(dx)), z.add(int(dz)))).toVar();
        If(fn.bitAnd(uint(7)).equal(uint((k ^ 1) + 1)), () => {
          const q = fn.shiftRight(uint(12));
          p.addAssign(q);
          sw.addAssign(float(q).mul(float(fn.shiftRight(uint(4)).bitAnd(uint(0xff)))));
        });
      });

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
      const fert = clamp(float(atomicLoad(mctr.element(MCTR.RES_SNAP))).div(M.resRef), M.fertMin, M.fertMax);
      // fertility drives arc melt only (slab-fed); hotspots are plumes and keep their own rate
      const melt = hotU.add(arcU.mul(fert)).toVar();

      // melt generation: reservoir → pending (gated on a deterministic reservoir snapshot)
      If(atomicLoad(mctr.element(MCTR.RES_SNAP)).greaterThan(int(M.reservoirMin)), () => {
        const cap = select(p.lessThan(uint(M.meltAccMax)), uint(M.meltAccMax).sub(p), uint(0));
        const n = tQuant(melt, tHash01(i, tick, 11), cap).toVar();
        If(n.greaterThan(uint(0)), () => {
          draw(n);
          p.addAssign(n);
          sw.addAssign(float(n).mul(clamp(arcU.mul(fert).div(max(melt, 1e-6)), 0, 1).mul(255)));
          If(hotU.greaterThan(arcU.mul(fert)), () => { hot.assign(1); });
        });
      });
      const sil = uint(clamp(round(sw.div(max(float(p), 1))), 0, 255));
      magCol.element(i).assign(uvec2(mc.x, p));
      volcano.element(i).assign(vec4(vs.x, vs.y, vs.z, tVwPack(tVwCarve(vs.w), sil, hot)));
    })().compute(NCOL);
  }

  /** Chamber emplacement, vent lifecycle (effusion into the lake, crater carve), crystallization. */
  private buildChamber(vox: StorageNode<'uint'>): THREE.ComputeNode {
    // storage buffers: vox, magCol, magFocus, volcano, lava, surfY, counters, magmaCtr = 8 (V23)
    const f = this.fields;
    const magCol = f.cur<'uvec2'>('magCol');
    const focus = f.cur<'uint'>('magFocus');
    const volcano = f.cur<'vec4'>('volcano');
    const lava = f.cur<'uvec2'>('lava');
    const surfY = f.cur('surfY');
    const ctr = f.cur<'int'>('counters');
    const mctr = f.cur<'int'>('magmaCtr');
    const { dt, tick } = this.uniforms;
    const M = MAGMA, V = VOLC;
    const MAGMA0 = uint(Mat.MAGMA); // an empty chamber voxel is MAGMA with fill 0 (a cavity; mass-free)
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const mc = magCol.element(i).toVar();
      const ch = mc.x.toVar();
      const p = mc.y.toVar();
      const vs = volcano.element(i).toVar();
      const psil = tVwSil(vs.w).toVar();
      const phot = tVwHot(vs.w).toVar();
      const act = vs.x.toVar();
      const ph = uint(vs.y).toVar();
      const tm = vs.z.toVar();
      const carveW = uint(0).toVar();
      const elig = focus.element(i).bitAnd(uint(8)).notEqual(uint(0));
      const seg = p.greaterThanEqual(uint(255)).or(p.greaterThanEqual(uint(M.segMin)).and(tHash01(i, tick, 15).lessThan(dt.div(M.segTau)))).toVar();

      If(seg.or(ch.greaterThan(uint(0))).or(ph.notEqual(uint(PHASE.IDLE))), () => {
        // column scan: top solid, lowest crust, chamber extent (incl. empty cavities), continental crust present
        const top = int(-1).toVar();
        const base = int(NY).toVar();
        const n = int(0).toVar();          // MAGMA voxels (full, partial or empty)
        const chTop = int(-1).toVar();
        const chBot = int(-1).toVar();
        const topFull = int(-1).toVar();   // highest chamber voxel holding magma (effusion source)
        const lowRoom = int(-1).toVar();   // lowest chamber voxel with room (refilled before inserting)
        const cont = uint(0).toVar();
        Loop(NY, ({ i: y }) => {
          const v = vox.element(tVoxIdx(x, y, z)).toVar();
          const m = tMat(v).toVar();
          If(m.notEqual(uint(Mat.AIR)), () => { top.assign(y); });
          If(isCrust(m), () => { base.assign(iMin(base, y)); cont.assign(cont.bitOr(tFlags(v).bitAnd(uint(FLAG_CONTINENTAL)))); });
          If(m.equal(MAGMA0), () => {
            n.addAssign(1); chTop.assign(y);
            If(chBot.lessThan(int(0)), () => { chBot.assign(y); });
            If(tFill(v).greaterThan(uint(0)), () => { topFull.assign(y); });
            If(tFill(v).lessThan(uint(255)).and(lowRoom.lessThan(int(0))), () => { lowRoom.assign(y); });
          });
        });
        const sil = select(cont.notEqual(uint(0)), psil.add(uint(255)).shiftRight(uint(1)), psil).toVar();
        const hotFlag = select(phot.equal(uint(1)), uint(FLAG_HOTSPOT), uint(0));

        // a chamber that ended up under all of its crust (crust flow / thrusting removed the rock below it) or near
        // the grid floor crystallizes at once: sinking columns drop their bottom voxel, which must never be magma
        // (tectonics does not account for it). Magma → pluton, cavity → mantle; both exact.
        If(chBot.greaterThanEqual(int(0)).and(chBot.lessThan(base).or(chBot.lessThanEqual(int(2)))), () => {
          const idx = tVoxIdx(x, chBot, z).toVar();
          const v = vox.element(idx).toVar();
          const fel = tAge(v).greaterThanEqual(uint(M.silAndesite));
          vox.element(idx).assign(select(tFill(v).equal(uint(0)), tPack(uint(Mat.PERIDOTITE), uint(255), uint(0), uint(0)),
            tPack(select(fel, uint(Mat.GRANITE), uint(Mat.GABBRO)), tFill(v), uint(0), select(fel, uint(FLAG_CONTINENTAL), uint(0)))));
          ch.subAssign(tFill(v));
          atomicAdd(mctr.element(MCTR.FROZEN), int(tFill(v)));
          If(tFill(v).greaterThan(uint(0)), () => { base.assign(iMin(base, chBot)); });
          n.subAssign(1);
          If(topFull.equal(chBot), () => { topFull.assign(-1); });
          If(lowRoom.equal(chBot), () => { lowRoom.assign(-1); });
          If(n.equal(int(0)), () => { chTop.assign(-1); });
          chBot.assign(select(n.greaterThan(int(0)), chBot.add(int(1)), int(-1)));
        });

        // a chamber roof within depthMin of the surface (erosion / talus stripped its lid) freezes to a pluton, an
        // empty roof cavity at the very top opens to air. Magma cannot slump, so exposed chambers used to pin
        // mountain fronts as 50-layer walls. Exact: magma → crust of the same fill, cavity (fill 0) → air.
        If(chTop.greaterThanEqual(int(0)).and(chTop.greaterThan(top.sub(int(M.depthMin)))), () => {
          const idx = tVoxIdx(x, chTop, z).toVar();
          const v = vox.element(idx).toVar();
          const fel = tAge(v).greaterThanEqual(uint(M.silAndesite));
          const gone = tFill(v).greaterThan(uint(0)).or(chTop.equal(top)).toVar(); // a buried empty cavity waits for collapse
          If(tFill(v).greaterThan(uint(0)), () => {
            vox.element(idx).assign(tPack(select(fel, uint(Mat.GRANITE), uint(Mat.GABBRO)), tFill(v), uint(0), select(fel, uint(FLAG_CONTINENTAL), uint(0))));
            ch.subAssign(tFill(v));
            atomicAdd(mctr.element(MCTR.FROZEN), int(tFill(v)));
          }).ElseIf(chTop.equal(top), () => {
            vox.element(idx).assign(uint(0));
            top.subAssign(1);
          });
          If(gone, () => {
            n.subAssign(1);
            If(topFull.equal(chTop), () => { topFull.assign(-1); });
            If(lowRoom.equal(chTop), () => { lowRoom.assign(-1); });
            chTop.assign(select(n.greaterThan(int(0)), chTop.sub(int(1)), int(-1)));
            If(n.equal(int(0)), () => { chBot.assign(-1); });
          });
        });

        // chamber growth: refill a cavity / partial voxel first, else insert a new MAGMA voxel (column inflates)
        If(seg.and(lowRoom.greaterThanEqual(int(0))), () => {
          const idx = tVoxIdx(x, lowRoom, z).toVar();
          const v = vox.element(idx).toVar();
          const put = uMin(p, uint(255).sub(tFill(v))).toVar();
          vox.element(idx).assign(tPack(MAGMA0, tFill(v).add(put), select(tFill(v).equal(uint(0)), sil, tAge(v)), tFlags(v).bitOr(hotFlag)));
          p.subAssign(put);
          ch.addAssign(put);
          topFull.assign(iMax(topFull, lowRoom));
        }).ElseIf(seg.and(n.lessThan(int(M.chamberMax))).and(top.greaterThanEqual(int(0))).and(top.lessThanEqual(int(NY - 3))), () => {
          const depth = iMin(iMax(top.sub(base).div(int(3)), int(M.depthMin)), int(M.depthMax));
          const ins = select(n.greaterThan(int(0)), chTop.add(int(1)), top.sub(depth)).toVar();
          // the roof stays ≥ depthMin below the surface: a chamber that reached it is full (like chamberMax)
          If(ins.greaterThan(base).and(ins.lessThanEqual(top.sub(int(M.depthMin)))), () => {
            Loop({ start: top, end: ins, condition: '>=' }, ({ i: y }) => {
              vox.element(tVoxIdx(x, y.add(int(1)), z)).assign(vox.element(tVoxIdx(x, y, z)));
            });
            const put = uMin(p, uint(255)).toVar();
            vox.element(tVoxIdx(x, ins, z)).assign(tPack(MAGMA0, put, sil, hotFlag));
            p.subAssign(put);
            ch.addAssign(put);
            If(n.equal(int(0)), () => { chBot.assign(ins); });
            n.addAssign(1);
            top.assign(iMax(top.add(int(1)), ins));
            chTop.assign(ins);
            topFull.assign(ins);
          });
        });

        const felsic = uint(0).toVar();
        If(topFull.greaterThanEqual(int(0)), () => { felsic.assign(select(tAge(vox.element(tVoxIdx(x, topFull, z))).greaterThanEqual(uint(M.silAndesite)), uint(1), uint(0))); });

        // effusion from the highest charged chamber voxel into the vent lake; the drained voxel stays as a cavity
        const effuse = (unitsIn: U) => {
          const units = uint(unitsIn).toVar();
          If(topFull.greaterThanEqual(int(0)).and(units.greaterThan(uint(0))), () => {
            const idx = tVoxIdx(x, topFull, z).toVar();
            const v = vox.element(idx).toVar();
            const t = uMin(units, tFill(v)).toVar();
            vox.element(idx).assign(tPack(MAGMA0, tFill(v).sub(t), tAge(v), tFlags(v)));
            ch.subAssign(t);
            const L = lava.element(i).toVar();
            lava.element(i).assign(tLavaMix(L.x, L.y, t, tEruptTemp(tAge(v)), tAge(v)));
          });
        };
        // caldera collapse: squeeze out empty cavities (the column above each drops one layer)
        const collapse = () => {
          Loop(M.chamberMax, () => {
            const hole = int(-1).toVar();
            Loop({ start: iMax(chBot, int(0)), end: iMax(chTop, int(-1)).add(int(1)), condition: '<' }, ({ i: y }) => {
              const v = vox.element(tVoxIdx(x, y, z));
              If(hole.lessThan(int(0)).and(tMat(v).equal(MAGMA0)).and(tFill(v).equal(uint(0))), () => { hole.assign(y); });
            });
            If(hole.greaterThanEqual(int(0)), () => {
              Loop({ start: hole, end: top, condition: '<' }, ({ i: y }) => {
                vox.element(tVoxIdx(x, y, z)).assign(vox.element(tVoxIdx(x, y.add(int(1)), z)));
              });
              vox.element(tVoxIdx(x, top, z)).assign(uint(0));
              top.subAssign(1); chTop.subAssign(1); n.subAssign(1);
            });
          });
        };
        // summit crater: carve the vent below its lowest rim neighbour (reservoir +); next tick's focus kernel deposits spatter.
        // Not under water: a crater of up to craterMax layers per episode took back everything a seamount gained,
        // so submarine volcanoes never grew past ~3 layers of relief (never breached).
        const carve = () => If(surfY.element(i).greaterThanEqual(this.uniforms.seaLevel.sub(0.5)), () => {
          const rimMin = float(1e9).toVar();
          for (const [dx, dz] of N8) rimMin.assign(min(rimMin, surfY.element(tColIdx(x.add(int(dx)), z.add(int(dz))))));
          const topV = vox.element(tVoxIdx(x, top, z)).toVar();
          const s = float(top).add(float(tFill(topV)).div(255));
          const want = uint(clamp(floor(s.sub(rimMin.sub(V.craterDepth)).mul(255)), 0, V.craterMax * 255)).toVar();
          const got = tRemoveTop(vox, x, z, top, want, iMax(chTop, base)).toVar();
          If(got.greaterThan(uint(0)), () => {
            atomicAdd(ctr.element(CTR_RESERVOIR), int(got));
            carveW.assign(uMin(got.mul(uint(2)).add(felsic), uint(2047)));
          });
        });
        const lid = top.sub(chTop);
        const charged = int(ch.add(uint(254)).div(uint(255))).toVar(); // full-voxel equivalents of magma
        const need = lid.div(int(M.lidPerVoxel)).add(int(1)).toVar();
        const rate = (scale: F) => uint(floor(max(float(V.effuseMin), float(ch).div(max(tm, 0).add(V.tWane))).mul(dt).mul(scale)));

        If(ph.equal(uint(PHASE.IDLE)), () => {
          act.assign(0);
          tm.subAssign(dt);
          If(tm.lessThanEqual(0).and(elig).and(charged.greaterThanEqual(int(VOLC.minCharge))).and(charged.greaterThanEqual(need)), () => {
            ph.assign(PHASE.BUILD); tm.assign(V.tBuild);
          }).ElseIf(topFull.greaterThanEqual(int(0)).and(tHash01(i, tick, 14).lessThan(float(charged).mul(dt).div(M.freezeTau))), () => {
            // chamber crystallizes from the bottom: magma → pluton (crust); much arc magma ends up intrusive
            const idx = tVoxIdx(x, chBot, z).toVar();
            const v = vox.element(idx).toVar();
            If(tFill(v).greaterThan(uint(0)), () => {
              const fel = tAge(v).greaterThanEqual(uint(M.silAndesite));
              vox.element(idx).assign(tPack(select(fel, uint(Mat.GRANITE), uint(Mat.GABBRO)), tFill(v), uint(0),
                select(fel, uint(FLAG_CONTINENTAL), uint(0))));
              ch.subAssign(tFill(v));
              atomicAdd(mctr.element(MCTR.FROZEN), int(tFill(v)));
            });
          });
        }).ElseIf(ph.equal(uint(PHASE.BUILD)), () => {
          tm.subAssign(dt);
          act.assign(float(V.buildAct).mul(clamp(float(1).sub(tm.div(V.tBuild)), 0, 1)));
          If(tm.lessThanEqual(0), () => {
            // explosive opening: crater, then the lake starts filling this very tick
            ph.assign(PHASE.ACTIVE);
            act.assign(1);
            tm.assign(float(V.tActive[0]).add(tHash01(i, tick, 16).mul(V.tActive[1] - V.tActive[0])));
            atomicAdd(mctr.element(MCTR.ERUPTIONS), int(1));
            carve();
            effuse(rate(float(1)));
          });
        }).ElseIf(ph.equal(uint(PHASE.ACTIVE)), () => {
          act.assign(1);
          effuse(rate(float(1)));
          tm.subAssign(dt);
          If(tm.lessThanEqual(0), () => { ph.assign(PHASE.WANING); tm.assign(V.tWane); })
            .ElseIf(ch.equal(uint(0)), () => { ph.assign(PHASE.WANING); tm.assign(V.tWane * 0.3); }); // drained early: brief waning
        }).ElseIf(ph.equal(uint(PHASE.WANING)), () => {
          act.assign(clamp(tm.div(V.tWane), 0, 1).mul(select(ch.greaterThan(uint(0)), float(1), float(0.5))));
          effuse(rate(act));
          tm.subAssign(dt);
          If(tm.lessThanEqual(0), () => { ph.assign(PHASE.POST); tm.assign(0); });
        }).Else(() => {
          // POST: once the lake has frozen (or after 3 My), the drained chamber collapses and the crater is carved
          act.assign(0);
          tm.subAssign(dt);
          If(lava.element(i).x.equal(uint(0)).or(tm.lessThan(-3)), () => {
            collapse();
            carve();
            ph.assign(PHASE.IDLE);
            tm.assign(float(V.repose[0]).add(tHash01(i, tick, 17).mul(V.repose[1] - V.repose[0])));
          });
        });
      });
      magCol.element(i).assign(uvec2(ch, p));
      volcano.element(i).assign(vec4(act, float(ph), tm, tVwPack(carveW, psil, phot)));
    })().compute(NCOL);
  }
}
