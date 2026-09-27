// §C pass 3-4: rigid plate advection on the torus, convergent/divergent resolution, isostasy.
//
// Plates translate rigidly. Each tick a plate accumulates a sub-cell offset; when it crosses a cell the
// plate's shift for that tick is ±1 on X and/or Z. A decide kernel (one thread per destination column)
// gathers every plate whose shifted source column lands here:
//   0 candidates → divergent gap: fresh oceanic crust drawn from the mantle reservoir
//   1 candidate  → column moves (or stays)
//   ≥2           → convergence: most buoyant column wins (continental > young oceanic > old oceanic);
//                  losers subduct into the mantle reservoir; continental losers thicken the winner (orogeny)
// Isostasy rides along: the winning column shifts ±1 layer toward buoyancy equilibrium.
// A voxel kernel then rebuilds every column from its action. All mass moves in integer fill units (V3).
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, int, uint, instanceIndex, Return, select, sqrt, uniform, uniformArray, atomicAdd, atomicStore, atomicMax, vec2, sin } from 'three/tsl';
import { PingPongKernel, type GpuFields } from '../core/gpu';
import { NCOL, NVOX, NX, NY, Mat, MAT_DENSITY, FLAG_CONTINENTAL, packVoxel } from './layout';
import { tColIdx, tColXZ, uMin, iMax, iMin, tMat, tFill } from './tslLayout';
import { MAX_PLATES, type Plate } from './worldData';
import { COL_CONTINENTAL } from './derive';
import { CTR_PLATE, CTR_RESERVOIR, CTR_SIZE, CTR_QUAKE } from './fields';
import { CrustFlow } from './crustFlow';
import { OCEAN_POOL } from './oceanLevel';

// Isostasy calibration (voxel-y units). Surface at equilibrium:
//   eq = Y_COMP + thickness * (1 - ρcrust/ρmantle) * ISO_GAIN - (oceanic ? SUBSIDENCE * sqrt(age) : 0)
// Gain exaggerates freeboard so 40-layer continents sit ~84 and fresh 7-layer ocean crust ~65.
export const ISO_GAIN = 2.83;
export const Y_COMP = 63.4;
export const SUBSIDENCE = 0.7; // layers per sqrt(My), thermal subsidence of oceanic lithosphere
const RHO_MANTLE = MAT_DENSITY[Mat.PERIDOTITE]!;
const RHO_CONT = MAT_DENSITY[Mat.GRANITE]!;
const RHO_OCEAN = MAT_DENSITY[Mat.BASALT]!;

// Fresh ridge crust profile: GABBRO [RIDGE_BASE, +3), BASALT [+3, +7). Mass drawn: 7 full layers.
export const RIDGE_BASE = 58;
export const RIDGE_GABBRO = 3;
export const RIDGE_BASALT = 4;
export const RIDGE_MASS = (RIDGE_GABBRO + RIDGE_BASALT) * 255;
/** Max layers of crustal root added to the winner per collision event (6-bit field). */
export const OROGENY_MAX = 63;
/** Crust thicker than this (layers) stops thickening; extra collided mass delaminates into the mantle. */
export const MAX_CRUST_LAYERS = 100;
/**
 * Fraction of oceanic crust consumed under a continent that is scraped off / underplated onto it
 * (accretionary wedge + arc root). The only steady return path for continental crust that erosion
 * sends to the sea; without it continents thin and drown within ~400 My (B8). The rest melts: it feeds arc
 * magma (granite on continental plates), the mantle's only steady supply once melting follows the reservoir
 * (at 0.9 volcanism starved).
 */
export const ACCRETE_FRAC = 0.6;
/**
 * Share accreted where the trench winner is oceanic (island arcs). Lower than ACCRETE_FRAC: ocean-ocean
 * subduction returns the rest to the mantle, which feeds arc magma; at 0.9 the reservoir starved and
 * volcanism stalled.
 */
export const ACCRETE_FRAC_OCEAN = 0.4;
/** Rift fill thickness as a fraction of the neighbouring continental crust (B18). */
export const RIFT_THIN = 0.6;
/**
 * Rift fill only while the stretched crust stays at least this thick (layers); thinner gaps open real ocean.
 * Without the floor every gap next to thin rift crust stayed continental and continents spread over the
 * whole world (cont 0.40 → 1.00 in 300 My), at the reservoir's expense.
 */
export const RIFT_MIN_THICK = 15;
/** Stretched crust is never thicker than normal continental crust (layers): orogens next to a rift do not copy. */
export const RIFT_MAX_THICK = 30;
/** Max immediate surface rise from orogeny per tectonics run (layers). */
export const MAX_UP_PER_RUN = 1;
/** Tectonics runs every N ticks with N·dtGeo of motion (cost control, V9/V19). */
export const TEC_EVERY = 4;

/** tecAct bits: 0-15 src col | 16 new ridge crust | 17-18 isostasy v+1 | 19-24 orogeny k | 25 oceanic slab consumed here | 26-31 root uplift. */
export const ACT_NEW = 1 << 16;
export const ACT_SUBDUCT = 1 << 25;
const V_SHIFT = 17, K_SHIFT = 19, UP_SHIFT = 26;
/** Root of k layers grows downward; the column rises only by its isostatic share, up = round(k·ROOT_UP/256). */
const ROOT_UP = Math.round(256 * (1 - RHO_CONT / RHO_MANTLE) * ISO_GAIN);
const rootUp = (k: THREE.Node<'uint'>) => k.mul(uint(ROOT_UP)).add(uint(128)).shiftRight(uint(8));

/** Winner score of a candidate column: continental > oceanic; thicker continent / younger ocean wins; ties → lower id. */
export function columnScore(ci: THREE.Node<'uvec2'>, age: THREE.Node<'float'>, plate: THREE.Node<'uint'>): THREE.Node<'uint'> {
  const cont = ci.x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL));
  const tie = uint(MAX_PLATES - 1).sub(plate);
  const ageKey = uint(0xffff).sub(uMin(uint(age.mul(10)), uint(0xffff)));
  return select(cont.equal(uint(1)), uint(1 << 28).bitOr(ci.y.shiftLeft(uint(4))).bitOr(tie), ageKey.shiftLeft(uint(4)).bitOr(tie));
}

export interface TectonicsStats { area: number[]; subducted: number[]; created: number[] }

export class Tectonics {
  readonly plates: Plate[];
  private decide: [THREE.ComputeNode, THREE.ComputeNode];
  private waterGather: [THREE.ComputeNode, THREE.ComputeNode];
  private voxel: PingPongKernel<'uint'>;
  private waterBack: THREE.ComputeNode;
  // (shiftX, shiftZ, alive, 0) per plate; uniform, not storage, to stay under the buffer limit (V23)
  private tableValues = Array.from({ length: MAX_PLATES }, () => new THREE.Vector4());
  private table = uniformArray(this.tableValues, 'vec4');
  private clearStats: THREE.ComputeNode;
  private quakeK: THREE.ComputeNode;
  private runIdU = uniform(0, 'uint');
  private ageDt = uniform(0);
  /**
   * 0..1 share of collided/accreted crust that may stack onto the winner. Set per window from the
   * reservoir snapshot: a reservoir in debt means ridges already consumed more than subduction returned,
   * so colliding crust delaminates back instead (mass constraint, not a param nudge: V3, V5, B10).
   */
  readonly stackGate = uniform(1);
  private readonly travel = new Float64Array(MAX_PLATES * 2);
  private isoOn = uniform(0);
  private lastRunTick = 0;
  /** Increments on every GPU tectonics run: tecAct (src column per dst) is valid for the latest run id. */
  runId = 0;
  private crustFlow: CrustFlow;

  constructor(private fields: GpuFields, plates: Plate[]) {
    this.plates = plates.map((p) => ({ ...p, vel: [...p.vel] as [number, number], accum: [...p.accum] as [number, number] }));
    const [pid0, pid1] = fields.pair<'uint'>('plateId');
    const [age0, age1] = fields.pair('crustAge');
    this.decide = [this.buildDecide(pid0, pid1, age0, age1), this.buildDecide(pid1, pid0, age1, age0)];
    this.waterGather = [this.buildWaterGather(pid0, age0), this.buildWaterGather(pid1, age1)];
    this.voxel = new PingPongKernel<'uint'>(fields, 'vox', (src, dst) => this.buildVoxel(src, dst));
    this.crustFlow = new CrustFlow(fields);
    this.quakeK = this.buildQuakeSites();
    const water = fields.cur('water');
    const sed = fields.cur('sedSusp');
    const waterTmp = fields.cur<'vec2'>('waterTmp');
    this.waterBack = Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const t = waterTmp.element(instanceIndex).toVar();
      water.element(instanceIndex).assign(t.x);
      sed.element(instanceIndex).assign(t.y);
    })().compute(NCOL);
    const ctr = fields.cur<'int'>('counters');
    this.clearStats = Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(CTR_SIZE - CTR_PLATE)), () => { Return(); });
      atomicStore(ctr.element(instanceIndex.add(CTR_PLATE)), int(0));
    })().compute(CTR_SIZE - CTR_PLATE);
  }

  private buildDecide(pidCur: THREE.StorageBufferNode<'uint'>, pidNext: THREE.StorageBufferNode<'uint'>,
    ageCur: THREE.StorageBufferNode<'float'>, ageNext: THREE.StorageBufferNode<'float'>): THREE.ComputeNode {
    // storage buffers: pidCur, pidNext, ageCur, ageNext, colInfo, surfY, tecAct, counters = 8 (V23)
    const f = this.fields;
    const table = this.table;
    const colInfo = f.cur<'uvec2'>('colInfo');
    const surfY = f.cur('surfY');
    const act = f.cur<'uint'>('tecAct');
    const ctr = f.cur<'int'>('counters');
    const { ageDt, isoOn, stackGate } = this;

    return Fn(() => {
      const d = instanceIndex;
      If(d.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(d);

      const count = uint(0).toVar();
      const massSum = uint(0).toVar();
      const contMass = uint(0).toVar();
      const contCount = uint(0).toVar();
      const oceanLost = uint(0).toVar();
      const best = uint(0).toVar();
      const bestSrc = uint(0).toVar();
      const bestPlate = uint(0).toVar();
      const bestMass = uint(0).toVar();
      const bestCont = uint(0).toVar();
      const bestAge = float(0).toVar();
      const bestInfo = uint(0).toVar();
      const bestSurf = float(0).toVar();

      Loop(MAX_PLATES, ({ i }) => {
        const pt = table.element(i) as unknown as THREE.Node<'vec4'>;
        If(pt.z.greaterThan(0.5), () => {
          const src = tColIdx(x.sub(int(pt.x)), z.sub(int(pt.y))).toVar();
          If(pidCur.element(src).equal(uint(i)), () => {
            const ci = colInfo.element(src).toVar();
            const info = ci.x;
            const mass = ci.y;
            const cont = info.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL)).toVar();
            const age = ageCur.element(src).toVar();
            const score = columnScore(ci, age, uint(i)).toVar();
            count.addAssign(1);
            massSum.addAssign(mass);
            If(cont.equal(uint(1)), () => { contMass.addAssign(mass); contCount.addAssign(1); });
            If(score.greaterThan(best), () => {
              best.assign(score); bestSrc.assign(src); bestPlate.assign(uint(i)); bestMass.assign(mass);
              bestCont.assign(cont); bestAge.assign(age); bestInfo.assign(info); bestSurf.assign(surfY.element(src));
            });
          });
        });
      });

      If(count.equal(uint(0)), () => {
        // divergent gap: new ridge crust, stays with the plate that left
        const owner = pidCur.element(d);
        pidNext.element(d).assign(owner);
        ageNext.element(d).assign(0);
        act.element(d).assign(uint(ACT_NEW).bitOr(d));
        atomicAdd(ctr.element(CTR_RESERVOIR), int(-RIDGE_MASS));
        atomicAdd(ctr.element(owner.mul(4).add(2 + CTR_PLATE)), int(1));
      }).Else(() => {
        const top = bestInfo.shiftRight(uint(8)).bitAnd(uint(0xff));
        const base = bestInfo.bitAnd(uint(0xff));
        // orogeny: continental losers stack onto the winner as crustal root (continents are not destroyed
        // in collisions), limited by max crust thickness and grid top; only the excess delaminates
        const k = uint(0).toVar();
        // oceanic winners accrete too (island arcs grow from their trench wedge): with continental winners only,
        // every ocean-ocean trench sent its slab sediment and slumped debris into the reservoir, which grew to
        // 7 layers/col while land shrank from 27 % to 13 %
        // only a column with crust can take a thrust sheet: a crustless winner (base = NY, e.g. a stripped ocean
        // column) put the insertion point at the grid top and read the sheet from past the column: a gneiss
        // voxel floating over ~60 layers of air (needle spikes over the sea, B28)
        If(count.greaterThan(uint(1)).and(base.lessThan(uint(NY))), () => {
          const room = uint(MAX_CRUST_LAYERS).sub(uMin(bestMass.div(uint(255)), uint(MAX_CRUST_LAYERS)));
          const floorRoom = base.sub(uMin(base, uint(1))); // root must stay above y=0
          // continental losers stack fully; oceanic losers accrete ACCRETE_FRAC of what they carry beyond
          // fresh ridge thickness (sediment + aged crust). The ridge share must return to the reservoir:
          // every gap draws RIDGE_MASS, and gaps pair 1:1 with losers, else the reservoir runs into debt (B10).
          // losers only: an oceanic winner's own mass is in the ocean sum, a continental winner's in contMass
          const isC = bestCont.equal(uint(1));
          const contLoss = select(isC, contMass.sub(bestMass), uint(0));
          const oceanMass = massSum.sub(contMass).sub(select(isC, uint(0), bestMass));
          const oceanLosers = count.sub(uint(1)).sub(contCount.sub(bestCont));
          const ridgeShare = oceanLosers.mul(uint(RIDGE_MASS));
          const excess = select(oceanMass.greaterThan(ridgeShare), oceanMass.sub(ridgeShare), uint(0));
          const gain0 = contLoss.add(excess.mul(select(isC, uint(Math.round(ACCRETE_FRAC * 256)), uint(Math.round(ACCRETE_FRAC_OCEAN * 256)))).shiftRight(uint(8)));
          const gain = uint(float(gain0).mul(stackGate));
          k.assign(uMin(uMin(gain.div(uint(255)), uint(OROGENY_MAX)), uMin(room, floorRoom)));
        });
        // near the ceiling the root still grows, just downward only (isostasy settles it later)
        // collided crust enters mostly as root; the surface rises at most MAX_UP_PER_RUN layers per run and
        // isostasy lifts the rest over time, so ranges grow gradually instead of popping up (user: B17)
        const up = uMin(uMin(rootUp(k), uint(MAX_UP_PER_RUN)), uint(NY - 3).sub(uMin(top, uint(NY - 3)))).toVar();
        const loserMass = massSum.sub(bestMass);
        // an oceanic loser means a slab went down here (arc volcanism input)
        oceanLost.assign(select(count.sub(uint(1)).greaterThan(contCount.sub(bestCont)), uint(ACT_SUBDUCT), uint(0)));
        If(loserMass.greaterThan(uint(0)).or(k.greaterThan(uint(0))), () => {
          atomicAdd(ctr.element(CTR_RESERVOIR), int(loserMass).sub(int(k.mul(255))));
        });
        If(count.greaterThan(uint(1)), () => {
          // credit each losing plate with a subducted column (slab pull input)
          Loop(MAX_PLATES, ({ i }) => {
            const pt = table.element(i) as unknown as THREE.Node<'vec4'>;
            If(pt.z.greaterThan(0.5).and(uint(i).notEqual(bestPlate)), () => {
              const src = tColIdx(x.sub(int(pt.x)), z.sub(int(pt.y)));
              If(pidCur.element(src).equal(uint(i)), () => { atomicAdd(ctr.element(uint(i).mul(4).add(1 + CTR_PLATE)), int(1)); });
            });
          });
        });

        // isostasy: one layer per run toward buoyancy equilibrium
        const v = int(0).toVar();
        If(isoOn.greaterThan(0.5), () => {
          const thick = float(bestMass.add(k.mul(255))).div(255);
          const isCont = bestCont.equal(uint(1));
          const rho = select(isCont, float(RHO_CONT), float(RHO_OCEAN));
          // thermal subsidence saturates (~150 My); uncapped, stripped continental columns sank through the mantle (B12)
          const sub = select(isCont, float(0), sqrt(bestAge.clamp(0, 150)).mul(SUBSIDENCE));
          const eq = float(Y_COMP).add(thick.mul(float(1).sub(rho.div(RHO_MANTLE))).mul(ISO_GAIN)).sub(sub);
          const diff = eq.sub(bestSurf.add(float(up)));
          If(diff.greaterThan(0.75).and(top.add(up).add(uint(1)).lessThan(uint(NY - 1))), () => { v.assign(1); });
          // sinking must keep the (possibly deepened) root above y=0
          If(diff.lessThan(-0.75).and(base.add(up).greaterThan(k.add(uint(1)))).and(base.lessThan(uint(NY))), () => { v.assign(-1); });
        });

        pidNext.element(d).assign(bestPlate);
        ageNext.element(d).assign(bestAge.add(ageDt));
        act.element(d).assign(bestSrc.bitOr(k.shiftLeft(uint(K_SHIFT))).bitOr(uint(v.add(1)).shiftLeft(uint(V_SHIFT))).bitOr(oceanLost)
          .bitOr(up.shiftLeft(uint(UP_SHIFT))));
        atomicAdd(ctr.element(bestPlate.mul(4).add(CTR_PLATE)), int(1));
      });
    })().compute(NCOL);
  }

  /**
   * Water and suspended sediment ride with their column; overlapping columns pool them, gaps start empty
   * (exactly conservative, V3/V4).
   */
  /**
   * Water and suspended sediment follow the WINNING column; losers' water drains into the ocean pool
   * (oceanSum[S_POOL], 2^-10 voxel units) that the ocean-leveling pass spreads over open ocean. Pooling all
   * candidates' water onto the winner piled ocean water onto continents overriding the sea (B14).
   * The sub-unit remainder of each loser's water stays with the winner, so Σ water is exact (V4).
   */
  private buildWaterGather(pidCur: THREE.StorageBufferNode<'uint'>, ageCur: THREE.StorageBufferNode<'float'>): THREE.ComputeNode {
    const water = this.fields.cur('water');
    const sed = this.fields.cur('sedSusp');
    const waterTmp = this.fields.cur<'vec2'>('waterTmp');
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const ocean = this.fields.cur<'int'>('oceanSum');
    const table = this.table;
    return Fn(() => {
      const d = instanceIndex;
      If(d.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(d);
      const sedSum = float(0).toVar();
      const wSum = float(0).toVar();
      const best = uint(0).toVar();
      const bestW = float(0).toVar();
      Loop(MAX_PLATES, ({ i }) => {
        const pt = table.element(i) as unknown as THREE.Node<'vec4'>;
        If(pt.z.greaterThan(0.5), () => {
          const src = tColIdx(x.sub(int(pt.x)), z.sub(int(pt.y))).toVar();
          If(pidCur.element(src).equal(uint(i)), () => {
            const ci = colInfo.element(src).toVar();
            const score = columnScore(ci, ageCur.element(src), uint(i)).toVar();
            const w = water.element(src).toVar();
            wSum.addAssign(w);
            sedSum.addAssign(sed.element(src));
            If(score.greaterThan(best), () => { best.assign(score); bestW.assign(w); });
          });
        });
      });
      // losers' water in whole 2^-10 units to the pool; the remainder stays here
      const lost = wSum.sub(bestW).max(0);
      const units = int(lost.mul(1024).floor()).toVar();
      If(units.greaterThan(int(0)), () => { atomicAdd(ocean.element(OCEAN_POOL), units); });
      waterTmp.element(d).assign(vec2(bestW.add(lost.sub(float(units).div(1024))), sedSum));
    })().compute(NCOL);
  }

  /**
   * Earthquake sites: each window keeps one deterministic pseudo-random subduction column and one collision
   * column (max hash key), plus event counts. Read-only for the sim; FX turn them into quakes/tsunamis.
   */
  private buildQuakeSites(): THREE.ComputeNode {
    const act = this.fields.cur<'uint'>('tecAct');
    const ctr = this.fields.cur<'int'>('counters');
    const run = this.runIdU;
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const a = act.element(c).toVar();
      const h = c.bitXor(run.mul(uint(0x9e3779b9))).mul(uint(747796405)).add(uint(2891336453));
      const key = int(h.shiftRight(uint(17)).shiftLeft(uint(16)).bitOr(c));
      If(a.bitAnd(uint(ACT_SUBDUCT)).notEqual(uint(0)), () => {
        atomicMax(ctr.element(CTR_QUAKE), key);
        atomicAdd(ctr.element(CTR_QUAKE + 1), int(1));
      });
      If(a.shiftRight(uint(K_SHIFT)).bitAnd(uint(63)).greaterThan(uint(0)).and(a.bitAnd(uint(ACT_NEW)).equal(uint(0))), () => {
        atomicMax(ctr.element(CTR_QUAKE + 2), key);
        atomicAdd(ctr.element(CTR_QUAKE + 3), int(1));
      });
    })().compute(NCOL);
  }

  private buildVoxel(src: THREE.StorageBufferNode<'uint'>, dst: THREE.StorageBufferNode<'uint'>): THREE.ComputeNode {
    const act = this.fields.cur<'uint'>('tecAct');
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const PERI = packVoxel(Mat.PERIDOTITE, 255, 0, 0);
    const GAB = packVoxel(Mat.GABBRO, 255, 0, 0);
    const BAS = packVoxel(Mat.BASALT, 255, 0, 0);
    const GNEISS = packVoxel(Mat.GNEISS, 255, 0, FLAG_CONTINENTAL);
    const GRAN = packVoxel(Mat.GRANITE, 255, 0, FLAG_CONTINENTAL);
    const SED = packVoxel(Mat.SEDIMENT, 255, 0, FLAG_CONTINENTAL);
    const ctr = this.fields.cur<'int'>('counters');
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NVOX)), () => { Return(); });
      const c = i.bitAnd(uint(NCOL - 1));
      const y = int(i.shiftRight(uint(Math.log2(NCOL))));
      const a = act.element(c).toVar();
      const out = uint(0).toVar();
      If(a.bitAnd(uint(ACT_NEW)).notEqual(uint(0)), () => {
        // Rifting inside a continent stretches continental crust instead of opening an ocean slit: a gap
        // with ≥ 2 continental neighbours fills with RIFT_THIN × their mean thickness (≥ ridge thickness).
        // Repeated gap fills thin geometrically across a widening rift → tapered margins, and only after
        // several cells does it become ocean crust. One-cell ridge slits in continents flickered on the cut
        // faces and nicked continents (B18). Extra mass over the decide kernel's RIDGE_MASS: reservoir.
        const { x: cx, z: cz } = tColXZ(c);
        const sum = uint(0).toVar();
        const nC = uint(0).toVar();
        for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const n = colInfo.element(tColIdx(cx.add(dx), cz.add(dz))).toVar();
          If(n.x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL)).equal(uint(1)), () => { sum.addAssign(n.y); nC.addAssign(1); });
        }
        const meanT = float(sum).div(float(nC).max(1)).div(255).mul(RIFT_THIN).min(RIFT_MAX_THICK).toVar();
        // extra mass comes from the reservoir: no rift fill while it is in debt (same gate as stacking)
        const rift = nC.greaterThanEqual(uint(2)).and(meanT.greaterThanEqual(RIFT_MIN_THICK)).and(this.stackGate.greaterThan(0.5));
        const T = int(select(rift, meanT.max(RIDGE_GABBRO + RIDGE_BASALT), float(RIDGE_GABBRO + RIDGE_BASALT))).toVar();
        If(rift, () => {
          const top = int(float(Y_COMP).add(float(T).mul((1 - RHO_CONT / RHO_MANTLE) * ISO_GAIN)).round());
          const base = top.sub(T);
          out.assign(select(y.lessThan(base), uint(PERI),
            select(y.lessThan(base.add(T.div(3))), uint(GNEISS),
              select(y.lessThan(top.sub(1)), uint(GRAN), select(y.lessThan(top), uint(SED), uint(0))))));
          If(y.equal(int(0)), () => { atomicAdd(ctr.element(CTR_RESERVOIR), T.mul(255).sub(int(RIDGE_MASS)).negate()); });
        }).Else(() => {
          out.assign(select(y.lessThan(int(RIDGE_BASE)), uint(PERI),
            select(y.lessThan(int(RIDGE_BASE + RIDGE_GABBRO)), uint(GAB),
              select(y.lessThan(int(RIDGE_BASE + RIDGE_GABBRO + RIDGE_BASALT)), uint(BAS), uint(0)))));
        });
      }).Else(() => {
        const s = a.bitAnd(uint(0xffff));
        const k = int(a.shiftRight(uint(K_SHIFT)).bitAnd(uint(63)));
        const v = int(a.shiftRight(uint(V_SHIFT)).bitAnd(uint(3))).sub(1);
        const info = colInfo.element(s).x;
        const base = int(info.bitAnd(uint(0xff)));
        const top = int(info.shiftRight(uint(8)).bitAnd(uint(0xff)));
        const up = int(a.shiftRight(uint(UP_SHIFT)).bitAnd(uint(63)));
        // Thrust stacking: the k collided layers enter mid-crust at a height that wanders along the orogen,
        // duplicating the strata just below it; crust below sinks by k - up, crust above rises by up.
        // Repeated collisions at shifting heights read as folded, repeated strata on the cut faces.
        const cx = float(c.bitAnd(uint(NX - 1))), cz = float(c.shiftRight(uint(Math.log2(NX))));
        const f = sin(cx.mul(0.21).add(cz.mul(0.13))).mul(0.18).add(sin(cx.mul(0.057).sub(cz.mul(0.083)).add(1.3)).mul(0.12)).add(0.42);
        const span = iMax(top.sub(base), int(0));
        // (decide never stacks onto a crustless column; hIns stays inside the column even if one slips through)
        const hIns = iMin(base.add(int(float(span).mul(f))), top).toVar(); // source y where the thrust sheet enters
        const dl = v.add(up).sub(k), du = v.add(up);
        const out0 = uint(0).toVar();
        If(y.lessThan(hIns.add(dl)), () => {
          const sy = y.sub(dl);
          out0.assign(select(sy.lessThan(int(0)), uint(PERI), src.element(s.add(uint(iMax(sy, int(0))).mul(uint(NCOL))))));
        }).ElseIf(y.lessThan(hIns.add(du)), () => {
          // duplicated sheet: copy of the k layers under hIns; anything not a full plain crust voxel → GNEISS
          const cy = iMax(base, hIns.sub(k).add(y.sub(hIns).sub(dl)));
          const cv = src.element(s.add(uint(cy).mul(uint(NCOL)))).toVar();
          const m = tMat(cv);
          const plain = tFill(cv).equal(uint(255)).and(m.notEqual(uint(Mat.MAGMA))).and(m.notEqual(uint(Mat.PERIDOTITE))).and(m.notEqual(uint(Mat.AIR)));
          out0.assign(select(plain, cv, uint(GNEISS)));
        }).Else(() => {
          const sy = y.sub(du);
          out0.assign(select(sy.greaterThanEqual(int(NY)), uint(0), src.element(s.add(uint(sy).mul(uint(NCOL))))));
        });
        out.assign(out0);
      });
      dst.element(i).assign(out);
    })().compute(NVOX);
  }

  /** CPU half: advance sub-cell offsets, return per-plate integer shift for this tick. */
  private computeShifts(dtGeo: number, speedMul: number): { shifts: [number, number][]; any: boolean } {
    let any = false;
    const shifts: [number, number][] = [];
    for (const p of this.plates) {
      let sx = 0, sz = 0;
      if (p.alive) {
        p.accum[0] += p.vel[0] * dtGeo * speedMul;
        p.accum[1] += p.vel[1] * dtGeo * speedMul;
        this.travel[p.id * 2]! += p.vel[0] * dtGeo * speedMul;
        this.travel[p.id * 2 + 1]! += p.vel[1] * dtGeo * speedMul;
        // whole cells only; the gather kernel handles any integer shift
        sx = Math.trunc(p.accum[0]); p.accum[0] -= sx;
        sz = Math.trunc(p.accum[1]); p.accum[1] -= sz;
        if (sx || sz) any = true;
      }
      shifts.push([sx, sz]);
    }
    return { shifts, any };
  }

  private writeTable(shifts: [number, number][]): void {
    for (let i = 0; i < MAX_PLATES; i++) this.tableValues[i]!.set(shifts[i]![0], shifts[i]![1], this.plates[i]!.alive ? 1 : 0, 0);
  }

  /**
   * One geo tick. Runs GPU kernels only when some plate crosses a cell or on isostasy ticks.
   * Caller runs the derive pass afterwards when this returns true.
   * Precondition: colInfo fresh (derive ran after the last voxel edit).
   */
  tick(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number, opts: { speedMul: number; isoEvery: number }): boolean {
    for (const p of this.plates) if (p.alive) p.age += dtGeo;
    if (tick % TEC_EVERY !== 0) return false;
    const { shifts, any } = this.computeShifts(dtGeo * TEC_EVERY, opts.speedMul);
    const iso = tick % opts.isoEvery === 0;
    if (!any && !iso) return false;
    this.writeTable(shifts);
    this.ageDt.value = (tick - this.lastRunTick) * dtGeo;
    this.lastRunTick = tick;
    this.isoOn.value = iso ? 1 : 0;
    const f = this.fields;
    if (f.parity('plateId') !== f.parity('crustAge')) throw new Error('Tectonics: plateId/crustAge parity diverged');
    this.runId++;
    const par = f.parity('plateId') as 0 | 1;
    renderer.compute(this.waterGather[par]);
    renderer.compute(this.decide[par]);
    this.runIdU.value = tick >>> 0; // seed by sim tick (saved), not runId (not saved): V2/V13
    renderer.compute(this.quakeK);
    this.voxel.run(renderer);
    renderer.compute(this.waterBack);
    f.swap('plateId');
    f.swap('crustAge');
    return true;
  }

  /**
   * Render continuity: fractional sub-cell offset of each plate (cells, −1..1 on X,Z). Content of a column on
   * plate p truly sits at grid + offset(p); when the sim shifts p by a whole cell the offset drops by that cell
   * in the same step, so rendering at grid + offset moves continuously.
   */
  plateOffsets(out: Float32Array): void {
    for (let i = 0; i < MAX_PLATES; i++) { const p = this.plates[i]!; out[i * 2] = p.alive ? p.accum[0] : 0; out[i * 2 + 1] = p.alive ? p.accum[1] : 0; }
  }

  /**
   * Render continuity: unwrapped distance (cells, X,Z) each plate has moved since this Tectonics was built.
   * Plates only move on tectonics runs (every TEC_EVERY ticks), so the display eases this in real time and
   * draws at accum + (eased − travel): smooth motion even when a run lands every few frames.
   */
  plateTravel(out: Float64Array): void { out.set(this.travel); }

  /** Lower-crust flow. Needs fresh colInfo: call after the post-tectonics derive, then derive again. */
  flow(renderer: THREE.WebGPURenderer): void { this.crustFlow.run(renderer); }

  /** Zero all per-window counters (call right after issuing a stats readback). */
  clearPlateStats(renderer: THREE.WebGPURenderer): void { renderer.compute(this.clearStats); }

  /** Parse a readback of the whole 'counters' buffer. */
  static parseStats(buf: ArrayBuffer): TectonicsStats {
    const a = new Int32Array(buf).subarray(CTR_PLATE);
    const s: TectonicsStats = { area: [], subducted: [], created: [] };
    for (let p = 0; p < MAX_PLATES; p++) { s.area.push(a[p * 4]!); s.subducted.push(a[p * 4 + 1]!); s.created.push(a[p * 4 + 2]!); }
    return s;
  }
}
