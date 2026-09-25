// §T.20 mantle field + §T.21 crust temperature (§C pass 2). GPU half; calibration in mantleModel.ts.
//
// Mantle (64×64×32, vec4 = T °C, velocity xyz in mantle cells/My):
//   velocity = procedural convection rolls (slowly drifting phase) + buoyancy B·(T − Tref) + plume conduit rise.
//   T ← semi-Lagrangian advection (trilinear, torus-wrapped X/Z, clamped Y) + explicit 6-point diffusion
//       + plume conduits relaxing toward Tref + excess + cold slab injection at the top where tectonics
//       consumed oceanic slabs (slabIn counts, written by magma.afterTectonics) + weak relaxation to Tref.
//   Bottom layer held at tBot. Two substeps per update (A→B, B→A) so buffer 0 is always current when step()
//   returns: render may bind fields.pair('mantle')[0] once.
// heatFlow (per column, mW/m²) = q0·mantleHeat + qPerC·(Ttop − tTop), bilinear on the top mantle layer.
//   Hotspot melt keys off heatFlow; plates drift over fixed plumes → hotspot chains.
// Crust temperature (128×128×64 f32, °C): each cell relaxes toward the column geotherm
//   Tgeo = surfTemp + (heatFlow / kCond)·depth (capped), plus diffusion, with MAGMA cells pinned hot and air
//   cells pinned to surfTemp. A 3D field (not a per-column formula) so chambers get thermal halos and burial /
//   uplift lags thermally — the inputs diagenesis/metamorphism (T26) and cut-face tint want. Half-res f32 is
//   4 MB and one update is ~1M cells × 9 reads: cheap when time-sliced. Also two iterations per update
//   (buffer 0 current after step()).
// Cost: mantle every MANTLE.every ticks (offset from tectonics), crust temp every CRUST_T.every ticks offset by
// half a period (V19).
// Race-free: every thread writes only its own cell (ping-pong for all neighbour reads). No float atomics (V2).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Return, float, int, uint, vec4, max, min, clamp, floor, sin, cos, exp, select, instanceIndex,
  uniform, uniformArray, atomicLoad, atomicStore,
} from 'three/tsl';
import { ReaderKernel, type GpuFields, type StorageNode } from '../core/gpu';
import type { Params } from '../core/params';
import { PCG32 } from '../core/rng';
import { NX, NZ, NCOL, Mat } from './layout';
import { tMat, tVoxIdx, tColIdx, tColXZ, iMin, iMax } from './tslLayout';
import type { WorldData } from './worldData';
import {
  MX, MZ, MY, MCELLS, M_SCALE, CX, CZ, CY, CT_CELLS, MANTLE, CRUST_T, MAX_PLUMES, PlumeSet, plumesFromWorld,
  plumeStrength, initMantleTemps, type Plume, type PlumeOptions,
} from './mantleModel';

type F = THREE.Node<'float'>;
type I = THREE.Node<'int'>;

/** Mantle-owned fields. Call before fields.freeze(). */
export function registerMantleFields(f: GpuFields): void {
  f.add('mantle', 'vec4', MCELLS, { pingPong: true }); // (T °C, vx, vy, vz mantle cells/My); read pair[0]
  f.add('heatFlow', 'float', NCOL);                    // mW/m², from the top mantle layer
  f.add('crustTemp', 'float', CT_CELLS, { pingPong: true }); // °C half-res; read pair[0]
  f.add('slabIn', 'int', MX * MZ, { atomic: true });   // subducted columns per top mantle cell since last update
}

export interface MantleOptions extends PlumeOptions {
  /** Explicit initial plumes (tests). Default: under worldgen hotspot chambers + seeded extras. */
  plumes?: Plume[];
}

const PI = Math.PI;

export class MantlePass {
  readonly plumes: PlumeSet;
  private plumeValues = Array.from({ length: MAX_PLUMES }, () => new THREE.Vector4());
  private plumeTable = uniformArray(this.plumeValues, 'vec4');
  private mantleK: [THREE.ComputeNode, THREE.ComputeNode];
  private heatK: THREE.ComputeNode;
  private crustK: [ReaderKernel<'uint'>, ReaderKernel<'uint'>];
  readonly uniforms = {
    dt: uniform(0.1),          // My per mantle substep
    phase: uniform(0),         // roll phase, rad (CPU f64 geoTime · ω mod 2π, V11)
    heat: uniform(1),          // 'mantleHeat'
    rollAmp: uniform(MANTLE.rollAmp as number), // procedural roll amplitude (tests set 0 for exact symmetry)
    ctRelax: uniform(CRUST_T.relax),
  };

  constructor(private fields: GpuFields, params: Params, world: WorldData, opts: MantleOptions = {}) {
    const rng = new PCG32(world.seed, 0x3a47e);
    this.plumes = new PlumeSet(rng, opts.plumes ?? plumesFromWorld(world, rng), opts);
    const u = this.uniforms;
    u.heat.value = params.get('mantleHeat') as number;
    params.onChange((k, v) => { if (k === 'mantleHeat') u.heat.value = v as number; });
    const [m0, m1] = fields.pair<'vec4'>('mantle');
    this.mantleK = [this.buildMantle(m0, m1), this.buildMantle(m1, m0)];
    this.heatK = this.buildHeatFlow(m0);
    const [c0, c1] = fields.pair('crustTemp');
    this.crustK = [
      new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildCrust(vox, c0, c1)),
      new ReaderKernel<'uint'>(fields, 'vox', (vox) => this.buildCrust(vox, c1, c0)),
    ];
    // initial state: reference profile with developed conduits; crust temp set by init()
    const t = initMantleTemps(this.plumes.plumes, u.heat.value);
    for (const which of [0, 1] as const) {
      const a = fields.cpuArray('mantle', which) as Float32Array;
      for (let i = 0; i < MCELLS; i++) { a[i * 4] = t[i]!; a[i * 4 + 1] = 0; a[i * 4 + 2] = 0; a[i * 4 + 3] = 0; }
    }
    fields.setParity('mantle', 0);
    fields.setParity('crustTemp', 0);
    fields.markDirty('mantle');
    this.writePlumes();
  }

  /** Heat flow + crust geotherm from the initial mantle. Needs fresh surfY (run derive first). */
  init(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.heatK);
    this.uniforms.ctRelax.value = 1; // snap to geotherm
    this.crustStep(renderer);
    this.uniforms.ctRelax.value = CRUST_T.relax;
    for (let k = 0; k < 8; k++) this.crustStep(renderer); // grow chamber halos a little
  }

  private writePlumes(): void {
    const ps = this.plumes.plumes;
    for (let i = 0; i < MAX_PLUMES; i++) {
      const p = ps[i];
      if (p) this.plumeValues[i]!.set(p.x, p.z, p.r, plumeStrength(p));
      else this.plumeValues[i]!.set(0, 0, 1, 0);
    }
  }

  private crustStep(renderer: THREE.WebGPURenderer): void {
    if (this.fields.parity('crustTemp') !== 0) throw new Error('MantlePass: crustTemp parity must be 0 between steps');
    this.crustK[0].run(renderer);
    this.crustK[1].run(renderer);
  }

  /**
   * §C pass 2. Time-sliced (V19): mantle on tick % every == 1 (off the tectonics tick, TEC_EVERY = 4 runs on 0),
   * crust temp half a period later.
   */
  step(renderer: THREE.WebGPURenderer, tick: number, dtGeo: number): void {
    const f = this.fields;
    if (tick % MANTLE.every === 1) {
      const dt = MANTLE.every * dtGeo;
      this.plumes.step(dt);
      this.writePlumes();
      this.uniforms.dt.value = dt / 2;
      this.uniforms.phase.value = (tick * dtGeo * MANTLE.rollOmega) % (2 * PI);
      if (f.parity('mantle') !== 0) throw new Error('MantlePass: mantle parity must be 0 between steps');
      renderer.compute(this.mantleK[0]); f.swap('mantle');
      renderer.compute(this.mantleK[1]); f.swap('mantle');
      renderer.compute(this.heatK);
    }
    if (tick % CRUST_T.every === (1 + (CRUST_T.every >> 1)) % CRUST_T.every) this.crustStep(renderer);
  }

  private buildMantle(src: StorageNode<'vec4'>, dst: StorageNode<'vec4'>): THREE.ComputeNode {
    // storage buffers: src, dst, slabIn = 3 (V23)
    const slabIn = this.fields.cur<'int'>('slabIn');
    const { dt, phase, heat, rollAmp } = this.uniforms;
    const table = this.plumeTable;
    const M = MANTLE;
    const LOGX = Math.log2(MX), LOGXZ = Math.log2(MX * MZ);
    const mIdx = (x: I, y: I, z: I) => uint(x.bitAnd(int(MX - 1)).add(z.bitAnd(int(MZ - 1)).mul(int(MX))).add(iMin(iMax(y, int(0)), int(MY - 1)).mul(int(MX * MZ))));
    const T = (x: I, y: I, z: I) => src.element(mIdx(x, y, z)).x;
    const tref = (y: F) => float(M.tTop).add(float(M.tBot - M.tTop).mul(float(MY - 1).sub(y)).div(MY - 1));
    return Fn(() => {
      const id = instanceIndex;
      If(id.greaterThanEqual(uint(MCELLS)), () => { Return(); });
      const mx = int(id.bitAnd(uint(MX - 1))), mz = int(id.shiftRight(uint(LOGX)).bitAnd(uint(MZ - 1))), my = int(id.shiftRight(uint(LOGXZ)));
      const T0 = src.element(id).x.toVar();
      const fy = float(my);
      const ref = tref(fy).toVar();
      // procedural rolls: two orthogonal families of cells, 2 per torus period, drifting phase
      const sy = sin(fy.add(0.5).mul(PI / MY)), cy = cos(fy.add(0.5).mul(PI / MY));
      const ax = float(mx).add(0.5).mul((4 * PI) / MX).add(phase);
      const az = float(mz).add(0.5).mul((4 * PI) / MZ).sub(phase.mul(0.7));
      const vx = rollAmp.mul(cy).mul(cos(ax)).toVar();
      const vz = rollAmp.mul(cy).mul(cos(az)).toVar();
      const vy = rollAmp.mul(sy).mul(sin(ax).add(sin(az))).mul(0.5).add(T0.sub(ref).mul(M.buoyancy)).toVar();
      // plume conduits (fixed in the mantle frame = grid frame)
      const px = float(mx).add(0.5).mul(M_SCALE), pz = float(mz).add(0.5).mul(M_SCALE);
      const hot = float(0).toVar();
      Loop(MAX_PLUMES, ({ i }) => {
        const p = table.element(i) as unknown as THREE.Node<'vec4'>;
        If(p.w.greaterThan(0), () => {
          const dx = px.sub(p.x).abs().toVar();
          const dz = pz.sub(p.y).abs().toVar();
          dx.assign(min(dx, float(NX).sub(dx)));
          dz.assign(min(dz, float(NZ).sub(dz)));
          const t = float(1).sub(dx.mul(dx).add(dz.mul(dz)).div(p.z.mul(p.z))).max(0);
          hot.assign(max(hot, t.mul(t).mul(p.w)));
        });
      });
      vy.addAssign(hot.mul(M.plumeRise));
      // semi-Lagrangian back-trace, trilinear
      const bx = float(mx).sub(vx.mul(dt)), by = clamp(fy.sub(vy.mul(dt)), 0, MY - 1), bz = float(mz).sub(vz.mul(dt));
      const x0 = int(floor(bx)), y0 = int(floor(by)), z0 = int(floor(bz));
      const tx = bx.sub(floor(bx)), ty = by.sub(floor(by)), tz = bz.sub(floor(bz));
      const lerp = (a: F, b: F, t: F) => a.add(b.sub(a).mul(t));
      const layer = (y: I) => lerp(lerp(T(x0, y, z0), T(x0.add(1), y, z0), tx), lerp(T(x0, y, z0.add(1)), T(x0.add(1), y, z0.add(1)), tx), tz);
      // advect the anomaly (potential temperature), not the absolute value: moving along the reference gradient
      // must not create heat, or rising parcels import deep heat to the top and the field runs away
      const adv = lerp(layer(y0), layer(y0.add(1)), ty).sub(tref(by)).add(ref).toVar();
      // explicit diffusion (Neumann in Y via clamped index)
      const lap = T(mx.add(1), my, mz).add(T(mx.sub(1), my, mz)).add(T(mx, my, mz.add(1))).add(T(mx, my, mz.sub(1)))
        // top: ghost cell holds the reference (zero anomaly) → lithosphere conducts plume/slab anomalies away
        .add(select(my.equal(int(MY - 1)), tref(fy.add(1)), T(mx, my.add(1), mz))).add(T(mx, my.sub(1), mz)).sub(T0.mul(6));
      const Tn = adv.add(lap.mul(float(M.kappa).mul(dt))).toVar();
      // plume conduit heating (not in the fixed bottom layer)
      const aP = float(1).sub(exp(dt.div(-M.plumeTau)));
      Tn.addAssign(ref.add(hot.mul(M.plumeExcess).mul(heat)).sub(Tn).mul(aP.mul(hot)));
      // slab: top-layer thread owns the count (reads + clears; the other substep sees 0)
      If(my.equal(int(MY - 1)), () => {
        const c = mx.add(mz.mul(int(MX)));
        const n = atomicLoad(slabIn.element(c)).toVar();
        If(n.greaterThan(int(0)), () => {
          Tn.addAssign(ref.sub(M.slabCold).sub(Tn).mul(min(float(n).mul(M.slabK), 1)));
          atomicStore(slabIn.element(c), int(0));
        });
      });
      Tn.addAssign(ref.sub(Tn).mul(float(1).sub(exp(dt.div(-M.bgTau)))));
      const out = select(my.equal(int(0)), float(M.tBot), Tn);
      dst.element(id).assign(vec4(out, vx, vy, vz));
    })().compute(MCELLS);
  }

  private buildHeatFlow(mantle: StorageNode<'vec4'>): THREE.ComputeNode {
    // storage buffers: mantle, heatFlow = 2
    const heatFlow = this.fields.cur('heatFlow');
    const { heat } = this.uniforms;
    const top = (MY - 1) * MX * MZ;
    const Tt = (x: I, z: I) => mantle.element(uint(x.bitAnd(int(MX - 1)).add(z.bitAnd(int(MZ - 1)).mul(int(MX))).add(int(top)))).x;
    return Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(i);
      // column centre in mantle-cell coordinates (cell centres at k + 0.5)
      const fx = float(x).add(0.5).div(M_SCALE).sub(0.5), fz = float(z).add(0.5).div(M_SCALE).sub(0.5);
      const x0 = int(floor(fx)), z0 = int(floor(fz));
      const tx = fx.sub(floor(fx)), tz = fz.sub(floor(fz));
      const a = Tt(x0, z0).add(Tt(x0.add(1), z0).sub(Tt(x0, z0)).mul(tx));
      const b = Tt(x0, z0.add(1)).add(Tt(x0.add(1), z0.add(1)).sub(Tt(x0, z0.add(1))).mul(tx));
      const Ttop = a.add(b.sub(a).mul(tz));
      heatFlow.element(i).assign(max(float(MANTLE.q0).mul(heat).add(Ttop.sub(MANTLE.tTop).mul(MANTLE.qPerC)), MANTLE.qMin));
    })().compute(NCOL);
  }

  private buildCrust(vox: StorageNode<'uint'>, src: StorageNode, dst: StorageNode): THREE.ComputeNode {
    // storage buffers: vox, surfY, heatFlow, surfTemp, src, dst = 6 (V23)
    const surfY = this.fields.cur('surfY');
    const heatFlow = this.fields.cur('heatFlow');
    const surfTemp = this.fields.cur('surfTemp');
    const { ctRelax } = this.uniforms;
    const C = CRUST_T;
    const LOGX = Math.log2(CX), LOGXZ = Math.log2(CX * CZ);
    const cIdx = (x: I, y: I, z: I) => uint(x.bitAnd(int(CX - 1)).add(z.bitAnd(int(CZ - 1)).mul(int(CX))).add(iMin(iMax(y, int(0)), int(CY - 1)).mul(int(CX * CZ))));
    const T = (x: I, y: I, z: I) => src.element(cIdx(x, y, z));
    return Fn(() => {
      const id = instanceIndex;
      If(id.greaterThanEqual(uint(CT_CELLS)), () => { Return(); });
      const cx = int(id.bitAnd(uint(CX - 1))), cz = int(id.shiftRight(uint(LOGX)).bitAnd(uint(CZ - 1))), cy = int(id.shiftRight(uint(LOGXZ)));
      const vx = cx.mul(2), vz = cz.mul(2), vy = cy.mul(2);
      const col = tColIdx(vx, vz);
      const ts = surfTemp.element(col).toVar();
      const depth = surfY.element(col).sub(float(vy).add(1)).toVar();
      const m0 = tMat(vox.element(tVoxIdx(vx, vy, vz)));
      const m1 = tMat(vox.element(tVoxIdx(vx.add(1), vy.add(1), vz.add(1))));
      const T0 = T(cx, cy, cz).toVar();
      const out = float(0).toVar();
      If(depth.lessThan(0), () => { out.assign(ts); })
        .ElseIf(m0.equal(uint(Mat.MAGMA)).or(m1.equal(uint(Mat.MAGMA))), () => { out.assign(float(C.tMagma)); })
        .Else(() => {
          const geo = min(ts.add(heatFlow.element(col).div(C.kCond).mul(depth)), C.tMax);
          const lap = T(cx.add(1), cy, cz).add(T(cx.sub(1), cy, cz)).add(T(cx, cy, cz.add(1))).add(T(cx, cy, cz.sub(1)))
            .add(T(cx, cy.add(1), cz)).add(T(cx, cy.sub(1), cz)).sub(T0.mul(6));
          const d = T0.add(lap.mul(C.kappa));
          out.assign(d.add(geo.sub(d).mul(ctRelax)));
        });
      dst.element(id).assign(out);
    })().compute(CT_CELLS);
  }
}

