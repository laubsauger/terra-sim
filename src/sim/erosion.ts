// §C pass 8b (hydrology): hydraulic erosion/deposition (scaled by kGeo) + thermal (talus) erosion.
//
// Mass units: 1 voxel of solid = 255 fill units. sedSusp stores suspended sediment in the same fill units,
// so crust mass (Σ fill of crust voxels) + Σ sedSusp is exactly conserved (V3): every exchange is an integer
// number of fill units, and the kernels credit/debit exactly what they actually removed/added to voxels.
// Fractional exchange rates become integers by deterministic dithered rounding (hash of column + `tick`),
// so tiny rates still act on average without extra state.
//
// vox editing: all kernels here write the CURRENT vox buffer in place (ReaderKernel, no swap). One thread per
// column, and a thread only writes voxels of its own column → no races, no full-grid copy. Only the top one or
// two crust voxels of a column are touched. Only the top voxel may be partial (fill < 255): erosion eats the
// top voxel down and turns it into AIR at 0, deposition first tops up a partial crust top voxel (keeping its
// mat — sediment filling rock pore space) and only then stacks a new SEDIMENT voxel above (≤ NY-1).
// PERIDOTITE / MAGMA are not crust (derive's crustMass) and never eroded or topped up; y = 0 is never eroded.
//
// Order in step(): hydraulic kernel → derive (internal, so thermal sees fresh surfY/top after hydraulic edits)
// → talus kernel (gather: each column computes its outflow bytes to 4 neighbours from the same pre-pass data)
// → thermal apply (each column applies net = Σ neighbours' bytes toward it - Σ own bytes). The caller re-runs
// derive after step(). Precondition: surfY is fresh (derived from current vox) when step() starts.
//
// kGeo: hydro substeps model minutes of flow, one geo tick is dtGeo = 0.05 My. kGeo is the fraction of the
// capacity deficit (C - sedSusp) picked up per geo tick: exchange = kGeo · erosionRate · erodibility · (C - s).
// Capacity C = Kc · √q · slope · min(depth, dMax) (fill units per column), q = |ΔW| = discharge per unit
// width (water volume per substep per cell, Mei's ΔW from the pipe fluxes). q is fixed by rain × catchment in
// steady state, so C does not depend on how fast the damped pipe model moves water (the substep velocity
// q/depth does). Carried concentration C/depth ∝ √q · S, so the saturated sediment flux ∝ q^1.5 · S grows
// faster than discharge: converging flow paths incise, sheet-flow flanks don't (channelisation).
// Vegetation: vegF = 1 - VEG_ERODIBILITY_K · veg multiplies both the detachment fraction (with MAT_ERODIBILITY
// of the top mat) and the capacity (roots bind soil, cover slows overland flow). Capacity matters because
// channels are mostly transport-limited here: scaling only the detachment fraction left forested troughs
// cutting 93% as fast as bare ones.
// The exchange fraction min(1, kGeo·erosionRate·erodibility) never overshoots capacity. With the damped pipe
// model, incision is transport-limited (load must be carried off), hence the large Kc and hydro's sedSpeed.
// Calibration (erosion.spec channel test, sim cadence: 2 substeps/tick, erosion every 2nd tick with kGeo × 2,
// realistic rain 3.5e-4 voxel/tick): a trough draining a ~40×140-cell sandstone slope (S 0.2) cuts ≈ 0.6
// voxel per 10 My (mean over a 17-cell band) at erosionRate 4, ≈ 0.5 per 10 My at the default 1 (detachment
// saturates at high rates: transport-limited), sheet-flow flanks ~6× less; vegetated (veg 1) ≈ 0.4× bare.
// Steep mountain channels hit maxExchange = 64 fill units per erosion step.
import type * as THREE from 'three/webgpu';
import { Fn, If, Loop, Break, Return, float, int, uint, vec2, max, min, clamp, floor, instanceIndex, uniform, uniformArray, select, ceil, exp } from 'three/tsl';
import { ReaderKernel, type GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { NCOL, NY, Mat, MAT_COUNT, MAT_ERODIBILITY, FLAG_CONTINENTAL } from './layout';
import { tMat, tFill, tAge, tFlags, tPack, tVoxIdx, tColIdx, tColXZ, uMin, uMax } from './tslLayout';
import { createDerivePass } from './derive';
import { VEG_ERODIBILITY_K } from './biomeModel';

export const EROSION_DEFAULTS = {
  kGeo: 0.25,        // capacity-deficit fraction exchanged per geo tick (see header)
  kCap: 6000,        // capacity: fill units per (√(voxel·cell/substep) · voxel/cell · voxel depth)
  minSlope: 0.05,    // voxel/cell floor so flat rivers still carry some load
  maxSlope: 4,
  dMax: 4,           // depth (voxels) above which capacity stops growing (deep water barely moves anyway)
  submarineTalusMul: 2, // talus multiplier where the sender is under > 1 voxel of water
  dDeep: 2,          // standing water deeper than this loses capacity: sediment settles in lakes/oceans (deltas, shelves)
  dDecay: 1.5,       // e-folding depth of that loss
  kDep: 0.5,         // fraction of excess suspended load settling per tick
  maxExchange: 64,   // fill units per column per tick, hydraulic
  talus: 2.5,        // layers per cell before slumping (≈ 7° real at 250 m layers, 20 km cells); 1.2 (≈0.9°) flattened every margin into the sea (B9)
  thermalRate: 0.5,  // fraction of talus excess relaxed per tick (× 'thermalErosion'), clamped to 1
  // fill units per direction per tick (byte-packed). 63 (0.25 layer) let convergent fronts rebuild 50-layer walls
  // faster than they slumped; receivers take up to 4 new voxels, senders give from their top 4 crust voxels
  thermalMaxDir: 255,
};

type U = THREE.Node<'uint'>;
type I = THREE.Node<'int'>;
type Vox = THREE.StorageBufferNode<'uint'>;

/** Crust erodibility per mat (PERIDOTITE/MAGMA forced to 0: not crust, V3 budget). */
const ERODIBILITY = MAT_ERODIBILITY.map((e, m) => (m === Mat.PERIDOTITE || m === Mat.MAGMA ? 0 : e));
if (ERODIBILITY.length !== MAT_COUNT) throw new Error('erosion: MAT_ERODIBILITY length mismatch');

const isCrust = (m: U) => m.notEqual(uint(Mat.AIR)).and(m.notEqual(uint(Mat.PERIDOTITE))).and(m.notEqual(uint(Mat.MAGMA)));

/** Integer hash → [0,1) dither (deterministic per column, tick, salt). */
export function tDither(a: U, tick: U, salt: number): THREE.Node<'float'> {
  const h = a.mul(uint(747796405)).add(tick.mul(uint(2891336453))).add(uint(salt * 0x9e3779b9 >>> 0)).toVar();
  h.assign(h.bitXor(h.shiftRight(uint(16))).mul(uint(0x7feb352d)));
  h.assign(h.bitXor(h.shiftRight(uint(15))).mul(uint(0x846ca68b)));
  h.assign(h.bitXor(h.shiftRight(uint(16))));
  return float(h.shiftRight(uint(8))).mul(1 / 16777216);
}

/** float ≥ 0 → uint(floor(v + dither)), capped. */
export const quantize = (v: THREE.Node<'float'>, dither: THREE.Node<'float'>, cap: number) =>
  uMin(uint(floor(max(v, 0).add(dither))), uint(cap));

/** Scan down for the top solid voxel. Returns y (-1 if column empty) and its packed value. */
export function scanTop(vox: Vox, x: I, z: I) {
  const topY = int(-1).toVar();
  const topV = uint(0).toVar();
  Loop({ start: int(NY - 1), end: int(0), condition: '>=' }, ({ i }) => {
    const v = vox.element(tVoxIdx(x, i, z));
    If(tMat(v).notEqual(uint(Mat.AIR)), () => { topY.assign(i); topV.assign(v); Break(); });
  });
  return { topY, topV };
}

/** Remove up to n (≤ 1020) fill units from the column's crust top, never below y = 1. Returns units removed. */
export function removeTop(vox: Vox, x: I, z: I, topY: I, n: U): U {
  const rem = n.toVar();
  const y = topY.toVar();
  Loop({ start: int(0), end: int(5), condition: '<' }, () => {
    If(rem.greaterThan(uint(0)).and(y.greaterThanEqual(int(1))), () => {
      const idx = tVoxIdx(x, y, z).toVar();
      const v = vox.element(idx).toVar();
      const m = tMat(v).toVar();
      If(isCrust(m).not(), () => { y.assign(int(-1)); }).Else(() => {
        const f = tFill(v).toVar();
        const t = uMin(rem, f);
        const nf = f.sub(t).toVar();
        rem.subAssign(t);
        If(nf.equal(uint(0)), () => {
          vox.element(idx).assign(uint(0)); // AIR
          y.subAssign(int(1));
        }).Else(() => {
          vox.element(idx).assign(tPack(m, nf, tAge(v), tFlags(v)));
        });
      });
    });
  });
  return n.sub(rem);
}

/** Add n (≤ 1020) fill units: top up a partial crust top voxel, rest as new SEDIMENT voxels above. Returns units added. */
function addTop(vox: Vox, x: I, z: I, topY: I, topV: U, n: U): U {
  const rem = n.toVar();
  If(rem.greaterThan(uint(0)).and(topY.greaterThanEqual(int(0))), () => {
    const m = tMat(topV);
    const f = tFill(topV).toVar();
    If(isCrust(m).and(f.lessThan(uint(255))), () => {
      const t = uMin(rem, uint(255).sub(f)).toVar();
      vox.element(tVoxIdx(x, topY, z)).assign(tPack(m, f.add(t), tAge(topV), tFlags(topV)));
      rem.subAssign(t);
    });
  });
  for (let k = 1; k <= 4; k++) {
    If(rem.greaterThan(uint(0)).and(topY.add(int(k)).lessThan(int(NY))), () => {
      const t = uMin(rem, uint(255)).toVar();
      vox.element(tVoxIdx(x, topY.add(int(k)), z)).assign(
        tPack(uint(Mat.SEDIMENT), t, uint(0), tFlags(topV).bitAnd(uint(FLAG_CONTINENTAL))));
      rem.subAssign(t);
    });
  }
  return n.sub(rem);
}

export interface ErosionPass {
  step(renderer: THREE.WebGPURenderer): void;
  uniforms: {
    kGeo: THREE.UniformNode<'float', number>;
    kCap: THREE.UniformNode<'float', number>;
    kDep: THREE.UniformNode<'float', number>;
    talus: THREE.UniformNode<'float', number>;
    thermalRate: THREE.UniformNode<'float', number>;
    erosionScale: THREE.UniformNode<'float', number>;
    thermalScale: THREE.UniformNode<'float', number>;
    /** Dither seed; auto-increments per step. Orchestrator should set it from the global tick (V2, V13). */
    tick: THREE.UniformNode<'uint', number>;
  };
}

export function createErosionPass(fields: GpuFields, params: Params): ErosionPass {
  const D = EROSION_DEFAULTS;
  const surfY = fields.cur('surfY');
  const water = fields.cur('water');
  const flux = fields.cur<'vec4'>('flux');
  const veg = fields.cur('veg');
  const sed = fields.cur('sedSusp');
  const talusOut = fields.cur<'uint'>('talusOut');

  const kGeo = uniform(D.kGeo);
  const kCap = uniform(D.kCap);
  const kDep = uniform(D.kDep);
  const talus = uniform(D.talus);
  const thermalRate = uniform(D.thermalRate);
  const erosionScale = uniform(params.get('erosionRate') as number);
  const thermalScale = uniform(params.get('thermalErosion') as number);
  const tick = uniform(0, 'uint') as unknown as THREE.UniformNode<'uint', number>;
  params.onChange((k, v) => {
    if (k === 'erosionRate') erosionScale.value = v as number;
    if (k === 'thermalErosion') thermalScale.value = v as number;
  });
  const erodTable = uniformArray(ERODIBILITY, 'float');

  // storage buffers: vox, surfY, water, flux, sedSusp, veg = 6 (V23)
  const hydraulic = new ReaderKernel<'uint'>(fields, 'vox', (vox) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const { x, z } = tColXZ(i);
    const sY = (dx: number, dz: number) => surfY.element(tColIdx(x.add(int(dx)), z.add(int(dz))));
    const grad = vec2(sY(1, 0).sub(sY(-1, 0)), sY(0, 1).sub(sY(0, -1))).mul(0.5);
    const slope = clamp(grad.length(), D.minSlope, D.maxSlope);
    // discharge per unit width: Mei's ΔW from own outflows and neighbours' inflows (same as hydro's velocity)
    const f = flux.element(i).toVar();
    const inR = flux.element(tColIdx(x.add(int(1)), z)).y;
    const inL = flux.element(tColIdx(x.sub(int(1)), z)).x;
    const inU = flux.element(tColIdx(x, z.add(int(1)))).w;
    const inD = flux.element(tColIdx(x, z.sub(int(1)))).z;
    const q = vec2(inL.sub(f.y).add(f.x).sub(inR), inD.sub(f.w).add(f.z).sub(inU)).mul(0.5).length();
    // capacity grows with flow depth in rivers but collapses in deep standing water; without this the
    // deep ocean kept eroding its floor and suspended sediment piled up forever (B6)
    const wd = water.element(i).toVar();
    const dEff = clamp(wd, 0, D.dMax).mul(exp(max(wd.sub(D.dDeep), 0).div(-D.dDecay)));
    const vegF = float(1).sub(clamp(veg.element(i), 0, 1).mul(VEG_ERODIBILITY_K)).toVar();
    const C = kCap.mul(vegF).mul(q.sqrt()).mul(slope).mul(dEff).toVar();
    const s = sed.element(i).toVar();
    const { topY, topV } = scanTop(vox, x, z);
    const m = tMat(topV);
    const erod = (select(topY.greaterThanEqual(int(0)), erodTable.element(m), float(0)) as THREE.Node<'float'>).mul(vegF);
    const dither = tDither(i, tick, 1);
    If(C.greaterThan(s), () => {
      // fraction of the deficit picked up this step, ≤ 1 (never overshoot capacity)
      const n = quantize(min(kGeo.mul(erosionScale).mul(erod), 1).mul(C.sub(s)), dither, D.maxExchange);
      If(n.greaterThan(uint(0)), () => {
        const got = removeTop(vox, x, z, topY, n);
        sed.element(i).assign(s.add(float(got)));
      });
    }).Else(() => {
      // settle excess load; never more than is (integrally) suspended
      const want = min(kDep.mul(s.sub(C)), floor(max(s, 0)));
      const n = quantize(want, dither, D.maxExchange);
      If(n.greaterThan(uint(0)), () => {
        const put = addTop(vox, x, z, topY, topV, n);
        sed.element(i).assign(s.sub(float(put)));
      });
    });
  })().compute(NCOL));

  const derive = createDerivePass(fields);

  // Thermal pass 1 (gather source): outflow bytes toward (+x,-x,+z,-z) from the same derived surfY.
  // A sender caps Σ out to what its top crust voxels can give; it skips receivers near the ceiling so every
  // byte sent is guaranteed to be accepted → apply is exact. storage buffers: vox, surfY, talusOut = 3 (V23)
  const talusK = new ReaderKernel<'uint'>(fields, 'vox', (vox) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const { x, z } = tColXZ(i);
    const h = surfY.element(i).toVar();
    // submarine slopes stand steeper (real continental slopes ≈ 4° ≈ 5 layers/cell here); a subaerial talus
    // underwater slumped every continental margin into the ocean (B9)
    const tal = select(water.element(i).greaterThan(1), talus.mul(D.submarineTalusMul), talus).toVar();
    const nbs: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const ex = nbs.map(([dx, dz]) => {
      const hn = surfY.element(tColIdx(x.add(int(dx)), z.add(int(dz))));
      return select(hn.lessThan(NY - 5.01), max(h.sub(hn).sub(tal), 0), float(0)).toVar(); // room for 4 new voxels
    });
    const exSum = ex[0]!.add(ex[1]!).add(ex[2]!).add(ex[3]!).toVar();
    const exMax = max(max(ex[0]!, ex[1]!), max(ex[2]!, ex[3]!));
    const rate = clamp(thermalRate.mul(thermalScale), 0, 1);
    const total = rate.mul(0.5).mul(exMax).mul(255).toVar();
    const q = ex.map((e, k) => quantize(select(exSum.greaterThan(0), total.mul(e).div(max(exSum, 1e-9)), float(0)),
      tDither(i.mul(uint(4)).add(uint(k)), tick, 2), D.thermalMaxDir).toVar());
    const qSum = q[0]!.add(q[1]!).add(q[2]!).add(q[3]!).toVar();
    // availability, same rules as removeTop: the contiguous top crust voxels (up to 4), all y ≥ 1
    const topY = int(ceil(h)).sub(int(1)).toVar();
    const avail = uint(0).toVar();
    const ok = uint(1).toVar();
    for (let k = 0; k < 4; k++) {
      const y = topY.sub(int(k));
      If(ok.equal(uint(1)).and(y.greaterThanEqual(int(1))), () => {
        const v = vox.element(tVoxIdx(x, y, z)).toVar();
        If(isCrust(tMat(v)), () => { avail.addAssign(tFill(v)); }).Else(() => { ok.assign(0); });
      }).Else(() => { ok.assign(0); });
    }
    If(qSum.greaterThan(avail), () => {
      for (const qk of q) qk.assign(qk.mul(avail).div(uMax(qSum, uint(1))));
    });
    talusOut.element(i).assign(q[0]!.bitOr(q[1]!.shiftLeft(uint(8))).bitOr(q[2]!.shiftLeft(uint(16))).bitOr(q[3]!.shiftLeft(uint(24))));
  })().compute(NCOL));

  // Thermal pass 2: net = Σ neighbour bytes toward me - Σ own bytes. storage buffers: vox, talusOut = 2 (V23)
  const byte = (v: U, k: number) => v.shiftRight(uint(8 * k)).bitAnd(uint(0xff));
  const thermalApply = new ReaderKernel<'uint'>(fields, 'vox', (vox) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const { x, z } = tColXZ(i);
    const own = talusOut.element(i).toVar();
    const out = byte(own, 0).add(byte(own, 1)).add(byte(own, 2)).add(byte(own, 3));
    // +x neighbour sends toward me on its -x byte (1), -x neighbour on +x (0), +z on -z (3), -z on +z (2)
    const inn = byte(talusOut.element(tColIdx(x.add(int(1)), z)), 1)
      .add(byte(talusOut.element(tColIdx(x.sub(int(1)), z)), 0))
      .add(byte(talusOut.element(tColIdx(x, z.add(int(1)))), 3))
      .add(byte(talusOut.element(tColIdx(x, z.sub(int(1)))), 2));
    const inV = inn.toVar();
    const outV = out.toVar();
    If(inV.notEqual(outV), () => {
      const { topY, topV } = scanTop(vox, x, z);
      If(outV.greaterThan(inV), () => { removeTop(vox, x, z, topY, outV.sub(inV)); })
        .Else(() => { addTop(vox, x, z, topY, topV, inV.sub(outV)); });
    });
  })().compute(NCOL));

  return {
    step(renderer) {
      hydraulic.run(renderer);
      derive.run(renderer);
      talusK.run(renderer);
      thermalApply.run(renderer);
      tick.value = (tick.value + 1) >>> 0;
    },
    uniforms: { kGeo, kCap, kDep, talus, thermalRate, erosionScale, thermalScale, tick },
  };
}
