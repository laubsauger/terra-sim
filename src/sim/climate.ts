// §C pass 7 (climate) on GPU: surface temperature, wind bands, evaporation, vapor transport,
// orographic precip / rain shadow, snow, melt, glaciers (T28-T30). Model, units and constants live in
// climateModel.ts; this file mirrors climateColumnCpu / climateTransportCpu op-for-op.
//
// Two kernels per geo tick, each thread writes only its own column (race-free):
//   1. column:    reads own water/vapor/ice, surfY (+4 neighbours for ∇h).
//                 writes own water (−evap +rain +melt), surfTemp, precip, climScratch = (vapor', ice').
//   2. transport: reads climScratch (+4 neighbours), surfY (+4 neighbours).
//                 writes own vapor (flux-form donor-cell advection + eddy diffusion) and ice (glacier flow).
// Face fluxes are evaluated from identical inputs by both columns sharing the face, so transport is
// exactly conservative; the column kernel moves amounts between water/vapor/ice of one column.
// ∴ Σwater + Σvapor + Σice changes only by f32 rounding (V4). Wind is analytic (no field).
// Storage buffers bound: column 7, transport 4 (V23 ≤ 8).
// Note: Dawn/Metal compiles with fast-math (verified: dw - ((w + dw) - w) folds to 0), so exactness
// tricks relying on float rounding identities are unsafe; the water-delta quantum uses integer bit ops.
import type * as THREE from 'three/webgpu';
import { Fn, If, Return, float, int, uint, vec2, max, min, clamp, select, cos, sin, exp, abs, round, floatBitsToUint, uintBitsToFloat, instanceIndex, uniform } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { NCOL, NZ } from './layout';
import { tColIdx, tColXZ, uMax } from './tslLayout';
import { CLIMATE as C, DEFAULT_CLIMATE_UNIFORMS } from './climateModel';

type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;

/** Climate-owned fields. Call before fields.freeze(). */
export function registerClimateFields(f: GpuFields): void {
  f.add('surfTemp', 'float', NCOL);    // °C-like; lapse by ground height above sea level
  f.add('vapor', 'float', NCOL);       // water-eq column vapor, voxel-y units (W_total member, V4)
  f.add('precip', 'float', NCOL);      // rain + snow that fell this tick, voxel-y units
  f.add('ice', 'float', NCOL);         // water-eq ice (snowpack, glaciers, sea ice), voxel-y units (W_total member)
  f.add('climScratch', 'vec2', NCOL);  // scratch: (vapor, ice) after column physics, before transport
  // biome pass (biome.ts)
  f.add('climAvg', 'vec2', NCOL);      // long-term EMA (surfTemp °C, precip per tick)
  f.add('veg', 'float', NCOL);         // vegetation cover 0..1; erosion multiplies erodibility by vegErodibilityFactor
  f.add('biome', 'uint', NCOL);        // Biome id (biomeModel.ts)
}

export interface ClimateUniformNodes {
  seaLevel: THREE.UniformNode<'float', number>;
  iceAge: THREE.UniformNode<'float', number>;
  iceAgeStrength: THREE.UniformNode<'float', number>;
  rainfall: THREE.UniformNode<'float', number>;
}

export interface ClimatePass {
  /** One climate relaxation step. dtGeo is fixed (V12); rates are per tick, so it is not used. */
  step(renderer: THREE.WebGPURenderer, dtGeo?: number): void;
  /** Ice-age forcing 0..1 (event scheduler). Cooling = ICE_AGE_DT · x · iceAgeStrength. */
  setIceAge(x: number): void;
  /** Emergent sea level (voxel y), e.g. from the stats readback. Default Y_SEA_NOMINAL. */
  setSeaLevel(y: number): void;
  uniforms: ClimateUniformNodes;
  dispose(): void;
}

// ---- TSL mirrors of the analytic model (climateModel.ts) ----

/** f32Quantum (climateModel.ts): ulp of the binade of mag, from the exponent bits. */
const tF32Quantum = (mag: F): F => {
  const ex = (floatBitsToUint(mag) as unknown as U).shiftRight(uint(23)).bitAnd(uint(0xff));
  return uintBitsToFloat(uMax(ex, uint(24)).sub(uint(23)).shiftLeft(uint(23))) as unknown as F;
};

/** Latitude φ and hemisphere sign at z position p ∈ [0, NZ). */
function tLatitude(p: F): { phi: F; hemi: F } {
  const d = min(p, float(NZ).sub(p));
  return { phi: d.mul(Math.PI / NZ), hemi: select(p.lessThan(float(NZ / 2)), float(1), float(-1)) };
}
const tWindU = (z: F): F => cos(tLatitude(z).phi.mul(4)).mul(-C.U0);
function tWindV(p: F): F {
  const { phi, hemi } = tLatitude(p);
  return sin(phi.mul(6)).mul(-C.V0).mul(hemi);
}
/** Z face k (wrapped int in [0,NZ)) between rows k-1 and k. */
const tFaceZPos = (k: THREE.Node<'int'>): F => {
  const p = float(k).sub(0.5);
  return select(p.lessThan(0), p.add(NZ), p);
};
/** vaporFlux(c, a, b) = donor-cell + eddy diffusion, a → b. */
const tVaporFlux = (c: F, a: F, b: F): F => select(c.greaterThan(0), c.mul(a), c.mul(b)).add(a.sub(b).mul(C.VAPOR_DIFF));
/** glacierFlux(hA, iA, hB, iB), antisymmetric bit-exactly. */
const tGlacierFlux = (hA: F, iA: F, hB: F, iB: F): F => {
  const dh = hA.sub(hB);
  return select(dh.greaterThan(0),
    min(dh.mul(C.GLACIER_K), C.GLACIER_MAX).mul(iA),
    min(dh.negate().mul(C.GLACIER_K), C.GLACIER_MAX).mul(iB).negate());
};

export function createClimatePass(fields: GpuFields, params: Params): ClimatePass {
  const surfY = fields.cur('surfY');
  const water = fields.cur('water');
  const vapor = fields.cur('vapor');
  const ice = fields.cur('ice');
  const surfTemp = fields.cur('surfTemp');
  const precip = fields.cur('precip');
  const scratch = fields.cur<'vec2'>('climScratch');

  const u: ClimateUniformNodes = {
    seaLevel: uniform(DEFAULT_CLIMATE_UNIFORMS.seaLevel),
    iceAge: uniform(DEFAULT_CLIMATE_UNIFORMS.iceAge),
    iceAgeStrength: uniform(params.get('iceAgeStrength') as number),
    rainfall: uniform(params.get('rainfall') as number),
  };
  const off = params.onChange((k, v) => {
    if (k === 'rainfall') u.rainfall.value = v as number;
    else if (k === 'iceAgeStrength') u.iceAgeStrength.value = v as number;
  });

  const nb = () => {
    const { x, z } = tColXZ(instanceIndex);
    return {
      x, z,
      l: tColIdx(x.sub(int(1)), z), r: tColIdx(x.add(int(1)), z),
      d: tColIdx(x, z.sub(int(1))), up: tColIdx(x, z.add(int(1))),
    };
  };

  const columnK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const n = nb();
    const zf = float(n.z);
    const sy = surfY.element(i).toVar();
    const w = water.element(i).toVar();
    const q = vapor.element(i).toVar();
    const ic = ice.element(i).toVar();
    // temperature: latitude + ice age + lapse above sea level
    const tSea = cos(zf.mul((2 * Math.PI) / NZ)).mul(C.T_AMP).add(C.T_BASE).sub(u.iceAge.mul(u.iceAgeStrength).mul(C.ICE_AGE_DT));
    const t = tSea.sub(max(sy.sub(u.seaLevel), 0).mul(C.LAPSE)).toVar();
    const cap = exp(t.mul(C.CAP_K)).mul(C.CAP0).toVar();
    // evaporation
    const wp = max(w, 0);
    const wet = min(wp.div(C.OCEAN_DEPTH), 1);
    const fT = clamp(t.sub(C.EVAP_T0).div(C.EVAP_TSPAN), 0, C.EVAP_FMAX);
    const e = min(u.rainfall.mul(C.EVAP_K).mul(fT).mul(wet).mul(max(float(1).sub(q.div(cap)), 0)), wp.mul(0.5)).toVar();
    const q1 = q.add(e).toVar();
    // forced ascent over orography (ocean flat at sea level)
    const oro = (c: U) => max(surfY.element(c), u.seaLevel);
    const gx = oro(n.r).sub(oro(n.l)).mul(0.5);
    const gz = oro(n.up).sub(oro(n.d)).mul(0.5);
    const lift = max(tWindU(zf).mul(gx).add(tWindV(zf).mul(gz)), 0);
    const rh = min(q1.div(cap), 1);
    const p = min(rh.mul(C.RAIN_K).mul(q1).add(max(q1.sub(cap), 0).mul(C.SAT_K)).add(lift.mul(C.ORO_K).mul(q1)), q1).toVar();
    const snow = p.mul(clamp(float(C.SNOW_T1).sub(t).div(C.SNOW_T1 - C.SNOW_T0), 0, 1)).toVar();
    const rain = p.sub(snow);
    const i1 = ic.add(snow).toVar();
    const calve = select(w.greaterThan(C.OCEAN_DEPTH), max(i1.sub(C.SEA_ICE_MAX), 0).mul(C.CALVE_K), float(0));
    const melt = min(max(t, 0).mul(C.MELT_K).add(i1.mul(C.BASAL_K)).add(calve), i1).toVar();
    // water takes Δw quantized to its own ulp grid (exact add); remainder stays in vapor (see climateColumnCpu)
    const dw = rain.add(melt).sub(e).toVar();
    const qu = tF32Quantum(abs(w).add(abs(dw))).toVar();
    const dwq = round(dw.div(qu)).mul(qu).toVar();
    water.element(i).assign(w.add(dwq));
    scratch.element(i).assign(vec2(q1.sub(p).add(dw.sub(dwq)), i1.sub(melt)));
    surfTemp.element(i).assign(t);
    precip.element(i).assign(p);
  })().compute(NCOL);

  const transportK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const n = nb();
    const s0 = scratch.element(i).toVar();
    const sL = scratch.element(n.l).toVar(), sR = scratch.element(n.r).toVar();
    const sD = scratch.element(n.d).toVar(), sU = scratch.element(n.up).toVar();
    // vapor: X faces share the row's zonal wind; Z faces evaluated at canonical wrapped face index
    const cx = tWindU(float(n.z)).toVar();
    const cDown = tWindV(tFaceZPos(n.z)).toVar();
    const cUp = tWindV(tFaceZPos(n.z.add(int(1)).bitAnd(int(NZ - 1)))).toVar();
    const fl = tVaporFlux(cx, sL.x, s0.x), fr = tVaporFlux(cx, s0.x, sR.x);
    const fd = tVaporFlux(cDown, sD.x, s0.x), fu = tVaporFlux(cUp, s0.x, sU.x);
    vapor.element(i).assign(s0.x.add(fl).sub(fr).add(fd).sub(fu));
    // glaciers: ice flows down the ice-surface gradient
    const h = surfY.element(i).add(s0.y).toVar();
    const hn = (c: U, s: THREE.Node<'vec2'>) => surfY.element(c).add(s.y);
    const out = tGlacierFlux(h, s0.y, hn(n.l, sL), sL.y).add(tGlacierFlux(h, s0.y, hn(n.r, sR), sR.y))
      .add(tGlacierFlux(h, s0.y, hn(n.d, sD), sD.y)).add(tGlacierFlux(h, s0.y, hn(n.up, sU), sU.y));
    ice.element(i).assign(s0.y.sub(out));
  })().compute(NCOL);

  return {
    step(renderer) {
      renderer.compute(columnK);
      renderer.compute(transportK);
    },
    setIceAge(x) { u.iceAge.value = Math.min(Math.max(x, 0), 1); },
    setSeaLevel(y) { u.seaLevel.value = y; },
    uniforms: u,
    dispose() { off(); },
  };
}
