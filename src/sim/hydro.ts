// §C pass 8a (hydrology): shallow-water pipe model (Mei et al. 2007 style) on the torus column grid.
//
// Units (dimensionless "substep" time, dx = 1 cell, cell area = 1):
//   water     depth above surfY, voxel-y units (= volume per cell)
//   flux      vec4 outflow volume per substep toward (+x, -x, +z, -z)
//   waterVel  vec2 cells per substep (Mei's ΔW / mean depth), |v| clamped to 1
//   sedSusp   suspended sediment per column, fill units (255 = one voxel of solid)
//
// Per substep, two in-place kernels (no ping-pong needed, each thread writes only its own column):
//   1. flux:  F_k ← max(0, (1-friction)·F_k + kFlow·(H - H_k)),  H = surfY + water.
//             K = min(1, water / ΣF) so a column never ships more than it holds; F ← K·F.
//             Also writes sedRatio = sedSusp / max(water, ΣF): sediment leaves with the water it rides in.
//             Reads: own flux, own+neighbour water/surfY. Writes: own flux, own sedRatio.
//   2. water: water += Σ inflow (neighbours' flux pointing at me) - Σ own outflow (+ test rain).
//             sedSusp moves the same way (sedRatio·flux), velocity from Mei's ΔW.
//             Reads: own+neighbour flux/sedRatio. Writes: own water, sedSusp, waterVel.
// Every unit of water leaving a column is read by exactly one neighbour from the same stored float,
// so flow conserves volume up to f32 rounding of the final add (V4). No clamping of depth (clamps create
// mass); K uses max(water, 0) so float-dust negatives stay inert.
//
// Stability: flux increments don't scale with depth (pipe area fixed), so the scheme is the symplectic
// Euler discretisation of a wave equation with c² = kFlow; stable for kFlow·λmax(=8)/4 ≤ 1 → kFlow ≤ 0.5
// regardless of ocean depth or mountain height. We use 0.25. Friction damps sloshing to quasi-steady.
import type * as THREE from 'three/webgpu';
import { Fn, If, Return, float, uint, int, vec2, vec4, max, clamp, select, instanceIndex, uniform } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { NCOL, NX, NZ } from './layout';
import { tColIdx, tColXZ } from './tslLayout';

export const HYDRO_DEFAULTS = {
  // Overdamped on purpose: geologic ticks want quasi-steady water, not sloshing waves. Steady flux per
  // neighbour ≈ kFlow/friction · Δh = 0.2·Δh (explicit-diffusion stable ≤ 0.25). Low friction made deep
  // oceans ring with persistent waves excited by every tectonic shift.
  kFlow: 0.1,     // flux gain per voxel of head difference per substep
  friction: 0.5,  // fraction of flux lost per substep
  rain: 0,        // TEST-ONLY uniform rain, depth per substep (× 'rainfall' param). Real precip is the climate pass.
  substeps: 4,    // default substeps per geo tick
};

type F = THREE.Node<'float'>;

/** Hydro + erosion owned fields. Call before fields.freeze(). */
export function registerHydroFields(f: GpuFields): void {
  f.add('flux', 'vec4', NCOL);      // pipe outflow per substep (+x,-x,+z,-z), water volume
  f.add('waterVel', 'vec2', NCOL);  // cells per substep
  f.add('sedSusp', 'float', NCOL);  // suspended sediment, fill units (V3 budget member)
  f.add('sedRatio', 'float', NCOL); // scratch: sediment per unit outflow, rebuilt every substep
  f.add('talusOut', 'uint', NCOL);  // scratch (erosion.ts): thermal outflow bytes per direction
}

export interface HydroPass {
  step(renderer: THREE.WebGPURenderer, substeps?: number): void;
  uniforms: {
    kFlow: THREE.UniformNode<'float', number>;
    friction: THREE.UniformNode<'float', number>;
    rain: THREE.UniformNode<'float', number>;
    rainScale: THREE.UniformNode<'float', number>;
  };
}

export function createHydroPass(fields: GpuFields, params: Params): HydroPass {
  const surfY = fields.cur('surfY');
  const water = fields.cur('water');
  const flux = fields.cur<'vec4'>('flux');
  const vel = fields.cur<'vec2'>('waterVel');
  const sed = fields.cur('sedSusp');
  const sedRatio = fields.cur('sedRatio');

  const kFlow = uniform(HYDRO_DEFAULTS.kFlow);
  const friction = uniform(HYDRO_DEFAULTS.friction);
  const rain = uniform(HYDRO_DEFAULTS.rain);
  const rainScale = uniform(params.get('rainfall') as number);
  params.onChange((k, v) => { if (k === 'rainfall') rainScale.value = v as number; });

  const neighbours = () => {
    const { x, z } = tColXZ(instanceIndex);
    return {
      r: tColIdx(x.add(int(1)), z), l: tColIdx(x.sub(int(1)), z),
      u: tColIdx(x, z.add(int(1))), d: tColIdx(x, z.sub(int(1))),
    };
  };

  const fluxK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const n = neighbours();
    const head = (c: THREE.Node<'uint'>) => surfY.element(c).add(max(water.element(c), 0));
    const d = max(water.element(i), 0).toVar();
    const h = head(i).toVar();
    const f = flux.element(i).toVar();
    const damp = float(1).sub(friction);
    const out = (fk: F, c: THREE.Node<'uint'>) => max(fk.mul(damp).add(kFlow.mul(h.sub(head(c)))), 0);
    const o = vec4(out(f.x, n.r), out(f.y, n.l), out(f.z, n.u), out(f.w, n.d)).toVar();
    const sum = o.x.add(o.y).add(o.z).add(o.w);
    const K = clamp(d.div(max(sum, 1e-12)), 0, 1);
    const os = o.mul(K).toVar();
    flux.element(i).assign(os);
    const moved = os.x.add(os.y).add(os.z).add(os.w);
    sedRatio.element(i).assign(sed.element(i).div(max(max(d, moved), 1e-6)));
  })().compute(NCOL);

  const waterK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const n = neighbours();
    const f = flux.element(i).toVar();
    // inflow = neighbour's pipe pointing at me: right neighbour's -x, left's +x, up's -z, down's +z
    const inR = flux.element(n.r).y.toVar();
    const inL = flux.element(n.l).x.toVar();
    const inU = flux.element(n.u).w.toVar();
    const inD = flux.element(n.d).z.toVar();
    const outSum = f.x.add(f.y).add(f.z).add(f.w);
    const inSum = inR.add(inL).add(inU).add(inD);
    const d0 = water.element(i).toVar();
    const d1 = d0.add(inSum.sub(outSum)).add(rain.mul(rainScale)).toVar();
    water.element(i).assign(d1);

    const r = sedRatio.element(i);
    const sIn = sedRatio.element(n.r).mul(inR).add(sedRatio.element(n.l).mul(inL))
      .add(sedRatio.element(n.u).mul(inU)).add(sedRatio.element(n.d).mul(inD));
    sed.element(i).assign(sed.element(i).add(sIn.sub(r.mul(outSum))));

    const dwx = inL.sub(f.y).add(f.x).sub(inR).mul(0.5);
    const dwz = inD.sub(f.w).add(f.z).sub(inU).mul(0.5);
    const dAvg = max(d0, 0).add(max(d1, 0)).mul(0.5);
    const v = vec2(dwx, dwz).div(max(dAvg, 1e-4)).toVar();
    const sp = v.length();
    const vc = select(sp.greaterThan(1), v.div(max(sp, 1e-6)), v);
    vel.element(i).assign(select(dAvg.greaterThan(1e-4), vc, vec2(0, 0)));
  })().compute(NCOL);

  return {
    step(renderer, substeps = HYDRO_DEFAULTS.substeps) {
      for (let s = 0; s < substeps; s++) {
        renderer.compute(fluxK);
        renderer.compute(waterK);
      }
    },
    uniforms: { kFlow, friction, rain, rainScale },
  };
}

// ---------------------------------------------------------------------------------------------
// CPU reference (tests). Same arithmetic in f64 with f32 storage.

export interface HydroCpuState {
  surfY: Float32Array; water: Float32Array; flux: Float32Array /* 4·NCOL */;
  vel: Float32Array /* 2·NCOL */; sed: Float32Array; sedRatio: Float32Array;
}

export function hydroStepCpu(st: HydroCpuState, c: { kFlow: number; friction: number; rain: number } = HYDRO_DEFAULTS): void {
  const { surfY, water, flux, vel, sed, sedRatio } = st;
  const idx = (x: number, z: number) => ((x + NX) % NX) + ((z + NZ) % NZ) * NX;
  const head = (i: number) => surfY[i]! + Math.max(water[i]!, 0);
  const damp = 1 - c.friction;
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const i = idx(x, z);
    const d = Math.max(water[i]!, 0), h = head(i);
    const nb = [idx(x + 1, z), idx(x - 1, z), idx(x, z + 1), idx(x, z - 1)];
    const o = nb.map((n, k) => Math.max(flux[i * 4 + k]! * damp + c.kFlow * (h - head(n)), 0));
    const sum = o[0]! + o[1]! + o[2]! + o[3]!;
    const K = Math.min(1, Math.max(0, d / Math.max(sum, 1e-12)));
    let moved = 0;
    for (let k = 0; k < 4; k++) { flux[i * 4 + k] = o[k]! * K; moved += flux[i * 4 + k]!; }
    sedRatio[i] = sed[i]! / Math.max(d, moved, 1e-6);
  }
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const i = idx(x, z);
    const r = idx(x + 1, z), l = idx(x - 1, z), u = idx(x, z + 1), dn = idx(x, z - 1);
    const f = [flux[i * 4]!, flux[i * 4 + 1]!, flux[i * 4 + 2]!, flux[i * 4 + 3]!];
    const inR = flux[r * 4 + 1]!, inL = flux[l * 4]!, inU = flux[u * 4 + 3]!, inD = flux[dn * 4 + 2]!;
    const outSum = f[0]! + f[1]! + f[2]! + f[3]!;
    const d0 = water[i]!;
    const d1 = d0 + (inR + inL + inU + inD - outSum) + c.rain;
    water[i] = d1;
    sed[i] = sed[i]! + (sedRatio[r]! * inR + sedRatio[l]! * inL + sedRatio[u]! * inU + sedRatio[dn]! * inD - sedRatio[i]! * outSum);
    const dAvg = (Math.max(d0, 0) + Math.max(d1, 0)) * 0.5;
    let vx = (inL - f[1]! + f[0]! - inR) * 0.5 / Math.max(dAvg, 1e-4);
    let vz = (inD - f[3]! + f[2]! - inU) * 0.5 / Math.max(dAvg, 1e-4);
    const sp = Math.hypot(vx, vz);
    if (sp > 1) { vx /= sp; vz /= sp; }
    if (!(dAvg > 1e-4)) { vx = 0; vz = 0; }
    vel[i * 2] = vx; vel[i * 2 + 1] = vz;
  }
}
