// Overlay prep: one small compute pass per active overlay turns sim fields into a per-column vec4 (ovA) that the
// overlay sheet samples bilinearly (smooth per pixel, never blocky). Values are 3×3 smoothed and, where the
// sim field flickers tick to tick, low-pass filtered over real time (no pulsing). Render-only buffers, read-only
// on sim state (V15); ≤ 8 storage buffers per kernel (V23).
//
// ovA layout per overlay (x drives the colour, y is the physical value for the hover readout):
//   continuous  (t 0..1, value, extra, 0)            age, temp, heat, moist, elev; flow: z = open sea mask
//   plates      (plate id, boundary 0|1, closing ratio −1..1, relative speed cells/My)
//   biome       (biome id, 0, 0, 0)
//   activity    (thickness-change t, subduction, ridge, collision)   EMAs over geo time
// ovB (activity only): (crust mass fill units last frame, thickness change EMA layers/My, 0, 0)
import * as THREE from 'three/webgpu';
import { Fn, If, Return, float, uint, vec2, vec4, instanceIndex, instancedArray, uniform, uniformArray, mix, max, length, dot, step, normalize } from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NCOL } from '../sim/layout';
import { tColIdx, tColXZ } from '../sim/tslLayout';
import { ACT_NEW, ACT_SUBDUCT } from '../sim/tectonics';
import { BIOME } from '../sim/biomeModel';
import { MAX_PLATES } from '../sim/worldData';
import { tScale } from './overlayTsl';
import type { OverlayDef } from './defs';

type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;
type I = THREE.Node<'int'>;

/** Orogeny layers field in tecAct (tectonics.ts K_SHIFT, 6 bits). */
const K_SHIFT = 19;
/** Open sea for the discharge overlay: water deeper than this (voxel layers). */
const SEA_DEPTH = 3;
/** Plate relative speeds below this (cells/My) still classify, but by a nudge-free unit direction. */
const EPS_SPEED = 1e-4;

export interface Prep {
  ovA: StorageNode<'vec4'>;
  ovB: StorageNode<'vec4'>;
  u: {
    /** EMA weight of this frame for real-time smoothing (1 = replace). */
    ema: THREE.UniformNode<'float', number>;
    /** EMA weight over geo time (activity) and 1 on reset. */
    emaGeo: THREE.UniformNode<'float', number>;
    geoDt: THREE.UniformNode<'float', number>;
    reset: THREE.UniformNode<'float', number>;
    seaLevel: THREE.UniformNode<'float', number>;
  };
  /** (vx, vz, alive, 0) per plate id, cells/My. */
  plateVel: THREE.Vector4[];
  /** Dispatch the prep of `def` (picks the ping-pong variant bound to the current parity). */
  run(renderer: THREE.WebGPURenderer, def: OverlayDef): void;
}

export function createPrep(fields: GpuFields): Prep {
  // allocated once with the overlays, fixed size (V10)
  const A = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  const B = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  A.setName('overlayA'); B.setName('overlayB');
  const u = { ema: uniform(1), emaGeo: uniform(1), geoDt: uniform(0), reset: uniform(1), seaLevel: uniform(76) };
  const plateVel = Array.from({ length: MAX_PLATES }, () => new THREE.Vector4());
  const plateTable = uniformArray(plateVel, 'vec4');
  const has = (n: string) => fields.names().includes(n);

  /** Kernel over all columns; body(i, x, z) returns the new ovA value. */
  const kernel = (body: (i: U, x: I, z: I) => THREE.Node<'vec4'>) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(instanceIndex);
    A.element(instanceIndex).assign(body(instanceIndex, x, z));
  })().compute(NCOL);

  /** 1-2-1 binomial 3×3 mean of f(dx, dz), wrapping (V1). */
  const blur3 = (f: (dx: number, dz: number) => F): F => {
    let s: F = float(0);
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) s = s.add(f(dx, dz).mul(((dx ? 1 : 2) * (dz ? 1 : 2)) / 16)) as F;
    return s;
  };
  /** Continuous overlay: blurred t (EMA'd), raw value at the column. */
  const continuous = (def: OverlayDef, read: (idx: U) => F, extra: (i: U) => F = () => float(0)) => kernel((i, x, z) => {
    const s = def.bar!.scale;
    const t = blur3((dx, dz) => tScale(s, read(tColIdx(x.add(dx), z.add(dz)))));
    const prev = A.element(i).toVar();
    return vec4(mix(prev.x, t, u.ema), read(i), extra(i), 0);
  });
  const f = <T extends 'float' | 'uint' | 'vec4' | 'uvec2' | 'vec2'>(n: string) => fields.cur<T>(n);
  const pp = <T extends 'float' | 'uint'>(n: string) => fields.pair<T>(n);

  const kernels = new Map<string, THREE.ComputeNode | [THREE.ComputeNode, THREE.ComputeNode]>();
  const build = (def: OverlayDef): THREE.ComputeNode | [THREE.ComputeNode, THREE.ComputeNode] => {
    switch (def.id) {
      case 'age': {
        const [a0, a1] = pp('crustAge');
        return [a0, a1].map((b) => continuous(def, (idx) => b.element(idx) as unknown as F)) as [THREE.ComputeNode, THREE.ComputeNode];
      }
      case 'temp': return continuous(def, (idx) => f('surfTemp').element(idx) as unknown as F);
      case 'heat': return continuous(def, (idx) => f('heatFlow').element(idx) as unknown as F);
      case 'moist': {
        const clim = f<'vec2'>('climAvg');
        return continuous(def, (idx) => (clim.element(idx) as unknown as THREE.Node<'vec2'>).y.div(BIOME.P_REF));
      }
      case 'elev': {
        const surf = f('surfY');
        return continuous(def, (idx) => (surf.element(idx) as unknown as F).sub(u.seaLevel));
      }
      case 'flow': {
        const flux = f<'vec4'>('flux'), water = f('water');
        const q = (idx: U) => { const v = flux.element(idx) as unknown as THREE.Node<'vec4'>; return v.x.add(v.y).add(v.z).add(v.w); };
        return kernel((i) => {
          const sea = step(SEA_DEPTH, water.element(i) as unknown as F);
          const t = tScale(def.bar!.scale, q(i)).mul(float(1).sub(sea));
          const prev = A.element(i).toVar();
          return vec4(mix(prev.x, t, u.ema), q(i), sea, 0);
        });
      }
      case 'biome': {
        const bio = f<'uint'>('biome');
        return kernel((i) => vec4(float(bio.element(i)), 0, 0, 0));
      }
      case 'plates': {
        const [p0, p1] = pp<'uint'>('plateId');
        return [p0, p1].map((pid) => kernel((i, x, z) => {
          const idAt = (dx: number, dz: number) => pid.element(tColIdx(x.add(dx), z.add(dz))) as unknown as U;
          const self = (pid.element(i) as unknown as U).toVar();
          // boundary: any 4-neighbour on another plate; `other` = that plate
          const other = self.toVar(), edge = float(0).toVar();
          for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
            const n = idAt(dx, dz).toVar();
            If(n.notEqual(self), () => { other.assign(n); edge.assign(1); });
          }
          // boundary normal (self → other side) from the 5×5 neighbourhood, nearer cells weigh more
          const nrm = vec2(0, 0).toVar();
          for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
            if (!dx && !dz) continue;
            const w = 1 / (dx * dx + dz * dz);
            nrm.addAssign(vec2(dx * w, dz * w).mul(idAt(dx, dz).notEqual(self).toFloat()));
          }
          const n = normalize(nrm.add(vec2(1e-6, 0)));
          const vs = (plateTable.element(self) as unknown as THREE.Node<'vec4'>).xy;
          const vo = (plateTable.element(other) as unknown as THREE.Node<'vec4'>).xy;
          const rel = vs.sub(vo).toVar();
          const speed = length(rel);
          const closing = dot(rel, n).div(max(speed, EPS_SPEED));
          return vec4(float(self), edge, closing.mul(edge), speed.mul(edge));
        })) as [THREE.ComputeNode, THREE.ComputeNode];
      }
      case 'activity': {
        const act = f<'uint'>('tecAct'), info = f<'uvec2'>('colInfo');
        const s = def.bar2!.scale;
        return kernel((i) => {
          const a = (act.element(i) as unknown as U).toVar();
          const mass = float((info.element(i) as unknown as THREE.Node<'uvec2'>).y);
          const prevB = B.element(i).toVar(), prevA = A.element(i).toVar();
          // thickness change: layers per My since last frame, EMA over geo time; frozen while paused
          const rate = mass.sub(prevB.x).div(255).div(max(u.geoDt, 1e-6)).mul(float(1).sub(u.reset)).mul(step(1e-9, u.geoDt));
          const rateE = mix(prevB.y, rate, u.emaGeo).mul(float(1).sub(u.reset));
          B.element(i).assign(vec4(mass, rateE, 0, 0));
          const sub = a.bitAnd(uint(ACT_SUBDUCT)).notEqual(uint(0)).toFloat();
          const ridge = a.bitAnd(uint(ACT_NEW)).notEqual(uint(0)).toFloat();
          const coll = a.shiftRight(uint(K_SHIFT)).bitAnd(uint(63)).notEqual(uint(0)).toFloat();
          const k = max(u.emaGeo, u.reset);
          return vec4(tScale(s, rateE), mix(prevA.y, sub, k), mix(prevA.z, ridge, k), mix(prevA.w, coll, k));
        });
      }
    }
  };

  return {
    ovA: A, ovB: B, u, plateVel,
    run(renderer, def) {
      if (def.id === 'moist' && !has('climAvg')) return;
      let k = kernels.get(def.id);
      if (!k) { k = build(def); kernels.set(def.id, k); }
      if (Array.isArray(k)) k = k[fields.parity(def.id === 'age' ? 'crustAge' : 'plateId') as 0 | 1];
      renderer.compute(k);
    },
  };
}
