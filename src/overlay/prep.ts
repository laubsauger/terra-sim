// Overlay prep: one small compute pass per active overlay turns sim fields into a per-column vec4 (ovA) that the
// overlay sheet samples bilinearly (smooth per pixel, never blocky). Values are 3×3 smoothed and, where the
// sim field flickers tick to tick, low-pass filtered over real time (no pulsing). Render-only buffers, read-only
// on sim state (V15); ≤ 8 storage buffers per kernel (V23).
//
// ovA layout per overlay (x drives the colour, y is the physical value for the hover readout):
//   continuous  (t 0..1, value, 0, 0)                age, temp, heat, moist, elev
//   flow        (river log net discharge 0..1, 3×3 max of it, sea, lake)   ovB.x = net discharge
//   plates      separate buffers T / TN (below), shared with the Tectonics layer
//   biome       (5×5 majority biome id, 0, 0, 0)
//   activity    (thickness-change t, subduction, ridge, collision)   geo-time EMA / peak-hold
// ovB activity: (5×5 mean crust mass last frame, thickness change EMA layers/My, 0, 0)
import * as THREE from 'three/webgpu';
import { Fn, If, Return, float, uint, vec2, vec3, vec4, floor, instanceIndex, instancedArray, uniform, uniformArray, mix, max, length, dot, step, normalize } from 'three/tsl';
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
/** Discharge overlay: standing water above sea level deeper than this is a lake, not a river. */
const LAKE_DEPTH = 0.6;
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
  /** Plate boundary buffers (see header). */
  tec: { T: StorageNode<'vec4'>; TN: StorageNode<'vec4'> };
  /** Refresh the plate boundary buffers from the current plateId + plateVel. */
  runPlates(renderer: THREE.WebGPURenderer): void;
  /** Dispatch the prep of `def` (picks the ping-pong variant bound to the current parity). */
  run(renderer: THREE.WebGPURenderer, def: OverlayDef): void;
}

export function createPrep(fields: GpuFields): Prep {
  // allocated once with the overlays, fixed size (V10)
  const A = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  const B = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;
  A.setName('overlayA'); B.setName('overlayB');
  // plate boundaries: shared by the Plates data overlay and the always-available Tectonics layer
  const TM = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>; // (5×5 majority plate id, 0, 0, 0)
  const T = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>;  // (plate id, boundary 0|1, closing ratio, relative speed)
  const TN = instancedArray(NCOL, 'vec4') as unknown as StorageNode<'vec4'>; // (boundary normal x, z, shear ratio along (−nz, nx), 0)
  TM.setName('overlayTecMajor'); T.setName('overlayTec'); TN.setName('overlayTecN');
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
  /**
   * 5×5 majority of a categorical uint field → out.x (drops 1-2 column speckles: plate slivers at
   * boundaries, flickering beach/ice cells). Votes are only counted where the window is mixed.
   */
  const majority = (src: StorageNode<'uint'>, out: StorageNode<'vec4'>) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(instanceIndex);
    const win: U[] = [];
    for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) win.push((src.element(tColIdx(x.add(dx), z.add(dz))) as unknown as U).toVar());
    const inner = [12, 7, 11, 13, 17, 6, 8, 16, 18]; // centre first: ties keep the column's own id
    const best = win[12]!.toVar();
    const mixed = win.reduce((s: THREE.Node<'bool'>, w) => s.or(w.notEqual(win[12]!)), win[0]!.notEqual(win[12]!));
    If(mixed, () => {
      const bestN = float(-1).toVar();
      for (const j of inner) {
        const n = win.reduce((s: F, w) => s.add(w.equal(win[j]!).toFloat()) as F, float(0));
        If(n.greaterThan(bestN), () => { bestN.assign(n); best.assign(win[j]!); });
      }
    });
    out.element(instanceIndex).assign(vec4(float(best), 0, 0, 0));
  })().compute(NCOL);

  /** Continuous overlay: blurred t (EMA'd), raw value at the column. */
  const continuous = (def: OverlayDef, read: (idx: U) => F, extra: (i: U) => F = () => float(0)) => kernel((i, x, z) => {
    const s = def.bar!.scale;
    const t = blur3((dx, dz) => tScale(s, read(tColIdx(x.add(dx), z.add(dz)))));
    const prev = A.element(i).toVar();
    return vec4(mix(prev.x, t, u.ema), read(i), extra(i), 0);
  });
  const f = <T extends 'float' | 'uint' | 'vec4' | 'uvec2' | 'vec2'>(n: string) => fields.cur<T>(n);
  const pp = <T extends 'float' | 'uint'>(n: string) => fields.pair<T>(n);

  // ---- plate boundaries (T, TN): majority id, then boundary flag / other plate / normal over 7×7 ----
  let plateK: [THREE.ComputeNode, THREE.ComputeNode, THREE.ComputeNode] | null = null;
  const buildPlates = (): [THREE.ComputeNode, THREE.ComputeNode, THREE.ComputeNode] => {
    const [p0, p1] = pp<'uint'>('plateId');
    const idAt = (x: I, z: I, dx: number, dz: number) => floor((TM.element(tColIdx(x.add(dx), z.add(dz))) as unknown as THREE.Node<'vec4'>).x.add(0.5));
    const offs: [number, number][] = [];
    for (let dz = -3; dz <= 3; dz++) for (let dx = -3; dx <= 3; dx++) if (dx || dz) offs.push([dx, dz]);
    offs.sort((a, b) => a[0] ** 2 + a[1] ** 2 - (b[0] ** 2 + b[1] ** 2));
    const edgeK = Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(instanceIndex);
      const self = idAt(x, z, 0, 0).toVar();
      const out = vec4(self, 0, 0, 0).toVar(), outN = vec4(0).toVar();
      const edge = float(0).toVar();
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) edge.assign(max(edge, idAt(x, z, dx, dz).notEqual(self).toFloat()));
      // boundary columns only: other plate (nearest differing id) and the boundary normal over 7×7
      If(edge.greaterThan(0.5), () => {
        const other = self.toVar(), found = float(0).toVar();
        const nrm = vec2(0, 0).toVar();
        for (const [dx, dz] of offs) {
          const id = idAt(x, z, dx, dz).toVar();
          const diff = id.notEqual(self).toFloat();
          If(diff.greaterThan(0.5).and(found.lessThan(0.5)), () => { other.assign(id); found.assign(1); });
          nrm.addAssign(vec2(dx, dz).mul(diff.div(dx * dx + dz * dz)));
        }
        const n = normalize(nrm.add(vec2(1e-6, 0))).toVar();
        const vs = (plateTable.element(uint(self)) as unknown as THREE.Node<'vec4'>).xy;
        const vo = (plateTable.element(uint(other)) as unknown as THREE.Node<'vec4'>).xy;
        const rel = vs.sub(vo).toVar();
        const speed = max(length(rel), EPS_SPEED);
        out.assign(vec4(self, 1, dot(rel, n).div(speed), length(rel)));
        outN.assign(vec4(n.x, n.y, dot(rel, vec2(n.y.negate(), n.x)).div(speed), 0));
      });
      T.element(instanceIndex).assign(out);
      TN.element(instanceIndex).assign(outN);
    })().compute(NCOL);
    return [majority(p0, TM), majority(p1, TM), edgeK];
  };

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
        // land rivers only: r = log discharge 0..1 (0 in the sea and in lakes), rmax = 3×3 max of r (line width),
        // sea / lake masks. Discharge itself → ovB.x for the readout.
        const flux = f<'vec4'>('flux'), water = f('water'), surf = f('surfY');
        const s = def.bar!.scale;
        const col = (idx: U) => {
          const v = (flux.element(idx) as unknown as THREE.Node<'vec4'>).toVar();
          const wd = (water.element(idx) as unknown as F).toVar();
          const top = (surf.element(idx) as unknown as F).add(wd);
          // sea: any water whose surface sits at sea level (shelves too); lake: deep standing water above it
          const sea = step(0.05, wd).mul(step(top, u.seaLevel.add(0.6)));
          const lake = step(LAKE_DEPTH, wd).mul(float(1).sub(sea));
          // net discharge: outflow vector (+x − −x, +z − −z); pipes sloshing in a still pond cancel out
          const q = length(vec2(v.x.sub(v.y), v.z.sub(v.w))).toVar();
          return { q, sea, lake, r: tScale(s, q).mul(float(1).sub(max(sea, lake))) };
        };
        return kernel((i, x, z) => {
          const c = col(i);
          const r = c.r.toVar(), rmax = c.r.toVar(), lakes = c.lake.toVar();
          for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dz) continue;
            const n = col(tColIdx(x.add(dx), z.add(dz)));
            rmax.assign(max(rmax, n.r));
            lakes.addAssign(n.lake);
          }
          // a lake needs most of its 3×3 wet (single-cell ponds would speckle the map)
          const lake = c.lake.mul(step(4.5, lakes));
          const prev = A.element(i).toVar();
          B.element(i).assign(vec4(c.q, 0, 0, 0));
          return vec4(mix(prev.x, r, u.ema), mix(prev.y, rmax, u.ema), c.sea, lake);
        });
      }
      case 'biome': return majority(f<'uint'>('biome'), A);
      case 'plates': throw new Error('plates prep runs through runPlates()');
      case 'activity': {
        const act = f<'uint'>('tecAct'), info = f<'uvec2'>('colInfo');
        const s = def.bar2!.scale;
        const b5 = [1, 4, 6, 4, 1];
        return kernel((i, x, z) => {
          const a = (act.element(i) as unknown as U).toVar();
          // crust mass, 5×5 binomial mean: the rate is a regional signal, not per-column erosion noise
          let m: F = float(0);
          for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
            m = m.add(float((info.element(tColIdx(x.add(dx), z.add(dz))) as unknown as THREE.Node<'uvec2'>).y).mul(b5[dx + 2]! * b5[dz + 2]! / 256)) as F;
          }
          const mass = m.toVar();
          const prevB = B.element(i).toVar(), prevA = A.element(i).toVar();
          // thickness change: layers per My since last frame, EMA over geo time; frozen while paused
          const live = float(1).sub(u.reset).mul(step(1e-9, u.geoDt));
          const rate = mass.sub(prevB.x).div(255).div(max(u.geoDt, 1e-6));
          const rateE = mix(prevB.y, rate, u.emaGeo.mul(live)).mul(float(1).sub(u.reset));
          B.element(i).assign(vec4(mass, rateE, 0, 0));
          // activity marks: peak-hold with decay over geo time (flags only show on plate-crossing ticks)
          // flags of this column and (weaker) its 4-neighbours: trench / ridge lines read continuous
          const flags = (v: U) => vec3(v.bitAnd(uint(ACT_SUBDUCT)).notEqual(uint(0)).toFloat(), v.bitAnd(uint(ACT_NEW)).notEqual(uint(0)).toFloat(),
            v.shiftRight(uint(K_SHIFT)).bitAnd(uint(63)).notEqual(uint(0)).toFloat());
          const fl = flags(a).toVar();
          for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) fl.assign(max(fl, flags(act.element(tColIdx(x.add(dx), z.add(dz))) as unknown as U).mul(0.5)));
          const sub = fl.x, ridge = fl.y, coll = fl.z;
          const keep = float(1).sub(u.emaGeo).mul(float(1).sub(u.reset));
          return vec4(tScale(s, rateE), max(prevA.y.mul(keep), sub), max(prevA.z.mul(keep), ridge), max(prevA.w.mul(keep), coll));
        });
      }
    }
  };

  return {
    ovA: A, ovB: B, u, plateVel,
    tec: { T, TN },
    runPlates(renderer) {
      plateK ??= buildPlates();
      renderer.compute(plateK[fields.parity('plateId') as 0 | 1]);
      renderer.compute(plateK[2]);
    },
    run(renderer, def) {
      if (def.id === 'plates') { this.runPlates(renderer); return; }
      if (def.id === 'moist' && !has('climAvg')) return;
      let k = kernels.get(def.id);
      if (!k) { k = build(def); kernels.set(def.id, k); }
      if (Array.isArray(k)) k = k[fields.parity('crustAge') as 0 | 1];
      renderer.compute(k);
    },
  };
}
