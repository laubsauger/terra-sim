// §T.18 plate lifecycle: split (rift), merge (suture), absorb tiny plates. Keeps plate count in [3,12] (V6).
// GPU: a stats kernel per window (exact area, torus centroid, continental contact matrix) and a relabel
// kernel that applies one CPU-chosen operation. CPU decides from the previous window's readback, so
// every choice is a pure function of deterministic GPU counters and the seeded rng (V2).
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, int, uint, instanceIndex, Return, select, sin, cos, uniform, uniformArray, atomicAdd } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL, NX, NZ } from './layout';
import { tColIdx, tColXZ } from './tslLayout';
import { MAX_PLATES, type Plate } from './worldData';
import { COL_CONTINENTAL } from './derive';
import { CTR_AREA, CTR_CENT, CTR_CONTACT } from './fields';
import type { PCG32 } from '../core/rng';

export const MIN_PLATES = 3;
export const MAX_ALIVE = 12;
export const ABSORB_AREA = Math.round(NCOL * 0.012);
export const SPLIT_AREA = Math.round(NCOL * 0.3);
export const SPLIT_COOLDOWN_MY = 40;
export const SUTURE_CONTACT = 48;     // continental boundary cells
export const SUTURE_WINDOWS = 5;      // sustained this many windows
export const RIFT_SPEED = 1.2;        // cells/My each side after a split
export const SUTURE_MIN_AGE = 30;     // My; young rift plates never re-suture

export interface LifecycleStats {
  area: number[];
  centroid: [number, number][]; // cells
  contact: number[][];          // continental contact p→q
}

export type LifecycleOp =
  | { kind: 'split'; plate: number; into: number; nx: number; nz: number; cx: number; cz: number }
  | { kind: 'merge'; from: number; into: number }
  | { kind: 'absorb'; plate: number }
  | { kind: 'kill'; plate: number };

const OP_NONE = 0, OP_SPLIT = 1, OP_REMAP = 2, OP_ABSORB = 3;

export class Lifecycle {
  private statsKernels: [THREE.ComputeNode, THREE.ComputeNode];
  private relabel: [THREE.ComputeNode, THREE.ComputeNode];
  private opKind = uniform(0);
  private opPlate = uniform(0);
  private opInto = uniform(0);
  private splitN = uniform(new THREE.Vector2());
  private splitC = uniform(new THREE.Vector2());
  private remapValues = Array.from({ length: MAX_PLATES }, (_, i) => new THREE.Vector4(i, 0, 0, 0));
  private remap = uniformArray(this.remapValues, 'vec4');
  private sutureRun = new Map<string, number>();
  private lastSplitMy = -Infinity;
  readonly log: { my: number; op: LifecycleOp }[] = [];

  constructor(private fields: GpuFields) {
    const [pid0, pid1] = fields.pair<'uint'>('plateId');
    const [age0, age1] = fields.pair('crustAge');
    this.statsKernels = [this.buildStats(pid0), this.buildStats(pid1)];
    this.relabel = [this.buildRelabel(pid0, pid1, age0, age1), this.buildRelabel(pid1, pid0, age1, age0)];
  }

  private buildStats(p: THREE.StorageBufferNode<'uint'>): THREE.ComputeNode {
    const colInfo = this.fields.cur<'uvec2'>('colInfo');
    const ctr = this.fields.cur<'int'>('counters');
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const id = p.element(c).toVar();
      atomicAdd(ctr.element(id.add(CTR_AREA)), int(1));
      const ax = float(x).mul((2 * Math.PI) / NX), az = float(z).mul((2 * Math.PI) / NZ);
      const cb = id.mul(4).add(CTR_CENT);
      atomicAdd(ctr.element(cb), int(cos(ax).mul(256).round()));
      atomicAdd(ctr.element(cb.add(1)), int(sin(ax).mul(256).round()));
      atomicAdd(ctr.element(cb.add(2)), int(cos(az).mul(256).round()));
      atomicAdd(ctr.element(cb.add(3)), int(sin(az).mul(256).round()));
      const cont = colInfo.element(c).x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL));
      If(cont.equal(uint(1)), () => {
        // +x and +z neighbours only: each boundary edge counted once per direction pair
        for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
          const n = tColIdx(x.add(dx), z.add(dz));
          const nid = p.element(n);
          const ncont = colInfo.element(n).x.shiftRight(uint(16)).bitAnd(uint(COL_CONTINENTAL));
          If(nid.notEqual(id).and(ncont.equal(uint(1))), () => {
            atomicAdd(ctr.element(id.mul(MAX_PLATES).add(nid).add(CTR_CONTACT)), int(1));
            atomicAdd(ctr.element(nid.mul(MAX_PLATES).add(id).add(CTR_CONTACT)), int(1));
          });
        }
      });
    })().compute(NCOL);
  }

  private buildRelabel(pidCur: THREE.StorageBufferNode<'uint'>, pidNext: THREE.StorageBufferNode<'uint'>,
    ageCur: THREE.StorageBufferNode<'float'>, ageNext: THREE.StorageBufferNode<'float'>): THREE.ComputeNode {
    const { opKind, opPlate, opInto, splitN, splitC, remap } = this;
    return Fn(() => {
      const c = instanceIndex;
      If(c.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(c);
      const id = pidCur.element(c).toVar();
      const out = id.toVar();
      If(opKind.equal(OP_REMAP), () => {
        out.assign(uint((remap.element(id) as unknown as THREE.Node<'vec4'>).x));
      }).ElseIf(opKind.equal(OP_SPLIT).and(id.equal(uint(opPlate))), () => {
        // wrapped offset from centroid, then a wobbly line so rifts are not ruler-straight
        const hx = float(NX / 2), hz = float(NZ / 2);
        const dx = float(x).sub(splitC.x).add(hx).mod(NX).sub(hx);
        const dz = float(z).sub(splitC.y).add(hz).mod(NZ).sub(hz);
        const along = dx.mul(splitN.y.negate()).add(dz.mul(splitN.x));
        const wob = sin(along.mul(0.09)).mul(7).add(sin(along.mul(0.23).add(1.7)).mul(3));
        const side = dx.mul(splitN.x).add(dz.mul(splitN.y)).add(wob);
        out.assign(select(side.greaterThan(0), uint(opInto), id));
      }).ElseIf(opKind.equal(OP_ABSORB).and(id.equal(uint(opPlate))), () => {
        // take the first neighbouring plate (fixed order → deterministic); interior cells wait for later windows
        const found = uint(0).toVar();
        Loop(4, ({ i }) => {
          const dx = select(i.equal(int(0)), int(1), select(i.equal(int(1)), int(-1), int(0)));
          const dz = select(i.equal(int(2)), int(1), select(i.equal(int(3)), int(-1), int(0)));
          const nid = pidCur.element(tColIdx(x.add(dx), z.add(dz)));
          If(found.equal(uint(0)).and(nid.notEqual(id)), () => { out.assign(nid); found.assign(1); });
        });
      });
      pidNext.element(c).assign(out);
      ageNext.element(c).assign(ageCur.element(c));
    })().compute(NCOL);
  }

  /** Accumulate this window's lifecycle stats into counters (call before the window readback). */
  gatherStats(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.statsKernels[this.fields.parity('plateId') as 0 | 1]);
  }

  static parseStats(buf: ArrayBuffer): LifecycleStats {
    const a = new Int32Array(buf);
    const s: LifecycleStats = { area: [], centroid: [], contact: [] };
    for (let p = 0; p < MAX_PLATES; p++) {
      s.area.push(a[CTR_AREA + p]!);
      const b = CTR_CENT + p * 4;
      const ax = Math.atan2(a[b + 1]!, a[b]!), az = Math.atan2(a[b + 3]!, a[b + 2]!);
      s.centroid.push([((ax / (2 * Math.PI)) * NX + NX) % NX, ((az / (2 * Math.PI)) * NZ + NZ) % NZ]);
      const row: number[] = [];
      for (let q = 0; q < MAX_PLATES; q++) row.push(a[CTR_CONTACT + p * MAX_PLATES + q]!);
      s.contact.push(row);
    }
    return s;
  }

  /** Pure decision: at most one op per window. `forceSplit` lets the Wilson controller request a rift. */
  decide(plates: Plate[], s: LifecycleStats, geoMy: number, rng: PCG32, forceSplit?: number): LifecycleOp | null {
    for (const p of plates) if (p.alive && s.area[p.id] === 0) return { kind: 'kill', plate: p.id };
    const alive = plates.filter((p) => p.alive);
    const n = alive.length;
    // suture: sustained continental contact between two plates
    const seen = new Set<string>();
    for (const p of alive) for (const q of alive) {
      if (q.id <= p.id) continue;
      const key = `${p.id}:${q.id}`;
      seen.add(key);
      // suture needs a real collision: plates approaching each other, neither a fresh rift sibling
      const dx = wrapDelta(s.centroid[q.id]![0] - s.centroid[p.id]![0], NX);
      const dz = wrapDelta(s.centroid[q.id]![1] - s.centroid[p.id]![1], NZ);
      const closing = (q.vel[0] - p.vel[0]) * dx + (q.vel[1] - p.vel[1]) * dz < 0;
      const mature = p.age >= SUTURE_MIN_AGE && q.age >= SUTURE_MIN_AGE;
      const run = (s.contact[p.id]![q.id]! >= SUTURE_CONTACT && closing && mature) ? (this.sutureRun.get(key) ?? 0) + 1 : 0;
      this.sutureRun.set(key, run);
    }
    for (const k of [...this.sutureRun.keys()]) if (!seen.has(k)) this.sutureRun.delete(k);
    if (n > MIN_PLATES) {
      for (const [key, run] of this.sutureRun) {
        if (run < SUTURE_WINDOWS) continue;
        const [a, b] = key.split(':').map(Number) as [number, number];
        const [from, into] = s.area[a]! < s.area[b]! ? [a, b] : [b, a];
        this.sutureRun.delete(key);
        return { kind: 'merge', from, into };
      }
      const tiny = alive.filter((p) => s.area[p.id]! < ABSORB_AREA).sort((p, q) => s.area[p.id]! - s.area[q.id]! || p.id - q.id)[0];
      if (tiny) return { kind: 'absorb', plate: tiny.id };
    }
    if (n < MAX_ALIVE && geoMy - this.lastSplitMy >= SPLIT_COOLDOWN_MY) {
      let target = forceSplit;
      if (target === undefined) {
        const big = alive.filter((p) => s.area[p.id]! > SPLIT_AREA).sort((p, q) => s.area[q.id]! - s.area[p.id]! || p.id - q.id)[0];
        target = big?.id;
      }
      const into = plates.find((p) => !p.alive)?.id;
      if (target !== undefined && into !== undefined && plates[target]?.alive) {
        const ang = rng.nextFloat() * Math.PI * 2;
        const [cx, cz] = s.centroid[target]!;
        return { kind: 'split', plate: target, into, nx: Math.cos(ang), nz: Math.sin(ang), cx, cz };
      }
    }
    return null;
  }

  // save/load (T50). sutureRun keeps insertion order: decide() merges the first qualifying pair.
  getState(): LifecycleState {
    return {
      sutureRun: [...this.sutureRun],
      lastSplitMy: Number.isFinite(this.lastSplitMy) ? this.lastSplitMy : null,
      log: this.log.map((e) => ({ my: e.my, op: { ...e.op } })),
    };
  }

  loadState(s: LifecycleState): void {
    if (!Array.isArray(s?.sutureRun) || !Array.isArray(s.log)) throw new Error('Lifecycle.loadState: invalid state');
    this.sutureRun = new Map(s.sutureRun);
    this.lastSplitMy = s.lastSplitMy ?? -Infinity;
    this.log.length = 0;
    for (const e of s.log) this.log.push({ my: e.my, op: { ...e.op } });
  }

  /** Apply op on GPU (relabel) and CPU (plate table). */
  apply(renderer: THREE.WebGPURenderer, plates: Plate[], op: LifecycleOp, geoMy: number): void {
    this.log.push({ my: geoMy, op });
    this.opKind.value = OP_NONE;
    for (let i = 0; i < MAX_PLATES; i++) this.remapValues[i]!.x = i;
    if (op.kind === 'kill') { plates[op.plate]!.alive = false; return; }
    if (op.kind === 'split') {
      const p = plates[op.plate]!, q = plates[op.into]!;
      q.alive = true; q.age = 0; q.continental = p.continental; q.accum = [0, 0];
      q.vel = [p.vel[0] + op.nx * RIFT_SPEED, p.vel[1] + op.nz * RIFT_SPEED];
      p.vel = [p.vel[0] - op.nx * RIFT_SPEED, p.vel[1] - op.nz * RIFT_SPEED];
      this.opKind.value = OP_SPLIT; this.opPlate.value = op.plate; this.opInto.value = op.into;
      this.splitN.value.set(op.nx, op.nz); this.splitC.value.set(op.cx, op.cz);
      this.lastSplitMy = geoMy;
    } else if (op.kind === 'merge') {
      const a = plates[op.from]!, b = plates[op.into]!;
      b.vel = [(a.vel[0] + b.vel[0]) / 2, (a.vel[1] + b.vel[1]) / 2];
      b.continental = a.continental || b.continental;
      a.alive = false;
      this.remapValues[op.from]!.x = op.into;
      this.opKind.value = OP_REMAP;
    } else {
      this.opKind.value = OP_ABSORB; this.opPlate.value = op.plate;
    }
    const f = this.fields;
    renderer.compute(this.relabel[f.parity('plateId') as 0 | 1]);
    f.swap('plateId');
    f.swap('crustAge');
  }
}

/** Serializable Lifecycle state (T50). lastSplitMy null = never split (JSON has no -Infinity). */
export interface LifecycleState { sutureRun: [string, number][]; lastSplitMy: number | null; log: { my: number; op: LifecycleOp }[] }

const wrapDelta = (d: number, n: number) => ((d + n / 2) % n + n) % n - n / 2;

/** CPU reference for tests: which side of a split a column lands on. */
export function splitSideCpu(x: number, z: number, op: Extract<LifecycleOp, { kind: 'split' }>): boolean {
  const dx = ((x - op.cx + NX / 2) % NX + NX) % NX - NX / 2;
  const dz = ((z - op.cz + NZ / 2) % NZ + NZ) % NZ - NZ / 2;
  const along = dx * -op.nz + dz * op.nx;
  const wob = Math.sin(along * 0.09) * 7 + Math.sin(along * 0.23 + 1.7) * 3;
  return dx * op.nx + dz * op.nz + wob > 0;
}
