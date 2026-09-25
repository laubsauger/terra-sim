// §T.42 volcanic plumes, impact dust, flood-basalt haze, hydrothermal vents: one GPU particle pool,
// compute-updated.
//
// Emitters:
//   lava vents   hot lava columns (lava temp > VENT_T0, sim 'lava' field): the vent kernel scans SUM²
//                cells of 4×4 columns and appends the hottest column of each active cell to a small list
//                (atomic counter). Cells with water next to the lava steam instead of ash.
//   hydrothermal columns on spreading ridges (crustAge < HYDRO_AGE, same cells the render seam glows
//                on). A deterministic hash keeps HYDRO_FRAC of the young-crust cells, each on its
//                lowest-hash young column. Underwater: bubble streams + dark shimmering "black smoker"
//                plumes that die before the surface. Subaerial rifts: faint steam wisps. Rate-capped.
//   bursts       CPU-fed event emitters (eruption / meteor / flood basalt from the event log) in a small
//                uniform table; each gets an exact spawn budget per frame.
// Dead particles pick an emitter at random and claim a spawn slot with an atomic counter, so the
// emission rate follows the emitters, not the pool size. Motion runs on real dt (ambient clock, V17):
// ash rises, billows and is sheared by the analytic zonal wind; steam rises and fades fast; dust rings
// rush outward and settle; haze drifts low and lingers.
// Two sprite draws over the pool: underwater kinds before the water (so its refraction shows them),
// everything else after it. Reads sim fields only (V15). Storage buffers bound: vents 6,
// particles 6, sprites 3 (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, int, uint, vec2, vec3, vec4, uniform, uniformArray, instanceIndex, instancedArray,
  hash, sin, cos, exp, mix, smoothstep, saturate, max, min, floor, abs, length, normalize, dot, pow, sqrt,
  atomicAdd, atomicLoad, atomicStore, uv, varying, cameraViewMatrix, step, round,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { ReaderKernel } from '../core/gpu';
import { NX, CELL } from '../sim/layout';
import { tColIdx, iMin } from '../sim/tslLayout';
import { tLavaTemp } from '../sim/magmaTsl';
import { tWorldY } from '../render/shared';
import { skyU } from '../render/sky';
import { ATMO, HALF, PK, BURST } from './atmoModel';
import { cloudShadowU, type Clouds } from './clouds';
import { atmoSeaLevel, tWind, tWindHF, tKeyColor, fxMRT } from './atmoTsl';

type F = THREE.Node<'float'>;
type V4 = THREE.Node<'vec4'>;

const S = ATMO.SUM;
const STEP = NX / S;
const TAU = Math.PI * 2;
/** ctr slots: lava vent count, lava spawns this frame, hydro vent count, hydro spawns, then burst spawns. */
const CTR_VENTS = 0, CTR_VSPAWN = 1, CTR_HVENTS = 2, CTR_HSPAWN = 3, CTR_BURST = 4;
const CTR_N = CTR_BURST + ATMO.BURSTS_MAX;
/** vents buffer: [0, VENTS_MAX) lava vents, [VENTS_MAX, +HYDRO_MAX) hydrothermal vents. */
const HV0 = ATMO.VENTS_MAX;

export interface Burst { x: number; y: number; z: number; kind: number; t0: number; dur: number; mag: number; radius: number; rate: number }


export interface Plumes {
  /** Above-water particles (drawn after the water). */
  object: THREE.Sprite;
  /** Underwater particles: bubbles, black smokers (drawn before the water). */
  under: THREE.Sprite;
  flash: THREE.Sprite;
  /** Particle state: P0 = (pos, age), P1 = (vel, life), P2 = (kind, seed, heat | strength | ceiling y, origin y). */
  P0: StorageNode<'vec4'>; P1: StorageNode<'vec4'>; P2: StorageNode<'vec4'>;
  /** Lava vents [0, VENTS_MAX): (x, y, z, heat + 2·steamQ); hydrothermal [VENTS_MAX, +HYDRO_MAX): (x, floor y, z, -(1 + water depth)). */
  vents: StorageNode<'vec4'>;
  ctr: StorageNode<'int'>;
  uniforms: {
    time: THREE.UniformNode<'float', number>;
    dt: THREE.UniformNode<'float', number>;
    frame: THREE.UniformNode<'float', number>;
    dither: THREE.UniformNode<'float', number>;
    seaLevel: THREE.UniformNode<'float', number>;
    flashPos: THREE.UniformNode<'vec3', THREE.Vector3>;
    flashAmt: THREE.UniformNode<'float', number>;
    flashSize: THREE.UniformNode<'float', number>;
  };
  /** CPU burst table (≤ BURSTS_MAX live). */
  bursts: Burst[];
  addBurst(b: Omit<Burst, 'rate'> & { rate?: number }): void;
  /** dt: real s; fxTime: monotonic FX clock (s) that burst t0 values refer to. */
  compute(renderer: THREE.WebGPURenderer, dt: number, fxTime: number): void;
  setHighQuality(v: boolean): void;
  dispose(): void;
}

const BURST_RATE: Record<number, number> = { [BURST.ASH]: 260, [BURST.DUST]: 900, [BURST.HAZE]: 70, [BURST.STEAM]: 120 };

export function createPlumes(fields: GpuFields, clouds: Clouds, opts: { highQuality: boolean; renderer: THREE.WebGPURenderer }): Plumes {
  const N = ATMO.PARTICLES_HIGH;
  const lava = fields.cur<'uvec2'>('lava'), water = fields.cur('water'), surfY = fields.cur('surfY');
  const mk = (n: number, name: string) => {
    const b = instancedArray(n, 'vec4') as unknown as StorageNode<'vec4'>;
    b.setName(name);
    return b;
  };
  const P0 = mk(N, 'plumeP0'), P1 = mk(N, 'plumeP1'), P2 = mk(N, 'plumeP2');
  // all dead at start: age 1 ≥ life 0
  for (let i = 0; i < N; i++) (P0.value.array as Float32Array)[i * 4 + 3] = 1;
  const vents = mk(ATMO.VENTS_MAX + ATMO.HYDRO_MAX, 'plumeVents');
  const ctr = instancedArray(CTR_N, 'int') as unknown as StorageNode<'int'>;
  ctr.setName('plumeCtr');
  ctr.toAtomic();

  const u = {
    time: uniform(0), dt: uniform(0), frame: uniform(0), dither: uniform(0.5),
    seaLevel: atmoSeaLevel,
    ventBudget: uniform(0), hydroBudget: uniform(0),
    flashPos: uniform(new THREE.Vector3()), flashAmt: uniform(0), flashSize: uniform(0.5),
  };
  const bA = uniformArray(Array.from({ length: ATMO.BURSTS_MAX }, () => new THREE.Vector4()), 'vec4');
  const bB = uniformArray(Array.from({ length: ATMO.BURSTS_MAX }, () => new THREE.Vector4()), 'vec4');
  const bAllow = uniformArray(new Array<number>(ATMO.BURSTS_MAX).fill(0), 'float');
  const bursts: Burst[] = [];
  const burstAcc = new Float32Array(ATMO.BURSTS_MAX);

  // ---- clear + vent scan ----
  const clearK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(CTR_N)), () => { Return(); });
    atomicStore(ctr.element(instanceIndex), int(0));
  })().compute(CTR_N);

  const buildVentK = (crustAge: StorageNode<'float'>) => Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(S * S)), () => { Return(); });
    const sx = int(instanceIndex.mod(uint(S))).mul(STEP), sz = int(instanceIndex.div(uint(S))).mul(STEP);
    const best = float(0).toVar(), bx = int(0).toVar(), bz = int(0).toVar(), by = float(0).toVar(), wet = float(0).toVar();
    const hkey = float(2).toVar(), hx = int(0).toVar(), hz = int(0).toVar(), hy = float(0).toVar(), hw = float(0).toVar();
    for (let dz = 0; dz < STEP; dz++) for (let dx = 0; dx < STEP; dx++) {
      const cx = sx.add(dx), cz = sz.add(dz);
      const c = tColIdx(cx, cz);
      const lv = lava.element(c).toVar();
      const m = float(lv.x);
      const heat = smoothstep(ATMO.VENT_T0, ATMO.VENT_T1, tLavaTemp(lv.y)).mul(min(m.div(255).mul(3).add(0.25), 1)).mul(step(0.5, m));
      const wd = (water.element(c) as unknown as F).toVar();
      const sy = (surfY.element(c) as unknown as F).toVar();
      wet.addAssign(step(ATMO.STEAM_WET, wd));
      If(heat.greaterThan(best), () => {
        best.assign(heat); bx.assign(cx); bz.assign(cz);
        by.assign(sy.add(m.div(255)));
      });
      // hydrothermal candidate: young crust, lowest per-column hash wins (deterministic)
      const key = hash(float(c).mul(0.917).add(41.3));
      If((crustAge.element(c) as unknown as F).lessThan(ATMO.HYDRO_AGE).and(key.lessThan(hkey)), () => {
        hkey.assign(key); hx.assign(cx); hz.assign(cz); hy.assign(sy); hw.assign(max(wd, 0));
      });
    }
    If(best.greaterThan(0.02), () => {
      const slot = atomicAdd(ctr.element(CTR_VENTS), int(1));
      If(slot.lessThan(int(ATMO.VENTS_MAX)), () => {
        const steamQ = round(wet.div(STEP * STEP).mul(7));
        vents.element(uint(slot)).assign(vec4(
          float(bx).add(0.5).mul(CELL).sub(HALF), tWorldY(by), float(bz).add(0.5).mul(CELL).sub(HALF), best.add(steamQ.mul(2))));
      });
    }).ElseIf(hkey.lessThan(1.5).and(hash(float(instanceIndex).mul(1.37).add(7.1)).lessThan(ATMO.HYDRO_FRAC)), () => {
      const slot = atomicAdd(ctr.element(CTR_HVENTS), int(1));
      If(slot.lessThan(int(ATMO.HYDRO_MAX)), () => {
        const floorY = tWorldY(hy).toVar();
        const depth = tWorldY(hy.add(hw)).sub(floorY);
        vents.element(uint(slot.add(HV0))).assign(vec4(
          float(hx).add(0.5).mul(CELL).sub(HALF), floorY, float(hz).add(0.5).mul(CELL).sub(HALF), depth.add(1).negate()));
      });
    });
  })().compute(S * S);
  const ventK = new ReaderKernel<'float'>(fields, 'crustAge', buildVentK);

  // ---- particles ----
  const buildPart = (count: number) => Fn(() => {
    const id = instanceIndex;
    If(id.greaterThanEqual(uint(count)), () => { Return(); });
    const p0 = P0.element(id).toVar(), p1 = P1.element(id).toVar(), p2 = P2.element(id).toVar();
    const fid = float(id);
    const rnd = (salt: number) => hash(fid.mul(0.7071).add(u.frame.mul(13.37)).add(salt * 17.13));
    If(p0.w.greaterThanEqual(p1.w), () => {
      const spawned = float(0).toVar();
      // lava vents
      const nV = iMin(atomicLoad(ctr.element(CTR_VENTS)) as unknown as THREE.Node<'int'>, int(ATMO.VENTS_MAX)).toVar();
      If(nV.greaterThan(int(0)), () => {
        const vi = iMin(int(floor(rnd(1).mul(float(nV)))), nV.sub(1));
        const v = vents.element(uint(vi)).toVar();
        const code = floor(v.w.mul(0.5));
        const heat = v.w.sub(code.mul(2)), steam = code.div(7);
        If(rnd(2).lessThan(heat), () => {
          const allowed = int(floor(float(nV).mul(u.ventBudget).add(u.dither)));
          const slot = atomicAdd(ctr.element(CTR_VSPAWN), int(1));
          If(slot.lessThan(allowed), () => {
            spawned.assign(1);
            const jx = rnd(4).sub(0.5).mul(0.035), jz = rnd(5).sub(0.5).mul(0.035);
            const isSteam = step(rnd(6), steam.mul(0.9));
            p0.assign(vec4(v.x.add(jx), v.y.add(0.008), v.z.add(jz), 0));
            const up = mix(rnd(7).mul(0.05).add(heat.mul(0.06)).add(0.1), float(0.06), isSteam);
            p1.assign(vec4(jx.mul(0.8), up, jz.mul(0.8), mix(rnd(8).mul(7).add(10), rnd(8).mul(1.5).add(2), isSteam)));
            p2.assign(vec4(mix(float(PK.ASH), float(PK.STEAM), isSteam), rnd(10), heat, v.y));
          });
        });
      });
      // hydrothermal vents
      If(spawned.lessThan(0.5), () => {
        const nH = iMin(atomicLoad(ctr.element(CTR_HVENTS)) as unknown as THREE.Node<'int'>, int(ATMO.HYDRO_MAX)).toVar();
        If(nH.greaterThan(int(0)), () => {
          const allowed = int(floor(min(float(nH).mul(ATMO.HYDRO_RATE), float(ATMO.HYDRO_CAP)).mul(u.hydroBudget).add(u.dither)));
          const slot = atomicAdd(ctr.element(CTR_HSPAWN), int(1));
          If(slot.lessThan(allowed), () => {
            spawned.assign(1);
            const vi = iMin(int(floor(rnd(21).mul(float(nH)))), nH.sub(1));
            const v = vents.element(uint(vi.add(HV0))).toVar();
            const depth = v.w.negate().sub(1);
            const ceil = v.y.add(depth);
            const jx = rnd(22).sub(0.5).mul(0.012), jz = rnd(23).sub(0.5).mul(0.012);
            If(depth.greaterThan(0.012), () => {
              const bubble = step(rnd(24), 0.55);
              const up = mix(rnd(25).mul(0.015).add(0.03), rnd(25).mul(0.05).add(0.08), bubble);
              // smokers dissipate by ~60 % of the water depth; bubbles pop at the surface
              const life = mix(min(depth.mul(0.6).div(up), 6), depth.div(up).mul(1.2), bubble);
              p0.assign(vec4(v.x.add(jx), v.y.add(0.002), v.z.add(jz), 0));
              p1.assign(vec4(0, up, 0, life));
              p2.assign(vec4(mix(float(PK.SMOKER), float(PK.BUBBLE), bubble), rnd(26), ceil, v.y));
            }).Else(() => { // subaerial rift: faint steam wisp
              p0.assign(vec4(v.x.add(jx.mul(2)), v.y.add(0.004), v.z.add(jz.mul(2)), 0));
              p1.assign(vec4(0, rnd(25).mul(0.03).add(0.04), 0, rnd(26).mul(1.5).add(2)));
              p2.assign(vec4(float(PK.STEAM), rnd(27), 0.35, v.y));
            });
          });
        });
      });
      // event bursts
      If(spawned.lessThan(0.5), () => {
        const bi = iMin(int(floor(rnd(11).mul(ATMO.BURSTS_MAX))), int(ATMO.BURSTS_MAX - 1)).toVar();
        const allow = bAllow.element(bi) as unknown as F;
        If(allow.greaterThan(0.5), () => {
          const slot = atomicAdd(ctr.element(uint(bi.add(CTR_BURST))), int(1));
          If(float(slot).lessThan(allow), () => {
            const A = (bA.element(bi) as unknown as V4).toVar(), B = (bB.element(bi) as unknown as V4).toVar();
            A.y.assign(max(A.y, tWorldY((clouds.sampleSummary(A.x, A.z) as V4).w)));
            const kind = A.w, mag = B.z, rad = B.w;
            const a = rnd(12).mul(TAU), rr = rnd(13);
            const dir = vec2(cos(a), sin(a));
            If(kind.lessThan(0.5), () => { // ASH eruption column
              const j = dir.mul(rr.mul(rad).mul(0.25));
              p0.assign(vec4(A.x.add(j.x), A.y.add(0.01), A.z.add(j.y), 0));
              p1.assign(vec4(j.x.mul(0.6), rnd(14).mul(0.12).add(0.28).mul(mag), j.y.mul(0.6), rnd(15).mul(8).add(12)));
              p2.assign(vec4(float(PK.ASH), rnd(16), mag.min(1.2), A.y));
            }).ElseIf(kind.lessThan(1.5), () => { // DUST ring (meteor)
              const ring = rad.mul(rr.mul(0.3).add(0.15));
              const col = step(rnd(17), 0.25); // a quarter rise as the impact column
              p0.assign(vec4(A.x.add(dir.x.mul(ring)), A.y.add(0.01), A.z.add(dir.y.mul(ring)), 0));
              const out = rnd(14).mul(0.5).add(0.35).mul(mag).mul(float(1).sub(col.mul(0.8)));
              p1.assign(vec4(dir.x.mul(out), mix(pow(rnd(18), 3).mul(0.18).add(0.02), rnd(18).mul(0.25).add(0.3), col), dir.y.mul(out), rnd(15).mul(4).add(3.5)));
              p2.assign(vec4(float(PK.DUST), rnd(16), 0.8, A.y));
            }).ElseIf(kind.lessThan(2.5), () => { // HAZE (flood basalt)
              const d = dir.mul(sqrt(rr).mul(rad));
              p0.assign(vec4(A.x.add(d.x), A.y.add(rnd(14).mul(0.12).add(0.03)), A.z.add(d.y), 0));
              p1.assign(vec4(0, 0, 0, rnd(15).mul(10).add(16)));
              p2.assign(vec4(float(PK.HAZE), rnd(16), 0.35, A.y));
            }).Else(() => { // STEAM burst
              const j = dir.mul(rr.mul(rad));
              p0.assign(vec4(A.x.add(j.x), A.y.add(0.01), A.z.add(j.y), 0));
              p1.assign(vec4(0, rnd(14).mul(0.06).add(0.08), 0, rnd(15).mul(2).add(2.5)));
              p2.assign(vec4(float(PK.STEAM), rnd(16), 1, A.y));
            });
          });
        });
      });
    }).Else(() => {
      const dt = u.dt;
      const kind = p2.x;
      const pos = p0.xyz.toVar(), vel = p1.xyz.toVar();
      const seaY = tWorldY(u.seaLevel);
      const seed = p2.y, heat = p2.z, oy = p2.w;
      const hAbove = pos.y.sub(oy);
      const wind = tWind(pos.z, tWindHF(pos.y)).mul(1.6); // visual exaggeration: the diorama's plumes read as bent
      const turb = vec2(sin(pos.y.mul(9).add(u.time.mul(1.3)).add(seed.mul(20))), cos(pos.x.mul(7).add(u.time.mul(1.1)).add(seed.mul(13)))).mul(0.03);
      const sdir = vec2(cos(seed.mul(TAU)), sin(seed.mul(TAU)));
      const underwater = step(3.5, kind);
      If(kind.lessThan(0.5), () => {
        // ASH: momentum-driven jet near the vent, buoyant rise that slows toward the neutral-buoyancy
        // height, entrainment bends it downwind as it climbs, then it spreads into a drifting umbrella.
        const hnb = min(heat.mul(0.3).add(0.22), max(seaY.add(0.95).sub(oy), 0.12));
        const rise = float(1).sub(smoothstep(hnb.mul(0.55), hnb, hAbove));
        const umb = smoothstep(hnb.mul(0.75), hnb, hAbove);
        const wUp = heat.mul(0.07).add(0.1).mul(rise).sub(umb.mul(0.008));
        vel.y.assign(mix(vel.y, wUp, float(1).sub(exp(dt.mul(-1.2)))));
        const entrain = smoothstep(0, hnb, hAbove).mul(1.1).add(0.25);
        const target = wind.add(turb).add(sdir.mul(umb.mul(0.07)));
        vel.xz.assign(mix(vel.xz, target, float(1).sub(exp(dt.mul(entrain).negate()))));
      }).ElseIf(kind.lessThan(1.5), () => { // STEAM: gentle rise, bends early, fades fast
        vel.y.assign(vel.y.mul(exp(dt.mul(-0.45))));
        vel.xz.assign(mix(vel.xz, wind.mul(0.8).add(turb), float(1).sub(exp(dt.mul(-1.2)))));
      }).ElseIf(kind.lessThan(2.5), () => { // DUST: rush out, drag, settle
        vel.xz.assign(vel.xz.mul(exp(dt.mul(-1.3))).add(wind.mul(0.3).mul(dt)));
        vel.y.assign(vel.y.mul(exp(dt.mul(-1.1))).sub(dt.mul(0.015)));
      }).ElseIf(kind.lessThan(3.5), () => { // HAZE: low, slow drift
        vel.xz.assign(mix(vel.xz, wind.mul(0.5).add(turb.mul(0.3)), float(1).sub(exp(dt.mul(-0.5)))));
        vel.y.assign(0);
      }).ElseIf(kind.lessThan(4.5), () => { // BUBBLE: steady rise, wobble
        vel.xz.assign(vec2(sin(u.time.mul(7).add(seed.mul(40))), cos(u.time.mul(6).add(seed.mul(31)))).mul(0.012));
      }).Else(() => { // SMOKER: slows, spreads a little
        vel.y.assign(vel.y.mul(exp(dt.mul(-0.15))));
        vel.xz.assign(turb.mul(0.1));
      });
      pos.addAssign(vel.mul(dt));
      const ground = tWorldY((clouds.sampleSummary(pos.x, pos.z) as V4).w).add(0.006);
      pos.y.assign(mix(max(pos.y, ground), pos.y, underwater));
      const age = p0.w.add(dt).toVar();
      // leaving the block or reaching the water surface: die (the render fades them before that)
      If(max(abs(pos.x), abs(pos.z)).greaterThan(HALF).or(underwater.greaterThan(0.5).and(pos.y.greaterThan(p2.z))), () => { age.assign(p1.w); });
      p0.assign(vec4(pos, age));
      p1.assign(vec4(vel, p1.w));
    });
    P0.element(id).assign(p0);
    P1.element(id).assign(p1);
    P2.element(id).assign(p2);
  })().compute(count);
  const partHigh = buildPart(ATMO.PARTICLES_HIGH);
  const partLow = buildPart(ATMO.PARTICLES_LOW);

  // ---- render ----
  const buildMaterial = (under: boolean) => {
    const mat = new THREE.SpriteNodeMaterial();
    mat.name = under ? 'plumesUnder' : 'plumes';
    mat.transparent = true;
    mat.depthWrite = false;
    mat.fog = true;
    fxMRT(mat, opts.renderer);
    mat.alphaTest = 0.01;
    const q0 = P0.element(instanceIndex) as unknown as V4, q1 = P1.element(instanceIndex) as unknown as V4, q2 = P2.element(instanceIndex) as unknown as V4;
    const kind = q2.x;
    const isUnder = step(3.5, kind);
    const alive = float(q0.w.lessThan(q1.w)).mul(under ? isUnder : float(1).sub(isUnder));
    const tl = saturate(q0.w.div(max(q1.w, 1e-3)));
    const kA = vec4(step(kind, 0.5), step(abs(kind.sub(1)), 0.5), step(abs(kind.sub(2)), 0.5), step(abs(kind.sub(3)), 0.5));
    const kB = vec2(step(abs(kind.sub(4)), 0.5), step(4.5, kind));
    // start size and growth per kind; ash swells again in the umbrella (height above its vent)
    const hAb = q0.y.sub(q2.w);
    const size0 = dot(kA, vec4(0.018, 0.012, 0.04, 0.18)).add(kA.y.mul(q2.z).mul(0.018)).add(kB.x.mul(hash(q2.y.mul(91)).mul(0.003).add(0.004))).add(kB.y.mul(0.01));
    const grow = dot(kA, vec4(0.05, 0.13, 0.15, 0.25)).add(dot(kB, vec2(0.002, 0.06)));
    // ash: narrow near the vent, widening with height, then swelling into the umbrella
    const size = size0.add(grow.mul(sqrt(tl))).add(kA.x.mul(saturate(hAb.mul(0.25))).add(kA.x.mul(smoothstep(0.25, 0.55, hAb)).mul(0.1)));
    mat.positionNode = q0.xyz;
    mat.scaleNode = vec2(size.mul(alive));
    mat.rotationNode = q2.y.mul(TAU).add(q0.w.mul(0.15));
    const vKA = varying(kA, 'vPlumeKA');
    const vKB = varying(kB, 'vPlumeKB');
    const vLife = varying(vec4(tl, q0.w, q2.z, q2.y), 'vPlumeLife');
    const vH = varying(hAb, 'vPlumeH');
    const vXZ = varying(q0.xz, 'vPlumeXZ');

    const shade = Fn(() => {
      const k = vKA, kb = vKB, t = vLife.x, age = vLife.y, heat = vLife.z, seed = vLife.w;
      const q = uv().mul(2).sub(1);
      const rr = length(q);
      const nse = sin(q.x.mul(4.7).add(seed.mul(17))).mul(sin(q.y.mul(5.3).add(seed.mul(11)))).mul(0.5)
        .add(sin(q.x.add(q.y).mul(9).add(seed.mul(5))).mul(0.2));
      const puff = smoothstep(1.0, 0.05, rr.add(nse.mul(0.25))).mul(smoothstep(1.0, 0.6, rr));
      const ring = smoothstep(1.0, 0.75, rr).mul(smoothstep(0.35, 0.8, rr)).add(smoothstep(0.35, 0.0, length(q.sub(vec2(-0.3, 0.3)))).mul(0.8));
      const shapeA = mix(puff, ring, kb.x);
      const nv = normalize(vec3(q.x, q.y, sqrt(max(float(1).sub(rr.mul(rr)), 0.05))));
      const keyV = cameraViewMatrix.mul(vec4(cloudShadowU.keyDir, 0)).xyz;
      const wrap = saturate(dot(nv, normalize(keyV)).add(0.5).div(1.5));
      const keyCol = tKeyColor();
      const amb = mix(skyU.horizon, skyU.zenith, q.y.mul(0.5).add(0.5)).mul(0.75);
      const albedo = mix(vec3(0.3, 0.26, 0.22), vec3(0.5, 0.47, 0.43), t).mul(k.x)
        .add(vec3(0.93, 0.94, 0.96).mul(k.y)).add(vec3(0.56, 0.45, 0.33).mul(k.z)).add(vec3(0.6, 0.55, 0.48).mul(k.w))
        .add(vec3(0.85, 0.97, 1.0).mul(kb.x.mul(1.6))).add(vec3(0.05, 0.05, 0.06).mul(kb.y));
      const lit = albedo.mul(amb.add(keyCol.mul(wrap).mul(0.3)));
      // orange underglow from the lava: young ash / hot ejecta, strongest on the underside
      const glow = vec3(1.0, 0.33, 0.07).mul(min(heat, 1).mul(exp(vH.mul(-22))).mul(k.x.add(k.z.mul(0.6).mul(exp(age.mul(-1.5))))).mul(q.y.negate().mul(0.5).add(0.6)).mul(6));
      const shimmer = sin(age.mul(9).add(seed.mul(30))).mul(0.2).add(0.8);
      const alpha = k.x.mul(pow(float(1).sub(t), 1.3).mul(0.34))
        .add(k.y.mul(pow(float(1).sub(t), 2.2).mul(0.32)).mul(min(heat.add(0.3), 1)))
        .add(k.z.mul(pow(float(1).sub(t), 1.5).mul(0.75)))
        .add(k.w.mul(sin(t.mul(Math.PI)).mul(0.2)))
        .add(kb.x.mul(0.7).mul(float(1).sub(pow(t, 4))))
        .add(kb.y.mul(pow(float(1).sub(t), 1.5).mul(0.55)).mul(shimmer));
      const edge = smoothstep(0, 0.2, float(HALF).sub(max(abs(vXZ.x), abs(vXZ.y))));
      const a = shapeA.mul(alpha).mul(smoothstep(0, 0.25, age)).mul(edge);
      return vec4(lit.add(glow), a);
    })();
    mat.colorNode = vec4(shade.rgb, 1);
    mat.opacityNode = shade.a;
    const sprite = new THREE.Sprite(mat);
    sprite.name = mat.name;
    sprite.count = opts.highQuality ? ATMO.PARTICLES_HIGH : ATMO.PARTICLES_LOW;
    sprite.frustumCulled = false;
    // water.ts draws at renderOrder 2 and refracts what is already in the frame: underwater first, rest after
    sprite.renderOrder = under ? 1 : 4; // underwater before the water sheet (2) so its refraction shows them; clouds (6) composite over the rest
    return { sprite, mat };
  };
  const above = buildMaterial(false), below = buildMaterial(true);

  // ---- impact flash (one additive sprite) ----
  const fmat = new THREE.SpriteNodeMaterial();
  fmat.transparent = true;
  fmat.depthWrite = false;
  fmat.blending = THREE.AdditiveBlending;
  fmat.positionNode = u.flashPos;
  fmat.scaleNode = vec2(u.flashSize.mul(step(0.001, u.flashAmt)));
  const fq = uv().mul(2).sub(1);
  const fall = pow(saturate(float(1).sub(length(fq))), 2.2);
  fmat.colorNode = vec4(vec3(1.0, 0.75, 0.45).mul(u.flashAmt.mul(30)), 1);
  fmat.opacityNode = fall;
  const flash = new THREE.Sprite(fmat);
  flash.name = 'impactFlash';
  flash.frustumCulled = false;
  flash.renderOrder = 7;

  let hq = opts.highQuality;
  const syncBursts = (dt: number, time: number) => {
    for (let b = bursts.length - 1; b >= 0; b--) if (time - bursts[b]!.t0 > bursts[b]!.dur + 1) { bursts.splice(b, 1); burstAcc.copyWithin(b, b + 1); burstAcc[ATMO.BURSTS_MAX - 1] = 0; }
    for (let b = 0; b < ATMO.BURSTS_MAX; b++) {
      const e = bursts[b];
      let allow = 0;
      if (e) {
        (bA.array[b] as THREE.Vector4).set(e.x, e.y, e.z, e.kind);
        (bB.array[b] as THREE.Vector4).set(e.t0, e.dur, e.mag, e.radius);
        const age = time - e.t0;
        if (age >= 0 && age <= e.dur) {
          // DUST front-loads its spawns (impact), the others emit evenly
          const env = e.kind === BURST.DUST ? Math.exp(-age * 2.5) * 2.5 : 1;
          burstAcc[b]! += e.rate * env * e.mag * dt * (hq ? 1 : 0.4);
          allow = Math.floor(burstAcc[b]!);
          burstAcc[b]! -= allow;
        }
      }
      (bAllow.array as number[])[b] = allow;
    }
  };

  return {
    object: above.sprite, under: below.sprite, flash, P0, P1, P2, vents, ctr, uniforms: u, bursts,
    addBurst(b) {
      if (bursts.length >= ATMO.BURSTS_MAX) { bursts.shift(); burstAcc.copyWithin(0, 1); burstAcc[ATMO.BURSTS_MAX - 1] = 0; }
      bursts.push({ ...b, rate: b.rate ?? BURST_RATE[b.kind] ?? 100 });
      burstAcc[bursts.length - 1] = 0;
    },
    compute(renderer, dt, fxTime) {
      u.dt.value = dt;
      u.time.value = fxTime % 1000;
      u.frame.value = (u.frame.value + 1) % 4096;
      u.dither.value = Math.random();
      u.ventBudget.value = ATMO.VENT_RATE * dt * (hq ? 1 : 0.4);
      u.hydroBudget.value = dt * (hq ? 1 : 0.5);
      syncBursts(dt, fxTime);
      renderer.compute(clearK);
      ventK.run(renderer);
      renderer.compute(hq ? partHigh : partLow);
    },
    setHighQuality(v) {
      hq = v;
      above.sprite.count = below.sprite.count = v ? ATMO.PARTICLES_HIGH : ATMO.PARTICLES_LOW;
    },
    dispose() { above.mat.dispose(); below.mat.dispose(); fmat.dispose(); },
  };
}
