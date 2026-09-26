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
  atomicAdd, atomicLoad, atomicStore, uv, varying, cameraViewMatrix, cameraPosition, step, round, atan, texture3D, screenUV,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { ReaderKernel } from '../core/gpu';
import { NX, CELL, VOXEL_H } from '../sim/layout';
import { tColIdx, iMin } from '../sim/tslLayout';
import { tLavaTemp, tLavaDepth } from '../sim/magmaTsl';
import { tWorldY, vertEx, sceneViewZ } from '../render/shared';
import { skyU } from '../render/sky';
import { ATMO, HALF, PK, BURST, VENT_DEEP, VENT_SHALLOW } from './atmoModel';
import { cloudShadowU, type Clouds } from './clouds';
import { cloudNoiseTexture } from './noise3d';
import { atmoSeaLevel, tWind, tWindHF, tKeyColor, fxMRT, tCutHard } from './atmoTsl';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

const S = ATMO.SUM;
const STEP = NX / S;
const TAU = Math.PI * 2;
/** ctr slots: lava vent count, lava spawns this frame, hydro vent count, hydro spawns, then burst spawns. */
const CTR_VENTS = 0, CTR_VSPAWN = 1, CTR_HVENTS = 2, CTR_HSPAWN = 3, CTR_BURST = 4;
const CTR_N = CTR_BURST + ATMO.BURSTS_MAX;
/** vents buffer: [0, VENTS_MAX) lava vents, [VENTS_MAX, +HYDRO_MAX) hydrothermal vents. */
const HV0 = ATMO.VENTS_MAX;
/** Share of PDC particles in the lofting ash cloud (flagged as heat + 2 in P2.z); the rest is the dense base. */
const PDC_LOFT = 0.4;

export interface Burst { x: number; y: number; z: number; kind: number; t0: number; dur: number; mag: number; radius: number; rate: number }


export interface Plumes {
  /** Above-water particles (drawn after the water). */
  object: THREE.Sprite;
  /** Underwater particles: bubbles, black smokers (drawn before the water). */
  under: THREE.Sprite;
  flash: THREE.Sprite;
  /** Particle state: P0 = (pos, age), P1 = (vel, life), P2 = (kind PK, seed, heat | strength | ceiling y, origin y). */
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

const BURST_RATE: Record<number, number> = { [BURST.ASH]: 260, [BURST.DUST]: 900, [BURST.HAZE]: 70, [BURST.STEAM]: 160, [BURST.FOUNTAIN]: 170, [BURST.BOMB]: 22, [BURST.PDC]: 110 };

export function createPlumes(fields: GpuFields, clouds: Clouds, opts: { highQuality: boolean; renderer: THREE.WebGPURenderer }): Plumes {
  const N = ATMO.PARTICLES_HIGH;
  const lava = fields.cur<'uvec2'>('lava'), water = fields.cur('water'), surfY = fields.cur('surfY');
  const volc = fields.cur<'vec4'>('volcano');
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
    const bw = float(0).toVar(), bpc = float(0).toVar();
    const hkey = float(2).toVar(), hx = int(0).toVar(), hz = int(0).toVar(), hy = float(0).toVar(), hw = float(0).toVar();
    for (let dz = 0; dz < STEP; dz++) for (let dx = 0; dx < STEP; dx++) {
      const cx = sx.add(dx), cz = sz.add(dz);
      const c = tColIdx(cx, cz);
      const lv = lava.element(c).toVar();
      const depth = tLavaDepth(lv).toVar(); // magmaTsl accessors only: the packing is theirs to change
      const wd = max(water.element(c) as unknown as F, 0).toVar();
      const vo = volc.element(c).toVar();
      // eruption episode at a vent column (magma.ts 'volcano': x activity, y phase 1 build, 2 active, 3 waning)
      const inEpisode = step(0.5, vo.y).mul(step(vo.y, 3.5));
      const erupt = vo.x.mul(inEpisode);
      // plain hot lava pools only degas gently (thin fissure films don't count)
      const pool = smoothstep(ATMO.VENT_T0, ATMO.VENT_T1, tLavaTemp(lv.y)).mul(smoothstep(0.04, 0.5, depth)).mul(0.35);
      const heat = max(erupt, pool);
      const sy = (surfY.element(c) as unknown as F).toVar();
      wet.addAssign(step(ATMO.STEAM_WET, wd));
      If(heat.greaterThan(best), () => {
        best.assign(heat); bx.assign(cx); bz.assign(cz); bw.assign(wd);
        bpc.assign(mix(float(0), round(vo.y), step(pool, erupt)));
        by.assign(max(sy.add(depth), sy.add(wd)));
      });
      // hydrothermal candidate: young crust, lowest per-column hash wins (deterministic)
      const key = hash(float(c).mul(0.917).add(41.3));
      If((crustAge.element(c) as unknown as F).lessThan(ATMO.HYDRO_AGE).and(key.lessThan(hkey)), () => {
        hkey.assign(key); hx.assign(cx); hz.assign(cz); hy.assign(sy); hw.assign(wd);
      });
    }
    If(best.greaterThan(0.02), () => {
      const slot = atomicAdd(ctr.element(CTR_VENTS), int(1));
      If(slot.lessThan(int(ATMO.VENTS_MAX)), () => {
        // w = heat + 2·(water depth code + 64·phase class + 256·coast): see decodeVent
        const dCode = min(round(bw.mul(4)), 63);
        const coast = round(wet.div(STEP * STEP).mul(3));
        vents.element(uint(slot)).assign(vec4(
          float(bx).add(0.5).mul(CELL).sub(HALF), tWorldY(by), float(bz).add(0.5).mul(CELL).sub(HALF),
          best.min(0.999).add(dCode.add(bpc.mul(64)).add(coast.mul(256)).mul(2))));
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
  // TSL codegen rule for these kernels: a plain node used in more than one If/ElseIf branch is emitted as a
  // temp var declared at the top of main() but assigned only inside the first branch that builds it; every
  // other branch reads it zero-initialised (it left the PDCs, tephra, bubbles, turbid clouds, pumice, slicks
  // with no jitter, no spawn velocity and seed 0, and haze / pumice / PDC with no wind). So every value
  // computed before a branch chain and read inside it is .toVar()'d: declared and assigned right there.
  const G = ATMO.GRAVITY;
  const groundAt = (x: F, z: F): F => tWorldY((clouds.sampleSummary(x, z) as V4).w);
  const buildPart = (count: number) => Fn(() => {
    const id = instanceIndex;
    If(id.greaterThanEqual(uint(count)), () => { Return(); });
    const p0 = P0.element(id).toVar(), p1 = P1.element(id).toVar(), p2 = P2.element(id).toVar();
    const fid = float(id);
    const rnd = (salt: number) => hash(fid.mul(0.7071).add(u.frame.mul(13.37)).add(salt * 17.13));
    /**
     * One volcanic particle of `kind` from origin o (vent surface), vent heat 0..1, magnitude and spread
     * radius. Shared by lava vents and eruption bursts so both look the same.
     */
    const emit = (kind: F, o: V3, heat: F, mag: F, rad: F, surf: F) => {
      // shared by the kind branches below, hence .toVar() (see the note at '---- particles ----')
      heat = heat.toVar(); mag = mag.toVar(); rad = rad.toVar();
      const a = rnd(12).mul(TAU).toVar(), rr = rnd(13).toVar();
      const dir = vec2(cos(a), sin(a)).toVar();
      const seed = rnd(16).toVar();
      const is = (k: number) => abs(kind.sub(k)).lessThan(0.5);
      If(is(PK.ASH), () => { // ASH: billowing column from a slim crater
        const j = dir.mul(rr.mul(rad).mul(0.3).add(0.003));
        p0.assign(vec4(o.x.add(j.x), o.y.add(0.006), o.z.add(j.y), 0));
        p1.assign(vec4(j.x.mul(0.8), rnd(14).mul(0.06).add(heat.mul(0.08)).add(0.1).mul(mag), j.y.mul(0.8), rnd(15).mul(7).add(11)));
        p2.assign(vec4(float(PK.ASH), seed, heat.mul(mag).min(1.3), o.y));
      }).ElseIf(is(PK.STEAM), () => { // STEAM: blasts / boiling sea / degassing (heat = strength)
        const j = dir.mul(rr.mul(rad).mul(0.6).add(0.004));
        const out = dir.mul(rnd(14).mul(0.06).mul(heat));
        p0.assign(vec4(o.x.add(j.x), o.y.add(0.006), o.z.add(j.y), 0));
        // weak sources (degassing) give short low wisps; strong ones (sea blasts) tall billows
        p1.assign(vec4(out.x, rnd(15).mul(0.16).mul(heat).add(0.05), out.y, rnd(17).mul(1.6).add(2).mul(heat.mul(0.8).add(0.35).min(1)).mul(ATMO.STEAM_LIFE)));
        p2.assign(vec4(float(PK.STEAM), seed, heat.max(0.35), o.y));
      }).ElseIf(is(PK.BUBBLE), () => { // BUBBLE: from a submarine vent up to the surface
        const j = dir.mul(rr.mul(0.02));
        const up = rnd(14).mul(0.05).add(0.08);
        p0.assign(vec4(o.x.add(j.x), o.y.add(0.003), o.z.add(j.y), 0));
        p1.assign(vec4(0, up, 0, surf.sub(o.y).div(up).mul(1.2).add(0.1)));
        p2.assign(vec4(float(PK.BUBBLE), seed, surf, o.y));
      }).ElseIf(is(PK.TURBID), () => { // TURBID: slim black-smoker column off a deep vent, gone just below the surface
        const j = dir.mul(rr.mul(0.004));
        const up = rnd(14).mul(0.01).add(0.02);
        p0.assign(vec4(o.x.add(j.x), o.y.add(0.003), o.z.add(j.y), 0));
        p1.assign(vec4(0, up, 0, min(surf.sub(o.y).mul(0.92).div(up), 9)));
        p2.assign(vec4(float(PK.TURBID), seed, surf, o.y));
      }).ElseIf(is(PK.FOUNTAIN), () => { // FOUNTAIN: bright spray arcing just above the vent
        const out = dir.mul(rr.mul(0.18).add(0.04)).mul(sqrt(heat));
        p0.assign(vec4(o.x.add(dir.x.mul(0.004)), o.y.add(0.004), o.z.add(dir.y.mul(0.004)), 0));
        p1.assign(vec4(out.x, rnd(14).mul(0.22).add(0.2).mul(sqrt(heat)).mul(mag.min(1.2)), out.y, 3));
        p2.assign(vec4(float(PK.FOUNTAIN), seed, heat, o.y));
      }).ElseIf(is(PK.BOMB), () => { // BOMB: glowing lava bomb on a ballistic arc
        const out = dir.mul(rr.mul(0.24).add(0.1)).mul(mag.min(1.2));
        p0.assign(vec4(o.x, o.y.add(0.01), o.z, 0));
        p1.assign(vec4(out.x, rnd(14).mul(0.25).add(0.35).mul(mag.min(1.2)), out.y, 6));
        p2.assign(vec4(float(PK.BOMB), seed, heat, o.y));
      }).ElseIf(is(PK.TEPHRA), () => { // TEPHRA: black cock's-tail jets fanning out of a shallow sea vent
        const out = dir.mul(rr.mul(0.22).add(0.06)).mul(heat.add(0.3).min(1));
        p0.assign(vec4(o.x, o.y.add(0.004), o.z, 0));
        p1.assign(vec4(out.x, rnd(14).mul(0.3).add(0.3).mul(heat.add(0.3).min(1)), out.y, 4));
        p2.assign(vec4(float(PK.TEPHRA), seed, heat, o.y));
      }).ElseIf(is(PK.SLICK), () => { // SLICK: milky discoloured water spreading on the surface above a submarine vent
        const d = dir.mul(rr.mul(0.04));
        p0.assign(vec4(o.x.add(d.x), surf.add(0.002), o.z.add(d.y), 0));
        p1.assign(vec4(0, 0, 0, rnd(15).mul(8).add(12)));
        p2.assign(vec4(float(PK.SLICK), seed, heat, surf));
      }).ElseIf(is(PK.PUMICE), () => { // PUMICE: specks floating on the sea above a submarine vent
        const d = dir.mul(sqrt(rr).mul(0.12));
        p0.assign(vec4(o.x.add(d.x), surf.add(0.001), o.z.add(d.y), 0));
        p1.assign(vec4(0, 0, 0, rnd(15).mul(10).add(12)));
        p2.assign(vec4(float(PK.PUMICE), seed, surf, surf));
      }).Else(() => { // PDC: pyroclastic density current from the crater rim. Most is the dense ground-hugging base
        // (a radial surge whose speed varies smoothly with azimuth: lobate fronts, a few fast fingers), a
        // PDC_LOFT share the ash cloud that lofts off its top
        const r0 = rr.mul(0.025).add(0.01);
        p0.assign(vec4(o.x.add(dir.x.mul(r0)), o.y.add(0.01), o.z.add(dir.y.mul(r0)), 0));
        const lobe = pow(sin(a.mul(5).add(o.x.mul(37)).add(o.z.mul(23))).mul(0.5).add(0.5), 2);
        const out = dir.mul(lobe.mul(0.15).add(0.04).mul(rnd(14).mul(0.6).add(0.7))).mul(mag);
        const loft = step(rnd(30), PDC_LOFT);
        p1.assign(vec4(out.x, 0, out.y, mix(rnd(15).mul(2).add(4), rnd(15).mul(3).add(6), loft)));
        p2.assign(vec4(float(PK.PDC), seed, heat.min(1).add(loft.mul(2)), o.y));
      });
    };
    If(p0.w.greaterThanEqual(p1.w), () => {
      const spawned = float(0).toVar();
      // lava vents, by eruption phase and water depth at the vent:
      //   dry:    active → ash column, fountain spray, bombs (steam blasts where lava meets the sea);
      //           build-up / plain pools → degassing steam wisps + a little ash; waning → ash only
      //   shallow (≤ 1 voxel, Surtseyan): black tephra jets, white steam billows, some ash
      //   medium (1-4 voxels): white steam puffs boiling off the sea surface, no ash column
      //   deep (> 4 voxels): nothing in the air; a slim dark smoker column (warm glow at its foot) + bubbles under water,
      //           pumice specks floating on the surface above
      const nV = iMin(atomicLoad(ctr.element(CTR_VENTS)) as unknown as THREE.Node<'int'>, int(ATMO.VENTS_MAX)).toVar();
      If(nV.greaterThan(int(0)), () => {
        const vi = iMin(int(floor(rnd(1).mul(float(nV)))), nV.sub(1));
        const v = vents.element(uint(vi)).toVar();
        const code = floor(v.w.mul(0.5)).toVar();
        const heat = v.w.sub(code.mul(2)).toVar();
        const wdep = code.mod(64).div(4).toVar(), pc = floor(code.div(64)).mod(4).toVar(), coast = floor(code.div(256)).div(3).toVar();
        // plain lava pools only puff now and then; eruption episodes get the budget
        If(rnd(2).lessThan(mix(heat.mul(0.2), heat.add(0.15), step(0.5, pc))), () => {
          // strict budget: beyond VENTS_EMIT vents the same total rate is shared (big eruptions stay readable)
          const allowed = int(floor(min(float(nV), float(ATMO.VENTS_EMIT)).mul(u.ventBudget).add(u.dither)));
          const slot = atomicAdd(ctr.element(CTR_VSPAWN), int(1));
          If(slot.lessThan(allowed), () => {
            spawned.assign(1);
            const r = rnd(6).toVar(), r2 = rnd(9).toVar();
            const kind = float(PK.ASH).toVar();
            const origin = v.xyz.toVar();
            const floorY = v.y.sub(wdep.mul(VOXEL_H).mul(vertEx)).toVar();
            const active = step(1.5, pc).mul(step(pc, 2.5)).toVar();
            If(wdep.greaterThan(VENT_DEEP), () => { // deep submarine: smoker column + bubbles below, slick + pumice on top
              If(r.lessThan(0.07), () => { kind.assign(PK.PUMICE); })
                .ElseIf(r.lessThan(0.16), () => { kind.assign(PK.SLICK); })
                .ElseIf(r.lessThan(0.4), () => { kind.assign(PK.BUBBLE); origin.y.assign(floorY); })
                .Else(() => { kind.assign(PK.TURBID); origin.y.assign(floorY); });
            }).ElseIf(wdep.greaterThan(VENT_SHALLOW), () => { // boiling sea: steam, a discoloured patch, pumice
              If(r.lessThan(0.12), () => { kind.assign(PK.SLICK); }).ElseIf(r.lessThan(0.18), () => { kind.assign(PK.PUMICE); }).Else(() => { kind.assign(PK.STEAM); });
            })
              .ElseIf(wdep.greaterThan(ATMO.STEAM_WET), () => { // Surtseyan: cock's-tail tephra jets + steam billows
                If(r.lessThan(0.4), () => { kind.assign(PK.TEPHRA); }).ElseIf(r.lessThan(0.8), () => { kind.assign(PK.STEAM); });
              })
              .Else(() => {
                If(r.lessThan(coast.mul(0.7)), () => { kind.assign(PK.STEAM); }) // lava hitting the sea
                  .ElseIf(pc.lessThan(1.5), () => { kind.assign(PK.STEAM); }) // pools / build-up: degassing wisps only
                  .ElseIf(active.greaterThan(0.5).and(r2.lessThan(heat.mul(ATMO.FOUNTAIN_SHARE))), () => { kind.assign(PK.FOUNTAIN); })
                  .ElseIf(active.greaterThan(0.5).and(r2.lessThan(heat.mul(ATMO.FOUNTAIN_SHARE).add(heat.mul(heat).mul(ATMO.BOMB_SHARE)))).and(heat.greaterThan(0.4)), () => { kind.assign(PK.BOMB); });
              });
            emit(kind, origin, heat, float(1), float(0.03), v.y);
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
            A.y.assign(max(A.y, groundAt(A.x, A.z)));
            const kind = A.w, mag = B.z, rad = B.w;
            const a = rnd(12).mul(TAU).toVar(), rr = rnd(13).toVar();
            const dir = vec2(cos(a), sin(a)).toVar();
            If(kind.lessThan(0.5), () => { emit(float(PK.ASH), A.xyz, float(1), mag, rad, A.y); })
              .ElseIf(kind.lessThan(1.5), () => { // DUST ring (meteor)
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
              }).ElseIf(kind.lessThan(3.5), () => { emit(float(PK.STEAM), A.xyz, mag.min(1), float(1), rad, A.y); })
              .ElseIf(kind.lessThan(4.5), () => { emit(float(PK.FOUNTAIN), A.xyz, float(1), mag, rad, A.y); })
              .ElseIf(kind.lessThan(5.5), () => { emit(float(PK.BOMB), A.xyz, float(1), mag, rad, A.y); })
              .Else(() => { emit(float(PK.PDC), A.xyz, float(1), mag, rad, A.y); });
          });
        });
      });
    }).Else(() => {
      const dt = u.dt;
      const kind = p2.x;
      const pos = p0.xyz.toVar(), vel = p1.xyz.toVar();
      const life = p1.w.toVar();
      const seaY = tWorldY(u.seaLevel).toVar();
      const seed = p2.y, heat = p2.z, oy = p2.w;
      const hAbove = pos.y.sub(oy).toVar();
      const wind = tWind(pos.z, tWindHF(pos.y)).mul(1.6).toVar(); // visual exaggeration: the diorama's plumes read as bent
      const turb = vec2(sin(pos.y.mul(9).add(u.time.mul(1.3)).add(seed.mul(20))), cos(pos.x.mul(7).add(u.time.mul(1.1)).add(seed.mul(13)))).mul(0.03).toVar();
      const sdir = vec2(cos(seed.mul(TAU)), sin(seed.mul(TAU))).toVar();
      const isK = (k: number) => step(abs(kind.sub(k)), 0.5);
      const underwater = step(3.5, kind).mul(step(kind, 6.5)).toVar();
      const ballistic = isK(PK.FOUNTAIN).add(isK(PK.BOMB)).add(isK(PK.TEPHRA)).toVar();
      const floating = isK(PK.PUMICE).add(isK(PK.SLICK)).toVar();
      If(kind.lessThan(0.5), () => {
        // ASH: momentum-driven jet near the vent, buoyant rise that slows toward the neutral-buoyancy
        // height, entrainment bends it downwind as it climbs, then it spreads into a drifting umbrella.
        const hnb = min(heat.mul(0.32).add(0.2), max(seaY.add(0.95).sub(oy), 0.12));
        const rise = float(1).sub(smoothstep(hnb.mul(0.55), hnb, hAbove));
        const umb = smoothstep(hnb.mul(0.75), hnb, hAbove);
        const wUp = heat.mul(0.08).add(0.1).mul(rise).sub(umb.mul(0.008));
        vel.y.assign(mix(vel.y, wUp, float(1).sub(exp(dt.mul(-1.2)))));
        const entrain = smoothstep(0, hnb, hAbove).mul(1.1).add(0.25);
        const target = wind.add(turb).add(sdir.mul(umb.mul(0.08)));
        vel.xz.assign(mix(vel.xz, target, float(1).sub(exp(dt.mul(entrain).negate()))));
      }).ElseIf(kind.lessThan(1.5), () => { // STEAM: rises, bends early, fades fast
        vel.y.assign(vel.y.mul(exp(dt.mul(-0.6))));
        vel.xz.assign(mix(vel.xz, wind.mul(0.8).add(turb), float(1).sub(exp(dt.mul(-1.2)))));
      }).ElseIf(kind.lessThan(2.5), () => { // DUST: rush out, drag, settle
        vel.xz.assign(vel.xz.mul(exp(dt.mul(-1.3))).add(wind.mul(0.3).mul(dt)));
        vel.y.assign(vel.y.mul(exp(dt.mul(-1.1))).sub(dt.mul(0.015)));
      }).ElseIf(kind.lessThan(3.5), () => { // HAZE: low, slow drift
        vel.xz.assign(mix(vel.xz, wind.mul(0.5).add(turb.mul(0.3)), float(1).sub(exp(dt.mul(-0.5)))));
        vel.y.assign(0);
      }).ElseIf(kind.lessThan(4.5), () => { // BUBBLE: steady rise, wobble
        vel.xz.assign(vec2(sin(u.time.mul(7).add(seed.mul(40))), cos(u.time.mul(6).add(seed.mul(31)))).mul(0.012));
      }).ElseIf(kind.lessThan(5.5), () => { // SMOKER: slow, spreading, billowing under water
        vel.y.assign(vel.y.mul(exp(dt.mul(-0.15))));
        vel.xz.assign(turb.mul(0.1));
      }).ElseIf(kind.lessThan(6.5), () => { // TURBID: steady rise, a little churn, bent gently downstream as it climbs
        const frac = saturate(hAbove.div(max(p2.z.sub(oy), 1e-3))); // p2.z: sea surface
        vel.y.assign(vel.y.mul(exp(dt.mul(-0.05))));
        vel.xz.assign(turb.mul(0.05).add(wind.mul(frac.mul(0.06))));
      }).ElseIf(ballistic.greaterThan(0.5), () => { // FOUNTAIN / BOMB / TEPHRA: ballistic, a touch of drag; landed bombs rest
        const flying = step(1e-5, dot(vel, vel));
        vel.y.subAssign(dt.mul(G).mul(flying));
        vel.mulAssign(exp(dt.mul(-0.15)));
      }).ElseIf(floating.greaterThan(0.5), () => { // PUMICE / SLICK: ride the sea surface downwind
        vel.xz.assign(mix(vel.xz, wind.mul(0.12).add(turb.mul(0.2)), float(1).sub(exp(dt.mul(-0.4)))));
        vel.y.assign(0);
        pos.y.assign(p2.w);
      }).Else(() => { // PDC: the base hugs the ground, runs downhill, drags, spreads and thickens; the loft rises off it
        const e = 0.05;
        const gx = groundAt(pos.x.add(e), pos.z).sub(groundAt(pos.x.sub(e), pos.z)).div(2 * e);
        const gz = groundAt(pos.x, pos.z.add(e)).sub(groundAt(pos.x, pos.z.sub(e))).div(2 * e);
        vel.xz.assign(vel.xz.sub(vec2(gx, gz).mul(dt.mul(0.4))).mul(exp(dt.mul(-0.85))).add(wind.mul(dt.mul(0.25))).add(turb.mul(dt.mul(4))).add(sdir.mul(dt.mul(0.02))));
        const g = groundAt(pos.x, pos.z);
        If(heat.lessThan(1.5), () => { // dense base (heat slot < 2: not lofting)
          pos.y.assign(mix(pos.y, g.add(seed.mul(0.04).add(0.01).add(p0.w.mul(0.015))), float(1).sub(exp(dt.mul(-5))))); // a billowing layer that thickens
          vel.y.assign(0);
        }).Else(() => { // co-ignimbrite ash: rides the current for ~1 s, then lifts off, slows and drifts downwind
          const lift = smoothstep(0.5, 1.5, p0.w);
          vel.y.assign(mix(vel.y, lift.mul(seed.mul(0.05).add(0.04)), float(1).sub(exp(dt.mul(-1.5)))));
          vel.xz.assign(mix(vel.xz, wind.mul(0.8).add(turb), float(1).sub(exp(dt.mul(lift.mul(-0.6))))));
          pos.y.assign(max(pos.y, mix(g.add(0.012), pos.y, lift)));
        });
      });
      pos.addAssign(vel.mul(dt));
      const ground = groundAt(pos.x, pos.z).add(0.004);
      const age = p0.w.add(dt).toVar();
      // ejecta: fountain spray dies when it falls back; a bomb lands, stops and glows out on the ground
      If(ballistic.greaterThan(0.5).and(pos.y.lessThan(ground)).and(vel.y.lessThan(0)), () => {
        pos.y.assign(ground);
        If(isK(PK.BOMB).lessThan(0.5), () => { age.assign(life); }).Else(() => { vel.assign(vec3(0)); life.assign(age.add(1.4)); });
      });
      pos.y.assign(mix(max(pos.y, ground), pos.y, max(underwater, floating)));
      // leaving the block or reaching the water surface: die (the render fades them before that)
      If(max(abs(pos.x), abs(pos.z)).greaterThan(HALF).or(underwater.greaterThan(0.5).and(pos.y.greaterThan(p2.z))), () => { age.assign(life); });
      p0.assign(vec4(pos, age));
      p1.assign(vec4(vel, life));
    });
    P0.element(id).assign(p0);
    P1.element(id).assign(p1);
    P2.element(id).assign(p2);
  })().compute(count);
  const partHigh = buildPart(ATMO.PARTICLES_HIGH);
  const partLow = buildPart(ATMO.PARTICLES_LOW);

  // ---- render ----
  const noise = cloudNoiseTexture();
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
    const isK = (k: number) => step(abs(kind.sub(k)), 0.5);
    const isUnder = step(3.5, kind).mul(step(kind, 6.5));
    const alive = float(q0.w.lessThan(q1.w)).mul(under ? isUnder : float(1).sub(isUnder));
    const tl = saturate(q0.w.div(max(q1.w, 1e-3)));
    const kA = vec4(isK(PK.ASH), isK(PK.STEAM), isK(PK.DUST), isK(PK.HAZE));
    const kB = vec4(isK(PK.BUBBLE), isK(PK.SMOKER), isK(PK.TURBID), isK(PK.PDC));
    const kC = vec3(isK(PK.FOUNTAIN), isK(PK.BOMB), isK(PK.TEPHRA));
    const kD = vec2(isK(PK.PUMICE), isK(PK.SLICK));
    const streak = kC.x.add(kC.y).add(kC.z);
    // start size and growth per kind; ash widens with height above its (slim) vent
    const hAb = q0.y.sub(q2.w);
    const pdcLoft = step(1.5, q2.z);
    const size0 = dot(kA, vec4(0.016, 0.012, 0.04, 0.18)).add(kA.y.mul(q2.z).mul(0.012))
      .add(kB.x.mul(hash(q2.y.mul(91)).mul(0.002).add(0.003))).add(dot(kB.yzw, vec3(0.01, 0.01, mix(float(0.04), float(0.05), pdcLoft))))
      .add(dot(kC, vec3(0.007, 0.011, 0.009))).add(dot(kD, vec2(0.004, 0.05)));
    // a PDC widens as it runs out (base) and billows up as it lofts
    const grow = dot(kA, vec4(0.05, 0.07, 0.15, 0.25)).add(dot(kB.xyz, vec3(0.001, 0.06, 0.03))).add(kB.w.mul(mix(float(0.17), float(0.26), pdcLoft))).add(kD.y.mul(0.28));
    const size = size0.add(grow.mul(sqrt(tl))).add(kA.x.mul(saturate(hAb.mul(0.3))));
    // ejecta streak along their screen-space velocity (ember trails)
    const vv = cameraViewMatrix.mul(vec4(q1.xyz, 0)).xyz;
    const speed = length(q1.xyz);
    const stretch = float(1).add(speed.mul(mix(float(6), float(9), kC.y)).mul(streak));
    const back = normalize(q1.xyz.add(vec3(0, 1e-5, 0))).mul(size.mul(stretch.sub(1)).mul(0.42)).mul(streak);
    // a slick lies on the sea: squash its billboard by the view elevation (reads as a flat patch)
    const viewY = abs(normalize(q0.xyz.sub(cameraPosition)).y);
    mat.positionNode = q0.xyz.sub(back);
    mat.scaleNode = vec2(size.mul(stretch), size.mul(mix(float(1), max(viewY, 0.08), kD.y))).mul(alive);
    mat.rotationNode = mix(mix(q2.y.mul(TAU).add(q0.w.mul(0.15)), atan(vv.y, vv.x), streak), float(0), kD.y);
    const vKA = varying(kA, 'vPlumeKA');
    const vKB = varying(kB, 'vPlumeKB');
    const vKC = varying(vec4(kC, 0), 'vPlumeKC');
    const vKD = varying(kD, 'vPlumeKD');
    const vLife = varying(vec4(tl, q0.w, q2.z, q2.y), 'vPlumeLife');
    const vH = varying(vec2(hAb, speed), 'vPlumeH');
    const vXZ = varying(q0.xz, 'vPlumeXZ');
    const vPos = varying(q0.xyz, 'vPlumePos');
    const vView = varying(vec2(cameraViewMatrix.mul(vec4(q0.xyz, 1)).z, size), 'vPlumeView');

    const shade = Fn(() => {
      const k = vKA, kb = vKB, kc = vKC, kd = vKD, t = vLife.x, age = vLife.y, heat = vLife.z, seed = vLife.w;
      const loftF = step(1.5, heat); // PDC: lofting ash cloud (heat + 2) vs dense base
      const quv = uv();
      const q = quv.mul(2).sub(1);
      const rr = length(q);
      // billowy puff: Perlin-Worley eats the disc edge, evolving with age (soft, noise-eroded sprites)
      // (PDC: slightly finer, faster-churning noise and deeper erosion: lumpy lobes, not smooth tubes)
      const n3 = texture3D(noise, vec3(q.mul(mix(float(0.32), float(0.38), kb.w)).add(seed.mul(5.7)), age.mul(mix(float(0.03), float(0.07), kb.w)).add(seed.mul(3.1)))).r;
      const puff = saturate(float(1).sub(rr).mul(1.3).sub(float(1).sub(n3).mul(mix(float(0.75), float(0.88), kb.w))).mul(2.4)).mul(smoothstep(1.0, 0.75, rr));
      const ring = smoothstep(1.0, 0.75, rr).mul(smoothstep(0.35, 0.8, rr)).add(smoothstep(0.35, 0.0, length(q.sub(vec2(-0.3, 0.3)))).mul(0.8));
      const speck = smoothstep(1.0, 0.4, rr);
      // ember streak: hot head at +x (direction of motion), trail fading behind
      const across = smoothstep(0.9, 0.0, abs(q.y));
      const trail = pow(saturate(quv.x), 1.6).mul(across).mul(smoothstep(1.0, 0.8, quv.x).mul(0.5).add(0.5));
      const isStreak = kc.x.add(kc.y).add(kc.z);
      const shapeA = mix(mix(mix(puff, ring, kb.x), trail, isStreak), speck, kd.x);
      const nv = normalize(vec3(q.x, q.y, sqrt(max(float(1).sub(rr.mul(rr)), 0.05))));
      const keyV = cameraViewMatrix.mul(vec4(cloudShadowU.keyDir, 0)).xyz;
      const wrap = saturate(dot(nv, normalize(keyV)).add(0.5).div(1.5));
      const keyCol = tKeyColor();
      const amb = mix(skyU.horizon, skyU.zenith, q.y.mul(0.5).add(0.5)).mul(0.75);
      // turbid smoker column: height fraction from the vent (vH.x above it) to the sea surface (heat slot)
      const colFrac = saturate(vH.x.div(max(heat.sub(vPos.y).add(vH.x), 1e-3)));
      // ash: near-black at the vent, lighter grey as it rises and thins
      const ashCol = mix(vec3(0.05, 0.045, 0.04), vec3(0.44, 0.42, 0.4), smoothstep(0.0, 0.3, vH.x));
      const albedo = ashCol.mul(k.x)
        .add(vec3(0.93, 0.94, 0.96).mul(k.y)).add(vec3(0.56, 0.45, 0.33).mul(k.z)).add(vec3(0.6, 0.55, 0.48).mul(k.w))
        .add(vec3(0.8, 0.92, 1.0).mul(kb.x.mul(1.1))).add(vec3(0.05, 0.05, 0.06).mul(kb.y)).add(mix(vec3(0.035, 0.03, 0.026), vec3(0.24, 0.215, 0.19), smoothstep(0, 0.9, colFrac)).mul(kb.z))
        .add(mix(vec3(0.19, 0.17, 0.155), vec3(0.44, 0.41, 0.38), loftF).mul(kb.w)).add(vec3(0.78, 0.74, 0.64).mul(kd.x)).add(vec3(0.42, 0.52, 0.38).mul(kd.y));
      const litC = albedo.mul(amb.add(keyCol.mul(wrap).mul(0.3)));
      // a PDC keeps its ash grey: a third of the warm sky / low-sun tint is taken out (no pink tubes)
      const lit = mix(litC, vec3(dot(litC, vec3(0.3, 0.55, 0.15))), kb.w.mul(0.35));
      // orange underglow from the lava: ash just above the crater, the foot of a submarine smoker column
      const glow = vec3(1.0, 0.33, 0.07).mul(exp(max(vH.x, 0).mul(-24)).mul(min(heat, 1).mul(k.x))
        .add(exp(max(vH.x, 0).mul(-45)).mul(kb.z.mul(0.35)))
        .add(k.z.mul(0.6).mul(exp(age.mul(-1.5))).mul(min(heat, 1)))
        .add(kb.w.mul(float(1).sub(loftF)).mul(0.1).mul(exp(age.mul(-1.4))).mul(min(heat, 1))) // PDC base: faint heat near the vent
        .mul(q.y.negate().mul(0.5).add(0.6)).mul(6));
      // incandescent ejecta cooling from yellow to deep red; tephra is black rock with a brief hot head
      const cool = mix(exp(age.mul(-1.1)), exp(age.mul(-0.3)), kc.y);
      const ember = mix(vec3(0.8, 0.1, 0.02), vec3(1.0, 0.5, 0.12), cool).mul(mix(float(1.5), float(4.5), cool)).mul(mix(float(0.7), float(1), kc.y));
      const tephra = mix(vec3(0.035, 0.03, 0.028).mul(amb.add(keyCol.mul(0.2))), vec3(1.0, 0.4, 0.1).mul(2), exp(age.mul(-5)).mul(0.5));
      const streakCol = mix(ember, tephra, kc.z);
      const shimmer = sin(age.mul(9).add(seed.mul(30))).mul(0.2).add(0.8);
      const alpha = k.x.mul(pow(float(1).sub(t), 1.3).mul(0.38)).mul(float(1).sub(smoothstep(0.12, 0.3, vH.x)))
        .add(k.y.mul(pow(float(1).sub(t), 1.6).mul(0.36)).mul(min(heat.add(0.3), 1)))
        .add(k.z.mul(pow(float(1).sub(t), 1.5).mul(0.75)))
        .add(k.w.mul(sin(t.mul(Math.PI)).mul(0.2)))
        .add(kb.x.mul(0.45).mul(float(1).sub(pow(t, 4))))
        .add(kb.y.mul(pow(float(1).sub(t), 1.5).mul(0.5)).mul(shimmer))
        .add(kb.z.mul(float(1).sub(smoothstep(0.65, 1, colFrac)).mul(0.7)))
        .add(kb.w.mul(mix(pow(float(1).sub(t), 1.4).mul(0.4), smoothstep(0, 0.2, t).mul(pow(float(1).sub(t), 1.2)).mul(0.3), loftF)))
        .add(isStreak.mul(mix(float(0.9), float(1), kc.y)).mul(float(1).sub(pow(t, 6))))
        .add(kd.x.mul(0.85).mul(smoothstep(1, 0.85, t)))
        .add(kd.y.mul(sin(t.mul(Math.PI)).mul(0.22)));
      const edge = smoothstep(0, 0.2, float(HALF).sub(max(abs(vXZ.x), abs(vXZ.y))));
      const fadeIn = mix(smoothstep(0, 0.25, age), float(1), isStreak);
      // drawn after the cloud composite: hidden by cloud opacity only where the particle is behind the cloud front
      const occ = under ? float(1) : clouds.occlusion(vPos);
      const near = smoothstep(0.12, 0.55, length(vPos.sub(cameraPosition))); // no balloon particles in the lens
      // soft contact with the terrain (ground-hugging currents, ash at the vent): no hard intersection lines
      const soft = under ? float(1) : mix(saturate(vView.x.sub(sceneViewZ(screenUV)).div(max(vView.y.mul(0.5), 1e-3))), float(1), isStreak.add(kd.x).add(kd.y).min(1));
      const a = shapeA.mul(alpha).mul(fadeIn).mul(edge).mul(occ).mul(near).mul(soft).mul(tCutHard(vPos));
      return vec4(mix(lit.add(glow), streakCol, isStreak), a);
    })();
    mat.colorNode = vec4(shade.rgb, 1);
    mat.opacityNode = shade.a;
    const sprite = new THREE.Sprite(mat);
    sprite.name = mat.name;
    sprite.count = opts.highQuality ? ATMO.PARTICLES_HIGH : ATMO.PARTICLES_LOW;
    sprite.frustumCulled = false;
    // underwater before the water sheet (2) so its refraction shows them; the rest after the clouds (6),
    // depth-aware against the cloud front (clouds.occlusion) so plumes in front of clouds stay in front
    sprite.renderOrder = under ? 1 : 7;
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
  flash.renderOrder = 8;

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
