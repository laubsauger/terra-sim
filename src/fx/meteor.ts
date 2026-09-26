// Meteor strike sequence (~3 s cinematic + aftermath), FX only (V15):
//  entry     a bolide streaks in from high in the sky along its travel azimuth (~38° elevation): HDR head
//            (blooms, lights the ground through a moving point light), an ionised glowing trail ribbon and a
//            smoke tail (particles laid along the path) that lingers and drifts;
//  impact    white-hot flash (sprite + light spike) and a short fireball (hot puffs cooling to soot in ~1 s),
//            a low ejecta skirt (dust thrown at 20–40° from the growing rim, biased downrange: butterfly
//            pattern with an uprange forbidden zone for oblique impacts), incandescent ballistic ejecta on
//            arcs (stretched along their velocity), a billowing dust column that stalls into a mushroom cap
//            and is sheared by the atmosphere's wind (atmoTsl.tWind, as the volcanic plumes);
//  aftermath the skirt lands and thins out, embers make small secondary impact puffs where they land, the
//            column fades within ~5 s; the crater floor glows and cools (quakeFx.scorch), the ground ring /
//            tint / shake / tsunami come from fx.ts at the impact moment.
// Heights, reach and counts scale with the impact (crater radius × magnitude): a small strike stays small.
// Puffs use the plume shading: noise-eroded soft sprites, sun-wrapped, dark at the column root. Everything
// honours the slice cut (atmoCut, set by the atmosphere each frame).
// One GPU particle pool (ring buffer; the CPU hands out spawn ranges), two sprite draws (dust: normal
// blending; embers: additive). Storage buffers: spawn 3, update 4 (+ display cols), sprites 4 (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, int, uint, vec2, vec3, vec4, uniform, uniformArray, instanceIndex, instancedArray, hash,
  sin, cos, exp, mix, smoothstep, saturate, max, min, floor, abs, length, normalize, dot, sqrt, step, pow, uv, varying,
  positionGeometry, positionWorld, cameraPosition, cameraViewMatrix, screenUV, texture3D,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { CELL } from '../sim/layout';
import { tColIdx } from '../sim/tslLayout';
import { displayBuffers, tWorldY, HALF } from '../render/space';
import { skyU } from '../render/sky';
import { sceneViewZ } from '../render/shared';
import { atmoCut, tCutHard, tWind, tWindHF } from '../atmo/atmoTsl';
import { cloudNoiseTexture } from '../atmo/noise3d';
import { fxMRT, type F, type V3 } from './fxTsl';

type V4 = THREE.Node<'vec4'>;
const TAU = Math.PI * 2;

export const METEOR = {
  POOL_HIGH: 6144, POOL_LOW: 3072,
  /** Spawns per frame (spawn kernel size). */
  SPAWN_MAX: 2048,
  /** Entry path: elevation (deg) of the incoming trajectory, speed (world/s). */
  ELEV_DEG: 38, SPEED: 2.6,
  /** Seconds from the start of a god-tool strike to the impact; a strike the sim already applied streaks this long. */
  ENTRY_S: 1.5, ENTRY_SHORT_S: 0.32,
  /** Ionised trail length (world) and how long it glows after the impact (s). */
  TRAIL_LEN: 2.4, TRAIL_FADE_S: 0.3,
  /** How long the sequence keeps anything alive after the impact (s): the entry smoke tail is the last to go. */
  AFTER_S: 9,
  GRAVITY: 1.5,
  /** Crater rim radius (world) at impact scale 1 (the god tool's default 12-cell crater). */
  CRATER_W: 12 * CELL,
  /** Dust column: seconds of emission after the impact, top height (world) at impact scale 1. */
  COLUMN_S: 2, COLUMN_H: 0.42,
} as const;

/** Particle kinds. */
const K = { SMOKE: 0, CURTAIN: 1, EMBER: 2, COLUMN: 3, SECONDARY: 4, FIREBALL: 5 } as const;
/** Column puffs with a seed above this form the mushroom cap, the rest the stem. */
const CAP_SEED = 0.65;
const SEGS = 6;

export interface StrikeInfo {
  x: number; z: number;
  /** Travel azimuth (rad), cells: (cos, sin) = (+x, +z). */
  dir: number;
  /** Meteor magnitude (0.5..1.5 random; god tool strength / 12). */
  mag: number;
  /** Crater radius (cells). */
  radius: number;
  /** Impact point (world): ground, or the sea surface. */
  point: THREE.Vector3;
  /** Water depth at the impact (voxel layers). */
  water: number;
}

export interface MeteorFx {
  object: THREE.Group;
  light: THREE.PointLight;
  P0: StorageNode<'vec4'>; P1: StorageNode<'vec4'>; P2: StorageNode<'vec4'>;
  /** Start a strike now; `entry` s until impact. onImpact fires on the impact frame. */
  strike(info: StrikeInfo, now: number, entry: number, onImpact: () => void): void;
  /** Advance (dt real s, now FX clock s); returns true while anything is alive. */
  update(renderer: THREE.WebGPURenderer, dt: number, now: number): boolean;
  readonly active: boolean;
  setHighQuality(v: boolean): void;
  dispose(): void;
}

interface Strike extends StrikeInfo {
  t0: number; tImpact: number; start: THREE.Vector3; v: THREE.Vector3; len: number;
  width0: number; headSize: number; onImpact: (() => void) | null; impacted: boolean; prevHead: THREE.Vector3; smokeAcc: number; columnAcc: number; curtainAcc: number;
}

export function createMeteorFx(fields: GpuFields, renderer: THREE.WebGPURenderer, opts: { highQuality: boolean }): MeteorFx {
  const { cols } = displayBuffers(fields);
  const N = METEOR.POOL_HIGH;
  const mk = (name: string) => { const b = instancedArray(N, 'vec4') as unknown as StorageNode<'vec4'>; b.setName(name); return b; };
  const P0 = mk('meteorP0'), P1 = mk('meteorP1'), P2 = mk('meteorP2');
  for (let i = 0; i < N; i++) (P0.value.array as Float32Array)[i * 4 + 3] = 1; // dead: age 1 ≥ life 0
  // spawn segments: A = (count, kind, first thread, launch progress), B = (pos xyz, scale), C = (pos2 xyz | dir x, dir z, obliquity; wet)
  const segA = uniformArray(Array.from({ length: SEGS }, () => new THREE.Vector4()), 'vec4');
  const segB = uniformArray(Array.from({ length: SEGS }, () => new THREE.Vector4()), 'vec4');
  const segC = uniformArray(Array.from({ length: SEGS }, () => new THREE.Vector4()), 'vec4');
  const u = { cursor: uniform(0), pool: uniform(N), seed: uniform(0), dt: uniform(0), time: uniform(0), total: uniform(0) };

  // ---------------------------------------------------------------- spawn
  const spawnK = Fn(() => {
    const i = instanceIndex;
    If(float(i).greaterThanEqual(u.total), () => { Return(); });
    const fi = float(i);
    const rnd = (salt: number) => hash(fi.mul(0.7131).add(u.seed.mul(17.31)).add(salt * 13.7));
    // which segment: unrolled scan over the cumulative starts
    const a = vec4(0).toVar(), b = vec4(0).toVar(), c = vec4(0).toVar();
    for (let k = 0; k < SEGS; k++) {
      const A = segA.element(k) as unknown as V4;
      If(fi.greaterThanEqual(A.z).and(fi.lessThan(A.z.add(A.x))), () => {
        a.assign(A); b.assign(segB.element(k) as unknown as V4); c.assign(segC.element(k) as unknown as V4);
      });
    }
    const pid = uint(float(u.cursor).add(fi).mod(u.pool)).toVar();
    const kind = a.y, s = b.w, wet = c.w;
    const pos = vec3(0).toVar(), vel = vec3(0).toVar();
    const life = float(1).toVar(), size = float(0.03).toVar(), heat = float(0).toVar();
    // ejecta azimuth: rejection-sampled butterfly around the travel direction (c.xy = dir, c.z = obliquity)
    const ejectDir = () => {
      const phi = float(0).toVar();
      const got = float(0).toVar();
      const base = c.y.atan(c.x);
      for (let t = 0; t < 4; t++) {
        If(got.lessThan(0.5), () => {
          const p = rnd(20 + t).mul(TAU);
          const rel = cos(p.sub(base)); // +1 downrange, −1 uprange
          const side = abs(sin(p.sub(base)));
          const w = mix(float(1), max(rel, 0).mul(0.85).add(pow(side, 3).mul(0.3)).add(0.04), c.z);
          If(rnd(30 + t).lessThan(w), () => { phi.assign(p); got.assign(1); });
        });
      }
      return vec2(cos(phi), sin(phi));
    };
    If(kind.lessThan(0.5), () => { // SMOKE along the head's path this frame
      const p = mix(b.xyz, c.xyz, rnd(1));
      pos.assign(p.add(vec3(rnd(2), rnd(3), rnd(4)).sub(0.5).mul(0.03)));
      vel.assign(vec3(rnd(5), rnd(6), rnd(7)).sub(0.5).mul(0.05).add(vec3(0, 0.012, 0)));
      life.assign(rnd(8).mul(3).add(5)); size.assign(pow(rnd(9), 2).mul(0.03).add(0.012)); heat.assign(1);
    }).ElseIf(kind.lessThan(1.5), () => { // CURTAIN: low skirt (~20–40°), launched from the growing crater rim
      const d = ejectDir();
      const f = a.w; // launch progress 0..1 over the excavation
      const rc = s.mul(METEOR.CRATER_W);
      const el = rnd(10).mul(0.35).add(0.35);
      const sp = rnd(11).mul(0.5).add(float(1.1).sub(f.mul(0.4))).mul(s);
      pos.assign(b.xyz.add(vec3(d.x, 0, d.y).mul(rc.mul(f.mul(0.65).add(0.35)))).add(vec3(0, 0.008, 0)));
      vel.assign(vec3(d.x.mul(cos(el)), sin(el), d.y.mul(cos(el))).mul(sp));
      life.assign(rnd(13).mul(1.4).add(1.8)); size.assign(rnd(14).mul(0.02).add(0.022).mul(s.mul(0.6).add(0.4))); heat.assign(0.35);
    }).ElseIf(kind.lessThan(2.5), () => { // EMBER: incandescent ballistic ejecta
      const d = ejectDir();
      const el = rnd(15).mul(0.7).add(0.45);
      const sp = rnd(16).mul(1.1).add(0.9).mul(s);
      pos.assign(b.xyz.add(vec3(0, 0.01, 0)));
      vel.assign(vec3(d.x.mul(cos(el)), sin(el), d.y.mul(cos(el))).mul(sp));
      life.assign(rnd(17).mul(1.2).add(2.2)); size.assign(rnd(18).mul(0.0025).add(0.0035)); heat.assign(1);
    }).ElseIf(kind.lessThan(3.5), () => { // COLUMN: dust boiling up out of the crater (stem + cap)
      const r = sqrt(rnd(19)).mul(0.4).mul(s.mul(METEOR.CRATER_W));
      const ang = rnd(20).mul(TAU);
      pos.assign(b.xyz.add(vec3(cos(ang).mul(r), rnd(21).mul(0.02).add(0.005), sin(ang).mul(r))));
      // launch speed follows the height the puff will stall at (seed, see the update): no overshoot into one blob
      vel.assign(vec3(cos(ang).mul(0.05), rnd(40).mul(0.3).add(0.1).add(rnd(22).mul(0.05)), sin(ang).mul(0.05)).mul(s));
      // heat slot = impact strength (scale / 1.5): the update's column height, the sprite's root glow
      life.assign(rnd(23).mul(1.6).add(2.8)); size.assign(rnd(24).mul(0.02).add(0.03).mul(s)); heat.assign(min(s, 1.5).div(1.5));
    }).ElseIf(kind.greaterThan(4.5), () => { // FIREBALL: hot gas bursting out of the crater
      const ang = rnd(25).mul(TAU), el = rnd(26).mul(0.8).add(0.6);
      const sp = rnd(27).mul(0.3).add(0.25).mul(s);
      pos.assign(b.xyz.add(vec3(cos(ang), 0, sin(ang)).mul(s.mul(METEOR.CRATER_W * 0.2))).add(vec3(0, 0.01, 0)));
      vel.assign(vec3(cos(ang).mul(cos(el)), sin(el), sin(ang).mul(cos(el))).mul(sp));
      life.assign(rnd(28).mul(0.5).add(0.9)); size.assign(rnd(29).mul(0.03).add(0.04).mul(s)); heat.assign(1);
    });
    P0.element(pid).assign(vec4(pos, 0));
    P1.element(pid).assign(vec4(vel, life));
    P2.element(pid).assign(vec4(kind, rnd(40), size, heat.add(wet.mul(2))));
  })().compute(METEOR.SPAWN_MAX);

  // ---------------------------------------------------------------- update
  const colAt = (p: V3): V4 => cols.element(tColIdx(int(floor(p.x.add(HALF).div(CELL))), int(floor(p.z.add(HALF).div(CELL))))) as unknown as V4;
  const groundAt = (p: V3): F => {
    const c = colAt(p);
    return tWorldY(max(c.x, c.y.add(c.z))) as F; // ground or sea surface
  };
  const buildUpdate = (count: number) => Fn(() => {
    const id = instanceIndex;
    If(id.greaterThanEqual(uint(count)), () => { Return(); });
    const p0 = P0.element(id).toVar(), p1 = P1.element(id).toVar(), p2 = P2.element(id).toVar();
    If(p0.w.greaterThanEqual(p1.w), () => { Return(); });
    const dt = u.dt;
    const kind = p2.x;
    const pos = p0.xyz.toVar(), vel = p1.xyz.toVar(), age = p0.w.add(dt).toVar(), life = p1.w.toVar(), k2 = kind.toVar();
    const size = p2.z.toVar(), tag = p2.w.toVar();
    const seed = p2.y;
    const drift = vec3(sin(u.time.mul(0.3).add(seed.mul(20))), 0, cos(u.time.mul(0.23).add(seed.mul(13)))).mul(0.01);
    If(kind.lessThan(0.5), () => { // SMOKE: drag to a slow drift, gentle rise
      vel.assign(mix(vel, drift.add(vec3(0.02, 0.01, 0.01)), float(1).sub(exp(dt.mul(-0.8)))));
    }).ElseIf(kind.lessThan(1.5), () => { // CURTAIN: ballistic with strong drag (a low skirt), lands and creeps on
      vel.y.subAssign(dt.mul(METEOR.GRAVITY));
      vel.assign(vel.mul(exp(dt.mul(-2.6))));
    }).ElseIf(kind.lessThan(2.5), () => { // EMBER: ballistic, little drag
      vel.y.subAssign(dt.mul(METEOR.GRAVITY));
      vel.assign(vel.mul(exp(dt.mul(-0.25))));
    }).ElseIf(kind.lessThan(3.5), () => { // COLUMN: buoyant rise that stalls, the cap spreads out, the wind shears it
      // stem puffs stall at their own height (a continuous stem up to ~0.8 of the top); the rest gather in a
      // flat band at the top and spread into the cap. Height, cap width and rise speed follow the impact
      // scale (heat slot = strength = scale / 1.5).
      const sc = tag.sub(step(1.5, tag).mul(2)).mul(1.5);
      const cap = step(CAP_SEED, seed);
      const hTop = mix(mix(float(0.06), float(0.8), seed.div(CAP_SEED)), mix(float(0.82), float(1), seed.sub(CAP_SEED).div(1 - CAP_SEED)), cap).mul(sc).mul(METEOR.COLUMN_H);
      const hAb = pos.y.sub(groundAt(pos));
      const rise = float(1).sub(smoothstep(hTop.mul(0.5), hTop, hAb));
      const out = vec2(sin(seed.mul(97)), cos(seed.mul(97))).mul(float(1).sub(rise).mul(cap.mul(0.12).add(0.006)).mul(sc));
      // same zonal wind as the plumes, felt more with height (the stem leans, the cap drifts off)
      const wind = tWind(pos.z, tWindHF(pos.y)).mul(smoothstep(0, 0.25, hAb).mul(1.4).add(0.2));
      vel.assign(mix(vel, vec3(out.x.add(wind.x), rise.mul(0.3).mul(sc).add(0.004), out.y.add(wind.y)).add(drift), float(1).sub(exp(dt.mul(-2.4)))));
    }).ElseIf(kind.lessThan(4.5), () => { // SECONDARY puff
      vel.assign(vel.mul(exp(dt.mul(-2))));
    }).Else(() => { // FIREBALL: fast drag, the hot gas keeps rising a little
      vel.assign(vel.mul(exp(dt.mul(-2.4))).add(vec3(0, dt.mul(0.18), 0)));
    });
    pos.addAssign(vel.mul(dt));
    const g = groundAt(pos);
    If(pos.y.lessThan(g), () => {
      pos.y.assign(g);
      If(kind.greaterThan(1.5).and(kind.lessThan(2.5)).and(colAt(pos).z.greaterThan(0.5)), () => {
        age.assign(life); // an ember falling into the sea just goes out (no dust puff on the water)
      }).ElseIf(kind.greaterThan(1.5).and(kind.lessThan(2.5)), () => {
        // ember lands: a small secondary impact puff
        k2.assign(K.SECONDARY); age.assign(0); life.assign(seed.mul(1).add(1.4));
        vel.assign(vec3(0, 0.05, 0)); size.assign(0.02);
      }).Else(() => { vel.assign(vec3(vel.x.mul(0.3), 0, vel.z.mul(0.3))); });
    });
    // ejecta leaving the block die (smoke high up in the sky may hang outside it)
    If(max(abs(pos.x), abs(pos.z)).greaterThan(HALF + 0.15).and(kind.greaterThan(0.5)), () => { age.assign(life); });
    P0.element(id).assign(vec4(pos, age));
    P1.element(id).assign(vec4(vel, life));
    P2.element(id).assign(vec4(k2, seed, size, tag));
  })().compute(count);
  const updHigh = buildUpdate(METEOR.POOL_HIGH), updLow = buildUpdate(METEOR.POOL_LOW);

  // ---------------------------------------------------------------- particle sprites
  const q0 = P0.element(instanceIndex) as unknown as V4, q1 = P1.element(instanceIndex) as unknown as V4, q2 = P2.element(instanceIndex) as unknown as V4;
  const tl = saturate(q0.w.div(max(q1.w, 1e-3)));
  const alive = float(q0.w.lessThan(q1.w));
  const kindOf = (k: number) => step(abs(q2.x.sub(k)), 0.5);
  const wetOf = step(1.5, q2.w);
  const heatOf = q2.w.sub(wetOf.mul(2));

  const dmat = new THREE.SpriteNodeMaterial();
  dmat.name = 'meteorDust';
  dmat.transparent = true; dmat.depthWrite = false; dmat.fog = true; dmat.alphaTest = 0.004;
  fxMRT(dmat, renderer);
  {
    const notEmber = float(1).sub(kindOf(K.EMBER));
    const capOf = step(CAP_SEED, q2.y); // column puffs that end up in the mushroom cap billow wider
    const grow = mix(mix(mix(mix(float(5), float(2.6), kindOf(K.CURTAIN)), capOf.mul(2).add(2.8), kindOf(K.COLUMN)), float(3), kindOf(K.SECONDARY)), float(2.2), kindOf(K.FIREBALL));
    const size = q2.z.mul(float(1).add(grow.mul(sqrt(tl)))).mul(alive).mul(notEmber);
    dmat.positionNode = q0.xyz;
    dmat.scaleNode = vec2(size);
    dmat.rotationNode = q2.y.mul(TAU).add(q0.w.mul(0.1));
    const vA = varying(vec4(tl, q2.y, q0.w, q2.x), 'vMetA');
    const vB = varying(vec4(cameraViewMatrix.mul(vec4(q0.xyz, 1)).z, size, heatOf, wetOf), 'vMetB');
    const vC = varying(vec4(q0.xyz, q0.y.sub(groundAt(q0.xyz))), 'vMetC'); // position + height above the ground
    const noise = cloudNoiseTexture();
    const sh = Fn(() => {
      const t = vA.x, seed = vA.y, age = vA.z, kind = vA.w, heat = vB.z, wet = vB.w, hAb = vC.w;
      const q = uv().mul(2).sub(1);
      const rr = length(q);
      // billowy puff as in the plumes: Perlin-Worley eats the disc edge, evolving with age (no hard discs)
      const n3 = texture3D(noise, vec3(q.mul(0.32).add(seed.mul(5.7)), age.mul(0.05).add(seed.mul(3.1)))).r;
      const puff = saturate(float(1).sub(rr).mul(1.3).sub(float(1).sub(n3).mul(0.75)).mul(2.4)).mul(smoothstep(1.0, 0.75, rr));
      const nv = normalize(vec3(q.x, q.y, sqrt(max(float(1).sub(rr.mul(rr)), 0.05))));
      const keyV = normalize(cameraViewMatrix.mul(vec4(skyU.sunDir, 0)).xyz);
      const wrap = saturate(dot(nv, keyV).add(0.5).div(1.5));
      const key = mix(skyU.sunColor.mul(skyU.sunIntensity), vec3(0.6, 0.7, 1.0).mul(0.3), step(0.02, skyU.night));
      const amb = mix(skyU.horizon, skyU.zenith, q.y.mul(0.5).add(0.5)).mul(0.7);
      const isSmoke = step(kind, 0.5), isCol = step(abs(kind.sub(K.COLUMN)), 0.5), isFire = step(abs(kind.sub(K.FIREBALL)), 0.5);
      const rock = mix(vec3(0.4, 0.33, 0.26), vec3(0.56, 0.48, 0.38), seed);
      // column: dark churned-up rock at its root, paler dust as it rises and thins (like the ash plumes)
      const colDust = mix(vec3(0.13, 0.11, 0.095), vec3(0.6, 0.55, 0.47), smoothstep(0.0, 0.28, hAb));
      const smoke = mix(vec3(0.32, 0.31, 0.3), vec3(0.5, 0.48, 0.46), seed);
      const albedo = mix(mix(mix(mix(rock, colDust, isCol), smoke, isSmoke), vec3(0.08, 0.07, 0.065), isFire),
        vec3(0.82, 0.85, 0.88), wet.mul(float(1).sub(isSmoke)).mul(float(1).sub(isFire)));
      const lit = albedo.mul(amb.add(key.mul(wrap).mul(0.3)));
      // incandescence: fresh smoke next to the head, the column's root, young curtain / secondary puffs
      const glowT = mix(exp(age.mul(-5)), exp(age.mul(-1.6)).mul(q.y.negate().mul(0.5).add(0.5)).mul(0.8), isCol).mul(heat).mul(float(1).sub(wet));
      const glow = vec3(1.0, 0.42, 0.12).mul(glowT.mul(glowT).mul(5));
      // fireball: white-hot core cooling through orange to soot within ~1 s (dimmer in a steam burst)
      const hot = exp(age.mul(-2.6)).mul(isFire);
      const fire = mix(vec3(1.0, 0.26, 0.05), vec3(1.0, 0.8, 0.5), hot).mul(hot.mul(hot).mul(14)).mul(smoothstep(1.0, 0.1, rr)).mul(float(1).sub(wet.mul(0.5)));
      const soft = saturate(vB.x.sub(sceneViewZ(screenUV)).div(max(vB.y.mul(0.6), 1e-3)));
      const edge = max(smoothstep(0, 0.15, float(HALF + 0.1).sub(max(abs(vC.x), abs(vC.z)))), isSmoke);
      const near = smoothstep(0.12, 0.55, length(vC.xyz.sub(cameraPosition))); // no balloon puffs in the lens (as the plumes)
      const alphaK = mix(mix(mix(float(0.45), float(0.42), isCol), float(0.22), isSmoke), float(0.85), isFire);
      const fadeIn = smoothstep(0, mix(float(0.12), float(0.03), isFire), age);
      // the column holds its shape, then thins out over the second half of its life
      const fade = mix(pow(float(1).sub(t), 1.3), float(1).sub(smoothstep(0.45, 1, t)), isCol);
      const a = puff.mul(alphaK).mul(fadeIn).mul(fade).mul(soft).mul(edge).mul(near).mul(tCutHard(vC.xyz));
      return vec4(lit.add(glow).add(fire), a);
    })();
    dmat.colorNode = vec4(sh.rgb, 1);
    dmat.opacityNode = sh.a;
  }
  const dust = new THREE.Sprite(dmat);
  dust.name = 'meteorDust'; dust.count = opts.highQuality ? METEOR.POOL_HIGH : METEOR.POOL_LOW; dust.frustumCulled = false; dust.renderOrder = 4.5; dust.visible = false;

  const emat = new THREE.SpriteNodeMaterial();
  emat.name = 'meteorEmbers';
  emat.transparent = true; emat.depthWrite = false; emat.blending = THREE.AdditiveBlending; emat.alphaTest = 0.002;
  fxMRT(emat, renderer);
  {
    const vv = cameraViewMatrix.mul(vec4(q1.xyz, 0)).xyz;
    const streak = length(vv.xy).mul(0.06);
    const on = alive.mul(kindOf(K.EMBER));
    emat.positionNode = q0.xyz;
    emat.scaleNode = vec2(q2.z.add(streak), q2.z).mul(on);
    emat.rotationNode = vv.y.atan(vv.x);
    const vE = varying(vec4(tl, q2.y, q0.x, q0.z), 'vEmb');
    const qq = uv().mul(2).sub(1);
    const core = pow(saturate(float(1).sub(length(qq.mul(vec2(1, 1.6))))), 1.6);
    const cool = vE.x;
    const col = mix(vec3(1.0, 0.62, 0.25), vec3(1.0, 0.18, 0.03), smoothstep(0.05, 0.6, cool)).mul(mix(float(7), float(0.6), smoothstep(0, 0.8, cool)));
    emat.colorNode = vec4(col, 1);
    emat.opacityNode = core.mul(pow(float(1).sub(cool), 1.5)).mul(tCutHard(vec3(vE.z, 0, vE.w)));
  }
  const embers = new THREE.Sprite(emat);
  embers.name = 'meteorEmbers'; embers.count = opts.highQuality ? METEOR.POOL_HIGH : METEOR.POOL_LOW; embers.frustumCulled = false; embers.renderOrder = 5; embers.visible = false;

  // ---------------------------------------------------------------- head, trail, flash
  const U = {
    head: uniform(new THREE.Vector3()), headAmt: uniform(0), headSize: uniform(0.08),
    start: uniform(new THREE.Vector3()), v: uniform(new THREE.Vector3(1, 0, 0)), dist: uniform(0), len: uniform(0), trailAmt: uniform(0), width: uniform(0.02),
    flash: uniform(new THREE.Vector3()), flashAmt: uniform(0), flashSize: uniform(0.3),
  };
  const sprite = (name: string, pos: THREE.Node, sizeN: THREE.Node, amt: THREE.Node, colr: THREE.Node, falloff: number, order: number) => {
    const m = new THREE.SpriteNodeMaterial();
    m.name = name; m.transparent = true; m.depthWrite = false; m.blending = THREE.AdditiveBlending; m.alphaTest = 0.002;
    fxMRT(m, renderer);
    m.positionNode = pos as V3;
    m.scaleNode = vec2(float(sizeN as F).mul(step(0.001, amt as F)));
    const qq = uv().mul(2).sub(1);
    const r = length(qq);
    const fall = pow(saturate(float(1).sub(r)), falloff).add(pow(saturate(float(1).sub(r.mul(4))), 2).mul(3));
    m.colorNode = vec4((colr as V3).mul(amt as F), 1);
    m.opacityNode = fall.mul(tCutHard(pos as V3));
    const s = new THREE.Sprite(m);
    s.name = name; s.frustumCulled = false; s.renderOrder = order; s.visible = false;
    return s;
  };
  const head = sprite('meteorHead', U.head, U.headSize, U.headAmt, vec3(1.0, 0.86, 0.62).mul(9), 2.4, 6);
  const flash = sprite('meteorFlash', U.flash, U.flashSize, U.flashAmt, vec3(1.0, 0.9, 0.75).mul(22), 2.0, 7);

  // trail: camera-facing ribbon along the straight path, 0 at the head … 1 at the tail
  const TSEG = 64;
  const tpos = new Float32Array((TSEG + 1) * 2 * 3);
  for (let i = 0; i <= TSEG; i++) for (let sd = 0; sd < 2; sd++) { const o = (i * 2 + sd) * 3; tpos[o] = i / TSEG; tpos[o + 1] = sd * 2 - 1; }
  const tidx: number[] = [];
  for (let i = 0; i < TSEG; i++) { const a = i * 2; tidx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  const tgeo = new THREE.BufferGeometry();
  tgeo.setAttribute('position', new THREE.BufferAttribute(tpos, 3));
  tgeo.setIndex(tidx);
  const tmat = new THREE.MeshBasicNodeMaterial();
  tmat.name = 'meteorTrail'; tmat.transparent = true; tmat.depthWrite = false; tmat.blending = THREE.AdditiveBlending; tmat.side = THREE.DoubleSide; tmat.fog = false;
  fxMRT(tmat, renderer);
  {
    const sT = positionGeometry.x, side = positionGeometry.y;
    const along = U.dist.sub(sT.mul(U.len));
    const p = U.start.add(U.v.mul(along));
    const toCam = normalize(cameraPosition.sub(p));
    const sideV = normalize(U.v.cross(toCam));
    const w = U.width.mul(pow(float(1).sub(sT), 0.7).mul(0.85).add(0.15));
    tmat.positionNode = p.add(sideV.mul(side.mul(w)));
    const vT = varying(vec2(sT, side), 'vTrail');
    const across = float(1).sub(abs(vT.y));
    const core = pow(across, 3);
    const hot = vec3(1.0, 0.9, 0.7).mul(8), ion = vec3(0.45, 1.0, 0.75).mul(2.2), dim = vec3(1.0, 0.35, 0.12).mul(0.9);
    const col = mix(mix(hot, ion, smoothstep(0.0, 0.25, vT.x)), dim, smoothstep(0.25, 1.0, vT.x));
    tmat.colorNode = vec4(col.mul(core.mul(0.8).add(across.mul(0.2))), 1);
    tmat.opacityNode = pow(float(1).sub(vT.x), 1.5).mul(U.trailAmt).mul(across).mul(tCutHard(positionWorld));
  }
  const trail = new THREE.Mesh(tgeo, tmat);
  trail.name = 'meteorTrail'; trail.frustumCulled = false; trail.renderOrder = 5.5; trail.visible = false;

  // moving light: the head lights the ground on the way in; a hard spike at the impact. Always in the
  // scene (intensity 0 when idle) so no material recompiles when a meteor starts.
  const light = new THREE.PointLight(0xffd9a8, 0, 0, 2);
  light.name = 'meteorLight';
  light.castShadow = false;

  const object = new THREE.Group();
  object.name = 'meteor';
  object.add(dust, embers, head, flash, trail, light);

  // ---------------------------------------------------------------- CPU sequence
  let hq = opts.highQuality;
  let cursor = 0;
  const strikes: Strike[] = [];
  let aliveUntil = -1;
  const segs: { n: number; kind: number; b: THREE.Vector4; c: THREE.Vector4; w: number }[] = [];
  const addSeg = (n: number, kind: number, b: THREE.Vector4, c: THREE.Vector4, w = 0) => {
    n = Math.floor(n * (hq ? 1 : 0.5));
    if (n > 0 && segs.length < SEGS) segs.push({ n, kind, b, c, w });
  };
  const flushSpawns = (r: THREE.WebGPURenderer) => {
    if (!segs.length) return;
    const pool = hq ? METEOR.POOL_HIGH : METEOR.POOL_LOW;
    let first = 0;
    for (let k = 0; k < SEGS; k++) {
      const s = segs[k];
      const n = s ? Math.min(s.n, METEOR.SPAWN_MAX - first) : 0;
      (segA.array[k] as THREE.Vector4).set(n, s?.kind ?? 0, first, s?.w ?? 0);
      if (s) { (segB.array[k] as THREE.Vector4).copy(s.b); (segC.array[k] as THREE.Vector4).copy(s.c); }
      first += n;
    }
    u.total.value = first;
    u.cursor.value = cursor;
    u.pool.value = pool;
    u.seed.value = (u.seed.value + 3.17) % 997;
    r.compute(spawnK);
    cursor = (cursor + first) % pool;
    segs.length = 0;
  };
  const tmp = new THREE.Vector3();
  /** The light follows the slice cut like the sprites (a strike in the cut-away part lights nothing). */
  const inCut = (p: THREE.Vector3) => (p.x <= atmoCut.value.x && p.z <= atmoCut.value.y ? 1 : 0);

  return {
    object, light, P0, P1, P2,
    get active() { return strikes.length > 0; },
    strike(info, now, entry, onImpact) {
      const el = THREE.MathUtils.degToRad(METEOR.ELEV_DEG);
      const v = new THREE.Vector3(Math.cos(info.dir) * Math.cos(el), -Math.sin(el), Math.sin(info.dir) * Math.cos(el));
      const len = METEOR.SPEED * entry;
      const start = info.point.clone().addScaledVector(v, -len);
      strikes.push({ ...info, t0: now, tImpact: now + entry, start, v, len, width0: 0.012 + 0.012 * Math.min(1.5, info.mag),
        headSize: 0.1 + 0.08 * Math.min(1.5, info.mag), onImpact, impacted: false, prevHead: start.clone(), smokeAcc: 0, columnAcc: 0, curtainAcc: 0 });
      dust.visible = embers.visible = true;
      aliveUntil = Math.max(aliveUntil, now + entry + METEOR.AFTER_S);
    },
    update(r, dt, now) {
      // every strike runs its own timeline (a second click never swallows the first one's impact); the
      // head / trail / flash / light show the latest one
      for (const s of [...strikes]) {
        const vis = s === strikes[strikes.length - 1];
        // impact scale: ejecta speed / reach / counts and the column height grow with the crater and the magnitude
        const scale = Math.sqrt(Math.max(2, s.radius) / 12) * (0.7 + 0.3 * Math.min(1.5, s.mag));
        if (!s.impacted) {
          const p = Math.min(1, (now - s.t0) / Math.max(1e-3, s.tImpact - s.t0));
          const d = s.len * p;
          const h = tmp.copy(s.start).addScaledVector(s.v, d);
          if (vis) {
            U.start.value.copy(s.start); U.v.value.copy(s.v); U.width.value = s.width0; U.headSize.value = s.headSize;
            U.head.value.copy(h);
            U.headAmt.value = 0.35 + 0.65 * Math.min(1, p * 3);
            U.dist.value = d;
            U.len.value = Math.min(METEOR.TRAIL_LEN, d);
            U.trailAmt.value = Math.min(1, p * 4);
            head.visible = trail.visible = true;
            light.position.copy(h);
            light.intensity = (3 + 5 * p) * inCut(h);
          }
          // smoke laid along the path covered this frame
          s.smokeAcc += h.distanceTo(s.prevHead) * 170 * Math.min(1.4, s.mag + 0.3);
          const n = Math.floor(s.smokeAcc);
          s.smokeAcc -= n;
          addSeg(n, K.SMOKE, new THREE.Vector4(s.prevHead.x, s.prevHead.y, s.prevHead.z, 1), new THREE.Vector4(h.x, h.y, h.z, 0));
          s.prevHead.copy(h);
          if (now >= s.tImpact) {
            s.impacted = true;
            if (vis) head.visible = false;
            try { s.onImpact?.(); } catch (e) { console.error('meteor onImpact failed', e); }
            s.onImpact = null;
            const wet = s.water > 0.5 ? 1 : 0;
            const b = new THREE.Vector4(s.point.x, s.point.y, s.point.z, scale);
            const c = new THREE.Vector4(Math.cos(s.dir), Math.sin(s.dir), 1, wet);
            addSeg((wet ? 90 : 280) * scale, K.EMBER, b, c);
            addSeg((wet ? 30 : 60) * scale, K.FIREBALL, b, c);
          }
        }
        if (s.impacted) {
          const t = now - s.tImpact;
          if (vis) {
            U.flash.value.copy(s.point).y += 0.03;
            flash.visible = t < 3;
            const env = Math.exp(-t / 0.13) * Math.min(1, t * 40 + 0.3);
            U.flashAmt.value = env;
            U.flashSize.value = (0.35 + 0.9 * t) * (0.7 + 0.5 * Math.min(1.5, s.mag));
            U.trailAmt.value = Math.exp(-t / METEOR.TRAIL_FADE_S); // the smoke tail stays, the glow goes
            U.width.value = s.width0 * (1 + 1.5 * t);
            U.dist.value = s.len; U.len.value = Math.min(METEOR.TRAIL_LEN, s.len);
            light.position.copy(s.point).y += 0.12;
            light.intensity = (14 * Math.exp(-t / 0.35) + 2 * Math.exp(-t / 1.5)) * inCut(s.point);
            trail.visible = t < METEOR.TRAIL_FADE_S * 8;
          }
          // curtain: launched over the excavation (~0.45 s), from the growing rim, slower as it widens
          if (t < 0.45) {
            s.curtainAcc += dt / 0.45 * 700 * scale;
            const n = Math.floor(s.curtainAcc);
            s.curtainAcc -= n;
            addSeg(n, K.CURTAIN, new THREE.Vector4(s.point.x, s.point.y, s.point.z, scale), new THREE.Vector4(Math.cos(s.dir), Math.sin(s.dir), 1, s.water > 0.5 ? 1 : 0), t / 0.45);
          }
          // rising column, fed hardest right after the impact and tapering off (~420 puffs at scale 1)
          if (t < METEOR.COLUMN_S) {
            s.columnAcc += dt * 2 * 420 / METEOR.COLUMN_S * scale * (1 - t / METEOR.COLUMN_S);
            const n = Math.floor(s.columnAcc);
            s.columnAcc -= n;
            addSeg(n, K.COLUMN, new THREE.Vector4(s.point.x, s.point.y, s.point.z, scale), new THREE.Vector4(0, 0, 0, s.water > 0.5 ? 1 : 0));
          }
          if (t > 3) {
            strikes.splice(strikes.indexOf(s), 1);
            if (vis) { light.intensity = 0; flash.visible = trail.visible = false; }
          }
        }
      }
      flushSpawns(r);
      const on = now < aliveUntil;
      if (on) {
        u.dt.value = dt;
        u.time.value = now % 1000;
        r.compute(hq ? updHigh : updLow);
      } else {
        dust.visible = embers.visible = false;
      }
      return on || strikes.length > 0;
    },
    setHighQuality(v) {
      hq = v;
      dust.count = embers.count = v ? METEOR.POOL_HIGH : METEOR.POOL_LOW;
    },
    dispose() {
      dmat.dispose(); emat.dispose(); tmat.dispose(); tgeo.dispose();
      (head.material as THREE.Material).dispose(); (flash.material as THREE.Material).dispose(); light.dispose();
    },
  };
}
