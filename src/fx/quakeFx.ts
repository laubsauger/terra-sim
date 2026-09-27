// Quake ground FX: a shockwave ring racing over the terrain (a translucent overlay on the terrain surface
// that re-draws the already rendered ground with a lifted, brightened, refracted band, and a brief dark
// "rumble" tint behind it), plus dust puffs shaken loose from steep slopes and mountains as the ring
// passes (soft GPU sprites that rise, drift and settle). Reads sim/display fields only (V15).
// Storage buffers bound: overlay 2 (display cols + motion), dust spawn 4, dust update 3, sprites 3 (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, int, uint, vec2, vec3, vec4, uniform, uniformArray, instanceIndex, instancedArray, hash,
  sin, cos, exp, mix, atan, smoothstep, saturate, max, min, floor, abs, length, normalize, dot, sqrt, step, pow, uv, varying, texture,
  positionGeometry, positionWorld, screenUV, viewportSharedTexture, cameraViewMatrix, transformNormalToView,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NX, CELL, VOXEL_H } from '../sim/layout';
import { tColIdx } from '../sim/tslLayout';
import { columnSampler, displayBuffers, tWorldToCell, tWorldY, vertEx, HALF } from '../render/space';
import { createColumnGrid } from '../render/terrain';
import { skyU } from '../render/sky';
import { sceneViewZ } from '../render/shared';
import { lookTextures } from '../render/textures';
import { RING, DUST, SCORCH_S, SCORCH_WET_S, ringRadius, ringWidth } from './fxModel';
import { tTorusD, fxMRT, type F, type V3 } from './fxTsl';

type V4 = THREE.Node<'vec4'>;
const TAU = Math.PI * 2;
export { SCORCH_S } from './fxModel';
/** Concurrent quakes drawn by the ground overlay. */
export const QUAKE_SLOTS = 2;

export interface QuakeSlot { x: number; z: number; mag: number; t0: number; strength: number }

export interface QuakeFx {
  ground: THREE.Mesh;
  dust: THREE.Sprite;
  /** Dust particle state: P0 = (pos, age), P1 = (vel, delay + life), P2 = (delay, seed, size, strength). */
  P0: StorageNode<'vec4'>; P1: StorageNode<'vec4'>; P2: StorageNode<'vec4'>;
  /** Start a quake's ground FX (now = FX clock s). Returns the slot index used. */
  start(renderer: THREE.WebGPURenderer, q: Omit<QuakeSlot, 't0'>, now: number, seaLevel: number, dust?: boolean): number;
  /**
   * Scorched, glowing crater floor + darkened dust blanket around an impact (x, z cells, r crater radius cells).
   * wet: an ocean impact (a short-lived faint glow and a stirred-up seabed instead, SCORCH_WET_S).
   */
  scorch(x: number, z: number, r: number, now: number, wet?: boolean): void;
  /** Advance; returns true while anything is still visible. */
  update(renderer: THREE.WebGPURenderer, dt: number, now: number): boolean;
  setHighQuality(v: boolean): void;
  dispose(): void;
}

export function createQuakeFx(fields: GpuFields, renderer: THREE.WebGPURenderer, opts: { highQuality: boolean }): QuakeFx {
  const S = columnSampler(fields);
  const { cols } = displayBuffers(fields);
  const slots: (QuakeSlot | null)[] = Array(QUAKE_SLOTS).fill(null);
  // (ex, ez cells, age s (<0 off), strength 0..1), (R max cells, band half-width cells, -, -)
  const qA = uniformArray(Array.from({ length: QUAKE_SLOTS }, () => new THREE.Vector4(0, 0, -1, 0)), 'vec4');
  const qB = uniformArray(Array.from({ length: QUAKE_SLOTS }, () => new THREE.Vector4(1, 1, 0, 0)), 'vec4');
  const scorchU = uniform(new THREE.Vector4(0, 0, 1, -1)); // (x, z, crater radius cells, age s; < 0 off)
  let scorchT0 = -1e9, scorchLife = SCORCH_S;
  const scorchLifeU = uniform(SCORCH_S);

  // ---------------------------------------------------------------- ground overlay
  const gu = tWorldToCell(positionGeometry.x), gv = tWorldToCell(positionGeometry.z);
  // terrain normal from the neighbour columns' display heights (grid reads at the column-centre vertices)
  const hsAt = (du: number, dv: number): F => (cols.element(tColIdx(int(floor(gu.add(du + 0.5))), int(floor(gv.add(dv + 0.5))))) as unknown as V4).x as F;
  /** Ring shape per slot at cell (u, v): x = crest (leading bright band − trailing dip), y = tint, zw = radial dir. */
  const ringAt = (u: F, v: F, k: number) => {
    const a = qA.element(k) as unknown as V4, b = qB.element(k) as unknown as V4;
    const on = step(0, a.z).mul(a.w);
    const dx = tTorusD(u, a.x), dz = tTorusD(v, a.y);
    const d = length(vec2(dx, dz));
    const r = min(a.z.mul(RING.SPEED), b.x);
    const w = b.y;
    const x = d.sub(r).div(w);
    const fade = float(1).sub(smoothstep(b.x.mul(0.55), b.x, r)).mul(float(1).div(r.div(60).add(1)));
    const crest = exp(x.mul(x).negate()).sub(exp(x.add(1.6).mul(x.add(1.6)).negate()).mul(0.45)).mul(fade).mul(on);
    const inside = smoothstep(r.add(w), r.sub(w.mul(2)), d).mul(float(1).sub(smoothstep(b.x.mul(0.75), b.x.mul(1.15), d)));
    const tint = inside.mul(exp(a.z.div(-RING.TINT_S))).mul(float(1).sub(exp(a.z.mul(-4)))).mul(on);
    return { crest, tint, dir: vec2(dx, dz).div(max(d, 1e-3)) };
  };
  const info = varying(Fn(() => {
    const c = S.corners(gu, gv);
    const h = S.height(c);
    return vec2(h, S.level(c).level.sub(h)); // (height, water depth; −2 on dry land)
  })(), 'vQkInfo');
  const nVary = varying(Fn(() => {
    const k = vertEx.mul(VOXEL_H);
    const hl = hsAt(-1, 0), hr = hsAt(1, 0), hd = hsAt(0, -1), hu = hsAt(0, 1);
    return normalize(vec3(hl.sub(hr).mul(k), 2 * CELL, hd.sub(hu).mul(k)));
  })(), 'vQkN');
  const gmat = new THREE.MeshBasicNodeMaterial();
  gmat.name = 'quakeGround';
  gmat.transparent = true;
  gmat.depthWrite = false;
  gmat.fog = false; // the re-drawn scene colour is already fogged
  gmat.alphaTest = 0.004;
  gmat.positionNode = Fn(() => {
    let bump: F = float(0);
    for (let k = 0; k < QUAKE_SLOTS; k++) bump = bump.add(max(ringAt(gu, gv, k).crest, 0));
    // lifted 0.3 layer against z-fighting; the crest heaves the ground ~1 layer
    return vec3(positionGeometry.x, tWorldY(info.x.add(0.3).add(bump.mul(2.2))), positionGeometry.z);
  })();
  const shade = Fn(() => {
    const u = tWorldToCell(positionWorld.x), v = tWorldToCell(positionWorld.z);
    const crest = float(0).toVar(), tint = float(0).toVar(), push = vec2(0).toVar();
    for (let k = 0; k < QUAKE_SLOTS; k++) {
      const r = ringAt(u, v, k);
      crest.addAssign(r.crest);
      tint.assign(max(tint, r.tint));
      push.addAssign(r.dir.mul(r.crest));
    }
    // refraction: shift the re-drawn ground along the ring's radial direction (in screen space)
    const dirV = cameraViewMatrix.mul(vec4(push.x, 0, push.y, 0)).xy;
    const base = viewportSharedTexture(screenUV.add(vec2(dirV.x, dirV.y.negate()).mul(0.012))).rgb;
    const bright = max(crest, 0);
    const lum = dot(base, vec3(0.3, 0.55, 0.15));
    const dusty = mix(base, vec3(0.66, 0.56, 0.43).mul(lum.mul(1.4).add(0.02)), saturate(tint.mul(0.4).add(bright.mul(0.45))));
    // lit dust haze on the crest (additive, so the band reads on shadowed ground too), dark dip behind it
    const haze = vec3(0.62, 0.52, 0.4).mul(skyU.sunIntensity.mul(0.05).add(0.03)).mul(bright);
    const col = dusty.mul(bright.mul(1.4).add(1)).add(haze).mul(float(1).sub(tint.mul(0.42))).mul(float(1).sub(max(crest.negate(), 0).mul(0.6)));
    // impact scorch: incandescent crater floor cooling to a dark scar within ~5 s; a charred crater and rim,
    // a sooty ejecta blanket out to ~2.8 radii and, for big strikes, dark ejecta rays out to ~5 radii. They
    // hold for a while, then fade out gradually by SCORCH_S.
    const sa = scorchU.w, on = step(0, sa);
    const du = tTorusD(u, scorchU.x), dv = tTorusD(v, scorchU.y);
    const sd = length(vec2(du, dv)).div(scorchU.z);
    const nz = texture(lookTextures().detail, positionWorld.xz.mul(7)).r;
    const nz2 = texture(lookTextures().detail, positionWorld.xz.mul(2.3)).g;
    const hot = smoothstep(1.05, 0.2, sd.add(nz.mul(0.35))).mul(exp(sa.div(-1.7))).mul(on);
    const hold = smoothstep(scorchLifeU, scorchLifeU.mul(0.25), sa).mul(smoothstep(0, 0.4, sa)).mul(on);
    const crater = smoothstep(2.0, 1.15, sd.add(nz.sub(0.5).mul(0.3))); // floor, walls and the raised rim
    const blanket = smoothstep(3.0, 1.3, sd.add(nz2.sub(0.5).mul(0.9)).add(nz.sub(0.5).mul(0.3))).mul(nz2.mul(0.45).add(0.55));
    const ang = atan(dv, du);
    const rayA = sin(ang.mul(7).add(nz2.mul(1.5))).mul(sin(ang.mul(11).add(1.3))).mul(0.5).add(0.5);
    const rays = pow(rayA, 5).mul(smoothstep(5, 1.5, sd.add(nz2.sub(0.5)))).mul(smoothstep(1, 1.6, sd))
      .mul(smoothstep(5, 9, scorchU.z)); // rays only for big craters (radius ≳ 5–9 cells)
    const char = max(max(crater.mul(0.85), blanket.mul(0.72)), rays.mul(0.6)).mul(hold);
    const glowCol = mix(vec3(1.0, 0.18, 0.03), vec3(1.0, 0.62, 0.22), hot).mul(hot.mul(hot).mul(6));
    // sooty, not only darker: toward a warm black
    const scorched = mix(col, col.mul(0.12).add(vec3(0.026, 0.022, 0.018)), char).add(glowCol);
    const dry = step(info.y, -0.02); // the quake ring and the soot are dry-land marks
    // on the seabed (seen through the water): a faint dim glow on the crater floor cooling in ~2–3 s, a
    // muddy darker silt ring at the rim and a turbid, silty tint around it; no soot, no rays
    const wet = float(1).sub(dry);
    const holdW = smoothstep(scorchLifeU, scorchLifeU.mul(0.3), sa).mul(smoothstep(0, 0.3, sa)).mul(on);
    const glowW = smoothstep(1.0, 0.25, sd.add(nz.mul(0.3))).mul(exp(sa.div(-1.1))).mul(on);
    const silt = smoothstep(0.75, 1.15, sd.add(nz.sub(0.5).mul(0.3))).mul(smoothstep(2.3, 1.3, sd.add(nz2.sub(0.5).mul(0.8))));
    const turbid = smoothstep(2.8, 0.6, sd.add(nz2.sub(0.5).mul(0.9))).mul(nz2.mul(0.4).add(0.6));
    const lumW = dot(col, vec3(0.3, 0.55, 0.15));
    const seabed = mix(mix(col, vec3(0.46, 0.4, 0.3).mul(lumW.mul(1.3).add(0.05)), turbid.mul(0.5).mul(holdW)), col.mul(0.5), silt.mul(0.55).mul(holdW))
      .add(vec3(1.0, 0.35, 0.08).mul(glowW.mul(glowW).mul(1.4)));
    const aW = saturate(turbid.mul(0.6).add(silt.mul(0.6)).mul(holdW).add(glowW.mul(1.5))).mul(wet);
    const aD = saturate(abs(crest).mul(3).add(tint.mul(4)).add(char.mul(3)).add(hot.mul(4))).mul(dry);
    return vec4(mix(seabed, scorched, dry), max(aD, aW));
  })();
  gmat.colorNode = vec4(shade.rgb, 1);
  gmat.opacityNode = shade.a;
  fxMRT(gmat, renderer, transformNormalToView(nVary) as V3); // terrain's own normal: AO stays correct under the band
  const ground = new THREE.Mesh(createColumnGrid(), gmat);
  ground.name = 'quakeGround';
  ground.frustumCulled = false;
  // after the underwater plumes (1), before the water sheet (2): the re-drawn ground is the opaque scene
  ground.renderOrder = 1.9;
  ground.visible = false;

  // ---------------------------------------------------------------- dust
  const N = DUST.POOL_HIGH;
  const mk = (name: string) => { const b = instancedArray(N, 'vec4') as unknown as StorageNode<'vec4'>; b.setName(name); return b; };
  const P0 = mk('quakeDustP0'), P1 = mk('quakeDustP1'), P2 = mk('quakeDustP2');
  for (let i = 0; i < N; i++) (P0.value.array as Float32Array)[i * 4 + 3] = 1; // age 1 ≥ life 0: dead
  const su = {
    epi: uniform(new THREE.Vector4()), // (ex, ez, R cells, magnitude factor 0..1)
    range: uniform(new THREE.Vector2()), // (offset, count)
    seed: uniform(0), sea: uniform(76), dt: uniform(0), time: uniform(0),
  };
  const colH = (x: THREE.Node<'int'>, z: THREE.Node<'int'>) => (cols.element(tColIdx(x, z)) as unknown as V4);
  const spawnK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(su.range.y)), () => { Return(); });
    const id = instanceIndex.add(uint(su.range.x)).toVar();
    const fid = float(id);
    const rnd = (salt: number) => hash(fid.mul(0.7071).add(su.seed.mul(13.37)).add(salt * 17.13));
    // slope / mountain puffs: up to 8 candidates in the disk, accepted by steepness + altitude;
    // otherwise (often) a low "skirt" puff on any dry column: the ring drags a dust skirt over the land
    const found = float(0).toVar(), cx = float(0).toVar(), cz = float(0).toVar(), cr = float(0).toVar();
    const ch = float(0).toVar(), cs = float(0).toVar(), gx = float(0).toVar(), gz = float(0).toVar();
    for (let k = 0; k < 8; k++) {
      If(found.lessThan(0.5), () => {
        const r = sqrt(rnd(k * 4 + 1)).mul(su.epi.z).toVar();
        const a = rnd(k * 4 + 2).mul(TAU);
        const px = su.epi.x.add(r.mul(cos(a))), pz = su.epi.y.add(r.mul(sin(a)));
        const xi = int(floor(px.add(0.5))), zi = int(floor(pz.add(0.5)));
        const c = colH(xi, zi).toVar();
        const dhx = colH(xi.add(1), zi).x.sub(colH(xi.sub(1), zi).x).mul(0.5);
        const dhz = colH(xi, zi.add(1)).x.sub(colH(xi, zi.sub(1)).x).mul(0.5);
        const slope = length(vec2(dhx, dhz)); // layers per cell
        const alt = c.x.sub(su.sea);
        const dry = step(c.z, 0.03);
        const score = smoothstep(0.5, 2.4, slope).add(smoothstep(8, 30, alt).mul(0.7)).mul(dry).toVar();
        const far = r.div(su.epi.z);
        // quieter far out: the ring weakens with distance
        If(rnd(k * 4 + 3).lessThan(score.mul(float(1).sub(far.mul(0.6)))), () => {
          found.assign(1); cx.assign(px); cz.assign(pz); cr.assign(r); ch.assign(c.x); cs.assign(min(score, 1.4));
          gx.assign(dhx); gz.assign(dhz);
        }).ElseIf(dry.greaterThan(0.5).and(cs.lessThan(0.01)).and(rnd(k * 4 + 4).lessThan(float(0.5).sub(far.mul(0.3)))), () => {
          cx.assign(px); cz.assign(pz); cr.assign(r); ch.assign(c.x); cs.assign(0.02); // skirt candidate
        });
      });
    }
    If(cs.greaterThan(0.01), () => {
      const skirt = float(1).sub(found);
      const wx = cx.sub(floor(cx.div(NX)).mul(NX)), wz = cz.sub(floor(cz.div(NX)).mul(NX));
      const pos = vec3(wx.add(0.5).mul(CELL).sub(HALF), tWorldY(ch).add(0.004), wz.add(0.5).mul(CELL).sub(HALF));
      const delay = cr.div(RING.SPEED).add(rnd(40).mul(0.25));
      const m = su.epi.w;
      const life = mix(rnd(41).mul(DUST.LIFE1 - DUST.LIFE0).add(DUST.LIFE0).mul(m.mul(0.4).add(0.6)), rnd(41).mul(1.5).add(2.2), skirt);
      // downslope + outward drift, up-kick grows with magnitude; skirts hug the ground and rush outward
      const outDir = normalize(vec2(tTorusD(cx, su.epi.x), tTorusD(cz, su.epi.y)).add(vec2(1e-3, 0)));
      const down = vec2(gx, gz).negate().mul(0.004);
      const j = vec2(rnd(42), rnd(43)).sub(0.5).mul(0.02);
      const hv = outDir.mul(mix(0.012, 0.05, skirt)).add(down).add(j);
      const vy = mix(rnd(44).mul(0.05).add(0.03).mul(m.mul(0.8).add(0.5)), rnd(44).mul(0.012).add(0.008), skirt);
      const size = mix(rnd(46).mul(0.035).add(0.03).add(m.mul(0.03)), rnd(46).mul(0.03).add(0.035), skirt);
      P0.element(id).assign(vec4(pos, 0));
      P1.element(id).assign(vec4(hv.x, vy, hv.y, delay.add(life)));
      P2.element(id).assign(vec4(delay, rnd(45), size, mix(cs, float(0.3), skirt)));
    }).Else(() => {
      P0.element(id).assign(vec4(0, 0, 0, 1));
      P1.element(id).assign(vec4(0, 0, 0, 0));
    });
  })().compute(N);

  const buildUpdate = (count: number) => Fn(() => {
    const id = instanceIndex;
    If(id.greaterThanEqual(uint(count)), () => { Return(); });
    const p0 = P0.element(id).toVar(), p1 = P1.element(id).toVar();
    If(p0.w.lessThan(p1.w), () => {
      const dt = su.dt;
      const age = p0.w.add(dt);
      const delay = (P2.element(id) as unknown as V4).x;
      const pos = p0.xyz.toVar(), vel = p1.xyz.toVar();
      If(age.greaterThan(delay), () => {
        const t = age.sub(delay);
        // billow up, drag, then settle back down
        const vy = vel.y.mul(exp(dt.mul(-0.9))).sub(dt.mul(smoothstep(1.0, 3.5, t)).mul(0.012));
        const vxz = vel.xz.mul(exp(dt.mul(-0.55))).add(vec2(sin(t.mul(1.7).add(float(id))), cos(t.mul(1.3).add(float(id).mul(0.37)))).mul(dt).mul(0.004));
        vel.assign(vec3(vxz.x, vy, vxz.y));
        pos.addAssign(vel.mul(dt));
      });
      P0.element(id).assign(vec4(pos, age));
      P1.element(id).assign(vec4(vel, p1.w));
    });
  })().compute(count);
  const updHigh = buildUpdate(DUST.POOL_HIGH), updLow = buildUpdate(DUST.POOL_LOW);

  const dmat = new THREE.SpriteNodeMaterial();
  dmat.name = 'quakeDust';
  dmat.transparent = true;
  dmat.depthWrite = false;
  dmat.fog = true;
  dmat.alphaTest = 0.004;
  fxMRT(dmat, renderer);
  {
    const q0 = P0.element(instanceIndex) as unknown as V4, q1 = P1.element(instanceIndex) as unknown as V4, q2 = P2.element(instanceIndex) as unknown as V4;
    const t = q0.w.sub(q2.x);
    const life = max(q1.w.sub(q2.x), 1e-3);
    const tl = saturate(t.div(life));
    const alive = step(0, t).mul(float(q0.w.lessThan(q1.w)));
    const size = q2.z.mul(sqrt(tl).mul(1.8).add(0.35)).mul(alive);
    dmat.positionNode = q0.xyz;
    dmat.scaleNode = vec2(size);
    dmat.rotationNode = q2.y.mul(TAU).add(t.mul(0.12));
    const centreZ = cameraViewMatrix.mul(vec4(q0.xyz, 1)).z; // sprite centre, view space
    const vA = varying(vec4(tl, q2.y, q2.w, q0.w.sub(q2.x)), 'vDustA');
    const vB = varying(vec4(centreZ, size, q0.x, q0.z), 'vDustB');
    const sh = Fn(() => {
      const tl2 = vA.x, seed = vA.y, strength = vA.z, tt = vA.w;
      const q = uv().mul(2).sub(1);
      const rr = length(q);
      const nse = sin(q.x.mul(4.3).add(seed.mul(17))).mul(sin(q.y.mul(5.1).add(seed.mul(11)))).mul(0.5)
        .add(sin(q.x.add(q.y).mul(8.5).add(seed.mul(5))).mul(0.2));
      // lumpy cauliflower edge: two octaves of angular noise eat into the disc
      const ang = q.y.atan(q.x);
      const lump = sin(ang.mul(5).add(seed.mul(31))).mul(0.12).add(sin(ang.mul(9).sub(seed.mul(17)).add(tt.mul(0.4))).mul(0.07));
      const puff = smoothstep(1.0, 0.05, rr.add(nse.mul(0.35)).add(lump)).mul(smoothstep(1.0, 0.7, rr));
      const nv = normalize(vec3(q.x, q.y, sqrt(max(float(1).sub(rr.mul(rr)), 0.05))));
      const keyV = normalize(cameraViewMatrix.mul(vec4(skyU.sunDir, 0)).xyz);
      const wrap = saturate(dot(nv, keyV).add(0.6).div(1.6));
      const key = mix(skyU.sunColor.mul(skyU.sunIntensity), vec3(0.6, 0.7, 1.0).mul(0.3), step(0.02, skyU.night));
      const amb = mix(skyU.horizon, skyU.zenith, q.y.mul(0.5).add(0.5)).mul(0.65);
      const albedo = mix(vec3(0.42, 0.33, 0.24), vec3(0.56, 0.47, 0.36), seed); // ochre rock dust, darker than cloud
      const lit = albedo.mul(amb.add(key.mul(wrap).mul(0.26)));
      // soft particles: fade where the puff meets the ground behind it
      const soft = saturate(vB.x.sub(sceneViewZ(screenUV)).div(max(vB.y.mul(0.6), 1e-3)));
      const edge = smoothstep(0, 0.15, float(HALF).sub(max(abs(vB.z), abs(vB.w))));
      const a = puff.mul(min(strength, 1).mul(0.42).add(0.16)).mul(smoothstep(0, 0.35, tt)).mul(pow(float(1).sub(tl2), 1.4)).mul(soft).mul(edge);
      return vec4(lit, a);
    })();
    dmat.colorNode = vec4(sh.rgb, 1);
    dmat.opacityNode = sh.a;
  }
  const dust = new THREE.Sprite(dmat);
  dust.name = 'quakeDust';
  dust.count = opts.highQuality ? DUST.POOL_HIGH : DUST.POOL_LOW;
  dust.frustumCulled = false;
  dust.renderOrder = 4;
  dust.visible = false;

  let hq = opts.highQuality;
  let dustUntil = -1, groundUntil = -1, nextHalf = 0;

  return {
    ground, dust, P0, P1, P2,
    start(r, q, now, seaLevel, withDust = true) {
      // free slot, else the oldest
      let k = slots.findIndex((s) => !s || now - s.t0 > RING.TAIL_S + ringRadius(s.mag) / RING.SPEED);
      if (k < 0) k = slots.reduce((b, s, i) => (s!.t0 < slots[b]!.t0 ? i : b), 0);
      slots[k] = { ...q, t0: now };
      const R = ringRadius(q.mag);
      (qA.array[k] as THREE.Vector4).set(q.x, q.z, 0, q.strength);
      (qB.array[k] as THREE.Vector4).set(R, ringWidth(q.mag), 0, 0);
      groundUntil = Math.max(groundUntil, now + R / RING.SPEED + RING.TAIL_S);
      ground.visible = true;
      if (!withDust) return k;
      // dust: half the pool per quake, alternating (a second quake does not wipe the first one's puffs)
      const pool = hq ? DUST.POOL_HIGH : DUST.POOL_LOW, half = pool / 2;
      su.epi.value.set(q.x, q.z, R * 0.85, Math.min(1, Math.max(0, (q.mag - 5) / 4)));
      su.range.value.set(nextHalf * half, half);
      su.seed.value = (su.seed.value + 7.31) % 1000;
      su.sea.value = seaLevel;
      nextHalf = 1 - nextHalf;
      r.compute(spawnK);
      dustUntil = Math.max(dustUntil, now + R / RING.SPEED + 0.35 + DUST.LIFE1 + 0.5);
      ground.visible = dust.visible = true;
      return k;
    },
    scorch(x, z, rc, now, wet = false) {
      // one scar at a time: a short-lived seabed mark never replaces a land scar that would outlast it
      if (wet && scorchLife - (now - scorchT0) > SCORCH_WET_S) return;
      scorchT0 = now;
      scorchLife = wet ? SCORCH_WET_S : SCORCH_S;
      scorchLifeU.value = scorchLife;
      scorchU.value.set(x, z, Math.max(2, rc), 0);
      groundUntil = Math.max(groundUntil, now + scorchLife);
      ground.visible = true;
    },
    update(r, dt, now) {
      scorchU.value.w = now - scorchT0 < scorchLife ? now - scorchT0 : -1;
      for (let k = 0; k < QUAKE_SLOTS; k++) {
        const s = slots[k];
        const a = qA.array[k] as THREE.Vector4;
        if (s && now - s.t0 <= RING.TAIL_S + ringRadius(s.mag) / RING.SPEED) a.z = now - s.t0;
        else { a.z = -1; slots[k] = null; }
      }
      ground.visible = now < groundUntil;
      const dustOn = now < dustUntil;
      dust.visible = dustOn;
      if (dustOn) {
        su.dt.value = dt;
        su.time.value = now % 1000;
        r.compute(hq ? updHigh : updLow);
      }
      return ground.visible || dustOn;
    },
    setHighQuality(v) { hq = v; dust.count = v ? DUST.POOL_HIGH : DUST.POOL_LOW; },
    dispose() { ground.geometry.dispose(); gmat.dispose(); dmat.dispose(); },
  };
}
