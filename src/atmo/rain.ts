// §T.42 rain + snow: streak particles that fall only from precipitating columns, plus a soft wet-haze
// curtain under raining cloudlets.
//
// A dead drop picks a random column; if that column's sim precip is above PRECIP_RAIN it spawns at
// the cloud base over it (rate ∝ precip), as rain, or as snow where surfTemp < SNOW_T. Drops fall
// with a little wind drift and die at the ground/water surface. R1.w keeps the spawn column so tests
// can check that every live drop came from a raining column.
// Reads sim fields only (V15). Storage buffers bound: kernel 7, drop sprite 1, curtain 2 (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, int, uint, vec2, vec3, vec4, uniform, instanceIndex, instancedArray,
  hash, sin, mix, smoothstep, max, floor, abs, pow, uv, varying, step, cameraPosition, normalize,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NX, NZ, CELL, BLOCK_SIZE } from '../sim/layout';
import { tColIdx, iMin } from '../sim/tslLayout';
import { tWorldY } from '../render/shared';
import { skyU } from '../render/sky';
import { ATMO, HALF } from './atmoModel';
import type { Clouds } from './clouds';
import { tWind, tWindHF, fxMRT, tCutHard } from './atmoTsl';

type F = THREE.Node<'float'>;
type V4 = THREE.Node<'vec4'>;

export interface Rain {
  drops: THREE.Sprite;
  curtain: THREE.Sprite;
  /** R0 = (pos, state 0 dead | 1 rain | 2 snow), R1 = (vel, spawn column index). */
  R0: StorageNode<'vec4'>; R1: StorageNode<'vec4'>;
  uniforms: { time: THREE.UniformNode<'float', number>; dt: THREE.UniformNode<'float', number>; frame: THREE.UniformNode<'float', number> };
  compute(renderer: THREE.WebGPURenderer, dt: number): void;
  setHighQuality(v: boolean): void;
  dispose(): void;
}

export function createRain(fields: GpuFields, clouds: Clouds, opts: { highQuality: boolean; renderer: THREE.WebGPURenderer }): Rain {
  const N = ATMO.RAIN_HIGH;
  const precip = fields.cur('precip'), surfTemp = fields.cur('surfTemp'), surfY = fields.cur('surfY'), water = fields.cur('water');
  const R0 = instancedArray(N, 'vec4') as unknown as StorageNode<'vec4'>;
  const R1 = instancedArray(N, 'vec4') as unknown as StorageNode<'vec4'>;
  R0.setName('rainR0'); R1.setName('rainR1');
  const u = { time: uniform(0), dt: uniform(0), frame: uniform(0), spawnP: uniform(0) };

  const build = (count: number) => Fn(() => {
    const id = instanceIndex;
    If(id.greaterThanEqual(uint(count)), () => { Return(); });
    const r0 = R0.element(id).toVar(), r1 = R1.element(id).toVar();
    const fid = float(id);
    const rnd = (salt: number) => hash(fid.mul(0.6180).add(u.frame.mul(7.77)).add(salt * 11.3));
    If(r0.w.lessThan(0.5), () => {
      const cx = iMin(int(floor(rnd(1).mul(NX))), int(NX - 1)), cz = iMin(int(floor(rnd(2).mul(NZ))), int(NZ - 1));
      const c = tColIdx(cx, cz);
      const p = precip.element(c) as unknown as F;
      const x = float(cx).add(rnd(4)).mul(CELL).sub(HALF), z = float(cz).add(rnd(5)).mul(CELL).sub(HALF);
      const wx = clouds.weatherAt(x, z).toVar();
      // falls only from raining columns, and only where a cloud actually hangs overhead
      const intensity = smoothstep(ATMO.PRECIP_RAIN, ATMO.PRECIP_RAIN * 5, p).mul(step(ATMO.PRECIP_RAIN, p)).mul(smoothstep(0.03, 0.3, wx.x));
      If(rnd(3).lessThan(intensity.mul(u.spawnP)), () => {
        const y = wx.w.add(0.02).sub(rnd(6).mul(0.04));
        const snow = step(surfTemp.element(c) as unknown as F, float(ATMO.SNOW_T));
        const w = tWind(z, tWindHF(y)).x;
        const fall = mix(float(ATMO.RAIN_SPEED).mul(rnd(7).mul(0.3).add(0.85)), float(ATMO.SNOW_SPEED).mul(rnd(7).mul(0.4).add(0.8)), snow);
        r0.assign(vec4(x, y, z, snow.add(1)));
        r1.assign(vec4(w.mul(mix(float(0.25), float(0.5), snow)), fall.negate(), 0, float(c)));
      });
    }).Else(() => {
      const pos = r0.xyz.toVar();
      pos.addAssign(r1.xyz.mul(u.dt));
      const snow = step(1.5, r0.w);
      pos.x.addAssign(sin(u.time.mul(1.7).add(fid.mul(0.37))).mul(0.03).mul(u.dt).mul(snow));
      const cx = int(floor(pos.x.add(HALF).div(CELL))), cz = int(floor(pos.z.add(HALF).div(CELL)));
      const c = tColIdx(cx, cz);
      const ground = tWorldY((surfY.element(c) as unknown as F).add(max(water.element(c) as unknown as F, 0)));
      const out = max(abs(pos.x), abs(pos.z)).greaterThan(HALF).or(pos.y.lessThan(ground));
      r0.assign(vec4(pos, mix(r0.w, float(0), float(out))));
    });
    R0.element(id).assign(r0);
    R1.element(id).assign(r1);
  })().compute(count);
  const kHigh = build(ATMO.RAIN_HIGH), kLow = build(ATMO.RAIN_LOW);

  // ---- drops ----
  const mat = new THREE.SpriteNodeMaterial();
  mat.name = 'rain';
  mat.transparent = true;
  mat.depthWrite = false;
  mat.fog = true;
  fxMRT(mat, opts.renderer);
  mat.alphaTest = 0.01;
  const d0 = R0.element(instanceIndex) as unknown as V4;
  const alive = step(0.5, d0.w), isSnow = step(1.5, d0.w);
  mat.positionNode = d0.xyz;
  mat.scaleNode = mix(vec2(0.0026, 0.055), vec2(0.011, 0.011), isSnow).mul(alive);
  const vSnow = varying(isSnow, 'vSnow');
  const vDrop = varying(d0.xyz, 'vDropPos');
  const q = uv().mul(2).sub(1);
  const streak = smoothstep(1, 0.1, abs(q.x)).mul(smoothstep(1, 0.2, abs(q.y)));
  const flake = smoothstep(1, 0.3, q.length());
  const amb = mix(skyU.horizon, skyU.zenith, 0.6);
  const keyCol = mix(skyU.sunColor.mul(skyU.sunIntensity), vec3(0.6, 0.7, 1.0).mul(0.3), step(0.02, skyU.night));
  mat.colorNode = vec4(mix(amb.mul(vec3(0.85, 0.9, 1.05)).mul(1.1), amb.mul(0.9).add(keyCol.mul(0.25)), vSnow), 1);
  mat.opacityNode = mix(streak.mul(0.42), flake.mul(0.9), vSnow).mul(clouds.occlusion(vDrop)).mul(tCutHard(vDrop));
  const drops = new THREE.Sprite(mat);
  drops.name = 'rain';
  drops.count = opts.highQuality ? ATMO.RAIN_HIGH : ATMO.RAIN_LOW;
  drops.frustumCulled = false;
  drops.renderOrder = 7; // after water (writes depth) and the cloud composite (6); depth-aware vs clouds

  // ---- wet-haze curtain: soft vertical sheets on a fixed jittered grid, shown under raining cloud ----
  const cmat = new THREE.SpriteNodeMaterial();
  cmat.name = 'rainCurtain';
  cmat.transparent = true;
  cmat.depthWrite = false;
  cmat.fog = true;
  fxMRT(cmat, opts.renderer);
  cmat.alphaTest = 0.005;
  const CG = ATMO.CURTAIN_GRID;
  const cid = float(instanceIndex);
  const cxw = float(instanceIndex.mod(uint(CG))).add(hash(cid.mul(1.7)).mul(0.8).add(0.1)).div(CG).mul(BLOCK_SIZE).sub(HALF);
  const czw = float(instanceIndex.div(uint(CG))).add(hash(cid.mul(2.3)).mul(0.8).add(0.1)).div(CG).mul(BLOCK_SIZE).sub(HALF);
  const cw = clouds.weatherAt(cxw, czw);
  const cs = clouds.sampleSummary(cxw, czw);
  const cground = tWorldY(cs.w);
  const h = max(cw.w.sub(cground), 0.001);
  const wet = smoothstep(ATMO.PRECIP_RAIN, ATMO.PRECIP_RAIN * 4, cs.y).mul(smoothstep(0.15, 0.6, cw.x));
  cmat.positionNode = vec3(cxw, cground.add(h.mul(0.5)), czw);
  cmat.scaleNode = vec2(float(BLOCK_SIZE / CG).mul(1.6), h).mul(step(0.02, wet));
  const vC = varying(vec4(wet, cxw, czw, cid), 'vCurtain');
  const cq = uv();
  const viewY = abs(normalize(vec3(vC.y, cground, vC.z).sub(cameraPosition)).y);
  const streaks = sin(cq.x.mul(31).add(vC.w.mul(1.7))).mul(0.25).add(sin(cq.x.mul(13).add(vC.w)).mul(0.2)).add(0.75);
  const prof = smoothstep(0, 0.35, cq.x).mul(smoothstep(1, 0.65, cq.x)).mul(pow(cq.y, 0.7)).mul(smoothstep(0, 0.12, cq.y));
  cmat.colorNode = vec4(mix(skyU.horizon, skyU.zenith, 0.5).mul(0.55).add(vec3(0.05)), 1);
  const vCPos = varying(vec3(cxw, cground.add(h.mul(0.5)), czw), 'vCurtainPos');
  cmat.opacityNode = prof.mul(streaks).mul(vC.x).mul(0.22).mul(float(1).sub(pow(viewY, 1.5))).mul(clouds.occlusion(vCPos)).mul(tCutHard(vCPos));
  const curtain = new THREE.Sprite(cmat);
  curtain.name = 'rainCurtain';
  curtain.count = CG * CG;
  curtain.frustumCulled = false;
  curtain.renderOrder = 7;

  let hq = opts.highQuality;
  return {
    drops, curtain, R0, R1, uniforms: u,
    compute(renderer, dt) {
      u.dt.value = dt;
      u.frame.value = (u.frame.value + 1) % 4096;
      u.spawnP.value = Math.min(1, dt * 10);
      renderer.compute(hq ? kHigh : kLow);
    },
    setHighQuality(v) { hq = v; drops.count = v ? ATMO.RAIN_HIGH : ATMO.RAIN_LOW; },
    dispose() { mat.dispose(); cmat.dispose(); },
  };
}
