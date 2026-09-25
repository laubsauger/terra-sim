// Diorama staging: the slice sits on a two-tier plinth (walnut tray over a dark slate base with a
// brass inlay) on an endless studio floor that dissolves into the sky backdrop. The backdrop
// (scene.backgroundNode) and the aerial haze (scene.fogNode) both evaluate the same sky in the view
// direction, so the floor fades into the background with no visible horizon seam.
import * as THREE from 'three/webgpu';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import {
  Fn, vec2, vec3, float, positionLocal, positionWorld, cameraPosition, normalize, length, max, min, abs,
  exp, pow, mix, smoothstep, saturate, texture, fog,
} from 'three/tsl';
import { BLOCK_SIZE } from '../sim/layout';
import { skyColor, skyU } from './sky';
import { lookTextures } from './textures';
import { voxelToWorldY, Y_RENDER_BOTTOM } from './space';

type V3 = THREE.Node<'vec3'>;

export const TRAY_H = 0.06;
export const BASE_H = 0.2;
const TRAY_W = BLOCK_SIZE + 0.14;
export const BASE_W = BLOCK_SIZE + 0.62;

export interface Backdrop {
  object: THREE.Object3D;
  /** Re-seat the plinth under the block (block bottom moves with vertical exaggeration). */
  update(): void;
  /** World y of the floor. */
  readonly floorY: number;
  dispose(): void;
}

/** Box SDF on xz (negative inside). */
const boxSdf = (p: THREE.Node<'vec2'>, half: number) => Fn(() => {
  const q = abs(p).sub(half);
  return length(max(q, vec2(0))).add(min(max(q.x, q.y), 0));
})();

/**
 * Studio backdrop in world direction `dir`: below the horizon a dark, cool sweep with a soft band of
 * sky-tinted haze at the horizon; above it the real sky (sun disc, stars). Dims with the night.
 */
export const studioColor = (dir: V3, discs: boolean): V3 => Fn(() => {
  const haze = mix(skyU.horizon, vec3(0.07, 0.09, 0.11), 0.8).mul(0.6);
  const sweep = mix(haze, vec3(0.016, 0.02, 0.026), smoothstep(-0.02, -0.28, dir.y));
  const below = sweep.mul(float(1).sub(skyU.night.mul(0.75)));
  return mix(below, skyColor(dir, discs), smoothstep(-0.01, 0.06, dir.y));
})() as V3;

/**
 * Aerial haze centred on the block: clear over the diorama, dense far out on the floor, where it
 * becomes exactly the studio backdrop → the floor has no edge.
 */
export function installAtmosphere(scene: THREE.Scene): void {
  const dir = normalize(positionWorld.sub(cameraPosition)) as V3;
  scene.backgroundNode = studioColor(normalize(positionLocal) as V3, true);
  const rCentre = length(positionWorld.xz);
  const far = float(1).sub(exp(pow(max(rCentre.sub(4.5), 0).mul(0.11), 1.6).negate()));
  const camD = length(positionWorld.sub(cameraPosition));
  const aerial = float(1).sub(exp(camD.mul(-0.006)));
  const f = float(1).sub(float(1).sub(far).mul(float(1).sub(aerial)));
  const col = mix(skyColor(dir, false), studioColor(dir, false), far.div(max(far.add(aerial), 1e-4)));
  scene.fogNode = fog(col, f);
}

export function createBackdrop(): Backdrop {
  const tex = lookTextures();
  const group = new THREE.Group();
  group.name = 'backdrop';

  // ---- walnut tray directly under the slice ----
  const trayGeo = new RoundedBoxGeometry(TRAY_W, TRAY_H, TRAY_W, 3, 0.018);
  const trayMat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  {
    const p = positionWorld;
    // grain runs along x on ±z faces and the top, along z on ±x faces: long stretched noise
    const g1 = texture(tex.detail, vec2(p.x.mul(0.08), p.z.mul(1.6).add(p.y.mul(3)))).g;
    const g2 = texture(tex.detail, vec2(p.x.mul(1.3).add(p.y.mul(2)), p.z.mul(0.1))).b;
    const grain = mix(g1, g2, 0.5);
    const rings = smoothstep(0.35, 0.65, grain).mul(0.45).add(texture(tex.detail, p.xz.mul(0.7)).a.mul(0.25));
    trayMat.colorNode = mix(vec3(0.085, 0.042, 0.022), vec3(0.2, 0.105, 0.055), rings);
    trayMat.roughnessNode = mix(float(0.32), float(0.55), grain);
  }
  const tray = new THREE.Mesh(trayGeo, trayMat);
  tray.name = 'plinth-tray';
  tray.position.y = -TRAY_H / 2;

  // ---- brass inlay between the tiers ----
  const brassGeo = new RoundedBoxGeometry(TRAY_W + 0.05, 0.022, TRAY_W + 0.05, 2, 0.008);
  const brassMat = new THREE.MeshStandardNodeMaterial({ color: 0xd9a54a, metalness: 1, roughness: 0.28 });
  const brass = new THREE.Mesh(brassGeo, brassMat);
  brass.name = 'plinth-brass';
  brass.position.y = -TRAY_H - 0.011;

  // ---- dark slate base ----
  const baseGeo = new RoundedBoxGeometry(BASE_W, BASE_H, BASE_W, 4, 0.05);
  const baseMat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  {
    const p = positionWorld;
    const s = texture(tex.detail, p.xz.mul(0.35).add(p.y.mul(0.5))).r;
    const speck = texture(tex.detail, vec2(p.x.add(p.z).mul(2.2), p.y.mul(2.2))).b;
    baseMat.colorNode = vec3(0.052, 0.055, 0.062).mul(s.mul(0.35).add(0.8)).mul(speck.mul(0.3).add(0.85));
    baseMat.roughnessNode = mix(float(0.42), float(0.7), s);
  }
  const base = new THREE.Mesh(baseGeo, baseMat);
  base.name = 'plinth-base';
  base.position.y = -TRAY_H - 0.022 - BASE_H / 2;

  // ---- studio floor ----
  const floorY = -TRAY_H - 0.022 - BASE_H;
  const floorGeo = new THREE.CircleGeometry(60, 96);
  floorGeo.rotateX(-Math.PI / 2);
  const floorMat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.92 });
  {
    const p = positionWorld;
    const d = boxSdf(p.xz, BASE_W / 2);
    // soft contact shadow hugging the plinth + a broad occlusion pool under the whole diorama
    const contact = exp(max(d, 0).mul(-9)).mul(0.55).add(exp(max(d, 0).mul(-1.4)).mul(0.25));
    const paper = mix(vec3(0.07, 0.076, 0.086), skyU.horizon.mul(0.1), 0.15);
    const mottled = texture(tex.detail, p.xz.mul(0.06)).r.mul(0.12).add(0.94);
    floorMat.colorNode = paper.mul(mottled).mul(float(1).sub(saturate(contact)));
  }
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.name = 'floor';
  floor.position.y = floorY;

  for (const m of [tray, brass, base]) { m.castShadow = true; m.receiveShadow = true; }
  floor.receiveShadow = true;
  group.add(tray, brass, base, floor);

  const update = () => { group.position.y = voxelToWorldY(Y_RENDER_BOTTOM); };
  update();

  return {
    object: group,
    update,
    get floorY() { return group.position.y + floorY; },
    dispose() {
      for (const g of [trayGeo, brassGeo, baseGeo, floorGeo]) g.dispose();
      for (const m of [trayMat, brassMat, baseMat, floorMat]) m.dispose();
    },
  };
}
