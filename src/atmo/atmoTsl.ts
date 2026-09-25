// TSL helpers shared by the atmosphere kernels and materials.
import * as THREE from 'three/webgpu';
import { uniform, float, vec2, vec3, vec4, cos, sin, min, floor, saturate, mix, step, select, mrt, fract, screenCoordinate, frameId } from 'three/tsl';
import { NZ, CELL, Y_SEA_NOMINAL } from '../sim/layout';
import { CLIMATE } from '../sim/climateModel';
import { tWorldY } from '../render/space';
import { skyU } from '../render/sky';
import { ATMO, HALF } from './atmoModel';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

/** Sea level (voxel y) seen by all atmosphere kernels; the atmosphere updates it from sim stats. */
export const atmoSeaLevel = uniform(Y_SEA_NOMINAL);

/** Height fraction of world y above sea level over WIND_H. */
export const tWindHF = (y: F): F => saturate(y.sub(tWorldY(atmoSeaLevel)).div(ATMO.WIND_H));

/** Wind (world/s, x and z) at world z and height fraction hf — TSL mirror of atmoModel.windProfile. */
export function tWind(z: F, hf: F): V2 {
  const c = z.add(HALF).div(CELL).sub(0.5);
  const q = c.sub(floor(c.div(NZ)).mul(NZ));
  const d = min(q, float(NZ).sub(q));
  const phi = d.mul(Math.PI / NZ);
  const hemi = select(q.lessThan(float(NZ / 2)), float(1), float(-1));
  const k = hf.mul((ATMO.WIND_TOP - 1) * ATMO.WIND_SCALE * CELL).add(ATMO.WIND_SCALE * CELL);
  const s2 = sin(phi.mul(2));
  const u = cos(phi.mul(4)).mul(-CLIMATE.U0).mul(k).add(s2.mul(s2).mul(hf).mul(hf).mul(ATMO.JET));
  const v = sin(phi.mul(6)).mul(-CLIMATE.V0).mul(hemi).mul(k);
  return vec2(u, v);
}

/** Key light colour × intensity as lighting.ts sets it (sun by day, dim moon at night). */
export const tKeyColor = (): V3 => mix(skyU.sunColor.mul(skyU.sunIntensity), vec3(0.6, 0.7, 1.0).mul(0.3), step(0.02, skyU.night)) as V3;

/** Interleaved gradient noise (Jimenez) per pixel, decorrelated per frame: cheap blue-ish jitter for TRAA. */
export const tPixelJitter = (): F => fract(float(52.9829189).mul(fract(
  screenCoordinate.x.add(float(frameId).mod(64).mul(5.588238)).mul(0.06711056)
    .add(screenCoordinate.y.add(float(frameId).mod(64).mul(5.588238)).mul(0.00583715))))) as F;

/**
 * Transparent FX in the post pipeline's MRT scene pass: three writes the non-'output' attachments of
 * a transparent material without blending, so a sprite quad or the cloud box would stamp its own
 * geometric normal into the GTAO/SSGI normal target (dark AO squares). FX instead write a
 * camera-facing normal and no albedo there (and discard their fully transparent pixels); velocity
 * stays the material's own (camera motion) for TRAA. Only applied when the active MRT has a normal
 * target: with no MRT a material-level MRT would replace the colour output (three NodeMaterial).
 */
export function fxMRT(mat: THREE.NodeMaterial, renderer: THREE.WebGPURenderer): void {
  const m = mrt({ normal: vec4(0.5, 0.5, 1, 1), diffuseColor: vec4(0, 0, 0, 0) });
  Object.defineProperty(mat, 'mrtNode', {
    configurable: true,
    get: () => { const r = renderer.getMRT() as { has?(n: string): boolean } | null; return r?.has?.('normal') ? m : null; },
    set: () => {},
  });
}

export type { F, V2, V3 };
