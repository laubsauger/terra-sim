// Placeholder water surface (proper water shading is T37): flat sheet at the wet water level,
// translucent teal → deep blue by depth, fresnel sky tint, a soft shore foam line.
import * as THREE from 'three/webgpu';
import {
  Fn, vec3, float, varying, positionGeometry, normalView, positionViewDirection,
  mix, smoothstep, saturate, pow, dot, normalize, mx_noise_float, mx_noise_vec3, positionWorld, transformNormalToView,
} from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { tWorldY, tWorldToCell, columnSampler } from './space';
import { createColumnGrid } from './terrain';
import { biomeColor } from './palette';

export function createWater(fields: GpuFields): { object: THREE.Object3D; dispose(): void } {
  const S = columnSampler(fields);
  const gu = tWorldToCell(positionGeometry.x), gv = tWorldToCell(positionGeometry.z);
  const levelAt = () => S.level(S.corners(gu, gv)).level;
  // Depth of the sheet above the (interpolated) seabed; negative where the flat level runs into land.
  const depthVary = varying(Fn(() => {
    const c = S.corners(gu, gv);
    return S.level(c).level.sub(S.height(c));
  })(), 'vWaterDepth');

  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.08, transparent: true, depthWrite: false });
  mat.positionNode = Fn(() => vec3(positionGeometry.x, tWorldY(levelAt()), positionGeometry.z))();

  const d = depthVary.max(0);
  const fres = pow(float(1).sub(saturate(dot(normalView, positionViewDirection))), 3);
  const ripple = mx_noise_float(positionWorld.mul(vec3(30, 0, 30))).mul(0.5).add(0.5);
  const foam = float(1).sub(smoothstep(0.1, 0.9, d.add(ripple.mul(0.35))));
  const body = mix(biomeColor('waterShallow'), biomeColor('waterDeep'), saturate(d.div(14)));
  mat.colorNode = mix(mix(body, vec3(0.45, 0.75, 0.95), fres.mul(0.4)), vec3(1), foam.mul(0.75));
  // Tiny ripple normals so the sun leaves a sparkle trail (real waves are T37).
  const rip = mx_noise_vec3(positionWorld.mul(vec3(28, 0, 28))).mul(vec3(0.06, 0, 0.06));
  mat.normalNode = transformNormalToView(normalize(vec3(0, 1, 0).add(rip)));
  mat.opacityNode = saturate(mix(0.4, 0.86, saturate(d.div(10))).add(fres.mul(0.2)).add(foam.mul(0.4)));
  // Where the flat level is below the land, the depth test already hides it; drop it outright.
  mat.maskNode = depthVary.greaterThan(-0.05);

  const mesh = new THREE.Mesh(createColumnGrid(), mat);
  mesh.name = 'water';
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;
  return { object: mesh, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}
