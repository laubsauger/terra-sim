// TSL helpers shared by the quake / tsunami layers.
import * as THREE from 'three/webgpu';
import { floor, vec4, mrt, packNormalToRGB } from 'three/tsl';
import { NX } from '../sim/layout';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/** Signed shortest torus offset a − b in cells (V1: the world wraps), in [−NX/2, NX/2). */
export const tTorusD = (a: F, b: F): F => {
  const d = a.sub(b);
  return d.sub(floor(d.div(NX).add(0.5)).mul(NX)) as F;
};

/**
 * Transparent FX in the post pipeline's MRT scene pass. three writes the non-'output' attachments of a
 * transparent material without blending, so an overlay would stamp its own normal into the GTAO/SSGI
 * normal target (dark AO patches). Same trick as atmoTsl.fxMRT (kept local: src/atmo is a separate
 * layer): the material writes `normalV` (view space; default a camera-facing normal) and no albedo.
 * Only applied when the active MRT has a normal target (with no MRT a material-level MRT would replace
 * the colour output).
 */
export function fxMRT(mat: THREE.NodeMaterial, renderer: THREE.WebGPURenderer, normalV?: V3): void {
  const n = normalV ? vec4(packNormalToRGB(normalV), 1) : vec4(0.5, 0.5, 1, 1);
  const m = mrt({ normal: n, diffuseColor: vec4(0, 0, 0, 0) });
  Object.defineProperty(mat, 'mrtNode', {
    configurable: true,
    get: () => { const r = renderer.getMRT() as { has?(n: string): boolean } | null; return r?.has?.('normal') ? m : null; },
    set: () => {},
  });
}

export type { F, V3 };
