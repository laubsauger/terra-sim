// Plate motion arrows: one instanced chunky 3D arrow per plate (≤ MAX_PLATES instances, 2 draw calls with the
// dark casing), floating above the terrain at the plate centroid, pointing along the plate velocity, longer
// when faster. Per-instance data lives in uniform arrays (no per-frame allocation, V10); height comes from
// the render columns on the GPU (1 storage buffer).
import * as THREE from 'three/webgpu';
import { Fn, float, vec3, vec4, positionGeometry, normalGeometry, instanceIndex, uniformArray, varying, select, max, cos, sin, normalize, dot } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { MAX_PLATES } from '../sim/worldData';
import { columnSampler, tWorldToCell, tWorldY, HALF, cellToWorld } from '../render/space';
import { NX, NZ } from '../sim/layout';
import { hexToLinear } from './colormaps';
import { untonemap } from './overlayTsl';
import type { SheetUniforms } from './sheet';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/** Arrow width (world units); length runs from MIN_LEN (still) to MAX_LEN at SPEED_FULL. */
export const ARROW_W = 0.105;
export const MIN_LEN = 0.15; // head plus a stub of shaft
export const MAX_LEN = 0.6;
/** cells/My at which an arrow reaches MAX_LEN (kinematics MAX_SPEED). */
export const SPEED_FULL = 2.4;
/** Height above the highest ground under the arrow (world units). */
const LIFT = 0.1;
const HEAD = 0.95, RIGID = 0.2; // local units (× ARROW_W): head length, unstretched shaft next to the head

/** Linear length scaling: relative plate speeds read directly off the arrows (√ scaling made them all look alike). */
export const arrowLength = (speed: number) => MIN_LEN + (MAX_LEN - MIN_LEN) * Math.min(1, Math.max(0, speed) / SPEED_FULL);

function arrowGeometry(): THREE.BufferGeometry {
  const s = new THREE.Shape();
  const hw = 0.26, hh = 0.62;
  s.moveTo(-1, -hw); s.lineTo(0, -hw); s.lineTo(0, -hh); s.lineTo(HEAD, 0); s.lineTo(0, hh); s.lineTo(0, hw); s.lineTo(-1, hw); s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.34, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.07, bevelSegments: 2, curveSegments: 1 });
  g.rotateX(-Math.PI / 2); // extrusion → +y (thickness), shape plane → xz; +x forward
  g.computeVertexNormals();
  return g;
}

export interface ArrowState { id: number; x: number; z: number; heading: number; length: number; speed: number; visible: boolean }

export function createArrows(fields: GpuFields, U: SheetUniforms) {
  const cs = columnSampler(fields);
  const dataA = Array.from({ length: MAX_PLATES }, () => new THREE.Vector4()); // (x, z, heading, length) world
  const dataB = Array.from({ length: MAX_PLATES }, () => new THREE.Vector4()); // (visible, 0, 0, 0)
  const A = uniformArray(dataA, 'vec4'), B = uniformArray(dataB, 'vec4');
  const geo = arrowGeometry();

  const build = (casing: boolean) => {
    const m = new THREE.MeshBasicNodeMaterial();
    m.fog = false;
    const a = A.element(instanceIndex) as unknown as THREE.Node<'vec4'>;
    const b = B.element(instanceIndex) as unknown as THREE.Node<'vec4'>;
    const heading = a.z, len = a.w;
    const ch = cos(heading), sh = sin(heading);
    m.positionNode = Fn(() => {
      const W = float(ARROW_W * (casing ? 1.32 : 1));
      // stretch only the shaft behind the rigid part so the head keeps its shape
      const k = max(len.div(W).sub(HEAD + RIGID).div(1 - RIGID), 0.1);
      const x = positionGeometry.x;
      const X = select(x.greaterThanEqual(-RIGID), x, select(x.lessThanEqual(-1), x.add(1).sub(RIGID).sub(k.mul(1 - RIGID)), x.add(RIGID).mul(k).sub(RIGID)));
      const mid = float(HEAD - RIGID).sub(k.mul(1 - RIGID)).mul(0.5);
      const lx = X.sub(mid).mul(W), lz = positionGeometry.z.mul(W), ly = positionGeometry.y.mul(W.mul(0.6));
      // float above the highest ground/water along the arrow
      const top = (d: number) => {
        const px = a.x.add(ch.mul(len.mul(d))), pz = a.y.add(sh.mul(len.mul(d)));
        const c = cs.corners(tWorldToCell(px), tWorldToCell(pz));
        return max(cs.height(c), cs.level(c).level);
      };
      const y0 = tWorldY(max(max(top(-0.5), top(0)), top(0.5))).add(LIFT).sub(casing ? 0.018 : 0);
      const vis = b.x.mul(U.arrowFade.mul(U.arrowFade).mul(float(3).sub(U.arrowFade.mul(2)))); // smoothstep ease
      return vec3(a.x.add(lx.mul(ch).sub(lz.mul(sh)).mul(vis)), y0.add(ly.mul(vis)), a.y.add(lx.mul(sh).add(lz.mul(ch)).mul(vis)));
    })();
    if (casing) {
      m.colorNode = vec4(untonemap(vec3(...hexToLinear('#14161d')) as V3, U.exposure), 1);
    } else {
      const n = varying(normalize(vec3(normalGeometry.x.mul(ch).sub(normalGeometry.z.mul(sh)), normalGeometry.y,
        normalGeometry.x.mul(sh).add(normalGeometry.z.mul(ch)))), 'vArrowN');
      const L = vec3(-0.45, 0.8, -0.4).normalize();
      m.colorNode = Fn(() => {
        const d = max(dot(normalize(n), L), 0) as F;
        const base = vec3(...hexToLinear('#e9e0cc')) as V3;
        // soft two-tone: lit top, darker flanks, faint warm rim so the silhouette reads on bright ground
        const up = normalize(n).y.max(0);
        const shade = d.mul(0.4).add(0.42).add(up.mul(0.12));
        return vec4(untonemap(base.mul(shade).min(vec3(0.8)) as V3, U.exposure), 1);
      })();
    }
    const mesh = new THREE.InstancedMesh(geo, m, MAX_PLATES);
    mesh.frustumCulled = false;
    mesh.renderOrder = casing ? 12 : 13;
    mesh.name = casing ? 'overlay-arrow-casing' : 'overlay-arrows';
    return mesh;
  };

  const group = new THREE.Group();
  group.name = 'overlay-arrows';
  group.add(build(true), build(false));

  // CPU state: smoothed centroids (wrap-aware) and headings
  const state: ArrowState[] = Array.from({ length: MAX_PLATES }, (_, id) => ({ id, x: 0, z: 0, heading: 0, length: MIN_LEN, speed: 0, visible: false }));
  const cell = [new Float64Array(MAX_PLATES), new Float64Array(MAX_PLATES)];
  const seen = new Uint8Array(MAX_PLATES);
  const wrapD = (d: number, n: number) => ((((d + n / 2) % n) + n) % n) - n / 2;

  return {
    object: group,
    state,
    /** plates + centroids (cells, torus mean) → instance data. dt real seconds (smoothing). */
    /** cut: slice inspection bounds (visible x ≤ cut.x, z ≤ cut.z); arrows anchored beyond it are hidden. */
    update(plates: readonly { id: number; alive: boolean; vel: [number, number] }[], centroids: ([number, number] | undefined)[] | null, dt: number,
      cut: { x: number; z: number } = { x: HALF, z: HALF }) {
      // ease everything (≈1 s): stats land once per window and must never make an arrow jump
      const a = 1 - Math.exp(-Math.max(0, dt) / 0.9);
      for (let i = 0; i < MAX_PLATES; i++) {
        const p = plates[i], c = centroids?.[i];
        const st = state[i]!;
        if (!p?.alive || !c || !Number.isFinite(c[0]) || !Number.isFinite(c[1])) { st.visible = false; seen[i] = 0; dataB[i]!.x = 0; continue; }
        const speedNow = Math.hypot(p.vel[0], p.vel[1]);
        const headNow = speedNow > 1e-6 ? Math.atan2(p.vel[1], p.vel[0]) : st.heading;
        if (!seen[i]) { cell[0]![i] = c[0]; cell[1]![i] = c[1]; st.speed = speedNow; st.heading = headNow; seen[i] = 1; }
        cell[0]![i] = (cell[0]![i]! + wrapD(c[0] - cell[0]![i]!, NX) * a + NX) % NX;
        cell[1]![i] = (cell[1]![i]! + wrapD(c[1] - cell[1]![i]!, NZ) * a + NZ) % NZ;
        st.speed += (speedNow - st.speed) * a;
        st.heading += Math.atan2(Math.sin(headNow - st.heading), Math.cos(headNow - st.heading)) * a;
        st.length = arrowLength(st.speed);
        // keep the whole arrow on the block
        const lim = HALF - st.length / 2 - 0.06;
        st.x = Math.max(-lim, Math.min(lim, cellToWorld(cell[0]![i]!)));
        st.z = Math.max(-lim, Math.min(lim, cellToWorld(cell[1]![i]!)));
        // anchor sliced away (slice.ts): hide, the arrow would float over the removed block (state keeps easing)
        st.visible = st.x <= cut.x && st.z <= cut.z;
        dataA[i]!.set(st.x, st.z, st.heading, st.length);
        dataB[i]!.x = st.visible ? 1 : 0;
      }
    },
    dispose() { geo.dispose(); for (const c of group.children) ((c as THREE.Mesh).material as THREE.Material).dispose(); },
  };
}
