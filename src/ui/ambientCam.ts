// Ambient mode camera (§T.49): slow cinematic orbit that drifts in and out, and glides to recent events
// (eruptions, impacts) as points of interest. Any user input hands control back for a while.
import * as THREE from 'three/webgpu';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { cellToWorld, voxelToWorldY } from '../render/space';
import type { GeoEvent } from '../sim/events';

export const IDLE_RESUME_S = 25;  // seconds without input before the auto camera takes over again
export const POI_HOLD_S = 14;
const ORBIT_RATE = 0.035;         // rad/s

export function createAmbientCam(camera: THREE.PerspectiveCamera, controls: OrbitControls, dom: HTMLElement) {
  let enabled = false;
  let idle = IDLE_RESUME_S;
  let t = 0;
  let poi: { target: THREE.Vector3; until: number } | null = null;
  let lastEventCount = 0;
  const target = new THREE.Vector3();
  const desired = new THREE.Vector3();
  const touch = () => { idle = 0; };
  for (const ev of ['pointerdown', 'wheel', 'keydown'] as const) dom.addEventListener(ev, touch, { passive: true });

  return {
    setEnabled(v: boolean) { enabled = v; idle = IDLE_RESUME_S; },
    get enabled() { return enabled; },
    /** Feed the event log; new eruption/impact-like events become points of interest. */
    notice(log: readonly GeoEvent[], surfYAt: (x: number, z: number) => number) {
      if (log.length === lastEventCount) return;
      const e = log[log.length - 1]!;
      lastEventCount = log.length;
      if (e.kind === 'iceAge') return;
      poi = { target: new THREE.Vector3(cellToWorld(e.x), voxelToWorldY(surfYAt(e.x, e.z)), cellToWorld(e.z)), until: t + POI_HOLD_S };
    },
    update(dt: number) {
      idle += dt;
      if (!enabled || idle < IDLE_RESUME_S) return;
      t += dt;
      const ang = t * ORBIT_RATE;
      const dist = 5.2 + Math.sin(t * 0.021) * 1.1;
      const elev = 0.5 + Math.sin(t * 0.013 + 1) * 0.12; // radians above horizon
      if (poi && t > poi.until) poi = null;
      target.lerp(poi ? poi.target : new THREE.Vector3(0, 0, 0), 1 - Math.exp(-dt * 0.35));
      desired.set(Math.cos(ang) * Math.cos(elev), Math.sin(elev), Math.sin(ang) * Math.cos(elev)).multiplyScalar(poi ? dist * 0.55 : dist).add(target);
      camera.position.lerp(desired, 1 - Math.exp(-dt * 0.5));
      controls.target.copy(target);
    },
  };
}
