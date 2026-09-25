import { describe, it, expect } from 'vitest';
import * as THREE from 'three/webgpu';
import { createAmbientCam, IDLE_RESUME_S } from '../../src/ui/ambientCam';

const fakeDom = () => { const l: Record<string, () => void> = {}; return { addEventListener: (k: string, f: () => void) => { l[k] = f; }, fire: (k: string) => l[k]?.() }; };

describe('ambient camera (T49)', () => {
  it('orbits on its own, never enters the block, and yields to user input', () => {
    const cam = new THREE.PerspectiveCamera();
    cam.position.set(4, 3, 4);
    const controls = { target: new THREE.Vector3() };
    const dom = fakeDom();
    const a = createAmbientCam(cam, controls as never, dom as unknown as HTMLElement);
    a.setEnabled(true);
    const p0 = cam.position.clone();
    for (let i = 0; i < 600; i++) {
      a.update(1 / 30);
      const inside = Math.abs(cam.position.x) < 2.05 && Math.abs(cam.position.z) < 2.05 && cam.position.y < 1.2;
      expect(inside).toBe(false);
    }
    expect(cam.position.distanceTo(p0)).toBeGreaterThan(0.3);
    dom.fire('pointerdown');
    const p1 = cam.position.clone();
    for (let i = 0; i < 30 * (IDLE_RESUME_S - 1); i++) a.update(1 / 30);
    expect(cam.position.distanceTo(p1)).toBe(0); // user in control
  });
});
