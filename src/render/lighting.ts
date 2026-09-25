// Warm key sun + soft sky/ground hemisphere fill. Fog-free for now; shadows arrive with T40.
import * as THREE from 'three/webgpu';

export interface Lighting {
  sun: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  dispose(): void;
}

export function createLighting(scene: THREE.Scene): Lighting {
  const sun = new THREE.DirectionalLight(0xffe1bd, 3.4);
  sun.position.set(4, 3.6, 1.4);
  const hemi = new THREE.HemisphereLight(0xc4dcff, 0x7a5a44, 0.95);
  scene.add(sun, sun.target, hemi);
  return {
    sun, hemi,
    dispose() { scene.remove(sun, sun.target, hemi); sun.dispose(); hemi.dispose(); },
  };
}
