import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export interface Stage {
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  onFrame(cb: (realDt: number) => void): void;
  /** CPU ms spent in the previous frame (callbacks + render submit). */
  readonly cpuMs: number;
  start(): void;
}

// Limits the voxel grid needs (32MB voxel buffers, many bindings per kernel). Take whatever the adapter offers.
function requiredLimits(adapter: GPUAdapter): Record<string, number> {
  const l = adapter.limits;
  return {
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxBufferSize: l.maxBufferSize,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
  };
}

export async function createStage(container: HTMLElement, adapter: GPUAdapter): Promise<Stage> {
  const renderer = new THREE.WebGPURenderer({
    antialias: false,
    trackTimestamp: true,
    requiredLimits: requiredLimits(adapter),
  } as THREE.WebGPURendererParameters);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x151924);

  const camera = new THREE.PerspectiveCamera(35, container.clientWidth / container.clientHeight, 0.1, 500);
  camera.position.set(3.2, 2.4, 3.2);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 1.5;
  controls.maxDistance = 12;

  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', resize);

  const frameCbs: ((dt: number) => void)[] = [];
  let last = performance.now();
  let cpuMs = 0;

  return {
    renderer, scene, camera, controls,
    get cpuMs() { return cpuMs; },
    onFrame(cb) { frameCbs.push(cb); },
    start() {
      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = (now - last) / 1000;
        last = now;
        for (const cb of frameCbs) cb(dt);
        controls.update();
        renderer.render(scene, camera);
        cpuMs = performance.now() - now;
      });
    },
  };
}
