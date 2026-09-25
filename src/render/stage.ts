import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { updateAllRenderColumns } from './space';

export interface Stage {
  renderer: THREE.WebGPURenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  onFrame(cb: (realDt: number) => void): void;
  /** Replace what a frame draws (e.g. a post pipeline); null restores renderer.render(scene, camera). */
  setRender(fn: (() => void) | null): void;
  /**
   * Volume the camera may never enter (the block + plinth) and a floor height. Applied after the
   * orbit controls every frame: the camera is pushed out through the nearest box face.
   */
  setKeepOut(box: THREE.Box3 | null, floorY?: number): void;
  /** Default 3/4 hero framing. */
  heroView(): void;
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

  const camera = new THREE.PerspectiveCamera(30, container.clientWidth / container.clientHeight, 0.1, 500);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.485;
  controls.minDistance = 1.2;
  controls.maxDistance = 16;
  const heroView = () => {
    camera.position.set(5.5, 2.7, 5.8);
    controls.target.set(0, -0.2, 0);
    controls.update();
  };
  heroView();

  let keepOut: THREE.Box3 | null = null;
  let floorY = -Infinity;
  const constrain = () => {
    const p = camera.position;
    if (p.y < floorY) p.y = floorY;
    if (!keepOut || !keepOut.containsPoint(p)) return;
    // push out through the nearest face (never up through the floor side)
    const b = keepOut;
    const opts: [number, () => void][] = [
      [p.x - b.min.x, () => { p.x = b.min.x; }], [b.max.x - p.x, () => { p.x = b.max.x; }],
      [p.z - b.min.z, () => { p.z = b.min.z; }], [b.max.z - p.z, () => { p.z = b.max.z; }],
      [b.max.y - p.y, () => { p.y = b.max.y; }],
    ];
    opts.sort((a, c) => a[0] - c[0]);
    opts[0]![1]();
  };
  let renderFn: (() => void) | null = null;

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
    setRender(fn) { renderFn = fn; },
    setKeepOut(box, fy) { keepOut = box; floorY = fy ?? -Infinity; },
    heroView,
    start() {
      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = (now - last) / 1000;
        last = now;
        for (const cb of frameCbs) cb(dt);
        controls.update();
        constrain();
        updateAllRenderColumns(renderer, dt); // display columns: advect + ease toward the latest sim fields
        if (renderFn) renderFn(); else renderer.render(scene, camera);
        cpuMs = performance.now() - now;
      });
    },
  };
}
