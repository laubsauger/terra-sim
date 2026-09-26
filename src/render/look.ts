// Look integration hook: everything visual in one call — sky/time of day, lighting + shadows,
// diorama staging (plinth, floor, backdrop, haze), terrain/sides/water, and the post pipeline.
//
// main.ts wiring (replaces createLighting + scene.add(terrain, sides, water)):
//   const look = createLook(stage, fields, { highQuality: params.get('highQuality') as boolean });
//   params.onChange((k, v) => { if (k === 'highQuality') look.setHighQuality(v as boolean); });
//   stage.onFrame(() => look.frame(clock.ambTime));
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Stage } from './stage';
import { createLighting, type Lighting } from './lighting';
import { createTerrain } from './terrain';
import { createSides } from './sides';
import { createWater } from './water';
import { createBackdrop, installAtmosphere, type Backdrop } from './backdrop';
import { createPost, type Post, type PostFeatures } from './post';
import { GOLDEN_HOUR, cycleToTod, todToCycle } from './sky';
import { setAmbTime, voxelToWorldY, HALF, setRenderMotion, Y_RENDER_BOTTOM, type RenderMotion } from './space';
import { NX, CELL } from '../sim/layout';

export interface LookOptions {
  highQuality?: boolean;
  /** Time of day at ambTime 0 (0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset). */
  timeOfDay?: number;
  /** Real seconds per full day/night cycle; 0 holds the time of day still. */
  dayLength?: number;
  /** Override individual post passes (perf triage); defaults follow highQuality. */
  postFeatures?: Partial<PostFeatures>;
  /** Plate motion source (Sim.tectonics): display follows plates continuously instead of jumping. */
  motion?: RenderMotion;
  /** Override the tone mapping operator (look-dev A/B). */
  toneMapping?: THREE.ToneMapping;
}

export interface Look {
  lighting: Lighting;
  post: Post;
  backdrop: Backdrop;
  terrain: THREE.Object3D;
  sides: THREE.Object3D;
  water: THREE.Object3D;
  /** Current time of day (0..1). */
  readonly timeOfDay: number;
  setTimeOfDay(tod: number): void;
  setDayLength(seconds: number): void;
  /** Real seconds per full day cycle; 0 = fixed time of day. */
  readonly dayLength: number;
  setHighQuality(v: boolean): void;
  /** Per frame: ambience clock (s, V17) drives waves, caustics, mantle flow and the day cycle. */
  frame(ambTime: number): void;
  dispose(): void;
}

export function createLook(stage: Stage, fields: GpuFields, opts: LookOptions = {}): Look {
  const { renderer, scene, camera } = stage;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const hq = opts.highQuality ?? true;

  const lighting = createLighting(scene, { shadows: true, shadowMapSize: hq ? 4096 : 2048, shadowExtent: 3.6 });
  installAtmosphere(scene);
  const backdrop = createBackdrop();
  const terrain = createTerrain(fields);
  const sides = createSides(fields);
  const water = createWater(fields);
  scene.add(backdrop.object, terrain.object, sides.object, water.object);

  if (opts.motion) setRenderMotion(fields, opts.motion);
  const post = createPost(renderer, scene, camera, { highQuality: hq, features: opts.postFeatures, toneMapping: opts.toneMapping });
  stage.setRender(() => post.render());

  let dayLength = opts.dayLength ?? 0;
  let tod = opts.timeOfDay ?? GOLDEN_HOUR;
  let tod0 = dayLength > 0 ? todToCycle(tod) : tod;
  let lastAmb = 0;
  let lastTod = -1;
  const applyTod = (t: number) => {
    tod = ((t % 1) + 1) % 1;
    if (Math.abs(tod - lastTod) < 1e-5) return;
    lastTod = tod;
    lighting.setTimeOfDay(tod);
  };
  applyTod(tod);

  const keepOut = new THREE.Box3();
  // Camera ground clamp: CPU copy of the ground / water surface (world y), refreshed ~2×/s from the sim fields.
  const GROUND_MARGIN = 0.025, GROUND_EVERY_MS = 500;
  let groundY: Float32Array | null = null, groundBusy = false, groundAt = -1e9;
  const refreshGround = () => {
    const now = performance.now();
    if (groundBusy || now - groundAt < GROUND_EVERY_MS) return;
    groundBusy = true; groundAt = now;
    void Promise.all([fields.read(stage.renderer, 'surfY'), fields.read(stage.renderer, 'water')]).then(([sb, wb]) => {
      const s = new Float32Array(sb), w = new Float32Array(wb);
      const g = groundY ?? new Float32Array(s.length);
      for (let i = 0; i < s.length; i++) g[i] = voxelToWorldY(s[i]! + Math.max(0, w[i]!));
      groundY = g;
    }).finally(() => { groundBusy = false; });
  };
  stage.setGround((x, z) => {
    if (!groundY || Math.abs(x) > HALF || Math.abs(z) > HALF) return null;
    // highest of the 2×2 cells around the point: never inside a slope between samples
    const u = (x + HALF) / CELL - 0.5, v = (z + HALF) / CELL - 0.5;
    const x0 = Math.floor(u), z0 = Math.floor(v);
    let m = -Infinity;
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
      const cx = Math.min(NX - 1, Math.max(0, x0 + dx)), cz = Math.min(NX - 1, Math.max(0, z0 + dz));
      m = Math.max(m, groundY[cx + cz * NX]!);
    }
    return m + GROUND_MARGIN;
  });
  return {
    lighting, post, backdrop, terrain: terrain.object, sides: sides.object, water: water.object,
    get timeOfDay() { return tod; },
    // while cycling, tod0 is the cycle phase at ambTime 0 (night compressed, sky.ts cycleToTod); keep the
    // current sun where it is when either changes
    setTimeOfDay(t) { tod0 = dayLength > 0 ? todToCycle(t) - lastAmb / dayLength : t; applyTod(t); },
    setDayLength(s) { tod0 = s > 0 ? todToCycle(tod) - lastAmb / s : tod; dayLength = s; },
    get dayLength() { return dayLength; },
    setHighQuality(v) {
      post.setHighQuality(v);
      lighting.sun.shadow.mapSize.setScalar(v ? 4096 : 2048);
      lighting.sun.shadow.map?.dispose();
      lighting.sun.shadow.map = null;
    },
    frame(ambTime) {
      setAmbTime(ambTime);
      lastAmb = ambTime;
      if (dayLength > 0) applyTod(cycleToTod(tod0 + ambTime / dayLength));
      backdrop.update();
      // Camera keep-out: over the block the ground clamp (stage.setGround); the plinth around it up to the
      // block bottom, and the floor
      refreshGround();
      keepOut.min.set(-HALF - 0.45, backdrop.floorY - 1, -HALF - 0.45);
      keepOut.max.set(HALF + 0.45, voxelToWorldY(Y_RENDER_BOTTOM) + 0.04, HALF + 0.45);
      stage.setKeepOut(keepOut, backdrop.floorY + 0.08);
      post.setFocus(camera.position.distanceTo(stage.controls.target));
    },
    dispose() {
      post.dispose();
      lighting.dispose();
      scene.remove(backdrop.object, terrain.object, sides.object, water.object);
      backdrop.dispose(); terrain.dispose(); sides.dispose(); water.dispose();
      scene.backgroundNode = null; scene.fogNode = null;
      stage.setRender(null);
    },
  };
}
