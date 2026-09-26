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
import { GOLDEN_HOUR } from './sky';
import { setAmbTime, voxelToWorldY, HALF, setRenderMotion, type RenderMotion } from './space';
import { NY } from '../sim/layout';

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

  let tod0 = opts.timeOfDay ?? GOLDEN_HOUR;
  let dayLength = opts.dayLength ?? 0;
  let tod = tod0;
  let lastAmb = 0;
  let lastTod = -1;
  const applyTod = (t: number) => {
    tod = ((t % 1) + 1) % 1;
    if (Math.abs(tod - lastTod) < 1e-5) return;
    lastTod = tod;
    lighting.setTimeOfDay(tod);
  };
  applyTod(tod0);

  const keepOut = new THREE.Box3();
  return {
    lighting, post, backdrop, terrain: terrain.object, sides: sides.object, water: water.object,
    get timeOfDay() { return tod; },
    // tod0 is the phase at ambTime 0 while cycling: keep the current sun where it is when either changes
    setTimeOfDay(t) { tod0 = dayLength > 0 ? t - lastAmb / dayLength : t; applyTod(t); },
    setDayLength(s) { tod0 = s > 0 ? tod - lastAmb / s : tod; dayLength = s; },
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
      if (dayLength > 0) applyTod(tod0 + ambTime / dayLength);
      backdrop.update();
      // Camera keep-out: the slice volume (grid top) + plinth, and the floor.
      keepOut.min.set(-HALF - 0.45, backdrop.floorY - 1, -HALF - 0.45);
      keepOut.max.set(HALF + 0.45, voxelToWorldY(NY) + 0.08, HALF + 0.45);
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
