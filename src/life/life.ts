// §T.45 tiny life: flora (GPU) + birds, critters, fish (CPU). Reads sim fields only (V15): the flora
// kernel reads biome/veg and the render column summary and writes only life-owned buffers;
// creatures steer on an async readback of surfY/water/biome refreshed every ~2 s.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import { createFlora, type Flora } from './flora';
import { createCreatures, type Creatures } from './creatures';
import { FLORA, CREATURES, type ColumnMap } from './lifeModel';

export interface LifeOptions {
  /** Low tier: half the flora slots (all three grids), fewer flocks/critters/schools. */
  highQuality?: boolean;
  seed?: number;
  /** Render camera: fine ground cover is frustum-culled and thinned with distance from it. */
  camera?: THREE.Camera;
}

export interface Life {
  object: THREE.Group;
  /** Per frame: real dt (s) and the ambience clock (s, V17 — keeps running while the geo clock is paused). */
  update(dt: number, ambTime: number): void;
  setEnabled(v: boolean): void;
  setHighQuality(v: boolean): void;
  /** Emergent sea level (voxel y, sim.stats.seaLevel) for the altitude treeline. Cheap; call per frame. */
  setSeaLevel(y: number): void;
  /** Ask for a flora refresh soon (sim ticked, load, god tools); rate-limited to one per 0.5 s. */
  invalidate(): void;
  /** Resolves once the first column map is read back and creatures have spawned. */
  whenReady(): Promise<void>;
  /** Tests / debug. */
  flora: Flora;
  creatures: Creatures;
  dispose(): void;
}

const MIN_INVALIDATE_S = 0.5;

export function createLife(fields: GpuFields, renderer: THREE.WebGPURenderer, scene: THREE.Scene, opts: LifeOptions = {}): Life {
  const flora = createFlora(fields, { highQuality: opts.highQuality });
  if (opts.camera) flora.setCamera(opts.camera);
  const creatures = createCreatures({ highQuality: opts.highQuality, seed: opts.seed });
  const object = new THREE.Group();
  object.name = 'life';
  object.add(flora.object, creatures.object);
  scene.add(object);

  let enabled = true;
  let sinceFlora = Infinity;   // refresh on the first update
  let sinceMap = Infinity;
  let dirty = false;
  let reading = false;
  let resolveReady: () => void = () => {};
  const ready = new Promise<void>((r) => { resolveReady = r; });

  async function readMap(): Promise<void> {
    reading = true;
    try {
      const [s, w, b, r] = await Promise.all([fields.read(renderer, 'surfY'), fields.read(renderer, 'water'), fields.read(renderer, 'biome'),
        renderer.getArrayBufferAsync(flora.buffers.renderH as unknown as THREE.StorageBufferAttribute)]);
      const m: ColumnMap = { surfY: new Float32Array(s), surfR: new Float32Array(r), water: new Float32Array(w), biome: new Uint32Array(b) };
      creatures.setMap(m);
      resolveReady();
    } finally {
      reading = false;
    }
  }

  return {
    object, flora, creatures,
    update(dt, ambTime) {
      if (!enabled) return;
      sinceFlora += dt; sinceMap += dt;
      if (sinceFlora >= FLORA.REFRESH_S || (dirty && sinceFlora >= MIN_INVALIDATE_S)) {
        flora.refresh(renderer, false);
        sinceFlora = 0;
        dirty = false;
      }
      if (!reading && sinceMap >= CREATURES.MAP_S) {
        sinceMap = 0;
        readMap().catch((e) => console.error('life: map readback failed: ' + (e as Error).message));
      }
      flora.frame(renderer, dt, ambTime);
      creatures.step(dt, ambTime);
    },
    setEnabled(v) {
      enabled = v;
      object.visible = v;
      if (v) { sinceFlora = Infinity; sinceMap = Infinity; } // catch up at once after a pause
    },
    setHighQuality(v) {
      flora.setQuality(v);
      creatures.setQuality(v);
      dirty = true;
    },
    setSeaLevel(y) { flora.setSeaLevel(y); },
    invalidate() { dirty = true; },
    whenReady: () => ready,
    dispose() {
      flora.dispose();
      creatures.dispose();
      object.removeFromParent();
    },
  };
}
