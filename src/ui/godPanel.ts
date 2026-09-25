// God tools UI (§T.48, §I.god tools): pick a tool, click the terrain. Everything becomes a sim event (V16).
import * as THREE from 'three/webgpu';
import type { FolderApi } from 'tweakpane';
import type { GpuFields } from '../core/gpu';
import type { Sim } from '../sim/sim';
import { colIdx } from '../sim/layout';
import { pickColumn } from './probe';

export type GodTool = 'inspect' | 'uplift' | 'subside' | 'volcano' | 'meteor' | 'storm' | 'split';

export function createGodPanel(folder: FolderApi, sim: Sim, renderer: THREE.WebGPURenderer, camera: THREE.Camera, fields: GpuFields) {
  const state = { tool: 'inspect' as GodTool, radius: 12, strength: 12 };
  folder.addBinding(state, 'tool', {
    options: { inspect: 'inspect', 'uplift ▲': 'uplift', 'subside ▼': 'subside', volcano: 'volcano', meteor: 'meteor', 'rain storm': 'storm', 'split plate': 'split' },
  });
  folder.addBinding(state, 'radius', { min: 3, max: 40, step: 1 });
  folder.addBinding(state, 'strength', { min: 1, max: 40, step: 1, label: 'strength (layers)' });

  async function apply(clientX: number, clientY: number) {
    const rect = renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1), camera);
    const hit = pickColumn(ray.ray, new Float32Array(await fields.read(renderer, 'surfY')));
    if (!hit) return;
    const base = { my: sim.geoMy, x: hit.x, z: hit.z, radius: state.radius };
    switch (state.tool) {
      case 'uplift': sim.events.enqueue({ ...base, kind: 'uplift', magnitude: state.strength }); break;
      case 'subside': sim.events.enqueue({ ...base, kind: 'uplift', magnitude: -state.strength }); break;
      case 'meteor': sim.events.enqueue({ ...base, kind: 'meteor', magnitude: state.strength / 12 }); break;
      case 'storm': sim.events.enqueue({ ...base, kind: 'storm', magnitude: Math.min(1, state.strength / 20) }); break;
      case 'volcano': sim.events.enqueue({ ...base, kind: 'volcano', magnitude: state.strength / 12 }); break;
      case 'split': {
        const pid = new Uint32Array(await fields.read(renderer, 'plateId'));
        sim.events.enqueue({ ...base, kind: 'split', magnitude: pid[colIdx(hit.x, hit.z)]! });
        break;
      }
    }
  }

  let down: [number, number] | null = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (state.tool !== 'inspect' && down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && e.button === 0) void apply(e.clientX, e.clientY);
    down = null;
  });
  return { state, isInspect: () => state.tool === 'inspect', apply };
}
