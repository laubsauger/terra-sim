// God tools UI (§T.48, §I.god tools): pick a tool, hover to see its footprint, click to apply.
// Everything becomes a sim event (V16), applied immediately (Sim.flushEvents runs every frame).
import * as THREE from 'three/webgpu';
import type { FolderApi } from 'tweakpane';
import type { GpuFields } from '../core/gpu';
import type { Sim } from '../sim/sim';
import { colIdx, CELL } from '../sim/layout';
import { pickColumn } from './probe';
import { cellToWorld, voxelToWorldY } from '../render/space';
import { showToast } from './savePanel';

export type GodTool = 'inspect' | 'uplift' | 'subside' | 'volcano' | 'meteor' | 'storm' | 'split';

const TOOL_COLOR: Record<GodTool, number> = {
  inspect: 0xffffff, uplift: 0xf2c46d, subside: 0x7fb2ff, volcano: 0xff6a2a, meteor: 0xffe0a0, storm: 0x9fd8ff, split: 0xff5ab4,
};
const TOOL_LABEL: Record<GodTool, string> = {
  inspect: 'inspect', uplift: 'uplift', subside: 'subside', volcano: 'volcano', meteor: 'meteor', storm: 'rain storm', split: 'split plate',
};

export function createGodPanel(folder: FolderApi, sim: Sim, renderer: THREE.WebGPURenderer, camera: THREE.Camera, fields: GpuFields, scene: THREE.Scene) {
  const state = { tool: 'inspect' as GodTool, radius: 12, strength: 12 };
  const toolBinding = folder.addBinding(state, 'tool', {
    options: { inspect: 'inspect', 'uplift ▲': 'uplift', 'subside ▼': 'subside', volcano: 'volcano', meteor: 'meteor', 'rain storm': 'storm', 'split plate': 'split' },
  });
  folder.addBinding(state, 'radius', { min: 3, max: 40, step: 1 });
  folder.addBinding(state, 'strength', { min: 1, max: 40, step: 1, label: 'strength (layers)' });

  // footprint ring: an outer ring + soft inner disc, drawn on top so it reads over terrain and water
  const ringMat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, opacity: 0.9 });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 96).rotateX(-Math.PI / 2), ringMat);
  const discMat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, opacity: 0.12 });
  const disc = new THREE.Mesh(new THREE.CircleGeometry(0.94, 96).rotateX(-Math.PI / 2), discMat);
  const cursor = new THREE.Group();
  cursor.add(ring, disc);
  cursor.renderOrder = 20;
  ring.renderOrder = disc.renderOrder = 20;
  cursor.visible = false;
  scene.add(cursor);
  let pulse = 0;

  const setColor = () => { const c = new THREE.Color(TOOL_COLOR[state.tool]); ringMat.color = c; discMat.color = c; };
  setColor();
  toolBinding.on('change', () => { setColor(); cursor.visible = false; renderer.domElement.style.cursor = state.tool === 'inspect' ? '' : 'crosshair'; });

  let surf: Float32Array | null = null;
  let surfAt = 0;
  async function pick(clientX: number, clientY: number) {
    const now = performance.now();
    if (!surf || now - surfAt > 400) { surf = new Float32Array(await fields.read(renderer, 'surfY')); surfAt = now; }
    const rect = renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1), camera);
    const hit = pickColumn(ray.ray, surf);
    return hit ? { hit, y: voxelToWorldY(surf[colIdx(hit.x, hit.z)]!) } : null;
  }

  async function apply(clientX: number, clientY: number) {
    const p = await pick(clientX, clientY);
    if (!p) return;
    const { hit } = p;
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
      default: return;
    }
    pulse = 1;
    surf = null; // terrain changed: refresh the pick cache
    showToast(`${TOOL_LABEL[state.tool]} at ${hit.x}, ${hit.z}${state.tool === 'split' ? ' — rifts at the next stats window' : ''}`, 'info');
  }

  let down: [number, number] | null = null;
  let moveBusy = false;
  const el = renderer.domElement;
  el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener('pointerup', (e) => {
    if (state.tool !== 'inspect' && down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && e.button === 0) void apply(e.clientX, e.clientY);
    down = null;
  });
  el.addEventListener('pointermove', (e) => {
    if (state.tool === 'inspect' || moveBusy) return;
    moveBusy = true;
    void pick(e.clientX, e.clientY).then((p) => {
      moveBusy = false;
      if (!p) { cursor.visible = false; return; }
      cursor.visible = true;
      cursor.position.set(cellToWorld(p.hit.x), p.y + 0.01, cellToWorld(p.hit.z));
    });
  });
  el.addEventListener('pointerleave', () => { cursor.visible = false; });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.tool !== 'inspect') { state.tool = 'inspect'; toolBinding.refresh(); setColor(); cursor.visible = false; el.style.cursor = ''; }
  });

  return {
    state,
    isInspect: () => state.tool === 'inspect',
    apply,
    /** Per frame: scale the footprint to the radius, pulse after a click. */
    update(dt: number) {
      if (!cursor.visible) return;
      const r = state.radius * CELL * (1 + 0.25 * pulse);
      cursor.scale.set(r, 1, r);
      ringMat.opacity = 0.9;
      discMat.opacity = 0.12 + 0.35 * pulse;
      pulse = Math.max(0, pulse - dt * 2.5);
    },
  };
}
