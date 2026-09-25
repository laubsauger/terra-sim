// Debug overlays (§T.47, §I.overlays): a translucent sheet over terrain/water coloured by one sim field.
// One lazily built material per overlay, each binding only the buffers it needs (V23).
import * as THREE from 'three/webgpu';
import { Fn, float, int, vec3, vec4, positionLocal, mix, clamp, fract, sin, uniformArray, floor, max } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { tColIdx } from '../sim/tslLayout';
import { createColumnGrid } from '../render/terrain';
import { columnSampler, tWorldToCell, tWorldY } from '../render/space';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

export interface OverlayDef { key: number; name: string; legend: string; field?: string; build?: (fields: GpuFields, c: THREE.Node<'uint'>) => V3 }

/** Perceptual-ish ramp (dark blue → teal → yellow), t in 0..1. */
const ramp = (t: F): V3 => {
  const a = vec3(0.13, 0.12, 0.35), b = vec3(0.12, 0.55, 0.55), c = vec3(0.98, 0.86, 0.3);
  const u = clamp(t, 0, 1);
  return mix(mix(a, b, u.mul(2).min(1)), c, u.mul(2).sub(1).max(0)) as V3;
};
const hashColor = (id: THREE.Node<'uint'>): V3 => {
  const h = float(id).mul(0.618034);
  const f = fract(h);
  return vec3(sin(f.mul(6.283)).mul(0.4).add(0.55), sin(f.mul(6.283).add(2.1)).mul(0.4).add(0.55), sin(f.mul(6.283).add(4.2)).mul(0.4).add(0.55)) as V3;
};
// ids: OCEAN ICE TUNDRA TAIGA TEMPERATE_FOREST GRASSLAND DESERT SAVANNA RAINFOREST BEACH ALPINE STEPPE SHRUBLAND COLD_DESERT
const BIOME_COLORS = [0x2b5fa8, 0xeaf2f7, 0x9aa38a, 0x37644a, 0x3f8a3a, 0x9bc15a, 0xe0c27c, 0xc7b35a, 0x1f6b2c, 0xf1dca2, 0x8f8a86, 0xc9b56e, 0x8d9a5b, 0xb8ab98]
  .map((h) => new THREE.Color(h).convertSRGBToLinear());

export const OVERLAYS: OverlayDef[] = [
  { key: 1, name: 'plates', legend: 'plate id', field: 'plateId', build: (f, c) => hashColor(f.cur<'uint'>('plateId').element(c) as unknown as THREE.Node<'uint'>) },
  { key: 2, name: 'crust age', legend: 'crust age 0 – 300 My', field: 'crustAge', build: (f, c) => ramp((f.cur('crustAge').element(c) as unknown as F).div(300)) },
  { key: 3, name: 'temperature', legend: 'surface temp −30 – 35°', field: 'surfTemp', build: (f, c) => ramp((f.cur('surfTemp').element(c) as unknown as F).add(30).div(65)) },
  { key: 4, name: 'mantle', legend: 'mantle heat (pending T20)' },
  { key: 5, name: 'stress', legend: 'plate stress (pending)' },
  { key: 6, name: 'water flux', legend: 'discharge (log)', field: 'flux', build: (f, c) => {
    const q = f.cur<'vec4'>('flux').element(c) as unknown as THREE.Node<'vec4'>;
    return ramp(q.x.add(q.y).add(q.z).add(q.w).add(1e-6).log2().add(14).div(14));
  } },
  { key: 7, name: 'moisture', legend: 'atmospheric vapor', field: 'vapor', build: (f, c) => ramp((f.cur('vapor').element(c) as unknown as F).div(0.3)) },
  { key: 8, name: 'biome', legend: 'biome', field: 'biome', build: (f, c) => {
    const pal = uniformArray(BIOME_COLORS, 'color');
    return pal.element(f.cur<'uint'>('biome').element(c)) as unknown as V3;
  } },
  { key: 9, name: 'precip', legend: 'precipitation (log)', field: 'precip', build: (f, c) => ramp((f.cur('precip').element(c) as unknown as F).add(1e-7).log2().add(18).div(10)) },
];

export function createOverlays(fields: GpuFields, scene: THREE.Scene) {
  const geo = createColumnGrid();
  const mats = new Map<number, THREE.MeshBasicNodeMaterial>();
  const mesh = new THREE.Mesh(geo);
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  mesh.visible = false;
  scene.add(mesh);
  const label = document.createElement('div');
  label.className = 'terra-overlay-label';
  label.style.display = 'none';
  document.body.appendChild(label);

  function material(def: OverlayDef): THREE.MeshBasicNodeMaterial {
    let m = mats.get(def.key);
    if (m) return m;
    const cs = columnSampler(fields);
    m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    const u = tWorldToCell(positionLocal.x), v = tWorldToCell(positionLocal.z);
    m.positionNode = Fn(() => {
      const c = cs.corners(u, v);
      const top = max(cs.height(c), cs.level(c).level);
      return vec3(positionLocal.x, tWorldY(top).add(0.004), positionLocal.z);
    })();
    m.colorNode = Fn(() => {
      const ci = tColIdx(int(floor(u.add(0.5))), int(floor(v.add(0.5))));
      return vec4(def.build!(fields, ci), 0.78);
    })();
    mats.set(def.key, m);
    return m;
  }

  let current = 0;
  return {
    /** n = 0 hides; 1-9 select (§I.keys). Returns false for overlays whose field does not exist yet. */
    set(n: number): boolean {
      const def = OVERLAYS.find((d) => d.key === n);
      // 0 or pressing the active key again turns overlays off
      if (!def || n === current) { current = 0; mesh.visible = false; label.style.display = 'none'; return true; }
      current = n;
      label.style.display = '';
      if (!def.build || (def.field && !fields.names().includes(def.field))) {
        mesh.visible = false;
        label.textContent = `${def.name}: not available yet`;
        return false;
      }
      mesh.material = material(def);
      mesh.visible = true;
      label.textContent = `${def.key} · ${def.legend}`;
      return true;
    },
    get current() { return current; },
    setLabelVisible(v: boolean) { if (current) label.style.display = v ? '' : 'none'; },
  };
}
