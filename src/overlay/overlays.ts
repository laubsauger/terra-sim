// Data overlays (§T.47, §I.overlays, keys 1-9, 0 off): a translucent sheet draped over terrain/water, coloured
// per pixel from a per-column prep pass (prep.ts), with a legend card, hover readout, plate motion arrows and,
// for mantle heat, a crust-temperature tint on the cut faces. Read-only on sim state (V15).
//
// Budget: when an overlay is on, one 65k-thread prep dispatch + one full-screen translucent sheet (4 storage
// reads per pixel) + ≤ 2 instanced draws for arrows; nothing runs while overlays are off.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Sim } from '../sim/sim';
import { Lifecycle } from '../sim/lifecycle';
import { NX, NZ, colIdx } from '../sim/layout';
import { MAX_PLATES } from '../sim/worldData';
import { createColumnGrid } from '../render/terrain';
import { worldToCell, voxelToWorldY } from '../render/space';
import { pickColumn } from '../ui/probe';
import { uniform } from 'three/tsl';
import { OVERLAYS, overlayByKey, readout, plateSpeedText, CM_PER_YR_PER_CELL_MY, type OverlayDef } from './defs';
import { createPrep } from './prep';
import { createSheetMaterial, createFaceMesh, type SheetUniforms } from './sheet';
import { createArrows } from './arrows';
import { createLegend } from './legend';
import { PLATE_COLORS } from './colormaps';

export { OVERLAYS } from './defs';

export interface PlateLike { id: number; alive: boolean; vel: [number, number] }

/** What the overlays read from the simulation (CPU side). */
export interface OverlaySource {
  plates(): readonly PlateLike[];
  /** Plate centroids in cells (torus mean) by plate id, or null before the first stats window. */
  centroids(): ([number, number] | undefined)[] | null;
  /** Sea level, voxel y. */
  seaLevel(): number;
  /** Geo time, My (activity rates). */
  geoMy(): number;
}

/** OverlaySource over the live Sim: plate table, lifecycle centroids from the last stats window. */
export function simSource(sim: Sim): OverlaySource {
  let seen: Int32Array | null = null;
  let cent: ([number, number] | undefined)[] | null = null;
  return {
    plates: () => sim.tectonics.plates,
    centroids() {
      const lc = sim.lastCounters;
      if (lc && lc !== seen) {
        seen = lc;
        const s = Lifecycle.parseStats(lc.buffer as ArrayBuffer);
        cent = s.centroid.map((c, i) => (s.area[i]! > 0 ? c : undefined));
      }
      return cent;
    },
    seaLevel: () => sim.stats?.seaLevel ?? 76,
    geoMy: () => sim.geoMy,
  };
}

export interface OverlayOptions {
  fields: GpuFields;
  scene: THREE.Scene;
  renderer: THREE.WebGPURenderer;
  camera: THREE.Camera;
  source: OverlaySource;
  /** Exposure the post chain applies before tone mapping (post.u.exposure); colours are pre-compensated. */
  exposure?: () => number;
  /** Per-frame hook (stage.onFrame). Without it, call frame(dt) yourself (tests). */
  onFrame?: (cb: (dt: number) => void) => void;
}

export function createOverlays(o: OverlayOptions) {
  const { fields, scene, renderer, camera, source } = o;
  const U: SheetUniforms = { opacity: uniform(0.85), exposure: uniform(1) };
  const prep = createPrep(fields);
  const legend = createLegend();
  const listeners = new Set<() => void>();

  const sheet = new THREE.Mesh(createColumnGrid());
  sheet.name = 'overlay-sheet';
  sheet.frustumCulled = false;
  sheet.renderOrder = 10;
  sheet.visible = false;
  scene.add(sheet);
  const faceDef = OVERLAYS.find((d) => d.id === 'heat')!;
  const faces = createFaceMesh(fields, faceDef.bar2!, U);
  if (faces) { faces.visible = false; scene.add(faces); }
  const arrows = createArrows(fields, U);
  arrows.object.visible = false;
  scene.add(arrows.object);

  const mats = new Map<number, THREE.MeshBasicNodeMaterial>();
  const material = (def: OverlayDef) => {
    let m = mats.get(def.key);
    if (!m) { m = createSheetMaterial(fields, prep.ovA, def, U); mats.set(def.key, m); }
    return m;
  };

  // plate tags (HTML, projected at the arrows)
  const tags = Array.from({ length: MAX_PLATES }, (_, id) => {
    const el = document.createElement('div');
    el.className = 'terra-plate-tag';
    el.style.display = 'none';
    el.innerHTML = `<b style="background:${PLATE_COLORS[id]}">${id}</b><span></span>`;
    document.body.appendChild(el);
    return { el, text: el.querySelector('span')!, shown: false, last: '' };
  });

  let current = 0;
  let uiVisible = true, legendOn = true, arrowsOn = true;
  let reset = true, lastGeo = NaN;
  const def = () => overlayByKey(current);
  const syncVisibility = () => {
    const d = def();
    sheet.visible = !!d;
    if (faces) faces.visible = d?.id === 'heat';
    arrows.object.visible = d?.id === 'plates' && arrowsOn;
    legend.setVisible(!!d && legendOn && uiVisible);
    if (!(d?.id === 'plates' && arrowsOn && uiVisible)) for (const t of tags) if (t.shown) { t.el.style.display = 'none'; t.shown = false; }
  };

  function select(n: number): void {
    const d = overlayByKey(n);
    const next = d ? n : 0;
    if (next === current) return;
    current = next;
    if (d) { sheet.material = material(d); legend.setOverlay(d); reset = true; lastGeo = NaN; hoverDirty = !!pointer; }
    syncVisibility();
    for (const l of listeners) l();
  }

  // ---- hover readout (≈10 Hz) against a CPU surfY copy refreshed about once a second -----------------
  let pointer: { x: number; y: number } | null = null;
  let hoverDirty = false, hoverBusy = false, lastPick = 0;
  let surf: Float32Array | null = null, surfAt = -1e9, surfBusy = false;
  const dom = renderer.domElement;
  dom.addEventListener('pointermove', (e) => { pointer = { x: e.clientX, y: e.clientY }; hoverDirty = true; });
  dom.addEventListener('pointerleave', () => { pointer = null; hoverDirty = false; legend.setReadout(null); });
  const refreshSurf = (now: number) => {
    if (surfBusy || now - surfAt < 1000) return;
    surfBusy = true;
    fields.read(renderer, 'surfY').then((b) => { surf = new Float32Array(b); surfAt = performance.now(); }).finally(() => { surfBusy = false; });
  };
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  async function hover(d: OverlayDef) {
    if (!pointer || !surf) return;
    hoverBusy = true;
    try {
      const r = dom.getBoundingClientRect();
      ndc.set(((pointer.x - r.left) / r.width) * 2 - 1, -((pointer.y - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = pickColumn(ray.ray, surf);
      if (!hit) { legend.setReadout(null); return; }
      const [a, b] = await Promise.all([readCol(prep.ovA, hit.x, hit.z), d.id === 'activity' ? readCol(prep.ovB, hit.x, hit.z) : Promise.resolve(new Float32Array(4))]);
      if (def() !== d) return;
      legend.setReadout(readout(d, { plates: source.plates(), a, b }), `${hit.x}, ${hit.z}`);
    } finally { hoverBusy = false; }
  }
  const readCol = async (buf: typeof prep.ovA, x: number, z: number) =>
    new Float32Array(await renderer.getArrayBufferAsync(buf.value as THREE.StorageInstancedBufferAttribute, null, colIdx(x, z) * 16, 16));

  // ---- per frame ---------------------------------------------------------------------------------------
  const v = new THREE.Vector3();
  function frame(dt: number) {
    const d = def();
    if (!d) return;
    U.exposure.value = o.exposure?.() ?? 1;
    const sea = source.seaLevel();
    prep.u.seaLevel.value = sea;
    const g = source.geoMy();
    const geoDt = Number.isFinite(lastGeo) ? Math.max(0, g - lastGeo) : 0;
    lastGeo = g;
    prep.u.ema.value = reset || d.smoothS <= 0 ? 1 : 1 - Math.exp(-dt / d.smoothS);
    prep.u.emaGeo.value = reset ? 1 : 1 - Math.exp(-geoDt / 5);
    prep.u.geoDt.value = geoDt;
    prep.u.reset.value = reset ? 1 : 0;
    const plates = source.plates();
    if (d.id === 'plates') {
      for (let i = 0; i < MAX_PLATES; i++) {
        const p = plates[i];
        prep.plateVel[i]!.set(p?.vel[0] ?? 0, p?.vel[1] ?? 0, p?.alive ? 1 : 0, 0);
      }
    }
    prep.run(renderer, d);
    reset = false;

    const now = performance.now();
    if (d.id === 'plates') {
      arrows.update(plates, source.centroids(), dt);
      let fastest = 0, n = 0;
      for (const s of arrows.state) if (s.visible) { n++; fastest = Math.max(fastest, s.speed); }
      legend.setNote(n ? `Arrows: plate motion, longer = faster · ${n} plates, fastest ${(fastest * CM_PER_YR_PER_CELL_MY).toFixed(1)} cm/yr.`
        : 'Arrows: plate motion, longer = faster (after the first stats window).');
      if (arrowsOn && uiVisible) {
        refreshSurf(now);
        placeTags(sea);
      }
    }
    if (legend.visible && pointer) {
      refreshSurf(now);
      if (hoverDirty && !hoverBusy && now - lastPick >= 100) { hoverDirty = false; lastPick = now; void hover(d); }
      // values under a still cursor change as the sim runs: re-read about once a second
      if (!hoverDirty && now - lastPick >= 1000) hoverDirty = true;
    }
  }

  function placeTags(sea: number) {
    const r = dom.getBoundingClientRect();
    for (let i = 0; i < MAX_PLATES; i++) {
      const st = arrows.state[i]!, t = tags[i]!;
      if (!st.visible) { if (t.shown) { t.el.style.display = 'none'; t.shown = false; } continue; }
      const cx = Math.round(worldToCell(st.x)), cz = Math.round(worldToCell(st.z));
      const h = surf ? Math.max(sea, surf[colIdx(Math.max(0, Math.min(NX - 1, cx)), Math.max(0, Math.min(NZ - 1, cz)))]!) : sea;
      v.set(st.x, voxelToWorldY(h) + 0.16, st.z).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) { if (t.shown) { t.el.style.display = 'none'; t.shown = false; } continue; }
      if (!t.shown) { t.el.style.display = ''; t.shown = true; }
      t.el.style.transform = `translate(${(r.left + ((v.x + 1) / 2) * r.width).toFixed(1)}px, ${(r.top + ((1 - v.y) / 2) * r.height).toFixed(1)}px) translate(-50%, -100%)`;
      const txt = plateSpeedText(source.plates()[i]!.vel);
      if (txt !== t.last) { t.text.textContent = txt; t.last = txt; }
    }
  }

  o.onFrame?.(frame);

  return {
    /** Keys (§I.keys): n = 0 or the active key again turns overlays off; 1-9 select. */
    set(n: number): boolean { select(n === current ? 0 : n); return true; },
    /** Select without toggling (panel). */
    select,
    get current() { return current; },
    get def() { return def(); },
    defs: OVERLAYS,
    get opacity() { return U.opacity.value; },
    set opacity(x: number) { U.opacity.value = Math.min(1, Math.max(0, x)); },
    get legendVisible() { return legendOn; },
    set legendVisible(x: boolean) { legendOn = x; syncVisibility(); },
    get arrowsVisible() { return arrowsOn; },
    set arrowsVisible(x: boolean) { arrowsOn = x; syncVisibility(); },
    /** UI hidden (H): hide legend + tags, keep the map colouring. */
    setLabelVisible(x: boolean) { uiVisible = x; syncVisibility(); },
    onChange(cb: () => void) { listeners.add(cb); return () => listeners.delete(cb); },
    frame,
    legend,
    arrows: arrows.state,
    /** Test/debug: prep values of one column (vec4 of the active overlay, see prep.ts). */
    readColumn: (x: number, z: number, which: 'a' | 'b' = 'a') => readCol(which === 'a' ? prep.ovA : prep.ovB, x, z),
    /** Test/debug: the whole prep buffer. */
    async readAll(which: 'a' | 'b' = 'a') {
      return new Float32Array(await renderer.getArrayBufferAsync((which === 'a' ? prep.ovA : prep.ovB).value as THREE.StorageInstancedBufferAttribute));
    },
    objects: { sheet, faces, arrows: arrows.object },
  };
}
export type Overlays = ReturnType<typeof createOverlays>;
