// Data overlays (§T.47, §I.overlays, keys 1-9, 0 off) and the Tectonics layer (game UI: 'Plates' pill, key T).
//  - Data overlay: a translucent sheet draped over terrain/water, coloured per pixel from a per-column prep pass
//    (prep.ts), with a legend card + hover readout; mantle heat also tints the cut faces.
//  - Tectonics layer: plate boundary lines + one 3D arrow and label per plate, no fill; can stay on under any
//    data overlay. Hidden in ambient mode unless switched on there.
// Read-only on sim state (V15). Budget: nothing runs while both are off; with an overlay on, one 65k-thread prep
// dispatch (plates / categorical ones at ≤ 10 Hz) + one full-screen translucent sheet + ≤ 2 instanced arrow draws.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Sim } from '../sim/sim';
import { Lifecycle, type LifecycleOp } from '../sim/lifecycle';
import { NX, NZ, colIdx } from '../sim/layout';
import { MAX_PLATES } from '../sim/worldData';
import { createColumnGrid } from '../render/terrain';
import { worldToCell, voxelToWorldY } from '../render/space';
import { pickColumn } from '../ui/probe';
import { uniform } from 'three/tsl';
import { OVERLAYS, overlayByKey, readout, plateSpeedText, CM_PER_YR_PER_CELL_MY, type OverlayDef } from './defs';
import { createPrep } from './prep';
import { createSheetMaterial, createTectonicsMaterial, createFaceMesh, type SheetUniforms, type SheetBuffers } from './sheet';
import { createArrows } from './arrows';
import { createLegend } from './legend';
import { createTectonicsPill } from './tectonicsUi';
import { PLATE_COLORS, childColor, hexToLinear } from './colormaps';

export { OVERLAYS } from './defs';

/** Geo-time memory of the activity overlay (My): marks fade and rates average over about this long. */
const ACTIVITY_TAU_MY = 5;
/** Overlays without temporal smoothing whose prep runs at ≤ 10 Hz. */
const THROTTLED = new Set(['plates', 'biome', 'elev']);

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
  /** Plate lifecycle log (splits give children a colour related to the parent). */
  lineage?(): readonly { op: LifecycleOp }[];
}

/** OverlaySource over the live Sim: plate table, lifecycle centroids from the last stats window, lifecycle log. */
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
    lineage: () => sim.lifecycle.log,
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
  /** Objects hidden while a data overlay is on (flora / fauna sit on top of the map). Restored after. */
  declutter?: THREE.Object3D[];
  /** Clouds: faded out while a data overlay is on (setCloudFade: 1 normal … 0 hidden; without it the layer is hidden). */
  clouds?: { object: THREE.Object3D; setCloudFade?: (v: number) => void };
  /** Ambient mode: the Tectonics layer is off there unless the user switches it on. */
  ambient?: () => boolean;
}

export function createOverlays(o: OverlayOptions) {
  const { fields, scene, renderer, camera, source } = o;
  const U: SheetUniforms = { opacity: uniform(0.85), exposure: uniform(1), tecOpacity: uniform(0), arrowFade: uniform(0) };
  const prep = createPrep(fields);
  const legend = createLegend();
  const listeners = new Set<() => void>();

  // plate colours by id, kept per plate identity; split children get a related colour (lineage replay)
  const plateHex = PLATE_COLORS.slice();
  const plateColors = plateHex.map((h) => new THREE.Color().setRGB(...hexToLinear(h), THREE.LinearSRGBColorSpace));
  let lineageSeen = 0;
  const children = new Uint8Array(MAX_PLATES);
  const setPlateColor = (id: number, hex: string) => { plateHex[id] = hex; plateColors[id]!.setRGB(...hexToLinear(hex), THREE.LinearSRGBColorSpace); };
  function syncLineage() {
    const log = source.lineage?.();
    if (!log) return;
    if (log.length < lineageSeen) { // loaded a save: replay from the start
      lineageSeen = 0; children.fill(0);
      PLATE_COLORS.forEach((h, i) => setPlateColor(i, h));
    }
    for (; lineageSeen < log.length; lineageSeen++) {
      const op = log[lineageSeen]!.op;
      if (op.kind === 'split') setPlateColor(op.into, childColor(plateHex[op.plate]!, children[op.plate]!++));
    }
  }
  const B: SheetBuffers = { ovA: prep.ovA, T: prep.tec.T, TN: prep.tec.TN, plateColors };

  const grid = createColumnGrid();
  const sheet = new THREE.Mesh(grid);
  sheet.name = 'overlay-sheet';
  sheet.frustumCulled = false;
  sheet.renderOrder = 10;
  sheet.visible = false;
  scene.add(sheet);
  const tecSheet = new THREE.Mesh(grid, createTectonicsMaterial(fields, B, U));
  tecSheet.name = 'overlay-tectonics';
  tecSheet.frustumCulled = false;
  tecSheet.renderOrder = 14;
  tecSheet.visible = false;
  scene.add(tecSheet);
  const faceDef = OVERLAYS.find((d) => d.id === 'heat')!;
  const faces = createFaceMesh(fields, faceDef.bar2!, U);
  if (faces) { faces.visible = false; scene.add(faces); }
  const arrows = createArrows(fields, U);
  arrows.object.visible = false;
  scene.add(arrows.object);

  const mats = new Map<number, THREE.MeshBasicNodeMaterial>();
  const material = (def: OverlayDef) => {
    let m = mats.get(def.key);
    if (!m) { m = createSheetMaterial(fields, B, def, U); mats.set(def.key, m); }
    return m;
  };

  // plate tags (HTML, projected above the arrows)
  const tags = Array.from({ length: MAX_PLATES }, () => {
    const el = document.createElement('div');
    el.className = 'terra-plate-tag';
    el.style.display = 'none';
    el.innerHTML = `<b></b><span></span>`;
    document.body.appendChild(el);
    return { el, dot: el.querySelector('b')!, text: el.querySelector('span')!, shown: false, last: '', color: '' };
  });

  let current = 0;
  let uiVisible = true, legendOn = true, arrowsOn = true, declutterOn = true;
  // Tectonics layer: explicit user choice per mode (normal: on by default; ambient: off unless switched on)
  let tecNormal = true, tecAmbient = false;
  const ambient = () => o.ambient?.() ?? false;
  const tecOn = () => (ambient() ? tecAmbient : tecNormal);
  let tecFade = 0, cloudFade = 0;

  // visibility we took away from decluttered objects (restored when the overlay goes off)
  const hidden = new Map<THREE.Object3D, boolean>();
  const declutter = (on: boolean) => {
    if (on) for (const ob of o.declutter ?? []) { if (!hidden.has(ob)) { hidden.set(ob, ob.visible); ob.visible = false; } }
    else { for (const [ob, v] of hidden) ob.visible = v; hidden.clear(); }
  };
  let reset = true, lastGeo = NaN, lastPrep = -1e9, lastPlates = -1e9;
  const def = () => overlayByKey(current);
  const showArrows = () => tecOn() || (def()?.id === 'plates' && arrowsOn);
  const syncVisibility = () => {
    const d = def();
    sheet.visible = !!d;
    if (faces) faces.visible = d?.id === 'heat';
    legend.setVisible(!!d && legendOn && uiVisible);
    declutter(!!d && declutterOn);
    pill.setActive(tecOn());
    pill.setVisible(uiVisible);
    if (!showArrows()) for (const t of tags) if (t.shown) { t.el.style.display = 'none'; t.shown = false; }
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
  function setTectonics(on: boolean) {
    if (ambient()) tecAmbient = on; else tecNormal = on;
    syncVisibility();
    for (const l of listeners) l();
  }
  const pill = createTectonicsPill(() => setTectonics(!tecOn()));
  // key T (keys.ts owns the other keys; same text-field / modifier rules)
  const onKey = (e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || e.key.toLowerCase() !== 't') return;
    const t = (e.target instanceof HTMLElement ? e.target : document.activeElement) as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
    setTectonics(!tecOn());
  };
  window.addEventListener('keydown', onKey);

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
      const a = await readCol(d.id === 'plates' ? prep.tec.T : prep.ovA, hit.x, hit.z);
      const b = d.id === 'activity' || d.id === 'flow' ? await readCol(prep.ovB, hit.x, hit.z) : new Float32Array(4);
      if (def() !== d) return;
      legend.setReadout(readout(d, { plates: source.plates(), a, b }), `${hit.x}, ${hit.z}`);
    } finally { hoverBusy = false; }
  }
  /** Readback of prep values; zeros while the buffer has never been bound (three allocates lazily). */
  const readBuf = async (buf: typeof prep.ovA, offset: number, bytes: number) => {
    const attr = buf.value as THREE.StorageInstancedBufferAttribute;
    const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } };
    if (!backend.get(attr).buffer) return new Float32Array(bytes / 4);
    return new Float32Array(await renderer.getArrayBufferAsync(attr, null, offset, bytes));
  };
  const readCol = (buf: typeof prep.ovA, x: number, z: number) => readBuf(buf, colIdx(x, z) * 16, 16);

  // ---- per frame ---------------------------------------------------------------------------------------
  const v = new THREE.Vector3();
  function frame(dt: number) {
    const d = def();
    const tec = tecOn();
    // eased fades: Tectonics layer in/out, clouds out under a data overlay
    const k = 1 - Math.exp(-Math.max(0, dt) / 0.18);
    tecFade += ((tec ? 1 : 0) - tecFade) * k;
    if (Math.abs(tecFade - (tec ? 1 : 0)) < 1e-3) tecFade = tec ? 1 : 0;
    U.tecOpacity.value = tecFade;
    tecSheet.visible = tecFade > 0 && d?.id !== 'plates'; // the Plates overlay draws the same lines itself
    fadeClouds(!!d, k);
    const plates = showArrows() || d?.id === 'plates' || tecFade > 0;
    arrows.object.visible = showArrows() || tecFade > 0;
    U.arrowFade.value = d?.id === 'plates' && arrowsOn ? 1 : tecFade;
    if (!d && !plates) return;

    U.exposure.value = o.exposure?.() ?? 1;
    const sea = source.seaLevel();
    prep.u.seaLevel.value = sea;
    const now = performance.now();
    const plateList = source.plates();
    if (plates) {
      syncLineage();
      for (let i = 0; i < MAX_PLATES; i++) {
        const p = plateList[i];
        prep.plateVel[i]!.set(p?.vel[0] ?? 0, p?.vel[1] ?? 0, p?.alive ? 1 : 0, 0);
      }
      // plate boundaries change only on tectonics ticks: ≤ 10 Hz
      if (now - lastPlates >= 100 || (reset && d?.id === 'plates')) { prep.runPlates(renderer); lastPlates = now; }
    }
    if (d && d.id !== 'plates') {
      const g = source.geoMy();
      const geoDt = Number.isFinite(lastGeo) ? Math.max(0, g - lastGeo) : 0;
      lastGeo = g;
      prep.u.ema.value = reset || d.smoothS <= 0 ? 1 : 1 - Math.exp(-dt / d.smoothS);
      prep.u.emaGeo.value = reset ? 1 : 1 - Math.exp(-geoDt / ACTIVITY_TAU_MY);
      prep.u.geoDt.value = geoDt;
      prep.u.reset.value = reset ? 1 : 0;
      if (reset || !THROTTLED.has(d.id) || now - lastPrep >= 100) { prep.run(renderer, d); lastPrep = now; }
    }
    reset = false;

    if (plates) {
      arrows.update(plateList, source.centroids(), dt);
      if (d?.id === 'plates') {
        let fastest = 0, n = 0;
        for (const s of arrows.state) if (s.visible) { n++; fastest = Math.max(fastest, s.speed); }
        legend.setNote(n ? `Arrows: plate motion, longer = faster · ${n} plates, fastest ${(fastest * CM_PER_YR_PER_CELL_MY).toFixed(1)} cm/yr.`
          : 'Arrows: plate motion, longer = faster (after the first stats window).');
      }
      if (showArrows() && uiVisible) { refreshSurf(now); placeTags(sea); }
    }
    if (d && legend.visible && pointer) {
      refreshSurf(now);
      if (hoverDirty && !hoverBusy && now - lastPick >= 100) { hoverDirty = false; lastPick = now; void hover(d); }
      // values under a still cursor change as the sim runs: re-read about once a second
      if (!hoverDirty && now - lastPick >= 1000) hoverDirty = true;
    }
  }

  function fadeClouds(on: boolean, k: number) {
    const c = o.clouds;
    if (!c) return;
    const target = on && declutterOn ? 1 : 0;
    if (cloudFade === target) return;
    cloudFade += (target - cloudFade) * k;
    if (Math.abs(cloudFade - target) < 1e-3) cloudFade = target;
    if (c.setCloudFade) c.setCloudFade(1 - cloudFade); // atmosphere: 1 = normal clouds, 0 = hidden
    else {
      // no fade hook on the atmosphere yet: hide / restore the whole layer at the midpoint
      if (cloudFade > 0.5 && !hidden.has(c.object)) { hidden.set(c.object, c.object.visible); c.object.visible = false; }
      if (cloudFade <= 0.5 && hidden.has(c.object)) { c.object.visible = hidden.get(c.object)!; hidden.delete(c.object); }
    }
  }

  function placeTags(sea: number) {
    const r = dom.getBoundingClientRect();
    const plateList = source.plates();
    for (let i = 0; i < MAX_PLATES; i++) {
      const st = arrows.state[i]!, t = tags[i]!;
      if (!st.visible) { if (t.shown) { t.el.style.display = 'none'; t.shown = false; } continue; }
      const cx = Math.round(worldToCell(st.x)), cz = Math.round(worldToCell(st.z));
      const h = surf ? Math.max(sea, surf[colIdx(Math.max(0, Math.min(NX - 1, cx)), Math.max(0, Math.min(NZ - 1, cz)))]!) : sea;
      v.set(st.x, voxelToWorldY(h) + 0.2, st.z).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 1.05 || Math.abs(v.y) > 1.05) { if (t.shown) { t.el.style.display = 'none'; t.shown = false; } continue; }
      if (!t.shown) { t.el.style.display = ''; t.shown = true; }
      t.el.style.opacity = `${Math.max(tecFade, def()?.id === 'plates' ? 1 : 0)}`;
      t.el.style.transform = `translate(${(r.left + ((v.x + 1) / 2) * r.width).toFixed(1)}px, ${(r.top + ((1 - v.y) / 2) * r.height).toFixed(1)}px) translate(-50%, -100%)`;
      const txt = plateSpeedText(plateList[i]!.vel);
      if (txt !== t.last) { t.text.textContent = txt; t.last = txt; }
      if (t.color !== plateHex[i]) { t.color = plateHex[i]!; t.dot.style.background = t.color; t.dot.textContent = `${i}`; }
    }
  }

  o.onFrame?.(frame);
  syncVisibility();

  return {
    /** Keys (§I.keys): n = 0 or the active key again turns overlays off; 1-9 select. */
    set(n: number): boolean { select(n === current ? 0 : n); return true; },
    /** Select without toggling (panel). */
    select,
    get current() { return current; },
    get def() { return def(); },
    defs: OVERLAYS,
    /** Tectonics layer (plate boundaries + arrows, no fill), 'Plates' pill / key T. */
    get tectonics() { return tecOn(); },
    set tectonics(x: boolean) { setTectonics(x); },
    get opacity() { return U.opacity.value; },
    set opacity(x: number) { U.opacity.value = Math.min(1, Math.max(0, x)); },
    get legendVisible() { return legendOn; },
    set legendVisible(x: boolean) { legendOn = x; syncVisibility(); },
    get arrowsVisible() { return arrowsOn; },
    set arrowsVisible(x: boolean) { arrowsOn = x; syncVisibility(); },
    /** Hide life and fade clouds while a data overlay is on. */
    get declutter() { return declutterOn; },
    set declutter(x: boolean) { declutterOn = x; syncVisibility(); },
    /** UI hidden (H): hide legend, tags and the pill; keep the map colouring. Also re-evaluates ambient mode. */
    setLabelVisible(x: boolean) { uiVisible = x; syncVisibility(); },
    onChange(cb: () => void) { listeners.add(cb); return () => listeners.delete(cb); },
    frame,
    legend,
    pill,
    arrows: arrows.state,
    plateColors: plateHex,
    /** Test/debug: prep values of one column (vec4, see prep.ts); 'tec' = plate boundary buffer T. */
    readColumn: (x: number, z: number, which: 'a' | 'b' | 'tec' = 'a') => readCol(which === 'a' ? prep.ovA : which === 'b' ? prep.ovB : prep.tec.T, x, z),
    /** Test/debug: a whole prep buffer. */
    readAll: (which: 'a' | 'b' | 'tec' | 'tecN' = 'a') =>
      readBuf({ a: prep.ovA, b: prep.ovB, tec: prep.tec.T, tecN: prep.tec.TN }[which], 0, NX * NZ * 16),
    objects: { sheet, tecSheet, faces, arrows: arrows.object },
    dispose() { window.removeEventListener('keydown', onKey); },
  };
}
export type Overlays = ReturnType<typeof createOverlays>;
