import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';
import { createPerfHud } from './ui/perfHud';
import { GpuFields } from './core/gpu';
import { Params, PARAM_DEFS } from './core/params';
import { DualClock } from './core/clock';
import { createPanel } from './ui/panel';
import { createTimebar, type ClockLike } from './ui/timebar';
import { bindKeys } from './ui/keys';
import { registerSimFields, uploadWorld } from './sim/fields';
import { generateWorld } from './sim/worldgen';
import { Sim, STATS_WINDOW } from './sim/sim';
import { SaveManager, SaveStore, SAVE_EXCLUDE } from './core/save';
import { NanGuard } from './core/nanGuard';
import { createSavePanel, showToast } from './ui/savePanel';
import { createTerrain } from './render/terrain';
import { createSides } from './render/sides';
import { createWater } from './render/water';
import { createLighting } from './render/lighting';
import { setVertEx } from './render/space';
import { createProbeUI } from './ui/probe';
import { createStatsPane } from './ui/statsPane';
import { createOverlays } from './overlay/overlays';
import { createAmbientCam } from './ui/ambientCam';
import { Ambience } from './audio/ambience';
import { TickBudget } from './core/tickBudget';
import { createGodPanel } from './ui/godPanel';

/** My per sim tick. Fixed for the life of a world (V12, V22). */
export const DT_GEO = 0.05;

async function main() {
  const gpu = await checkWebGPU();
  if (!gpu.ok) {
    showErrorScreen('WebGPU unavailable', gpu.reason);
    return;
  }
  const app = document.getElementById('app')!;
  let stage;
  try {
    stage = await createStage(app, gpu.adapter);
  } catch (e) {
    showErrorScreen('Renderer failed to start', (e as Error).message);
    return;
  }

  // Params are the single source of truth (V20); clock follows the 'speed' param.
  const params = new Params();
  params.applyURL(location.search);
  const clock = new DualClock({ dtGeo: DT_GEO, requestedSpeed: params.get('speed') as number });
  params.onChange((key, v) => { if (key === 'speed') clock.setSpeed(v as number); });

  const speedDef = PARAM_DEFS.find((d) => d.key === 'speed')!;
  const clockUi: ClockLike = {
    get geoTime() { return clock.geoTime; },
    get paused() { return clock.paused; },
    get requestedSpeed() { return clock.requestedSpeed; },
    get effectiveSpeed() { return clock.effectiveSpeed; },
    setSpeed: (v) => params.set('speed', v),
    togglePause: () => clock.togglePause(),
  };

  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  const world = generateWorld(params.get('seed') as number, { plates: params.get('initialPlates') as number });
  uploadWorld(fields, world);
  const sim = new Sim(stage.renderer, fields, world, params, DT_GEO);
  // T50/T35: saves, autosave ring, NaN guard + rollback
  const guard = new NanGuard(fields, SAVE_EXCLUDE);
  let store: SaveStore | null = null;
  try { store = await SaveStore.open(); } catch (e) { console.warn('IndexedDB unavailable: autosave keeps an in-memory snapshot only', e); }
  const saves = new SaveManager({ renderer: stage.renderer, fields, sim, params, clock, guard }, store, {
    windowTicks: STATS_WINDOW, notice: showToast, pause: () => { clock.paused = true; },
  });
  const loadSlot = new URLSearchParams(location.search).get('load'); // ?load=<slot key>|latest
  if (loadSlot) await saves.loadSlot(loadSlot).catch(() => {}); // failure is toasted; the fresh world keeps running

  createLighting(stage.scene);
  stage.scene.add(createTerrain(fields).object, createSides(fields).object, createWater(fields).object);
  setVertEx(params.get('verticalExaggeration') as number);
  params.onChange((key, v) => { if (key === 'verticalExaggeration') setVertEx(v as number); });

  const panel = createPanel(params);
  const savePanel = createSavePanel(panel.folders.Save, saves);
  const timebar = createTimebar(clockUi, { minSpeed: speedDef.min!, maxSpeed: speedDef.max! });
  // sim GPU budget per frame (V9): speed beyond this shows as effective < requested (V22)
  const budget = new TickBudget(5);
  let ticksSinceSample = 0;
  const hud = createPerfHud(stage.renderer, () => fields.bytes(), (ms) => { budget.observe(ms, ticksSinceSample); ticksSinceSample = 0; });
  const god = createGodPanel(panel.folders.God, sim, stage.renderer, stage.camera, fields);
  const probe = createProbeUI(stage.renderer, stage.camera, fields, god.isInspect);
  const statsPane = createStatsPane(panel.folders.Stats, sim);
  const overlays = createOverlays(fields, stage.scene);
  const ambientCam = createAmbientCam(stage.camera, stage.controls, stage.renderer.domElement);
  const audio = new Ambience();
  let volcanic = 0, seenEvents = 0;

  let uiVisible = !(params.get('ambientMode') as boolean);
  const applyUi = () => { panel.setVisible(uiVisible); timebar.setVisible(uiVisible); hud.setVisible(uiVisible); probe.setVisible(uiVisible); overlays.setLabelVisible(uiVisible);
    ambientCam.setEnabled(!uiVisible || (params.get('ambientMode') as boolean)); };
  params.onChange((key) => { if (key === 'ambientMode') applyUi(); });
  applyUi();
  bindKeys({
    toggleUI: () => { uiVisible = !uiVisible; applyUi(); },
    togglePause: () => clock.togglePause(),
    faster: () => params.set('speed', (params.get('speed') as number) * 2),
    slower: () => params.set('speed', (params.get('speed') as number) / 2),
    overlay: (n) => { overlays.set(n); },
    mute: () => audio.toggleMute(),
  });

  stage.onFrame((dt) => {
    panel.fps.begin();
    const ticks = clock.frame(dt, budget.cap);
    const ran = sim.runTicks(ticks);
    ticksSinceSample += ran;
    if (ran < ticks) clock.unrun(ticks - ran);
    saves.frame();
    savePanel.update();
    timebar.update();
    probe.update(dt);
    statsPane.update();
    ambientCam.notice(sim.events.log, () => sim.stats?.seaLevel ?? 76);
    ambientCam.update(dt);
    // audio levels from sim state only (V15)
    const log = sim.events.log;
    for (; seenEvents < log.length; seenEvents++) if (log[seenEvents]!.kind !== 'iceAge') volcanic = Math.min(1, volcanic + 0.5);
    volcanic *= Math.exp(-dt / 20);
    const land = sim.stats?.landFrac ?? 0.3;
    const camH = stage.camera.position.y;
    audio.set({ ocean: 1 - land, wind: 0.35 + 0.3 * Math.min(1, camH / 6), volcanic, life: land * 1.2, closeness: Math.max(0, 1 - (camH - 0.3) / 4) });
    audio.update(dt);
    hud.frame(dt, stage.cpuMs);
    panel.fps.end();
  });
  stage.start();

  (window as unknown as { terra: unknown }).terra = { params, clock, fields, stage, sim, world, probe, overlays, audio, god, saves,
    save: async () => (await saves.exportFile()).blob, load: (blob: Blob) => saves.loadBlob(blob) };
  (window as unknown as { terraReady: boolean }).terraReady = true;
}

main();
