import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';
import { createPerfHud } from './ui/perfHud';
import { GpuFields } from './core/gpu';
import { Params, PARAM_DEFS } from './core/params';
import { DualClock } from './core/clock';
import { createPanel } from './ui/panel';
import { createTimebar, type ClockLike } from './ui/timebar';
import { createDayPill } from './ui/dayPill';
import { bindKeys } from './ui/keys';
import { registerSimFields, uploadWorld } from './sim/fields';
import { generateWorld } from './sim/worldgen';
import { Sim, STATS_WINDOW } from './sim/sim';
import { SaveManager, SaveStore, SAVE_EXCLUDE } from './core/save';
import { NanGuard } from './core/nanGuard';
import { createSavePanel, showToast } from './ui/savePanel';
import { createLook } from './render/look';
import { createLife } from './life/life';
import { createAtmosphere } from './atmo/atmosphere';
import { createSlice } from './slice/slice';
import { createFx } from './fx/fx';
import { setVertEx } from './render/space';
import { createProbeUI } from './ui/probe';
import { createStatsPane } from './ui/statsPane';
import { createOverlays, simSource } from './overlay/overlays';
import { createOverlayPanel } from './ui/overlayPanel';
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

  const look = createLook(stage, fields, { highQuality: params.get('highQuality') as boolean, motion: sim.tectonics });
  const life = createLife(fields, stage.renderer, stage.scene, { highQuality: params.get('highQuality') as boolean, seed: params.get('seed') as number, camera: stage.camera });
  const slice = createSlice(stage, fields);
  const atmo = createAtmosphere(fields, stage.renderer, stage.scene, stage.camera, { highQuality: params.get('highQuality') as boolean });
  params.onChange((k, v) => { if (k === 'highQuality') { look.setHighQuality(v as boolean); life.setHighQuality(v as boolean); atmo.setHighQuality(v as boolean); } });
  setVertEx(params.get('verticalExaggeration') as number);
  params.onChange((key, v) => { if (key === 'verticalExaggeration') setVertEx(v as number); });

  const panel = createPanel(params);
  const savePanel = createSavePanel(panel.folders.Save, saves);
  const timebar = createTimebar(clockUi, { minSpeed: speedDef.min!, maxSpeed: speedDef.max! });
  const dayPill = createDayPill(look);
  // sim GPU budget per frame (V9): speed beyond this shows as effective < requested (V22)
  const budget = new TickBudget(5);
  let ticksSinceSample = 0;
  const hud = createPerfHud(stage.renderer, () => fields.bytes(), (ms) => { budget.observe(ms, ticksSinceSample); ticksSinceSample = 0; });
  const fx = createFx(fields, stage.renderer, stage.scene, stage.camera, sim, {
    highQuality: params.get('highQuality') as boolean,
    controls: stage.controls,
    speed: () => clock.effectiveSpeed,
  });
  params.onChange((k, v) => { if (k === 'highQuality') fx.setHighQuality(v as boolean); });
  const god = createGodPanel(panel.folders.God, sim, stage.renderer, stage.camera, fields, stage.scene, fx);
  const probe = createProbeUI(stage.renderer, stage.camera, fields, god.isInspect);
  const statsPane = createStatsPane(panel.folders.Stats, sim);
  // overlays (T47) + Tectonics layer (key T / 'Plates' pill): per-frame prep via stage.onFrame; colours pre-compensate
  // the post exposure + tone map; life hidden and clouds faded while a data overlay is on
  const overlays = createOverlays({ fields, scene: stage.scene, renderer: stage.renderer, camera: stage.camera, source: simSource(sim),
    exposure: () => look.post.u.exposure.value, onFrame: (cb) => stage.onFrame(cb), declutter: [life.object], clouds: atmo,
    ambient: () => params.get('ambientMode') as boolean, cut: () => slice.cut });
  createOverlayPanel(panel.folders.Overlays, overlays);
  const ambientCam = createAmbientCam(stage.camera, stage.controls, stage.renderer.domElement);
  const audio = new Ambience();
  let volcanic = 0, seenEvents = 0;

  let uiVisible = !(params.get('ambientMode') as boolean);
  const applyUi = () => { panel.setVisible(uiVisible); timebar.setVisible(uiVisible); dayPill.setVisible(uiVisible); hud.setVisible(uiVisible); probe.setVisible(uiVisible); overlays.setLabelVisible(uiVisible); slice.setVisible(uiVisible);
    ambientCam.setEnabled(!uiVisible || (params.get('ambientMode') as boolean)); };
  // ambient mode runs a slow day; leaving it stops the cycle (the Day pill can restart it). Not on H: that
  // must not undo the pill's choice.
  const applyAmbientDay = () => look.setDayLength((params.get('ambientMode') as boolean) ? 600 : 0);
  params.onChange((key) => { if (key === 'ambientMode') { applyUi(); applyAmbientDay(); } });
  applyUi();
  applyAmbientDay();
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
    sim.flushEvents(); // god tools act immediately, even while paused
    const ticks = clock.frame(dt, budget.cap);
    const ran = sim.runTicks(ticks);
    ticksSinceSample += ran;
    if (ran < ticks) clock.unrun(ticks - ran);
    look.frame(clock.ambTime);
    life.setSeaLevel(sim.stats?.seaLevel ?? 76);
    if (ran > 0) life.invalidate();
    life.update(dt, clock.ambTime);
    atmo.setCut(slice.cut.x, slice.cut.z);
    atmo.update(dt, clock.ambTime, { events: sim.events.log, seaLevel: sim.stats?.seaLevel });
    slice.update(dt);
    saves.frame();
    savePanel.update();
    timebar.update();
    dayPill.update();
    probe.update(dt);
    god.update(dt);
    statsPane.update();
    ambientCam.notice(sim.events.log, () => sim.stats?.seaLevel ?? 76);
    ambientCam.update(dt);
    fx.update(dt, clock.ambTime);
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

  (window as unknown as { terra: unknown }).terra = { params, clock, fields, stage, sim, world, probe, overlays, audio, god, saves, look, life, atmo, slice, fx,
    save: async () => (await saves.exportFile()).blob, load: (blob: Blob) => saves.loadBlob(blob) };
  (window as unknown as { terraReady: boolean }).terraReady = true;
}

main();
