import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';
import { createPerfHud } from './ui/perfHud';
import { GpuFields } from './core/gpu';
import { Params, PARAM_DEFS } from './core/params';
import { DualClock } from './core/clock';
import { createPanel } from './ui/panel';
import { createTimebar, type ClockLike } from './ui/timebar';
import { bindKeys } from './ui/keys';

/** My per sim tick. Fixed for the life of a world (V12, V22). */
export const DT_GEO = 0.05;
/** Ticks per frame the sim may run before speed is capped (refined by perf budget later, V9). */
const MAX_TICKS_PER_FRAME = 8;

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
  fields.freeze();

  const panel = createPanel(params);
  const timebar = createTimebar(clockUi, { minSpeed: speedDef.min!, maxSpeed: speedDef.max! });
  const hud = createPerfHud(stage.renderer, () => fields.bytes());

  let uiVisible = !(params.get('ambientMode') as boolean);
  const applyUi = () => { panel.setVisible(uiVisible); timebar.setVisible(uiVisible); hud.setVisible(uiVisible); };
  applyUi();
  bindKeys({
    toggleUI: () => { uiVisible = !uiVisible; applyUi(); },
    togglePause: () => clock.togglePause(),
    faster: () => params.set('speed', (params.get('speed') as number) * 2),
    slower: () => params.set('speed', (params.get('speed') as number) / 2),
  });

  stage.onFrame((dt) => {
    panel.fps.begin();
    const ticks = clock.frame(dt, MAX_TICKS_PER_FRAME);
    void ticks; // sim passes hook in here (M2+)
    timebar.update();
    hud.frame(dt, stage.cpuMs);
    panel.fps.end();
  });
  stage.start();

  (window as unknown as { terra: unknown }).terra = { params, clock, fields, stage };
  (window as unknown as { terraReady: boolean }).terraReady = true;
}

main();
