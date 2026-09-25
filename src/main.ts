import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';
import { createPerfHud } from './ui/perfHud';
import { GpuFields } from './core/gpu';

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
  const fields = new GpuFields();
  fields.freeze();
  const hud = createPerfHud(stage.renderer, () => fields.bytes());
  stage.onFrame((dt) => hud.frame(dt, stage.cpuMs));
  stage.start();
  (window as unknown as { terraReady: boolean }).terraReady = true;
}

main();
