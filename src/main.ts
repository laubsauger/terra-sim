import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';

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
  stage.start();
  (window as unknown as { terraReady: boolean }).terraReady = true;
}

main();
