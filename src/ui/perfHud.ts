import type * as THREE from 'three/webgpu';

// Frame timing HUD. GPU times come from timestamp queries when the adapter supports them (V9 budget check).
export function createPerfHud(renderer: THREE.WebGPURenderer, gpuBytes: () => number) {
  const el = document.createElement('div');
  el.className = 'terra-perf';
  document.body.appendChild(el);
  let frames = 0, acc = 0, cpuMs = 0, gpuCompute = 0, gpuRender = 0, lastText = '';
  const hasTs = renderer.hasFeature('timestamp-query');

  async function sample() {
    if (!hasTs) return;
    await renderer.resolveTimestampsAsync('compute');
    await renderer.resolveTimestampsAsync('render');
    gpuCompute = renderer.info.compute.timestamp;
    gpuRender = renderer.info.render.timestamp;
  }

  return {
    el,
    frame(realDt: number, frameCpuMs: number) {
      frames++; acc += realDt; cpuMs += frameCpuMs;
      if (frames % 4 === 0) void sample();
      if (acc < 0.5) return;
      const fps = frames / acc;
      const ts = hasTs ? `gpu sim ${gpuCompute.toFixed(2)}ms · render ${gpuRender.toFixed(2)}ms` : 'gpu timing n/a';
      const text = `${fps.toFixed(0)} fps · cpu ${(cpuMs / frames).toFixed(2)}ms · ${ts} · ${(gpuBytes() / 1048576).toFixed(0)} MB`;
      if (text !== lastText) { el.textContent = text; lastText = text; }
      frames = 0; acc = 0; cpuMs = 0;
    },
    setVisible(v: boolean) { el.style.display = v ? '' : 'none'; },
  };
}
