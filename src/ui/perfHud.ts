import type * as THREE from 'three/webgpu';

// Frame timing HUD. GPU times come from timestamp queries when the adapter supports them (V9 budget check).
/** onGpuSample: compute ms resolved since the previous sample (covers the frames since then). */
export function createPerfHud(renderer: THREE.WebGPURenderer, gpuBytes: () => number, onGpuSample?: (computeMs: number) => void) {
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
    onGpuSample?.(gpuCompute);
  }

  return {
    el,
    frame(realDt: number, frameCpuMs: number) {
      frames++; acc += realDt; cpuMs += frameCpuMs;
      void sample(); // every frame: the query pool holds only 256 timestamps between resolves
      if (acc < 0.5) return;
      const fps = frames / acc;
      // render timestamps sum overlapping pass intervals on tile-based GPUs (Apple) → shown as Σpass, not frame time
      const ts = hasTs ? `gpu sim ${gpuCompute.toFixed(2)}ms · render Σpass ${gpuRender.toFixed(1)}ms` : 'gpu timing n/a';
      const text = `${fps.toFixed(0)} fps · cpu ${(cpuMs / frames).toFixed(2)}ms · ${ts} · ${(gpuBytes() / 1048576).toFixed(0)} MB`;
      if (text !== lastText) { el.textContent = text; lastText = text; }
      frames = 0; acc = 0; cpuMs = 0;
    },
    setVisible(v: boolean) { el.style.display = v ? '' : 'none'; },
  };
}
