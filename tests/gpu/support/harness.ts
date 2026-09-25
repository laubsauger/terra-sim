// Loaded inside the browser via dynamic import from Playwright page.evaluate.
import * as THREE from 'three/webgpu';

export async function makeRenderer(): Promise<THREE.WebGPURenderer> {
  const adapter = await navigator.gpu.requestAdapter();
  const l = adapter!.limits;
  const r = new THREE.WebGPURenderer({
    requiredLimits: {
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxBufferSize: l.maxBufferSize,
      maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    },
  } as THREE.WebGPURendererParameters);
  await r.init();
  return r;
}

// Bare specifiers do not resolve inside page.evaluate; re-export through this Vite-served module.
export * as TSL from 'three/tsl';
export * as THREE from 'three/webgpu';
