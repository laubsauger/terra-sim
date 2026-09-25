// V18: missing WebGPU must produce a clear message, never a blank page.
export function showErrorScreen(title: string, reason: string): void {
  const el = document.createElement('div');
  el.className = 'terra-error';
  el.innerHTML = `<div class="terra-error-card"><h1></h1><p class="reason"></p>
    <p class="hint">terra-sim needs WebGPU. Use a recent Chrome or Edge on desktop with hardware acceleration enabled.</p></div>`;
  el.querySelector('h1')!.textContent = title;
  el.querySelector('.reason')!.textContent = reason;
  document.body.appendChild(el);
}

export type GpuCheck = { ok: true; adapter: GPUAdapter } | { ok: false; reason: string };

export async function checkWebGPU(): Promise<GpuCheck> {
  if (!('gpu' in navigator) || !navigator.gpu) return { ok: false, reason: 'navigator.gpu is not available in this browser.' };
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { ok: false, reason: `requestAdapter() failed: ${(e as Error).message}` };
  }
  if (!adapter) return { ok: false, reason: 'No WebGPU adapter found (GPU blocklisted or disabled).' };
  return { ok: true, adapter };
}
