// Startup loader (index.html #terra-loader): step text while the app boots, fade-out once the first frames (and so
// the shader pipelines) are done. Everything is optional: pages without the loader element just no-op.

const el = () => document.getElementById('terra-loader');

export function loaderStep(text: string): void {
  const s = document.getElementById('terra-loader-step');
  if (s) s.textContent = text;
}

/** Show a step and let the browser paint it before a long synchronous stretch. */
export async function loaderYield(text: string): Promise<void> {
  loaderStep(text);
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
}

/** Fade the loader out after `frames` animation frames (the first renders compile the pipelines). */
export function loaderDone(frames = 3): void {
  let n = 0;
  const tick = () => {
    if (++n < frames) { requestAnimationFrame(tick); return; }
    const l = el();
    if (!l) return;
    l.style.opacity = '0';
    l.style.pointerEvents = 'none';
    setTimeout(() => l.remove(), 600);
  };
  requestAnimationFrame(tick);
}

/** Remove at once (error screen takes over). */
export function loaderRemove(): void { el()?.remove(); }
