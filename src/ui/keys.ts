// Global key bindings (§I.keys). Returns an unbind function.

export interface KeyHandlers {
  toggleUI(): void;
  togglePause(): void;
  faster(): void;
  slower(): void;
  inspect?(): void;
  overlay?(n: number): void;
  mute?(): void;
}

function isTextTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
}

export function bindKeys(h: KeyHandlers, target: Window | HTMLElement = window): () => void {
  const onKey = (ev: Event) => {
    const e = ev as KeyboardEvent;
    if (e.ctrlKey || e.metaKey || e.altKey) return; // leave browser shortcuts alone
    if (isTextTarget(e.target) || isTextTarget(document.activeElement)) return;

    const k = e.key;
    // toggles must not flicker on key auto-repeat; speed steps may repeat
    if (k === ' ' || k === 'Spacebar') {
      e.preventDefault();
      if (!e.repeat) h.togglePause();
      return;
    }
    if (k === ']') return void h.faster();
    if (k === '[') return void h.slower();
    if (e.repeat) return;
    switch (k.toLowerCase()) {
      case 'h':
        return void h.toggleUI();
      case 'i':
        return void h.inspect?.();
      case 'm':
        return void h.mute?.();
    }
    if (k.length === 1 && k >= '0' && k <= '9') h.overlay?.(Number(k));
  };
  target.addEventListener('keydown', onKey);
  return () => target.removeEventListener('keydown', onKey);
}
