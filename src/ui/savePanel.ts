// Save folder (I.ui Save): save now, slot list + load, export/import .terra, autosave indicator, toasts.
import type { FolderApi, ListBladeApi } from 'tweakpane';
import type { SaveManager, SlotMeta } from '../core/save';

export interface SavePanel { refresh(): Promise<void>; update(): void; dispose(): void }

const NO_SLOT = '';

export function createSavePanel(folder: FolderApi, mgr: SaveManager): SavePanel {
  const view = { autosave: '', lastGood: '' };
  folder.addBinding(view, 'autosave', { readonly: true, label: 'autosave' });
  folder.addBinding(view, 'lastGood', { readonly: true, label: 'last good' });
  folder.addButton({ title: 'save now' }).on('click', () => { void mgr.saveSlot().then(() => refresh()).catch(() => {}); });
  const list = folder.addBlade({ view: 'list', label: 'slot', options: [{ text: '(none)', value: NO_SLOT }], value: NO_SLOT }) as ListBladeApi<string>;
  folder.addButton({ title: 'load slot' }).on('click', () => {
    if (list.value !== NO_SLOT) void mgr.loadSlot(list.value).catch(() => {});
  });
  folder.addButton({ title: 'export file' }).on('click', () => {
    void mgr.exportFile().then(({ blob, filename }) => download(blob, filename)).catch(() => {});
  });
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.terra';
  input.style.display = 'none';
  document.body.appendChild(input);
  input.addEventListener('change', () => {
    const f = input.files?.[0];
    input.value = '';
    if (f) void mgr.loadBlob(f, f.name).catch(() => {});
  });
  folder.addButton({ title: 'import file' }).on('click', () => input.click());

  async function refresh(): Promise<void> {
    let slots: SlotMeta[] = [];
    try { slots = await mgr.listSlots(); } catch (e) { console.warn('save panel: slot list failed', e); }
    const prev = list.value;
    list.options = slots.length
      ? slots.map((m) => ({ text: `${m.key} · ${m.geoTime.toFixed(1)} My · ${new Date(m.savedAt).toLocaleTimeString()}`, value: m.key }))
      : [{ text: '(none)', value: NO_SLOT }];
    list.value = slots.some((m) => m.key === prev) ? prev : (slots[0]?.key ?? NO_SLOT);
  }

  let lastAuto: number | null = null;
  let lastUpdate = -Infinity;
  /** Call every frame; refreshes the indicator at most twice a second. */
  function update(): void {
    const now = performance.now();
    if (now - lastUpdate < 500) return;
    lastUpdate = now;
    const s = mgr.status();
    const when = s.lastAutosave ? `last ${new Date(s.lastAutosave.savedAt).toLocaleTimeString()} (${s.lastAutosave.key})` : 'none yet';
    view.autosave = s.autosaveMinutes <= 0 ? 'off' : `${s.autosaveMinutes} min · ${s.idb ? when : 'memory only (no IDB)'}${s.busy ? ' · …' : ''}`;
    view.lastGood = s.lastGoodTick === null ? '—' : `tick ${s.lastGoodTick}`;
    if (s.lastAutosave && s.lastAutosave.savedAt !== lastAuto) { lastAuto = s.lastAutosave.savedAt; void refresh(); }
  }

  void refresh();
  update();
  return { refresh, update, dispose() { input.remove(); } };
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

let toastEl: HTMLDivElement | null = null;
let toastTimer = 0;

/** Small bottom-centre toast for save/load/NaN notices. One at a time; errors stay longer. */
export function showToast(msg: string, level: 'info' | 'warn' | 'error' = 'info'): void {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.setAttribute('role', 'status');
    Object.assign(toastEl.style, {
      position: 'fixed', left: '50%', bottom: '64px', transform: 'translateX(-50%)', zIndex: '1000',
      padding: '8px 14px', borderRadius: '6px', font: '12px/1.4 system-ui, sans-serif', color: '#fff',
      maxWidth: 'min(90vw, 560px)', pointerEvents: 'none', transition: 'opacity 0.3s', boxShadow: '0 2px 10px rgba(0,0,0,0.35)',
    } satisfies Partial<CSSStyleDeclaration>);
    document.body.appendChild(toastEl);
  }
  toastEl.textContent = msg;
  toastEl.style.background = level === 'error' ? '#a8322d' : level === 'warn' ? '#9a6a12' : '#2d4f7c';
  toastEl.style.opacity = '1';
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { if (toastEl) toastEl.style.opacity = '0'; }, level === 'error' ? 8000 : 3500);
}
