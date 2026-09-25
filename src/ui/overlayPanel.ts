// Tweakpane 'Overlays' folder (§I.ui): overlay selector (mirrors keys 1-9 / 0), opacity, legend / arrow / declutter
// toggles, and the Tectonics layer (also the 'Plates' pill and key T).
import type { FolderApi } from 'tweakpane';
import type { Overlays } from '../overlay/overlays';

export function createOverlayPanel(folder: FolderApi, ov: Overlays) {
  const state = { overlay: ov.current, tectonics: ov.tectonics, opacity: ov.opacity, legend: ov.legendVisible, arrows: ov.arrowsVisible, declutter: ov.declutter };
  const options = [{ text: 'off', value: 0 }, ...ov.defs.map((d) => ({ text: `${d.key} · ${d.title}`, value: d.key }))];
  const sel = folder.addBinding(state, 'overlay', { label: 'overlay', options });
  sel.on('change', (ev) => ov.select(ev.value as number));
  const tec = folder.addBinding(state, 'tectonics', { label: 'plate lines (T)' });
  tec.on('change', (ev) => { ov.tectonics = ev.value as boolean; });
  folder.addBinding(state, 'opacity', { label: 'opacity', min: 0.2, max: 1, step: 0.01 }).on('change', (ev) => { ov.opacity = ev.value as number; });
  folder.addBinding(state, 'legend', { label: 'show legend' }).on('change', (ev) => { ov.legendVisible = ev.value as boolean; });
  folder.addBinding(state, 'arrows', { label: 'show plate arrows' }).on('change', (ev) => { ov.arrowsVisible = ev.value as boolean; });
  folder.addBinding(state, 'declutter', { label: 'hide life & clouds' }).on('change', (ev) => { ov.declutter = ev.value as boolean; });
  // keys change the overlay too; select() ignores the echo (same value)
  const off = ov.onChange(() => {
    if (state.overlay !== ov.current) { state.overlay = ov.current; sel.refresh(); }
    if (state.tectonics !== ov.tectonics) { state.tectonics = ov.tectonics; tec.refresh(); }
  });
  return { dispose: () => { off(); } };
}
