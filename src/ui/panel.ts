// Tweakpane shell built from the param schema (V20). Later tasks append to `folders`.
import { Pane, type FolderApi } from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';
import type { FpsGraphBladeApi } from '@tweakpane/plugin-essentials';
import type { ParamDef, ParamGroup, Params, ParamValue } from '../core/params';

export type PanelFolder = ParamGroup | 'Overlays' | 'God' | 'Stats' | 'Save';

export interface Panel {
  pane: Pane;
  folders: Record<PanelFolder, FolderApi>;
  /** FPS graph in Stats; call fps.begin()/fps.end() around each frame. */
  fps: FpsGraphBladeApi;
  setVisible(v: boolean): void;
  dispose(): void;
}

const PARAM_GROUPS: readonly ParamGroup[] = ['Sim', 'Tectonics', 'Climate', 'Events', 'Render'];
const EXTRA_FOLDERS = ['Overlays', 'God', 'Stats', 'Save'] as const;
/** Ranges wider than this get a plain number field instead of a slider (e.g. u32 seed). */
const MAX_SLIDER_SPAN = 1e6;

function fmtLog(x: number): string {
  return String(Number((10 ** x).toPrecision(3)));
}

export function createPanel(params: Params, container?: HTMLElement): Panel {
  const pane = new Pane({ container, title: 'terra-sim' });
  pane.registerPlugin(EssentialsPlugin);

  const folders = {} as Record<PanelFolder, FolderApi>;
  for (const g of PARAM_GROUPS) folders[g] = pane.addFolder({ title: g, expanded: g === 'Sim' });
  for (const f of EXTRA_FOLDERS) folders[f] = pane.addFolder({ title: f, expanded: f === 'Stats' });

  // Proxy object tweakpane binds to; log params hold log10(value).
  const state: Record<string, ParamValue> = {};
  const toUI = (d: ParamDef, v: ParamValue): ParamValue => (d.log && typeof v === 'number' ? Math.log10(v) : v);
  const byKey = new Map<string, ParamDef>();
  // pane.refresh() can re-emit 'change' for constrained values; ignore those echoes
  let syncing = false;

  for (const d of params.defs()) {
    byKey.set(d.key, d);
    state[d.key] = toUI(d, params.get(d.key));
    const label = d.unit ? `${d.label} (${d.unit})` : d.label;
    const folder = folders[d.group];
    let opts: Record<string, unknown>;
    if (typeof d.default === 'boolean') {
      opts = { label };
    } else if (d.log) {
      opts = { label, min: Math.log10(d.min!), max: Math.log10(d.max!), format: fmtLog };
    } else if (d.min !== undefined && d.max !== undefined && d.max - d.min <= MAX_SLIDER_SPAN) {
      opts = { label, min: d.min, max: d.max, step: d.step };
    } else {
      opts = { label, step: d.step };
    }
    folder.addBinding(state, d.key, opts).on('change', (ev) => {
      if (syncing) return;
      const v = ev.value as ParamValue;
      params.set(d.key, d.log && typeof v === 'number' ? 10 ** v : v);
    });
  }

  // External params.set (keys, url, load) -> UI. params.set is a no-op on unchanged values, so no feedback loop.
  const off = params.onChange((key, v) => {
    const d = byKey.get(key);
    if (!d) return;
    state[key] = toUI(d, v);
    syncing = true;
    try {
      pane.refresh();
    } finally {
      syncing = false;
    }
  });

  const fps = folders.Stats.addBlade({ view: 'fpsgraph', label: 'fps', rows: 2 }) as FpsGraphBladeApi;

  return {
    pane,
    folders,
    fps,
    setVisible(v: boolean) {
      pane.hidden = !v;
    },
    dispose() {
      off();
      pane.dispose();
    },
  };
}
