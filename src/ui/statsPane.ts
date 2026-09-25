// Stats pane (§T.34): live read-only monitors of world balance, fed by the deterministic window snapshot.
import type { FolderApi } from 'tweakpane';
import type { Sim } from '../sim/sim';
import { NCOL } from '../sim/layout';

export function createStatsPane(folder: FolderApi, sim: Sim) {
  const s = {
    phase: '', cycles: 0, plates: 0, seaLevel: 0, landPct: 0, landGraph: 0,
    crustRes: 0, water: 0, vaporIce: 0, events: '', unhandled: 0,
  };
  const ro = { readonly: true } as const;
  folder.addBinding(s, 'phase', { ...ro, label: 'wilson' });
  folder.addBinding(s, 'cycles', { ...ro, format: (v: number) => v.toFixed(0) });
  folder.addBinding(s, 'plates', { ...ro, format: (v: number) => v.toFixed(0) });
  folder.addBinding(s, 'seaLevel', { ...ro, label: 'sea level', format: (v: number) => v.toFixed(1) });
  folder.addBinding(s, 'landPct', { ...ro, label: 'land %', format: (v: number) => v.toFixed(1) });
  folder.addBinding(s, 'landGraph', { ...ro, label: 'land', view: 'graph', min: 0, max: 60 });
  folder.addBinding(s, 'crustRes', { ...ro, label: 'mantle res.', format: (v: number) => `${v.toFixed(2)} L/col` });
  folder.addBinding(s, 'water', { ...ro, format: (v: number) => v.toFixed(0) });
  folder.addBinding(s, 'vaporIce', { ...ro, label: 'vapor+ice', format: (v: number) => v.toFixed(0) });
  folder.addBinding(s, 'events', { ...ro, label: 'last event' });
  folder.addBinding(s, 'unhandled', { ...ro, label: 'unhandled ev', format: (v: number) => v.toFixed(0) });
  let last = -1;
  return {
    update() {
      if (sim.tick === last) return;
      last = sim.tick;
      s.phase = sim.wilson.state.phase;
      s.cycles = sim.wilson.state.cycles;
      s.plates = sim.alivePlates();
      const st = sim.stats;
      if (st) { s.seaLevel = st.seaLevel; s.landPct = st.landFrac * 100; s.landGraph = s.landPct; s.water = st.water; s.vaporIce = st.vaporIce; }
      s.crustRes = sim.reservoir() / 255 / NCOL;
      const e = sim.events.log[sim.events.log.length - 1];
      s.events = e ? `${e.kind} @ ${e.my.toFixed(0)} My` : '—';
      s.unhandled = sim.unhandledEvents;
    },
  };
}
