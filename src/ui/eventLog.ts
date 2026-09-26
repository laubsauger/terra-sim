// Little event feed (game UI, bottom right): what just happened on the planet, in plain words. Polls the sim's
// event log, the plate lifecycle log and the Wilson phase each frame (cheap: length / value compares only).
import type { GeoEvent } from '../sim/events';
import type { LifecycleOp } from '../sim/lifecycle';
import type { WilsonPhase } from '../sim/wilson';
import { geoTimeParts } from './timebar';

export interface EventSource {
  readonly geoMy: number;
  readonly events: readonly GeoEvent[];
  readonly lifecycle: readonly { my: number; op: LifecycleOp }[];
  readonly phase: WilsonPhase;
}

const STYLE_ID = 'terra-events-style';
const UNIT_SHORT: Record<string, string> = { 'thousand years': 'k', 'million years': 'M', 'billion years': 'B' };
const MAX = 5;
const FADE_S = 25; // real seconds an entry stays fully visible
const CSS = `
.te-log { position: fixed; right: 18px; bottom: 68px; z-index: 19; display: flex; flex-direction: column; gap: 6px; align-items: flex-end;
  pointer-events: none; max-width: min(340px, calc(100vw - 36px)); }
.te-item { display: flex; gap: 8px; align-items: center; padding: 6px 12px 6px 8px; border-radius: 999px;
  background: rgba(14, 16, 22, 0.55); backdrop-filter: blur(12px) saturate(1.4); -webkit-backdrop-filter: blur(12px) saturate(1.4);
  border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  color: rgba(236, 240, 247, 0.9); font: 500 12px/1.3 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  transition: opacity 1.2s ease, transform 300ms ease; animation: te-in 320ms ease-out; }
.te-item i { font-style: normal; font-size: 15px; width: 18px; text-align: center; }
.te-item b { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.te-item small { color: rgba(236, 240, 247, 0.5); font-size: 10.5px; margin-left: 2px; white-space: nowrap; }
@keyframes te-in { from { opacity: 0; transform: translateX(12px); } to { opacity: 1; transform: none; } }
`;

const EVENT_TEXT: Record<GeoEvent['kind'], [string, string]> = {
  meteor: ['☄️', 'A meteor struck'],
  volcano: ['🌋', 'A volcano erupted'],
  floodBasalt: ['🌋', 'Flood basalts poured out'],
  hotspot: ['🔥', 'A mantle plume rose'],
  iceAge: ['❄️', 'An ice age began'],
  storm: ['⛈️', 'A great storm'],
  uplift: ['⛰️', 'The land was pushed up'],
  split: ['✂️', 'A plate was torn apart'],
};
const OP_TEXT: Partial<Record<LifecycleOp['kind'], [string, string]>> = {
  split: ['💥', 'A plate tore in two'],
  merge: ['🤝', 'Two plates welded together'],
  absorb: ['🫧', 'A small plate was swallowed'],
};
const PHASE_TEXT: Record<WilsonPhase, [string, string]> = {
  drift: ['🧭', 'Continents drift'],
  assemble: ['🧲', 'Continents are closing in'],
  super: ['🌍', 'A supercontinent formed'],
  rift: ['🌋', 'The supercontinent is rifting'],
  disperse: ['🌊', 'New oceans are opening'],
};

export function createEventLog(src: EventSource) {
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style'); st.id = STYLE_ID; st.textContent = CSS; document.head.appendChild(st);
  }
  const el = document.createElement('div');
  el.className = 'te-log';
  document.body.appendChild(el);
  // start after what already happened (a loaded save does not replay its history)
  let seenEv = src.events.length, seenOps = src.lifecycle.length, phase = src.phase;
  const items: { node: HTMLElement; born: number }[] = [];
  let now = 0;

  const push = ([icon, text]: [string, string], my: number, god = false) => {
    const t = geoTimeParts(my);
    const node = document.createElement('div');
    node.className = 'te-item';
    node.innerHTML = `<i>${icon}</i><b>${god ? text.replace(/^A |^An /, 'Your ') : text}</b><small>${t.value}${UNIT_SHORT[t.unit] ?? ''} yrs</small>`;
    el.prepend(node);
    items.unshift({ node, born: now });
    while (items.length > MAX) items.pop()!.node.remove();
  };

  return {
    el,
    update(dt: number) {
      now += dt;
      const ev = src.events;
      if (ev.length < seenEv) seenEv = 0; // log trimmed / world reset
      for (; seenEv < ev.length; seenEv++) { const e = ev[seenEv]!; push(EVENT_TEXT[e.kind], e.my, e.source === 'god'); }
      const ops = src.lifecycle;
      if (ops.length < seenOps) seenOps = 0;
      for (; seenOps < ops.length; seenOps++) { const o = ops[seenOps]!; const t = OP_TEXT[o.op.kind]; if (t) push(t, o.my); }
      if (src.phase !== phase) { phase = src.phase; push(PHASE_TEXT[phase], src.geoMy); }
      items.forEach((it, i) => { const age = now - it.born; it.node.style.opacity = String(age > FADE_S ? 0 : Math.max(0.35, 1 - i * 0.16)); });
      while (items.length && now - items[items.length - 1]!.born > FADE_S + 1.5) items.pop()!.node.remove();
    },
    setVisible(v: boolean) { el.style.display = v ? '' : 'none'; },
  };
}
