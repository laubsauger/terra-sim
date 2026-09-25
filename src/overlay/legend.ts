// Overlay legend: bottom-left glass card with title, one-line explanation, colour bar with ticks + units or
// named swatches, and the hover readout. Built from the same ramp/scale specs as the GPU colouring.
import './overlay.css';
import { rampCss, scaleT } from './colormaps';
import type { OverlayDef, RampLegend } from './defs';

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

function barHtml(b: RampLegend): string {
  const ticks = b.ticks.map((v, i) => {
    const cls = i === 0 ? 'first' : i === b.ticks.length - 1 ? 'last' : '';
    return `<span class="${cls}" style="left:${(scaleT(b.scale, v) * 100).toFixed(2)}%">${esc((b.fmt ?? String)(v))}</span>`;
  }).join('');
  return `${b.caption ? `<div class="tl-cap">${esc(b.caption)}</div>` : ''}<div class="tl-bar" style="background:${rampCss(b.ramp)}"></div>`
    + `<div class="tl-scale">${ticks}</div><div class="tl-unit">${esc(b.unit)}</div>`;
}

const ARROW_ICON = '<svg width="18" height="10" viewBox="0 0 18 10"><path d="M1 3.2h10V0.6L17 5l-6 4.4V6.8H1z" fill="#f1e9d6" stroke="#14161d" stroke-width="1"/></svg>';

export function createLegend() {
  const el = document.createElement('div');
  el.className = 'terra-legend';
  el.setAttribute('role', 'status');
  document.body.appendChild(el);
  let body: HTMLDivElement | null = null;
  let readEl: HTMLDivElement | null = null;
  let noteEl: HTMLDivElement | null = null;
  let shown = false;

  return {
    el,
    /** Rebuild for an overlay (animated content swap). */
    setOverlay(def: OverlayDef) {
      const sw = def.swatches ? `<div class="tl-sw${def.swatches.length > 6 ? ' cols' : ''}">${def.swatches.map((s) =>
        `<div><i class="${s.line ? 'line' : ''}" style="background:${s.color}"></i><span title="${esc(s.name)}">${esc(s.name)}</span></div>`).join('')}</div>` : '';
      const html = `<div class="tl-head"><span class="tl-key">${def.key}</span><span class="tl-title">${esc(def.title)}</span></div>`
        + `<div class="tl-blurb">${esc(def.blurb)}</div>`
        + (def.bar ? barHtml(def.bar) : '') + sw + (def.bar2 ? barHtml(def.bar2) : '')
        + (def.id === 'plates' ? `<div class="tl-note">${ARROW_ICON}<span class="tl-arrows">Arrows: plate motion, longer = faster.</span></div>` : '')
        + `<div class="tl-read"><span class="muted">Hover the map for values</span></div>`;
      body = document.createElement('div');
      body.className = 'tl-body';
      body.innerHTML = html;
      el.replaceChildren(body);
      readEl = body.querySelector('.tl-read');
      noteEl = body.querySelector('.tl-arrows');
    },
    /** Hover value; null → hint. */
    setReadout(text: string | null, loc?: string) {
      if (!readEl) return;
      readEl.innerHTML = text === null ? '<span class="muted">Hover the map for values</span>'
        : `<b>${esc(text)}</b>${loc ? `<span class="loc">${esc(loc)}</span>` : ''}`;
    },
    /** Extra line for the plates legend (arrow summary). */
    setNote(text: string) { if (noteEl && noteEl.textContent !== text) noteEl.textContent = text; },
    setVisible(v: boolean) {
      if (v === shown) return;
      shown = v;
      el.classList.toggle('show', v);
    },
    get visible() { return shown; },
    get readoutText() { return readEl?.textContent ?? ''; },
  };
}
export type Legend = ReturnType<typeof createLegend>;
