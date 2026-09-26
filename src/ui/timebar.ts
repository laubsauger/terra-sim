// Always-visible time control bar (§I.time ctl, V22). Plain DOM, CSS injected once.

export interface ClockLike {
  geoTime: number;
  paused: boolean;
  requestedSpeed: number;
  readonly effectiveSpeed: number;
  setSpeed(v: number): void;
  togglePause(): void;
}

export interface Timebar {
  el: HTMLElement;
  /** Call each frame; only touches the DOM when displayed values change. */
  update(): void;
  setVisible(v: boolean): void;
  dispose(): void;
}

const STYLE_ID = 'terra-timebar-style';
const SLIDER_RES = 1000;

const CSS = `
.tb {
  position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%);
  display: flex; align-items: center; gap: 14px;
  padding: 8px 16px 8px 8px; border-radius: 999px;
  background: rgba(14, 16, 22, 0.55);
  backdrop-filter: blur(14px) saturate(1.4); -webkit-backdrop-filter: blur(14px) saturate(1.4);
  border: 1px solid rgba(255, 255, 255, 0.09);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  color: rgba(236, 240, 247, 0.88);
  font: 500 11px/1 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-variant-caps: all-small-caps; letter-spacing: 0.08em;
  font-variant-numeric: tabular-nums;
  user-select: none; z-index: 20; white-space: nowrap;
  max-width: calc(100vw - 32px); box-sizing: border-box;
}
.tb[hidden] { display: none; }
.tb-btn {
  width: 30px; height: 30px; flex: none; border-radius: 50%; border: 0; padding: 0;
  display: grid; place-items: center; cursor: pointer;
  background: rgba(255, 255, 255, 0.1); color: inherit;
  transition: background 120ms ease;
}
.tb-btn:hover { background: rgba(255, 255, 255, 0.18); }
.tb-btn svg { width: 12px; height: 12px; fill: currentColor; }
.tb-time { min-width: 96px; font-size: 13px; letter-spacing: 0.04em; color: #fff; display: flex; flex-direction: column; gap: 2px; }
.tb-time small { font-size: 10px; letter-spacing: 0.08em; color: rgba(236, 240, 247, 0.6); }
.tb-range {
  -webkit-appearance: none; appearance: none; width: 150px; height: 3px; margin: 0;
  border-radius: 2px; background: rgba(255, 255, 255, 0.18); outline: none; cursor: pointer;
}
.tb-range::-webkit-slider-thumb {
  -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%;
  background: #e9eef7; box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12);
}
.tb-range::-moz-range-thumb {
  width: 12px; height: 12px; border: 0; border-radius: 50%; background: #e9eef7;
}
.tb-speed { display: flex; flex-direction: column; gap: 3px; min-width: 92px; }
.tb-speed .tb-eff { opacity: 0.6; transition: color 150ms ease, opacity 150ms ease; }
.tb-speed.capped .tb-eff { color: #ffb454; opacity: 1; }
.tb.paused .tb-time { opacity: 0.55; }
`;

const ICON_PLAY = '<svg viewBox="0 0 12 12"><path d="M2.5 1.2v9.6L10.5 6z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 12 12"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';

/** Geologic time in plain words: { value, unit } e.g. 550 / "thousand years", 12.3 / "million years". */
export function geoTimeParts(my: number): { value: string; unit: string } {
  const a = Math.abs(my);
  if (a < 1) return { value: String(Math.round(my * 1000)), unit: 'thousand years' };
  if (a < 1000) return { value: my.toFixed(a < 10 ? 2 : a < 100 ? 1 : 0), unit: 'million years' };
  return { value: (my / 1000).toFixed(2), unit: 'billion years' };
}

/** "550 thousand years" / "12.3 million years" / "1.23 billion years". */
export function formatGeoTime(my: number): string {
  const p = geoTimeParts(my);
  return `${p.value} ${p.unit}`;
}

/** Speed as years of geology per real second: "300k yrs/s", "2.7M yrs/s". */
export function formatSpeed(v: number): string {
  if (v < 1) return `${Number((v * 1000).toPrecision(3))}k yrs/s`;
  return `${Number(v.toPrecision(3))}M yrs/s`;
}

export function createTimebar(clock: ClockLike, opts: { minSpeed: number; maxSpeed: number }): Timebar {
  if (!(opts.minSpeed > 0 && opts.maxSpeed > opts.minSpeed)) {
    throw new Error(`timebar: invalid speed range ${opts.minSpeed}..${opts.maxSpeed}`);
  }
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  const lmin = Math.log10(opts.minSpeed);
  const lmax = Math.log10(opts.maxSpeed);
  const toSlider = (v: number) => Math.round(((Math.log10(v) - lmin) / (lmax - lmin)) * SLIDER_RES);
  const fromSlider = (s: number) => 10 ** (lmin + (s / SLIDER_RES) * (lmax - lmin));

  const el = document.createElement('div');
  el.className = 'tb';

  const btn = document.createElement('button');
  btn.className = 'tb-btn';
  btn.type = 'button';
  // keep focus off the button so Space goes to the global key handler, not a button click (double toggle)
  btn.addEventListener('pointerdown', (e) => e.preventDefault());
  btn.addEventListener('click', () => {
    clock.togglePause();
    update();
  });

  const time = document.createElement('span');
  time.className = 'tb-time';
  time.title = 'Geologic time simulated so far (1 My = 1 million years)';

  const range = document.createElement('input');
  range.type = 'range';
  range.className = 'tb-range';
  range.min = '0';
  range.max = String(SLIDER_RES);
  range.step = '1';
  range.title = 'Speed: years of geology per real second';
  let dragging = false;
  range.addEventListener('pointerdown', () => (dragging = true));
  range.addEventListener('pointerup', () => (dragging = false));
  range.addEventListener('pointercancel', () => (dragging = false));
  range.addEventListener('input', () => {
    clock.setSpeed(fromSlider(Number(range.value)));
    update();
  });
  range.addEventListener('change', () => {
    dragging = false;
    range.blur(); // release focus so keys (Space, [, ]) work again
  });

  const speed = document.createElement('div');
  speed.className = 'tb-speed';
  const req = document.createElement('span');
  const eff = document.createElement('span');
  eff.className = 'tb-eff';
  speed.append(req, eff);

  el.append(btn, time, range, speed);
  document.body.appendChild(el);

  let lastPaused: boolean | undefined;
  let lastTime = '';
  let lastReq = '';
  let lastEff = '';
  let lastCapped: boolean | undefined;
  let lastSlider = -1;

  function update(): void {
    const paused = clock.paused;
    if (paused !== lastPaused) {
      lastPaused = paused;
      btn.innerHTML = paused ? ICON_PLAY : ICON_PAUSE;
      btn.title = paused ? 'play (Space)' : 'pause (Space)';
      el.classList.toggle('paused', paused);
    }
    const t = formatGeoTime(clock.geoTime);
    if (t !== lastTime) {
      lastTime = t;
      const p = geoTimeParts(clock.geoTime);
      time.innerHTML = `<span>${p.value}</span><small>${p.unit} since start</small>`;
    }

    const r = clock.requestedSpeed;
    const rs = `${formatSpeed(r)}`;
    if (rs !== lastReq) req.textContent = lastReq = rs;
    const e = clock.effectiveSpeed;
    const es = paused ? 'paused' : `now ${formatSpeed(e)}`;
    if (es !== lastEff) eff.textContent = lastEff = es;
    const capped = !paused && e < 0.95 * r;
    if (capped !== lastCapped) {
      lastCapped = capped;
      speed.classList.toggle('capped', capped);
      speed.title = capped ? 'the GPU can not keep up: running slower than requested' : 'years of geology per real second';
    }

    if (!dragging) {
      const s = Math.min(SLIDER_RES, Math.max(0, toSlider(r)));
      if (s !== lastSlider) range.value = String((lastSlider = s));
    }
  }

  update();

  return {
    el,
    update,
    setVisible(v: boolean) {
      el.hidden = !v;
    },
    dispose() {
      el.remove();
      if (!document.querySelector('.tb')) document.getElementById(STYLE_ID)?.remove();
    },
  };
}
