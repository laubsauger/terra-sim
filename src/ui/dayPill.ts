// 'Day' pill left of the time bar (game UI): drag the sun through the day, or let it cycle.
// Matches the time bar glass. Time of day: 0..1, 0.5 = noon (sky.ts).

export interface DayControl {
  readonly timeOfDay: number;
  readonly dayLength: number;
  setTimeOfDay(tod: number): void;
  setDayLength(seconds: number): void;
}

/** Real seconds per day when the pill's cycle is on (ambient mode sets its own, slower day). */
export const GAME_DAY_S = 180;
const STYLE_ID = 'terra-day-style';
const RES = 1000;

const CSS = `
.terra-day {
  position: fixed; bottom: 18px; display: flex; align-items: center; gap: 10px;
  padding: 0 14px 0 6px; border-radius: 999px; box-sizing: border-box;
  background: rgba(14, 16, 22, 0.55);
  backdrop-filter: blur(14px) saturate(1.4); -webkit-backdrop-filter: blur(14px) saturate(1.4);
  border: 1px solid rgba(255, 255, 255, 0.09);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  color: rgba(236, 240, 247, 0.88);
  font: 500 11px/1 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-variant-caps: all-small-caps; letter-spacing: 0.08em; font-variant-numeric: tabular-nums;
  user-select: none; z-index: 20; white-space: nowrap;
}
.terra-day button {
  width: 30px; height: 30px; flex: none; border-radius: 50%; border: 0; padding: 0;
  display: grid; place-items: center; cursor: pointer; color: inherit;
  background: rgba(255, 255, 255, 0.1); transition: background 120ms ease, color 120ms ease;
}
.terra-day button:hover { background: rgba(255, 255, 255, 0.18); }
.terra-day.cycling button { color: #ffcf7a; background: rgba(255, 190, 90, 0.18); }
.terra-day input {
  -webkit-appearance: none; appearance: none; width: 110px; height: 3px; margin: 0; border-radius: 2px; outline: none; cursor: pointer;
  background: linear-gradient(90deg, #1b2440 0%, #2a3a6e 20%, #f0a060 27%, #9cc6f0 40%, #cfe2f6 50%, #9cc6f0 60%, #f0a060 73%, #2a3a6e 80%, #1b2440 100%);
}
.terra-day input::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #e9eef7; box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12); }
.terra-day input::-moz-range-thumb { width: 12px; height: 12px; border: 0; border-radius: 50%; background: #e9eef7; }
.terra-day .td-clock { min-width: 36px; font-size: 12px; letter-spacing: 0.04em; color: #fff; }
`;

const SUN = '<svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="2.8" fill="currentColor"/><g stroke="currentColor" stroke-width="1.3" stroke-linecap="round"><path d="M7 .8v1.6M7 11.6v1.6M.8 7h1.6M11.6 7h1.6M2.6 2.6l1.1 1.1M10.3 10.3l1.1 1.1M2.6 11.4l1.1-1.1M10.3 3.7l1.1-1.1"/></g></svg>';
const MOON = '<svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true"><path d="M9.8 10.6A5 5 0 0 1 5.2 2 5 5 0 1 0 12 8.9a5 5 0 0 1-2.2 1.7z" fill="currentColor"/></svg>';

/** "16:55" from a 0..1 time of day. */
export function formatClock(tod: number): string {
  const m = Math.floor((((tod % 1) + 1) % 1) * 24 * 60);
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

export function createDayPill(day: DayControl) {
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style'); st.id = STYLE_ID; st.textContent = CSS; document.head.appendChild(st);
  }
  const el = document.createElement('div');
  el.className = 'terra-day';
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.addEventListener('pointerdown', (e) => e.preventDefault()); // keep Space for the global pause key
  btn.addEventListener('click', (e) => { e.stopPropagation(); day.setDayLength(day.dayLength > 0 ? 0 : GAME_DAY_S); update(); });
  const range = document.createElement('input');
  range.type = 'range'; range.min = '0'; range.max = String(RES); range.step = '1'; range.title = 'time of day';
  let dragging = false;
  range.addEventListener('pointerdown', () => (dragging = true));
  range.addEventListener('pointerup', () => (dragging = false));
  range.addEventListener('input', () => { day.setTimeOfDay(Number(range.value) / RES); update(); });
  range.addEventListener('change', () => { dragging = false; range.blur(); });
  const clock = document.createElement('span');
  clock.className = 'td-clock';
  el.append(btn, range, clock);
  document.body.appendChild(el);

  let lastIcon = '', lastClock = '', lastCycling: boolean | undefined;
  function update(): void {
    const tod = day.timeOfDay;
    const night = tod < 0.25 || tod > 0.77;
    const icon = night ? MOON : SUN;
    if (icon !== lastIcon) btn.innerHTML = lastIcon = icon;
    const cycling = day.dayLength > 0;
    if (cycling !== lastCycling) {
      lastCycling = cycling;
      el.classList.toggle('cycling', cycling);
      btn.title = cycling ? 'stop the day cycle' : 'run the day cycle';
    }
    const c = formatClock(tod);
    if (c !== lastClock) clock.textContent = lastClock = c;
    if (!dragging) range.value = String(Math.round(tod * RES));
  }
  // sit just left of the time bar (its width changes with the speed text); on narrow windows there is no room
  // there, so stack above it instead of sliding off-screen
  const place = () => {
    const tb = document.querySelector('.tb') as HTMLElement | null;
    const r = tb && !tb.hidden ? tb.getBoundingClientRect() : null;
    if (r && r.width > 0) {
      const w = el.getBoundingClientRect().width;
      const left = r.left - 10 - w;
      const fits = left >= 12;
      el.style.left = `${Math.round(fits ? left : r.left)}px`;
      el.style.bottom = `${Math.round(window.innerHeight - r.bottom + (fits ? 0 : r.height + 8))}px`;
      el.style.height = `${Math.round(r.height)}px`;
    }
  };
  update(); place();
  const timer = setInterval(place, 500);
  window.addEventListener('resize', place);
  return {
    el,
    update,
    setVisible(v: boolean) { el.style.display = v ? '' : 'none'; if (v) place(); },
    dispose() { clearInterval(timer); window.removeEventListener('resize', place); el.remove(); },
  };
}
