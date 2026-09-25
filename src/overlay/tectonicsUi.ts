// 'Plates' pill next to the time bar: game UI toggle for the Tectonics layer (key T). Matches the time bar glass.
import './overlay.css';

const ICON = `<svg viewBox="0 0 20 14" width="20" height="14" aria-hidden="true">
  <path d="M1 9.5 C5 8 7 11 10 8.5 S15 5.5 19 7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M3.5 4 h5 M6.8 2.3 L8.8 4 L6.8 5.7" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M16.5 11.5 h-5 M13.2 9.8 L11.2 11.5 L13.2 13.2" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" opacity=".7"/>
</svg>`;

export function createTectonicsPill(onToggle: () => void) {
  const el = document.createElement('button');
  el.className = 'terra-tec-pill';
  el.type = 'button';
  el.title = 'Plate boundaries and motion (T)';
  el.setAttribute('aria-pressed', 'false');
  el.innerHTML = `${ICON}<span>Plates</span><kbd>T</kbd>`;
  el.addEventListener('click', (e) => { e.stopPropagation(); onToggle(); el.blur(); });
  document.body.appendChild(el);
  let visible = true;
  // sit just right of the time bar (its width changes with the speed text): re-measure cheaply
  const place = () => {
    const tb = document.querySelector('.tb') as HTMLElement | null;
    const r = tb && !tb.hidden ? tb.getBoundingClientRect() : null;
    if (r && r.width > 0) {
      el.style.left = `${Math.round(r.right + 10)}px`;
      el.style.bottom = `${Math.round(window.innerHeight - r.bottom)}px`;
      el.style.height = `${Math.round(r.height)}px`;
    } else { el.style.left = 'calc(50% + 250px)'; }
  };
  place();
  const timer = setInterval(place, 500);
  window.addEventListener('resize', place);
  return {
    el,
    setActive(on: boolean) { el.classList.toggle('on', on); el.setAttribute('aria-pressed', String(on)); },
    setVisible(v: boolean) { if (v !== visible) { visible = v; el.style.display = v ? '' : 'none'; if (v) place(); } },
    dispose() { clearInterval(timer); window.removeEventListener('resize', place); el.remove(); },
  };
}
