// First-visit guide (game UI): what the diorama is doing and the quick controls. Shows once, reopens from the '?'
// corner button. Plus the corner cluster itself: sound toggle (audio starts muted) and help. Matches the time bar glass.

const SEEN_KEY = 'terra-sim.guideSeen';
const STYLE_ID = 'terra-guide-style';

const CSS = `
.tg-back { position: fixed; left: 18px; bottom: 76px; z-index: 40; pointer-events: none;
  opacity: 0; transform: translateY(8px); transition: opacity 220ms ease, transform 220ms ease; }
.tg-back.show { opacity: 1; transform: none; }
.tg-card { pointer-events: auto; width: min(300px, calc(100vw - 36px)); box-sizing: border-box;
  padding: 16px 18px 14px; border-radius: 16px; color: rgba(236, 240, 247, 0.92);
  background: rgba(16, 19, 27, 0.72); backdrop-filter: blur(16px) saturate(1.4); -webkit-backdrop-filter: blur(16px) saturate(1.4);
  border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 14px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
  font: 400 12.5px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.tg-card h2 { margin: 0 0 6px; font-size: 15px; font-weight: 650; color: #fff; display: flex; align-items: center; gap: 10px; }
.tg-card h3 { margin: 12px 0 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #ffcf7a; }
.tg-card p { margin: 0; color: rgba(236, 240, 247, 0.78); }
.tg-card ul { margin: 0; padding: 0; list-style: none; display: grid; gap: 5px; }
.tg-card li { display: grid; grid-template-columns: 78px 1fr; gap: 8px; align-items: baseline; }
.tg-card li > span:first-child { white-space: nowrap; }
.tg-card kbd { display: inline-block; min-width: 18px; padding: 1px 6px; border-radius: 6px; text-align: center;
  font: 600 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: #fff;
  background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.14); border-bottom-width: 2px; }
.tg-card .tg-k { color: rgba(236, 240, 247, 0.6); font-size: 12px; }
.tg-foot { margin-top: 12px; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.tg-foot small { color: rgba(236, 240, 247, 0.5); }
.tg-btn { border: 0; border-radius: 999px; padding: 8px 18px; cursor: pointer; font: 600 13px/1 ui-sans-serif, system-ui, sans-serif;
  color: #1a1206; background: linear-gradient(180deg, #ffd88e, #f0a64a); box-shadow: 0 4px 14px rgba(240, 166, 74, 0.35); }
.tg-btn:hover { filter: brightness(1.06); }
.tg-dock { position: fixed; right: 18px; bottom: 18px; z-index: 21; display: flex; gap: 8px; }
.tg-round { width: 38px; height: 38px; border-radius: 50%; border: 1px solid rgba(255, 255, 255, 0.09); cursor: pointer;
  display: grid; place-items: center; color: rgba(236, 240, 247, 0.88); background: rgba(14, 16, 22, 0.55);
  backdrop-filter: blur(14px) saturate(1.4); -webkit-backdrop-filter: blur(14px) saturate(1.4);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06); font: 700 15px/1 ui-sans-serif, system-ui, sans-serif; }
.tg-round:hover { background: rgba(40, 44, 56, 0.7); }
.tg-round.on { color: #ffcf7a; }
.tg-round svg { width: 16px; height: 16px; }
`;

const PLANET = `<svg viewBox="0 0 64 64" width="24" height="24" aria-hidden="true"><path d="M6 22 L32 36 L32 58 L6 44 Z" fill="#6b4a36"/>
<path d="M58 22 L32 36 L32 58 L58 44 Z" fill="#553827"/><path d="M6 40 L32 54 L32 58 L6 44 Z M58 40 L32 54 L32 58 L58 44 Z" fill="#f0763a"/>
<path d="M6 22 L32 8 L58 22 L32 36 Z" fill="#3fb8d4"/><path d="M22 21 C24 16 33 13 39 17 C44 20 42 25 36 26 C30 27 23 26 22 21 Z" fill="#4f9a3c"/></svg>`;
const SPEAKER_ON = '<svg viewBox="0 0 16 16"><path d="M2 6h3l4-3v10L5 10H2z" fill="currentColor"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.6a6 6 0 0 1 0 8.8" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
const SPEAKER_OFF = '<svg viewBox="0 0 16 16"><path d="M2 6h3l4-3v10L5 10H2z" fill="currentColor"/><path d="M11 6l4 4M15 6l-4 4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';

const BODY = `
<h2>${PLANET}<span>A little planet, fast-forwarded</span></h2>
<p>Plates drift, mountains rise and erode, volcanoes build islands. It runs on its own.</p>
<h3>Quick keys</h3>
<ul>
  <li><span><kbd>Space</kbd></span><span>pause</span></li>
  <li><span><kbd>[</kbd> <kbd>]</kbd></span><span>slower / faster</span></li>
  <li><span><kbd>T</kbd></span><span>plate lines</span></li>
  <li><span><kbd>1</kbd>–<kbd>9</kbd></span><span>data maps</span></li>
  <li><span><kbd>H</kbd></span><span>hide UI</span></li>
  <li><span><kbd>M</kbd></span><span>sound</span></li>
</ul>
<p style="margin-top:10px">Drag to orbit · brass knobs cut the slice · sun slider sets the time of day.</p>
<div class="tg-foot"><small>? reopens this</small><button type="button" class="tg-btn">Got it</button></div>
`;

function safeGet(k: string): string | null { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k: string, v: string): void { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

export interface Guide { open(): void; close(): void; setVisible(v: boolean): void; updateSound(): void; readonly isOpen: boolean }

export function createGuide(sound: { readonly isMuted: boolean; toggleMute(): void }, opts: { autoOpen?: boolean } = {}): Guide {
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style'); st.id = STYLE_ID; st.textContent = CSS; document.head.appendChild(st);
  }
  let back: HTMLElement | null = null;
  const close = () => {
    if (!back) return;
    const b = back; back = null;
    b.classList.remove('show');
    setTimeout(() => b.remove(), 230);
    safeSet(SEEN_KEY, '1');
  };
  const onKey = (e: KeyboardEvent) => { if (back && (e.key === 'Escape' || e.key === 'Enter')) { e.preventDefault(); close(); } };
  window.addEventListener('keydown', onKey);
  const open = () => {
    if (back) return;
    back = document.createElement('div');
    back.className = 'tg-back';
    back.innerHTML = `<div class="tg-card" role="dialog" aria-label="Welcome guide">${BODY}</div>`;
    back.querySelector('.tg-btn')!.addEventListener('click', close);
    document.body.appendChild(back);
    requestAnimationFrame(() => back?.classList.add('show'));
  };

  const dock = document.createElement('div');
  dock.className = 'tg-dock';
  const snd = document.createElement('button'); snd.type = 'button'; snd.className = 'tg-round';
  const help = document.createElement('button'); help.type = 'button'; help.className = 'tg-round'; help.textContent = '?'; help.title = 'How this works';
  for (const b of [snd, help]) b.addEventListener('pointerdown', (e) => e.preventDefault()); // keep Space for pause
  snd.addEventListener('click', () => { sound.toggleMute(); updateSound(); });
  help.addEventListener('click', open);
  dock.append(snd, help);
  document.body.appendChild(dock);
  const updateSound = () => {
    const on = !sound.isMuted;
    snd.innerHTML = on ? SPEAKER_ON : SPEAKER_OFF;
    snd.classList.toggle('on', on);
    snd.title = on ? 'Sound on (M)' : 'Sound off (M)';
  };
  updateSound();

  if ((opts.autoOpen ?? true) && safeGet(SEEN_KEY) !== '1') open();
  return {
    open, close, updateSound,
    get isOpen() { return back !== null; },
    setVisible(v) { dock.style.display = v ? '' : 'none'; },
  };
}
