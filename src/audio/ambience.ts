// Procedural ambient audio (§T.46): wind, surf, rain, deep rumble, lava pops, birdsong. No assets.
// Reads sim-derived levels only (V15). Starts on the first user gesture; mute persists (V14).
const MUTE_KEY = 'terra-sim.muted';

export interface AmbienceLevels {
  wind: number;     // 0..1
  ocean: number;    // 0..1 share of ocean near the view
  rain: number;     // 0..1
  volcanic: number; // 0..1 recent eruption activity
  life: number;     // 0..1 vegetation / biodiversity
  closeness: number; // 0..1, camera near the surface
}

function noiseBuffer(ctx: AudioContext, kind: 'white' | 'pink' | 'brown', seconds = 4): AudioBuffer {
  const n = ctx.sampleRate * seconds;
  const buf = ctx.createBuffer(1, n, ctx.sampleRate);
  const d = buf.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, last = 0;
  let s = 0x12345678;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) / 4294967296) * 2 - 1; };
  for (let i = 0; i < n; i++) {
    const w = rnd();
    if (kind === 'white') d[i] = w;
    else if (kind === 'pink') { b0 = 0.997 * b0 + w * 0.029; b1 = 0.985 * b1 + w * 0.032; b2 = 0.95 * b2 + w * 0.048; d[i] = (b0 + b1 + b2) * 3.5; }
    else { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; }
  }
  return buf;
}

function safeGet(k: string): string | null { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k: string, v: string): void { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

export class Ambience {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private layers: Record<'wind' | 'surf' | 'rain' | 'rumble', { gain: GainNode; filter: BiquadFilterNode }> = {} as never;
  private muted = safeGet(MUTE_KEY) === '1';
  private t = 0;
  private nextBird = 4;
  private nextPop = 1;
  private nextWave = 3;
  private padGain: GainNode | null = null;
  private levels: AmbienceLevels = { wind: 0.4, ocean: 0.5, rain: 0, volcanic: 0, life: 0.3, closeness: 0.3 };

  constructor() {
    // V14: browsers require a gesture; start lazily on the first one
    const start = () => { this.start(); window.removeEventListener('pointerdown', start); window.removeEventListener('keydown', start); };
    window.addEventListener('pointerdown', start);
    window.addEventListener('keydown', start);
  }

  get started(): boolean { return this.ctx !== null; }
  get isMuted(): boolean { return this.muted; }

  toggleMute(): void {
    this.muted = !this.muted;
    safeSet(MUTE_KEY, this.muted ? '1' : '0');
    if (this.ctx) this.master.gain.setTargetAtTime(this.muted ? 0 : 0.8, this.ctx.currentTime, 0.3);
  }

  private start(): void {
    if (this.ctx) return;
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 0.8;
    const comp = ctx.createDynamicsCompressor();
    this.master.connect(comp).connect(ctx.destination);
    const loop = (kind: 'white' | 'pink' | 'brown', type: BiquadFilterType, freq: number, q = 0.7) => {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuffer(ctx, kind);
      src.loop = true;
      const filter = ctx.createBiquadFilter();
      filter.type = type; filter.frequency.value = freq; filter.Q.value = q;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      src.connect(filter).connect(gain).connect(this.master);
      src.start(ctx.currentTime + Math.random() * 0.1);
      return { gain, filter };
    };
    // beds are brown/pink noise, heavily low-passed: the earlier white/highpass layers read as hiss
    this.layers.wind = loop('brown', 'bandpass', 320, 0.8);
    this.layers.surf = loop('brown', 'lowpass', 260, 0.6);
    this.layers.rain = loop('pink', 'bandpass', 1400, 0.5);
    this.layers.rumble = loop('brown', 'lowpass', 55, 0.9);
    this.startPad();
  }

  set(levels: Partial<AmbienceLevels>): void { Object.assign(this.levels, levels); }

  update(dt: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    this.t += dt;
    const L = this.levels, now = ctx.currentTime, tc = 0.8;
    const gust = 0.6 + 0.4 * Math.sin(this.t * 0.23) * Math.sin(this.t * 0.071 + 1);
    this.layers.wind.gain.gain.setTargetAtTime(0.1 * L.wind * gust * (0.6 + 0.4 * (1 - L.closeness)), now, tc);
    this.layers.wind.filter.frequency.setTargetAtTime(220 + 260 * gust, now, tc);
    // surf: a slow swell bed plus separate wave-break swooshes (see wave())
    const swell = 0.5 + 0.5 * Math.sin(this.t * 0.45) ** 2;
    this.layers.surf.gain.gain.setTargetAtTime(0.12 * L.ocean * swell * (0.4 + 0.6 * L.closeness), now, 0.6);
    this.layers.rain.gain.gain.setTargetAtTime(0.06 * L.rain, now, 1.5);
    if (L.ocean > 0.2 && this.t > this.nextWave) { this.wave(0.5 + 0.5 * L.closeness); this.nextWave = this.t + 5 + Math.random() * 6; }
    this.padGain?.gain.setTargetAtTime(0.05, now, 3);
    this.layers.rumble.gain.gain.setTargetAtTime(0.5 * L.volcanic, now, 1.2);
    if (L.volcanic > 0.05 && this.t > this.nextPop) { this.pop(); this.nextPop = this.t + 0.3 + Math.random() * 2.5 / (0.2 + L.volcanic); }
    if (L.life > 0.2 && this.t > this.nextBird) { this.chirp(); this.nextBird = this.t + 3 + Math.random() * 14 / L.life; }
  }

  /** A single wave breaking: band-limited noise swell up and down over ~3 s. */
  private wave(strength: number): void {
    const ctx = this.ctx!, t = ctx.currentTime;
    const src = ctx.createBufferSource(); src.buffer = noiseBuffer(ctx, 'pink', 4);
    const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(300, t); f.frequency.linearRampToValueAtTime(1100, t + 1.1); f.frequency.linearRampToValueAtTime(250, t + 3.2);
    const g = ctx.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.09 * strength, t + 1.0); g.gain.exponentialRampToValueAtTime(0.0001, t + 3.4);
    const pan = ctx.createStereoPanner(); pan.pan.value = Math.random() * 1.2 - 0.6;
    src.connect(f).connect(g).connect(pan).connect(this.master); src.start(t); src.stop(t + 3.6);
  }

  /** Quiet evolving pad (open fifths, slow filter drift) that glues the soundscape into something musical. */
  private startPad(): void {
    const ctx = this.ctx!;
    const g = ctx.createGain(); g.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 700; lp.Q.value = 0.3;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.03;
    const lfoGain = ctx.createGain(); lfoGain.gain.value = 350; lfo.connect(lfoGain).connect(lp.frequency); lfo.start();
    for (const [f, det] of [[110, -6], [164.8, 4], [220, 7], [329.6, -3]] as const) {
      const o = ctx.createOscillator(); o.type = 'triangle'; o.frequency.value = f; o.detune.value = det;
      const og = ctx.createGain(); og.gain.value = 0.25;
      o.connect(og).connect(lp); o.start();
    }
    lp.connect(g).connect(this.master);
    this.padGain = g;
  }

  /** Lava bubble: short low sine drop. */
  private pop(): void {
    const ctx = this.ctx!, o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime;
    o.frequency.setValueAtTime(90 + Math.random() * 60, t);
    o.frequency.exponentialRampToValueAtTime(35, t + 0.25);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.18 * this.levels.volcanic, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
    o.connect(g).connect(this.master); o.start(t); o.stop(t + 0.35);
  }

  /** Tiny FM birdsong phrase. */
  private chirp(): void {
    const ctx = this.ctx!, t0 = ctx.currentTime, notes = 2 + Math.floor(Math.random() * 4), base = 2200 + Math.random() * 1800;
    const pan = ctx.createStereoPanner(); pan.pan.value = Math.random() * 1.6 - 0.8; pan.connect(this.master);
    for (let i = 0; i < notes; i++) {
      const t = t0 + i * (0.09 + Math.random() * 0.06), o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.setValueAtTime(base * (1 + Math.random() * 0.3), t);
      o.frequency.exponentialRampToValueAtTime(base * (0.7 + Math.random() * 0.6), t + 0.07);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.04 * this.levels.life, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
      o.connect(g).connect(pan); o.start(t); o.stop(t + 0.1);
    }
  }
}
