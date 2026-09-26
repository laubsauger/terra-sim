// Procedural ambient audio (§T.46): nature (breaking waves, rain while it rains, birdsong) and event sounds
// (lava pops, eruption blasts, meteor impacts). No continuous drones: the wind / surf / rumble noise beds and the
// pad read as a constant hum and were removed (user). No assets.
// Reads sim-derived levels only (V15). Starts on the first user gesture, muted by default; mute persists (V14).
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
  private layers: Record<'rain', { gain: GainNode; filter: BiquadFilterNode }> = {} as never;
  // muted until the listener turns sound on (M / sound button); the choice persists (V14)
  private muted = safeGet(MUTE_KEY) !== '0';
  private t = 0;
  private nextBird = 4;
  private nextPop = 1;
  private nextWave = 3;
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
    // the only bed: rain, audible only while it rains
    this.layers.rain = loop('pink', 'bandpass', 1400, 0.5);
  }

  set(levels: Partial<AmbienceLevels>): void { Object.assign(this.levels, levels); }

  update(dt: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    this.t += dt;
    const L = this.levels, now = ctx.currentTime;
    this.layers.rain.gain.gain.setTargetAtTime(0.04 * L.rain, now, 1.5);
    if (L.ocean > 0.2 && this.t > this.nextWave) { this.wave(0.5 + 0.5 * L.closeness); this.nextWave = this.t + 5 + Math.random() * 6; }
    if (L.volcanic > 0.2 && this.t > this.nextPop) { this.pop(); this.nextPop = this.t + 0.8 + Math.random() * 4 / (0.2 + L.volcanic); }
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

  /** Meteor impact: a deep falling boom, a noise blast and a rolling tail (mag ~5-9). */
  impact(mag: number): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const k = Math.min(1, Math.max(0.2, (mag - 4) / 4)), t = ctx.currentTime;
    const o = ctx.createOscillator(), og = ctx.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(90, t); o.frequency.exponentialRampToValueAtTime(28, t + 1.6);
    og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.5 * k, t + 0.03); og.gain.exponentialRampToValueAtTime(0.0001, t + 2.4);
    o.connect(og).connect(this.master); o.start(t); o.stop(t + 2.5);
    this.noiseBurst(t, 0.35 * k, 1800, 220, 2.8);
  }

  /** Eruption blast (explosive opening): a shorter boom and crack. */
  blast(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime, o = ctx.createOscillator(), og = ctx.createGain();
    o.frequency.setValueAtTime(70, t); o.frequency.exponentialRampToValueAtTime(32, t + 0.9);
    og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.28, t + 0.02); og.gain.exponentialRampToValueAtTime(0.0001, t + 1.3);
    o.connect(og).connect(this.master); o.start(t); o.stop(t + 1.4);
    this.noiseBurst(t, 0.18, 1400, 300, 1.6);
  }

  /** Band-limited noise burst whose low-pass sweeps down (debris / rolling tail). */
  private noiseBurst(t: number, peak: number, f0: number, f1: number, len: number): void {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource(); src.buffer = noiseBuffer(ctx, 'brown', Math.ceil(len + 0.2));
    const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + len);
    const g = ctx.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    src.connect(f).connect(g).connect(this.master); src.start(t); src.stop(t + len + 0.1);
  }

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
