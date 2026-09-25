// Dual clock (V11, V12, V17, V22).
// geo clock: fixed dtGeo per sim tick; speed only changes ticks per frame, never dtGeo.
// ambience clock: real seconds, always advances (pause affects geo only).
// geoTime is derived from an integer tick count so it depends on tick count only,
// never on how ticks were chunked into frames (30fps vs 60fps give bit-identical geoTime).

export interface ClockState {
  geoTime: number;
  ambTime: number;
  paused: boolean;
  requestedSpeed: number;
}

export class DualClock {
  static readonly minSpeed = 0.001; // My per real second
  static readonly maxSpeed = 1000; // My per real second
  static readonly maxRealDt = 0.25; // s; clamp for tab switches / stalls
  static readonly speedSmoothingTau = 0.5; // s; EMA time constant for effectiveSpeed

  readonly dtGeo: number;
  ambTime = 0;
  paused = false;
  requestedSpeed: number;

  private readonly ambPeriod: number;
  private geoBase = 0; // only non-zero if a loaded geoTime was not a multiple of dtGeo
  private geoTicks = 0; // integer
  private acc = 0; // fractional ticks owed, always < maxTicks worth
  private effSpeed = 0;

  constructor(opts: { dtGeo: number; requestedSpeed?: number; ambPeriod?: number }) {
    if (!(Number.isFinite(opts.dtGeo) && opts.dtGeo > 0)) {
      throw new RangeError(`DualClock: dtGeo must be finite > 0, got ${opts.dtGeo}`);
    }
    const period = opts.ambPeriod ?? 3600;
    if (!(Number.isFinite(period) && period > 0)) {
      throw new RangeError(`DualClock: ambPeriod must be finite > 0, got ${period}`);
    }
    this.dtGeo = opts.dtGeo;
    this.ambPeriod = period;
    this.requestedSpeed = DualClock.clampSpeed(opts.requestedSpeed ?? 1);
  }

  get geoTime(): number {
    return this.geoBase + this.geoTicks * this.dtGeo;
  }

  set geoTime(v: number) {
    if (!Number.isFinite(v)) throw new RangeError(`DualClock: geoTime must be finite, got ${v}`);
    const k = Math.round(v / this.dtGeo);
    if (k * this.dtGeo === v) {
      this.geoBase = 0;
      this.geoTicks = k;
    } else {
      this.geoBase = v;
      this.geoTicks = 0;
    }
  }

  /** Smoothed My/s actually achieved (≤ requestedSpeed when the sim budget caps ticks). */
  get effectiveSpeed(): number {
    return this.effSpeed;
  }

  /**
   * Advance by one real frame. Returns the number of sim ticks the caller must run.
   * Ticks are capped by maxTicks; excess requested time is dropped, not owed.
   */
  frame(realDt: number, maxTicks: number): number {
    const dt = Number.isFinite(realDt) ? Math.min(Math.max(realDt, 0), DualClock.maxRealDt) : 0;
    const cap = Number.isNaN(maxTicks) ? 0 : Math.max(0, Math.floor(maxTicks));

    this.ambTime += dt;

    let ticks = 0;
    if (!this.paused) {
      this.acc += (dt * this.requestedSpeed) / this.dtGeo;
      ticks = Math.floor(this.acc);
      if (ticks > cap) {
        ticks = cap;
        this.acc = 0; // drop the debt: no death spiral, no burst when the cap lifts
      } else {
        this.acc -= ticks;
      }
      this.geoTicks += ticks;
    }

    if (dt > 0) {
      const inst = (ticks * this.dtGeo) / dt;
      const a = 1 - Math.exp(-dt / DualClock.speedSmoothingTau);
      this.effSpeed += (inst - this.effSpeed) * a;
    }
    return ticks;
  }

  /** ambTime mod ambPeriod, for f32 GPU uniforms (V11). */
  ambTimeWrapped(): number {
    const p = this.ambPeriod;
    return ((this.ambTime % p) + p) % p;
  }

  setSpeed(mySec: number): void {
    this.requestedSpeed = DualClock.clampSpeed(mySec);
  }

  faster(): void {
    this.setSpeed(this.requestedSpeed * 2);
  }

  slower(): void {
    this.setSpeed(this.requestedSpeed / 2);
  }

  togglePause(): void {
    this.paused = !this.paused;
  }

  getState(): ClockState {
    return {
      geoTime: this.geoTime,
      ambTime: this.ambTime,
      paused: this.paused,
      requestedSpeed: this.requestedSpeed,
    };
  }

  loadState(s: ClockState): void {
    if (
      typeof s !== 'object' ||
      s === null ||
      typeof s.geoTime !== 'number' ||
      typeof s.ambTime !== 'number' ||
      typeof s.paused !== 'boolean' ||
      typeof s.requestedSpeed !== 'number' ||
      !Number.isFinite(s.ambTime)
    ) {
      throw new Error(`DualClock.loadState: invalid state ${JSON.stringify(s)}`);
    }
    this.geoTime = s.geoTime;
    this.ambTime = s.ambTime;
    this.paused = s.paused;
    this.requestedSpeed = DualClock.clampSpeed(s.requestedSpeed);
    this.acc = 0;
    this.effSpeed = 0;
  }

  private static clampSpeed(v: number): number {
    if (!Number.isFinite(v)) throw new RangeError(`DualClock: speed must be finite, got ${v}`);
    return Math.min(Math.max(v, DualClock.minSpeed), DualClock.maxSpeed);
  }
}
