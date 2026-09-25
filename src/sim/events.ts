// §T.33 seeded random event scheduler. Poisson arrivals per stats window from a dedicated PCG32 stream (V2).
// God tools enqueue into the same queue (V16), so every event goes through one handler path.
import { PCG32 } from '../core/rng';
import { NX, NZ } from './layout';

export type EventKind = 'hotspot' | 'meteor' | 'iceAge' | 'floodBasalt';

export interface GeoEvent {
  kind: EventKind;
  my: number;          // geo time it fires
  x: number; z: number; // column
  magnitude: number;   // 0.5..1.5, kind-specific meaning
  durationMy?: number; // ice ages
  source: 'random' | 'god';
}

/** Mean events per My at eventRate = 1. */
export const BASE_RATES: Record<EventKind, number> = {
  hotspot: 1 / 150,
  meteor: 1 / 220,
  iceAge: 1 / 300,
  floodBasalt: 1 / 260,
};

export interface EventSchedulerState { rng: ReturnType<PCG32['getState']>; queue: GeoEvent[]; log: GeoEvent[] }

export class EventScheduler {
  private rng: PCG32;
  queue: GeoEvent[] = [];
  readonly log: GeoEvent[] = [];
  static readonly LOG_MAX = 200;

  constructor(seed: number) { this.rng = new PCG32(seed, 0xe7e7); }

  /** Draw this window's random events (Poisson thinning per kind). rates: per-kind multipliers. */
  roll(my: number, windowMy: number, rates: Partial<Record<EventKind, number>>): void {
    for (const kind of Object.keys(BASE_RATES) as EventKind[]) {
      const lambda = BASE_RATES[kind] * (rates[kind] ?? 1) * windowMy;
      // Knuth Poisson; lambda is tiny so this is almost always 0 or 1 draws
      let k = 0, p = 1;
      const L = Math.exp(-lambda);
      for (;;) { p *= this.rng.nextFloat(); if (p <= L) break; k++; }
      for (let i = 0; i < k; i++) {
        this.queue.push({
          kind, my, source: 'random',
          x: this.rng.int(NX), z: this.rng.int(NZ),
          magnitude: 0.5 + this.rng.nextFloat(),
          durationMy: kind === 'iceAge' ? 10 + this.rng.nextFloat() * 25 : undefined,
        });
      }
    }
  }

  enqueue(e: Omit<GeoEvent, 'source'>): void { this.queue.push({ ...e, source: 'god' }); }

  /** Remove and return every queued event; the orchestrator dispatches them to handlers. */
  drain(): GeoEvent[] {
    const q = this.queue;
    this.queue = [];
    for (const e of q) { this.log.push(e); if (this.log.length > EventScheduler.LOG_MAX) this.log.shift(); }
    return q;
  }

  getState(): EventSchedulerState { return { rng: this.rng.getState(), queue: [...this.queue], log: [...this.log] }; }
  loadState(s: EventSchedulerState): void {
    this.rng = PCG32.fromState(s.rng);
    this.queue = [...s.queue];
    this.log.length = 0;
    this.log.push(...s.log);
  }
}

/** Ice-age envelope: smooth ramp in/out over 20% of duration each side. Returns forcing 0..1. */
export function iceAgeForcing(active: { start: number; durationMy: number; magnitude: number }[], my: number): number {
  let f = 0;
  for (const a of active) {
    const t = (my - a.start) / a.durationMy;
    if (t < 0 || t > 1) continue;
    const ramp = Math.min(1, t / 0.2, (1 - t) / 0.2);
    f = Math.max(f, ramp * ramp * (3 - 2 * ramp) * Math.min(1, a.magnitude));
  }
  return f;
}
