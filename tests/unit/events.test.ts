import { describe, it, expect } from 'vitest';
import { EventScheduler, BASE_RATES, iceAgeForcing } from '../../src/sim/events';

describe('event scheduler (T33)', () => {
  it('is deterministic per seed and survives a state roundtrip (V2, V13)', () => {
    const a = new EventScheduler(5), b = new EventScheduler(5);
    for (let i = 0; i < 100; i++) { a.roll(i * 2, 2, {}); b.roll(i * 2, 2, {}); }
    expect(a.drain()).toEqual(b.drain());
    const c = new EventScheduler(5);
    for (let i = 0; i < 50; i++) c.roll(i * 2, 2, {});
    const d = new EventScheduler(0);
    d.loadState(JSON.parse(JSON.stringify(c.getState())));
    for (let i = 50; i < 100; i++) { c.roll(i * 2, 2, {}); d.roll(i * 2, 2, {}); }
    expect(d.drain()).toEqual(c.drain());
  });
  it('long-run event frequency matches the configured rates (keeps the world eventful)', () => {
    const s = new EventScheduler(9);
    const my = 200_000;
    for (let t = 0; t < my; t += 2) s.roll(t, 2, {});
    const ev = s.drain();
    for (const k of Object.keys(BASE_RATES) as (keyof typeof BASE_RATES)[]) {
      const n = ev.filter((e) => e.kind === k).length;
      const expected = BASE_RATES[k]! * my;
      expect(Math.abs(n - expected) / expected).toBeLessThan(0.15);
    }
  });
  it('rate multiplier 0 disables a kind', () => {
    const s = new EventScheduler(2);
    for (let t = 0; t < 50_000; t += 2) s.roll(t, 2, { meteor: 0 });
    expect(s.drain().some((e) => e.kind === 'meteor')).toBe(false);
  });
  it('ice age forcing ramps smoothly in and out', () => {
    const a = [{ start: 100, durationMy: 20, magnitude: 1 }];
    expect(iceAgeForcing(a, 99)).toBe(0);
    expect(iceAgeForcing(a, 110)).toBeCloseTo(1);
    expect(iceAgeForcing(a, 101)).toBeGreaterThan(0);
    expect(iceAgeForcing(a, 101)).toBeLessThan(0.5);
    expect(iceAgeForcing(a, 121)).toBe(0);
  });
});
