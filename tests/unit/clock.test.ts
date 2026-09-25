import { describe, expect, it } from 'vitest';
import { DualClock } from '../../src/core/clock';

const BIG = 1_000_000; // effectively uncapped sim budget

function runFor(clock: DualClock, seconds: number, fps: number, maxTicks = BIG): number {
  let total = 0;
  const frames = Math.round(seconds * fps);
  for (let i = 0; i < frames; i++) total += clock.frame(1 / fps, maxTicks);
  return total;
}

describe('DualClock', () => {
  it('pause stops geoTime but ambTime keeps advancing (waves/clouds keep moving, V17)', () => {
    const c = new DualClock({ dtGeo: 0.01, requestedSpeed: 1 });
    runFor(c, 1, 60);
    const geo0 = c.geoTime;
    const amb0 = c.ambTime;
    c.togglePause();
    let ticks = 0;
    for (let i = 0; i < 120; i++) ticks += c.frame(1 / 60, BIG);
    expect(ticks).toBe(0);
    expect(c.geoTime).toBe(geo0);
    expect(c.ambTime).toBeCloseTo(amb0 + 2, 9);
    c.togglePause();
    expect(c.frame(0.1, BIG)).toBeGreaterThan(0);
  });

  it('speed changes never change dtGeo, only ticks per frame (V22/V12)', () => {
    const c = new DualClock({ dtGeo: 0.02, requestedSpeed: 1 });
    const dt0 = c.dtGeo;
    const t1 = runFor(c, 1, 60);
    c.faster();
    c.faster();
    expect(c.requestedSpeed).toBe(4);
    expect(c.dtGeo).toBe(dt0);
    const t4 = runFor(c, 1, 60);
    c.setSpeed(0.5);
    expect(c.dtGeo).toBe(dt0);
    c.slower();
    expect(c.requestedSpeed).toBe(0.25);
    expect(c.dtGeo).toBe(dt0);
    // 1 My/s / 0.02 = 50 ticks/s; 4 My/s = 200 ticks/s (±1 from fractional carry)
    expect(Math.abs(t1 - 50)).toBeLessThanOrEqual(1);
    expect(Math.abs(t4 - 200)).toBeLessThanOrEqual(1);
  });

  it('speed is clamped to [minSpeed, maxSpeed]', () => {
    const c = new DualClock({ dtGeo: 0.01 });
    c.setSpeed(1e9);
    expect(c.requestedSpeed).toBe(DualClock.maxSpeed);
    c.faster();
    expect(c.requestedSpeed).toBe(DualClock.maxSpeed);
    c.setSpeed(0);
    expect(c.requestedSpeed).toBe(DualClock.minSpeed);
    c.slower();
    expect(c.requestedSpeed).toBe(DualClock.minSpeed);
    expect(() => c.setSpeed(NaN)).toThrow();
  });

  it('30fps vs 60fps: same total ticks (±1) and geoTime is exactly ticks*dtGeo (fps-independent sim, V12)', () => {
    const a = new DualClock({ dtGeo: 0.001, requestedSpeed: 0.37 });
    const b = new DualClock({ dtGeo: 0.001, requestedSpeed: 0.37 });
    const ta = runFor(a, 1, 30);
    const tb = runFor(b, 1, 60);
    expect(Math.abs(ta - tb)).toBeLessThanOrEqual(1);
    expect(Math.abs(ta - 370)).toBeLessThanOrEqual(1);
    expect(a.geoTime).toBe(ta * a.dtGeo);
    expect(b.geoTime).toBe(tb * b.dtGeo);
  });

  it('geoTime depends on tick count only, not on how ticks were chunked into frames', () => {
    // An event scheduler comparing geoTime against thresholds must fire on the same tick
    // regardless of fps; summing dtGeo per frame would drift by ulps with frame chunking.
    const a = new DualClock({ dtGeo: 0.1, requestedSpeed: 10 }); // 100 ticks/s
    const b = new DualClock({ dtGeo: 0.1, requestedSpeed: 10 });
    let ta = 0;
    let tb = 0;
    while (ta < 10000) ta += a.frame(0.01, 1); // 1 tick at a time
    while (tb < 10000) tb += b.frame(0.07, 7); // 7 at a time
    while (tb > ta) ta += a.frame(0.01, 1);
    expect(ta).toBe(tb);
    expect(a.geoTime).toBe(b.geoTime);
  });

  it('slow speeds still tick eventually via fractional carry-over', () => {
    const c = new DualClock({ dtGeo: 1, requestedSpeed: DualClock.minSpeed }); // 1 tick / 1000 s
    let ticks = 0;
    for (let i = 0; i < 60 * 1001; i++) ticks += c.frame(1 / 60, BIG);
    expect(ticks).toBe(1);
  });

  it('maxTicks cap: effectiveSpeed < requestedSpeed, and no debt accumulates (no burst after cap lifts)', () => {
    const c = new DualClock({ dtGeo: 0.01, requestedSpeed: 6 }); // 10 ticks per 60fps frame
    for (let i = 0; i < 600; i++) expect(c.frame(1 / 60, 2)).toBeLessThanOrEqual(2);
    // 2 ticks/frame * 60fps * 0.01 = 1.2 My/s achieved
    expect(c.effectiveSpeed).toBeLessThan(c.requestedSpeed);
    expect(c.effectiveSpeed).toBeCloseTo(1.2, 2);
    const first = c.frame(1 / 60, BIG);
    expect(first).toBeLessThanOrEqual(11); // normal per-frame amount, not 10 s of backlog
    for (let i = 0; i < 300; i++) c.frame(1 / 60, BIG);
    expect(c.effectiveSpeed).toBeCloseTo(6, 1);
  });

  it('huge realDt (tab switch) is clamped, not turned into a tick burst', () => {
    const c = new DualClock({ dtGeo: 0.01, requestedSpeed: 1 });
    const ticks = c.frame(30, BIG);
    expect(ticks).toBeLessThanOrEqual(Math.ceil(DualClock.maxRealDt / 0.01));
    expect(c.ambTime).toBe(DualClock.maxRealDt);
  });

  it('ambTimeWrapped stays in [0, period) after days of runtime (f32 uniform precision, V11)', () => {
    const c = new DualClock({ dtGeo: 0.01, requestedSpeed: 1 });
    c.ambTime = 86400 * 30 + 123.5;
    const w = c.ambTimeWrapped();
    expect(w).toBeGreaterThanOrEqual(0);
    expect(w).toBeLessThan(3600);
    expect(w).toBeCloseTo(123.5, 6);
    const c2 = new DualClock({ dtGeo: 0.01, ambPeriod: 100 });
    for (let i = 0; i < 10000; i++) c2.frame(0.25, 0);
    expect(c2.ambTime).toBeCloseTo(2500, 6);
    expect(c2.ambTimeWrapped()).toBeGreaterThanOrEqual(0);
    expect(c2.ambTimeWrapped()).toBeLessThan(100);
  });

  it('state roundtrip through JSON restores geoTime/ambTime/paused/speed and ticking continues identically', () => {
    const a = new DualClock({ dtGeo: 0.003, requestedSpeed: 2 });
    runFor(a, 3.3, 60);
    a.togglePause();
    runFor(a, 0.5, 60);
    const saved = JSON.parse(JSON.stringify(a.getState()));
    const b = new DualClock({ dtGeo: 0.003 });
    b.loadState(saved);
    expect(b.getState()).toEqual(a.getState());
    // after load, N more ticks give exactly the same geoTime on both (save->load determinism, V13)
    a.togglePause();
    b.togglePause();
    for (let i = 0; i < 50; i++) {
      a.frame(1, 3);
      b.frame(1, 3);
    }
    expect(b.geoTime).toBe(a.geoTime);
  });

  it('loadState rejects malformed state', () => {
    const c = new DualClock({ dtGeo: 0.01 });
    expect(() => c.loadState({ geoTime: 1 } as never)).toThrow();
  });
});
