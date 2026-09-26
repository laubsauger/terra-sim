import { describe, it, expect } from 'vitest';
import { cycleToTod, todToCycle, NIGHT_SHARE } from '../../src/render/sky';

// Ambient mode looked "always dark": a linear day cycle spends half its time at night. A running cycle must
// spend only NIGHT_SHARE in the dark, move the sun continuously, and round-trip so toggles don't jump the sun.
describe('day cycle', () => {
  const night = (t: number) => t < 0.25 || t >= 0.75;
  it('spends only NIGHT_SHARE of the cycle at night', () => {
    let n = 0; const N = 10000;
    for (let i = 0; i < N; i++) if (night(cycleToTod(i / N))) n++;
    expect(n / N).toBeCloseTo(NIGHT_SHARE, 2);
    expect(NIGHT_SHARE).toBeLessThanOrEqual(0.25);
  });
  it('is continuous (no jump of the sun anywhere in the cycle)', () => {
    let prev = cycleToTod(0);
    for (let i = 1; i <= 2000; i++) {
      const t = cycleToTod(i / 2000);
      const d = Math.min(Math.abs(t - prev), 1 - Math.abs(t - prev)); // wrap at midnight
      expect(d).toBeLessThan(0.01);
      prev = t;
    }
  });
  it('round-trips time of day', () => {
    for (const t of [0, 0.1, 0.25, 0.34, 0.5, 0.705, 0.75, 0.9, 0.999]) {
      const d = Math.abs(cycleToTod(todToCycle(t)) - t);
      expect(Math.min(d, 1 - d)).toBeLessThan(1e-9); // circular: midnight is 0 and 1
    }
  });
});
