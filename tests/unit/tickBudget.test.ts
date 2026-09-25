import { describe, it, expect } from 'vitest';
import { TickBudget } from '../../src/core/tickBudget';

describe('TickBudget (V9, V22)', () => {
  it('converges so ticks × cost stays within the sim budget', () => {
    const b = new TickBudget(5);
    for (let i = 0; i < 200; i++) { const n = b.cap; b.observe(n * 0.4, n); }
    expect(b.cap * 0.4).toBeLessThanOrEqual(5 + 1e-9);
    expect(b.cap).toBeGreaterThanOrEqual(11);
  });
  it('never drops below one tick so the world keeps moving on slow GPUs', () => {
    const b = new TickBudget(5);
    for (let i = 0; i < 200; i++) b.observe(40, 1);
    expect(b.cap).toBe(1);
  });
  it('ignores frames without ticks or bogus timings', () => {
    const b = new TickBudget(5);
    const c0 = b.cap;
    b.observe(0, 5); b.observe(NaN, 5); b.observe(3, 0);
    expect(b.cap).toBe(c0);
  });
});
