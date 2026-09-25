import { describe, expect, it } from 'vitest';
import { PeriodicNoise2D } from '../../src/core/noise';
import { PCG32 } from '../../src/core/rng';

// The world is a torus (V1): every noise-driven field must tile exactly, or the
// wrap edge shows a cliff in terrain, strata and plate borders.

describe('PeriodicNoise2D determinism (V2)', () => {
  it('same seed gives bit-identical values; different seeds give different fields', () => {
    const a = new PeriodicNoise2D(7);
    const b = new PeriodicNoise2D(new PCG32(7));
    const c = new PeriodicNoise2D(8);
    let diff = 0;
    for (let i = 0; i < 500; i++) {
      const u = i * 0.01731;
      const v = i * 0.02917;
      expect(a.fbm(u, v, 4, 4, 5, 0.5)).toBe(b.fbm(u, v, 4, 4, 5, 0.5));
      diff += Math.abs(a.fbm(u, v, 4, 4, 5, 0.5) - c.fbm(u, v, 4, 4, 5, 0.5));
    }
    expect(diff / 500).toBeGreaterThan(0.05);
  });
});

describe('PeriodicNoise2D tiling (V1)', () => {
  const n = new PeriodicNoise2D(123);

  it('base noise repeats exactly with its integer periods, also for negative coords', () => {
    for (let i = 0; i < 300; i++) {
      const x = i * 0.377 - 40;
      const z = i * 0.611 - 25;
      const ref = n.noise(x, z, 5, 3);
      expect(n.noise(x + 5, z, 5, 3)).toBeCloseTo(ref, 12);
      expect(n.noise(x, z + 3, 5, 3)).toBeCloseTo(ref, 12);
      expect(n.noise(x - 10, z - 6, 5, 3)).toBeCloseTo(ref, 12);
    }
  });

  it('fbm, ridged and warp are periodic with period 1 in torus coords', () => {
    const out1 = new Float64Array(2);
    const out2 = new Float64Array(2);
    for (let i = 0; i < 200; i++) {
      const u = i * 0.00731;
      const v = 1 - i * 0.00493;
      expect(n.fbm(u + 1, v, 3, 5, 6, 0.5)).toBeCloseTo(n.fbm(u, v, 3, 5, 6, 0.5), 9);
      expect(n.fbm(u, v - 1, 3, 5, 6, 0.5)).toBeCloseTo(n.fbm(u, v, 3, 5, 6, 0.5), 9);
      expect(n.ridged(u - 1, v + 1, 4, 4, 5, 0.5)).toBeCloseTo(n.ridged(u, v, 4, 4, 5, 0.5), 9);
      n.warp(u, v, 3, 3, 4, 0.5, 0.1, out1);
      n.warp(u + 1, v, 3, 3, 4, 0.5, 0.1, out2);
      // warped coords shift by exactly the period, i.e. same point on the torus
      expect(out2[0]! - 1).toBeCloseTo(out1[0]!, 9);
      expect(out2[1]!).toBeCloseTo(out1[1]!, 9);
    }
  });

  it('is continuous across the wrap edge (no seam jump larger than interior steps)', () => {
    const eps = 1 / 1024;
    let maxInterior = 0;
    for (let i = 1; i < 1024; i++) {
      const d = Math.abs(n.fbm(i * eps, 0.3, 4, 4, 6, 0.5) - n.fbm((i - 1) * eps, 0.3, 4, 4, 6, 0.5));
      if (d > maxInterior) maxInterior = d;
    }
    const seam = Math.abs(n.fbm(1 - eps, 0.3, 4, 4, 6, 0.5) - n.fbm(0, 0.3, 4, 4, 6, 0.5));
    expect(seam).toBeLessThanOrEqual(maxInterior * 1.01);
  });

  it('rejects non-integer periods, which would silently break tiling', () => {
    expect(() => n.fbm(0.1, 0.1, 2.5, 2, 3, 0.5)).toThrow(RangeError);
    expect(() => n.ridged(0.1, 0.1, 2, 0, 3, 0.5)).toThrow(RangeError);
  });
});

describe('PeriodicNoise2D range', () => {
  // Worldgen maps noise to layer heights assuming ≈[-1,1] (fbm) and [0,1] (ridged).
  it('fbm stays in ≈[-1,1] with useful variance; ridged in [0,1]', () => {
    const n = new PeriodicNoise2D(99);
    let sum = 0;
    let sum2 = 0;
    const N = 20000;
    for (let i = 0; i < N; i++) {
      const u = (i % 141) / 141;
      const v = Math.floor(i / 141) / 141;
      const f = n.fbm(u, v, 4, 4, 5, 0.5);
      const r = n.ridged(u, v, 4, 4, 5, 0.5);
      expect(Math.abs(f)).toBeLessThanOrEqual(1.0001);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1);
      sum += f;
      sum2 += f * f;
    }
    const mean = sum / N;
    const sd = Math.sqrt(sum2 / N - mean * mean);
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(sd).toBeGreaterThan(0.1);
  });
});
