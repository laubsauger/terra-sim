// Seeded periodic (tileable) 2D gradient noise for the torus world (V1: no seams).
//
// Periodicity: lattice coordinates are reduced mod integer periods (px, pz) before
// hashing, so noise(x + px, z) === noise(x, z) exactly. fbm/ridged/warp take torus
// coordinates u,v (period 1) and a base period in lattice cells; lacunarity is fixed
// at 2 so every octave's period stays an integer and the sum stays periodic.
//
// Determinism (V2): permutation from PCG32, gradients from a literal 16-direction
// table (no Math.sin/cos, whose last-bit results are not specified across engines).
// Hot path allocates nothing.

import { PCG32 } from './rng';

// 16 unit gradients at 22.5° steps, as literals.
const C1 = 0.9238795325112867;
const C2 = 0.7071067811865476;
const C3 = 0.3826834323650898;
const GX = new Float64Array([1, C1, C2, C3, 0, -C3, -C2, -C1, -1, -C1, -C2, -C3, 0, C3, C2, C1]);
const GZ = new Float64Array([0, C3, C2, C1, 1, C1, C2, C3, 0, -C3, -C2, -C1, -1, -C1, -C2, -C3]);
// Max |perlin 2D| with unit gradients is sqrt(1/2); rescale to roughly [-1, 1].
const NORM = Math.SQRT2;

function checkPeriod(p: number, name: string): void {
  if (!Number.isInteger(p) || p < 1) throw new RangeError(`noise: ${name} must be an integer >= 1, got ${p}`);
}

export class PeriodicNoise2D {
  /** Doubled permutation so perm[perm[i] + j] never needs a mask. */
  private readonly perm = new Uint8Array(512);

  /** Seeded from a PCG32 (draws 255 values) or directly from a seed. */
  constructor(src: PCG32 | number) {
    const rng = typeof src === 'number' ? new PCG32(src) : src;
    const p = this.perm;
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      const j = rng.int(i + 1);
      const t = p[i]!;
      p[i] = p[j]!;
      p[j] = t;
    }
    for (let i = 0; i < 256; i++) p[i + 256] = p[i]!;
  }

  /**
   * Gradient noise at lattice coords (x, z), periodic with integer periods px, pz.
   * Range ≈ [-1, 1]. Periods > 256 stay periodic but the gradient hash repeats every 256 cells.
   * Caller guarantees integer periods (checked in fbm/ridged/warp, not here: hot path).
   */
  noise(x: number, z: number, px: number, pz: number): number {
    const fx = Math.floor(x);
    const fz = Math.floor(z);
    const tx = x - fx;
    const tz = z - fz;
    let ix0 = fx % px;
    if (ix0 < 0) ix0 += px;
    let iz0 = fz % pz;
    if (iz0 < 0) iz0 += pz;
    let ix1 = ix0 + 1;
    if (ix1 >= px) ix1 = 0;
    let iz1 = iz0 + 1;
    if (iz1 >= pz) iz1 = 0;
    const p = this.perm;
    const a0 = p[ix0 & 255]!;
    const a1 = p[ix1 & 255]!;
    const zb0 = iz0 & 255;
    const zb1 = iz1 & 255;
    const h00 = p[a0 + zb0]! & 15;
    const h10 = p[a1 + zb0]! & 15;
    const h01 = p[a0 + zb1]! & 15;
    const h11 = p[a1 + zb1]! & 15;
    const tx1 = tx - 1;
    const tz1 = tz - 1;
    const n00 = GX[h00]! * tx + GZ[h00]! * tz;
    const n10 = GX[h10]! * tx1 + GZ[h10]! * tz;
    const n01 = GX[h01]! * tx + GZ[h01]! * tz1;
    const n11 = GX[h11]! * tx1 + GZ[h11]! * tz1;
    // quintic fade
    const sx = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
    const sz = tz * tz * tz * (tz * (tz * 6 - 15) + 10);
    const nx0 = n00 + sx * (n10 - n00);
    const nx1 = n01 + sx * (n11 - n01);
    return (nx0 + sz * (nx1 - nx0)) * NORM;
  }

  /**
   * Fractal sum on torus coords (u, v) (period 1 in each). pu, pv = base period in lattice
   * cells (integer). Lacunarity 2. Normalized by amplitude sum → ≈ [-1, 1].
   */
  fbm(u: number, v: number, pu: number, pv: number, octaves: number, gain: number): number {
    checkPeriod(pu, 'pu');
    checkPeriod(pv, 'pv');
    let sum = 0;
    let amp = 1;
    let norm = 0;
    let fu = pu;
    let fv = pv;
    for (let i = 0; i < octaves; i++) {
      sum += amp * this.noise(u * fu, v * fv, fu, fv);
      norm += amp;
      amp *= gain;
      fu *= 2;
      fv *= 2;
    }
    return norm > 0 ? sum / norm : 0;
  }

  /** Ridged fbm: sum of (1 - |n|)² → [0, 1], sharp crests along noise zero-lines. */
  ridged(u: number, v: number, pu: number, pv: number, octaves: number, gain: number): number {
    checkPeriod(pu, 'pu');
    checkPeriod(pv, 'pv');
    let sum = 0;
    let amp = 1;
    let norm = 0;
    let fu = pu;
    let fv = pv;
    for (let i = 0; i < octaves; i++) {
      let n = this.noise(u * fu, v * fv, fu, fv);
      n = 1 - (n < 0 ? -n : n);
      if (n < 0) n = 0;
      sum += amp * n * n;
      norm += amp;
      amp *= gain;
      fu *= 2;
      fv *= 2;
    }
    return norm > 0 ? sum / norm : 0;
  }

  /**
   * Domain warp: out[0] = u + amp * fbm_a(u, v), out[1] = v + amp * fbm_b(u, v).
   * amp in torus units. Constant offsets decorrelate the two channels and keep periodicity.
   * Writes into caller-owned `out` (no allocation).
   */
  warp(u: number, v: number, pu: number, pv: number, octaves: number, gain: number, amp: number, out: Float64Array): void {
    const a = this.fbm(u + 0.3137, v + 0.7291, pu, pv, octaves, gain);
    const b = this.fbm(u + 0.5813, v + 0.1759, pu, pv, octaves, gain);
    out[0] = u + amp * a;
    out[1] = v + amp * b;
  }
}
