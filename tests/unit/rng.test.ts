import { describe, expect, it } from 'vitest';
import { PCG32 } from '../../src/core/rng';

// Independent BigInt transcription of the pcg32 reference (pcg_basic.c), used to
// cross-check the u32-halves fast path over many outputs, not just the first six.
function refPcg32(initstate: bigint, initseq: bigint) {
  const M = (1n << 64n) - 1n;
  let state = 0n;
  const inc = ((initseq << 1n) | 1n) & M;
  const next = (): number => {
    const old = state;
    state = (old * 6364136223846793005n + inc) & M;
    const xorshifted = Number((((old >> 18n) ^ old) >> 27n) & 0xffffffffn);
    const rot = Number(old >> 59n);
    return ((xorshifted >>> rot) | (xorshifted << ((32 - rot) & 31))) >>> 0;
  };
  next();
  state = (state + initstate) & M;
  next();
  return next;
}

describe('PCG32', () => {
  it('matches the official pcg32-demo output for pcg32_srandom_r(42, 54)', () => {
    // Values from pcg-c-basic pcg32-demo "Round 1: 32bit:" line (6th is 0xcbed606e).
    // Worldgen and event scheduling must be reproducible from a seed (V2); matching the
    // published reference proves the generator is PCG32 and not an accidental variant.
    const rng = new PCG32(42, 54);
    const got = Array.from({ length: 6 }, () => rng.nextU32());
    expect(got).toEqual([0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]);
  });

  it('matches a BigInt reference over many outputs, seeds and streams (carry/limb edge cases)', () => {
    const cases: [bigint, bigint][] = [
      [42n, 54n],
      [0n, 0n],
      [(1n << 64n) - 1n, (1n << 63n) - 1n],
      [0xdeadbeefcafebaben, 12345n],
    ];
    for (const [s, q] of cases) {
      const ref = refPcg32(s, q);
      const rng = new PCG32(s, q);
      for (let i = 0; i < 5000; i++) expect(rng.nextU32()).toBe(ref());
    }
  });

  it('number and bigint seeds are equivalent', () => {
    const a = new PCG32(123456789, 7);
    const b = new PCG32(123456789n, 7n);
    for (let i = 0; i < 100; i++) expect(a.nextU32()).toBe(b.nextU32());
  });

  it('save -> JSON -> load continues the identical sequence (save/load determinism, V2/V13)', () => {
    const rng = new PCG32(2024, 3);
    for (let i = 0; i < 777; i++) rng.nextU32();
    const json = JSON.stringify(rng.getState());
    const expected = Array.from({ length: 1000 }, () => rng.nextU32());
    const restored = PCG32.fromState(JSON.parse(json));
    const got = Array.from({ length: 1000 }, () => restored.nextU32());
    expect(got).toEqual(expected);
  });

  it('fromState rejects corrupt state instead of silently producing a different stream', () => {
    expect(() => PCG32.fromState({ state: 'zz', inc: '1' })).toThrow();
    expect(() => PCG32.fromState({ state: '0', inc: '2' })).toThrow(); // inc must be odd
  });

  it('nextFloat is in [0,1) and range is in [min,max)', () => {
    const rng = new PCG32(9);
    for (let i = 0; i < 10000; i++) {
      const f = rng.nextFloat();
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(1);
      const r = rng.range(-3, 5);
      expect(r).toBeGreaterThanOrEqual(-3);
      expect(r).toBeLessThan(5);
    }
  });

  it('int(n) stays in [0,n), is an integer, and hits every bucket roughly uniformly', () => {
    const rng = new PCG32(77, 1);
    for (const n of [1, 2, 3, 7, 1000, 2 ** 31 + 1, 2 ** 32]) {
      for (let i = 0; i < 2000; i++) {
        const v = rng.int(n);
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(n);
      }
    }
    const counts = new Array<number>(6).fill(0);
    const N = 60000;
    for (let i = 0; i < N; i++) counts[rng.int(6)]!++;
    for (const c of counts) expect(Math.abs(c - N / 6)).toBeLessThan(N / 6 * 0.05);
  });

  it('int(n) rejects invalid bounds', () => {
    const rng = new PCG32(1);
    expect(() => rng.int(0)).toThrow();
    expect(() => rng.int(2.5)).toThrow();
    expect(() => rng.int(2 ** 32 + 1)).toThrow();
  });
});
