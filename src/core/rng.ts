// PCG32 (pcg32 XSH RR, O'Neill). 64-bit LCG state held as two u32 halves so the
// hot path needs no BigInt. BigInt is used only for seeding and (de)serialization.
// Serializable state is what makes save -> load deterministic (V2, V13).

export interface RngState {
  state: string; // 16 hex digits, u64
  inc: string; // 16 hex digits, u64 (always odd)
}

// 6364136223846793005 = 0x5851F42D_4C957F2D
const MUL_HI = 0x5851f42d;
const MUL_LO = 0x4c957f2d;
const TWO_32 = 4294967296;
const U64_MASK = (1n << 64n) - 1n;

/** High 32 bits of the 64-bit product of two u32 (exact, via 16-bit limbs). */
function mulHiU32(a: number, b: number): number {
  const a0 = a & 0xffff;
  const a1 = a >>> 16;
  const b0 = b & 0xffff;
  const b1 = b >>> 16;
  const a0b0 = a0 * b0;
  const a1b0 = a1 * b0;
  const a0b1 = a0 * b1;
  const mid = (a0b0 >>> 16) + (a1b0 & 0xffff) + (a0b1 & 0xffff);
  return (a1 * b1 + (a1b0 >>> 16) + (a0b1 >>> 16) + (mid >>> 16)) >>> 0;
}

function toU64(v: number | bigint, name: string): bigint {
  if (typeof v === 'number') {
    if (!Number.isSafeInteger(v)) throw new RangeError(`PCG32: ${name} must be a safe integer, got ${v}`);
    v = BigInt(v);
  }
  return BigInt.asUintN(64, v);
}

function hex64(hi: number, lo: number): string {
  return hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
}

function parseHex64(s: string, name: string): bigint {
  if (typeof s !== 'string' || !/^[0-9a-fA-F]{1,16}$/.test(s)) {
    throw new Error(`PCG32: invalid ${name} hex string: ${String(s)}`);
  }
  return BigInt('0x' + s);
}

export class PCG32 {
  private sHi = 0;
  private sLo = 0;
  private iHi = 0;
  private iLo = 1;

  /**
   * Same semantics as pcg32_srandom_r(rng, initstate, initseq).
   * Default stream is an arbitrary fixed constant (PCG's default 64-bit increment value).
   */
  constructor(seed: number | bigint, stream: number | bigint = 0x14057b7ef767814fn) {
    const initstate = toU64(seed, 'seed');
    const inc = ((toU64(stream, 'stream') << 1n) | 1n) & U64_MASK;
    this.setInc(inc);
    this.setState(0n);
    this.step();
    this.setState((this.getStateBig() + initstate) & U64_MASK);
    this.step();
  }

  private setState(v: bigint): void {
    this.sHi = Number(v >> 32n);
    this.sLo = Number(v & 0xffffffffn);
  }

  private setInc(v: bigint): void {
    this.iHi = Number(v >> 32n);
    this.iLo = Number(v & 0xffffffffn);
  }

  private getStateBig(): bigint {
    return (BigInt(this.sHi) << 32n) | BigInt(this.sLo);
  }

  /** state = state * MUL + inc  (mod 2^64) */
  private step(): void {
    const hi = this.sHi;
    const lo = this.sLo;
    const pLo = Math.imul(lo, MUL_LO) >>> 0;
    const pHi = (mulHiU32(lo, MUL_LO) + Math.imul(lo, MUL_HI) + Math.imul(hi, MUL_LO)) >>> 0;
    const sumLo = pLo + this.iLo;
    const carry = sumLo >= TWO_32 ? 1 : 0;
    this.sLo = sumLo >>> 0;
    this.sHi = (pHi + this.iHi + carry) >>> 0;
  }

  nextU32(): number {
    const hi = this.sHi;
    const lo = this.sLo;
    this.step();
    // xorshifted = ((old >> 18) ^ old) >> 27, truncated to u32
    const xHi = hi ^ (hi >>> 18);
    const xLo = lo ^ ((lo >>> 18) | (hi << 14));
    const xs = ((xLo >>> 27) | (xHi << 5)) >>> 0;
    const rot = hi >>> 27;
    return ((xs >>> rot) | (xs << ((32 - rot) & 31))) >>> 0;
  }

  /** Float in [0,1) with 32-bit resolution. */
  nextFloat(): number {
    return this.nextU32() / TWO_32;
  }

  /** Float in [min,max). */
  range(min: number, max: number): number {
    return min + (max - min) * this.nextFloat();
  }

  /** Unbiased integer in [0,n) via rejection (pcg32_boundedrand_r). n integer in [1, 2^32]. */
  int(n: number): number {
    if (!Number.isInteger(n) || n < 1 || n > TWO_32) {
      throw new RangeError(`PCG32.int: n must be an integer in [1, 2^32], got ${n}`);
    }
    const threshold = TWO_32 % n;
    for (;;) {
      const r = this.nextU32();
      if (r >= threshold) return r % n;
    }
  }

  getState(): RngState {
    return { state: hex64(this.sHi, this.sLo), inc: hex64(this.iHi, this.iLo) };
  }

  static fromState(s: RngState): PCG32 {
    const state = parseHex64(s.state, 'state');
    const inc = parseHex64(s.inc, 'inc');
    if ((inc & 1n) === 0n) throw new Error(`PCG32: inc must be odd, got ${s.inc}`);
    const rng = new PCG32(0);
    rng.setState(state);
    rng.setInc(inc);
    return rng;
  }
}
