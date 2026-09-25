import { describe, it, expect } from 'vitest';
import { hydroStepCpu, HYDRO_DEFAULTS, type HydroCpuState } from '../../src/sim/hydro';
import { NCOL, NX, NZ } from '../../src/sim/layout';

const idx = (x: number, z: number) => ((x + NX) % NX) + ((z + NZ) % NZ) * NX;
const state = (surf: (x: number, z: number) => number, water: (x: number, z: number) => number): HydroCpuState => {
  const s: HydroCpuState = {
    surfY: new Float32Array(NCOL), water: new Float32Array(NCOL), flux: new Float32Array(NCOL * 4),
    vel: new Float32Array(NCOL * 2), sed: new Float32Array(NCOL), sedRatio: new Float32Array(NCOL),
  };
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) { s.surfY[idx(x, z)] = surf(x, z); s.water[idx(x, z)] = water(x, z); }
  return s;
};
const sum = (a: Float32Array) => a.reduce((t, v) => t + v, 0);

// The CPU reference is what the GPU kernel is checked against, so it must itself honour V4 and V1.
describe('hydro CPU reference', () => {
  it('conserves water and sediment and keeps depths non-negative on rough terrain', () => {
    const st = state((x, z) => 70 + 20 * Math.sin(x * 0.09) * Math.cos(z * 0.13), (x, z) => (x < 64 ? 3 : 0.2) + (z % 7) * 0.1);
    st.sed.fill(5);
    const w0 = sum(st.water), s0 = sum(st.sed);
    for (let i = 0; i < 100; i++) hydroStepCpu(st);
    expect(Math.abs(sum(st.water) - w0) / w0).toBeLessThan(1e-5);
    expect(Math.abs(sum(st.sed) - s0) / s0).toBeLessThan(1e-5);
    expect(Math.min(...st.water)).toBeGreaterThan(-1e-5);
  });

  it('spreads across the torus seam exactly like across any other edge', () => {
    const st = state(() => 80, (x, z) => (x === 0 && z === 10 ? 10 : 0));
    for (let i = 0; i < 3; i++) hydroStepCpu(st, HYDRO_DEFAULTS);
    expect(st.water[idx(NX - 1, 10)]!).toBeGreaterThan(0.1);
    expect(st.water[idx(NX - 1, 10)]).toBeCloseTo(st.water[idx(1, 10)]!, 6);
    expect(st.water[idx(0, 9)]).toBeCloseTo(st.water[idx(0, 11)]!, 6);
  });
});
