import { describe, it, expect } from 'vitest';
import { NZ, NCOL } from '../../src/sim/layout';
import {
  CLIMATE, DEFAULT_CLIMATE_UNIFORMS, windU, windV, faceZPos, seaLevelTemp, glacierFlux, vaporFlux, f32Quantum,
  makeClimateState, makeClimateScratch, climateTransportCpu,
} from '../../src/sim/climateModel';
import { Biome, BIOME_COUNT, BIOME_VEG_TARGET, classifyBiome, vegErodibilityFactor } from '../../src/sim/biomeModel';

let seed = 987654321;
const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);

describe('face fluxes are antisymmetric (V4)', () => {
  // Two GPU threads evaluate each shared face independently; if f(A,B) ≠ -f(B,A) bit-exactly,
  // every tick leaks or creates water.
  it('glacier flux A→B equals minus B→A exactly', () => {
    for (let n = 0; n < 10000; n++) {
      const hA = Math.fround(60 + rnd() * 60), hB = Math.fround(60 + rnd() * 60);
      const iA = Math.fround(rnd() * 5), iB = Math.fround(rnd() * 5);
      expect(glacierFlux(hA, iA, hB, iB) + glacierFlux(hB, iB, hA, iA) === 0).toBe(true);
    }
  });
  it('glacier flux never takes more than GLACIER_MAX of the donor ice per face', () => {
    for (let n = 0; n < 1000; n++) {
      const iA = rnd() * 5, f = glacierFlux(200 * rnd(), iA, 0, 0);
      expect(f).toBeLessThanOrEqual(CLIMATE.GLACIER_MAX * iA + 1e-12);
    }
  });
});

describe('vapor transport', () => {
  // Donor-cell + diffusion must move vapor without creating it and never drive it negative
  // (negative vapor would rain negative water).
  it('conserves total vapor and ice and keeps them non-negative on random fields', () => {
    const s = makeClimateState(), k = makeClimateScratch();
    for (let i = 0; i < NCOL; i++) {
      s.surfY[i] = 60 + rnd() * 50; k.vaporSrc[i] = rnd() < 0.1 ? 0 : rnd() * 0.3; k.iceSrc[i] = rnd() < 0.5 ? 0 : rnd() * 3;
    }
    let q0 = 0, i0 = 0; for (let i = 0; i < NCOL; i++) { q0 += k.vaporSrc[i]!; i0 += k.iceSrc[i]!; }
    climateTransportCpu(s, k);
    let q1 = 0, i1 = 0, minQ = Infinity, minI = Infinity;
    for (let i = 0; i < NCOL; i++) { q1 += s.vapor[i]!; i1 += s.ice[i]!; minQ = Math.min(minQ, s.vapor[i]!); minI = Math.min(minI, s.ice[i]!); }
    expect(Math.abs(q1 - q0) / q0).toBeLessThan(1e-6);
    expect(Math.abs(i1 - i0) / i0).toBeLessThan(1e-6);
    expect(minQ).toBeGreaterThanOrEqual(0);
    expect(minI).toBeGreaterThanOrEqual(0);
  });
  it('worst-case outflow fraction per tick stays below 1 (positivity bound)', () => {
    let worst = 0;
    for (let z = 0; z < NZ; z++) {
      const out = Math.abs(windU(z)) + Math.abs(windV(faceZPos(z))) + Math.abs(windV(faceZPos((z + 1) % NZ))) + 4 * CLIMATE.VAPOR_DIFF;
      worst = Math.max(worst, out);
    }
    expect(worst).toBeLessThan(1);
  });
  it('vapor flux of equal neighbours in still air is zero', () => {
    expect(vaporFlux(0, 0.1, 0.1)).toBe(0);
  });
});

describe('deep-water delta quantum (V4)', () => {
  // A 16-voxel ocean has ulp ≈ 2e-6; slow evaporation would round away every tick and create water.
  // Adding only the quantized part must be exact so the remainder can stay in vapor.
  it('small deltas on deep water: w + round(dw/q)·q is exact in f32, remainder ≤ half a quantum', () => {
    let tested = 0;
    for (let n = 0; n < 20000; n++) {
      const w = Math.fround(0.5 + rnd() * 120);
      const dw = Math.fround((rnd() - 0.5) * (rnd() < 0.5 ? 1e-6 : 1e-2));
      const q = f32Quantum(Math.abs(w) + Math.abs(dw));
      const dwq = Math.round(dw / q) * q;
      if (Math.abs(w + dwq) >= 2 ** (Math.floor(Math.log2(w)) + 1)) continue; // binade crossing: rounds, unbiased
      tested++;
      expect(Math.fround(w + dwq)).toBe(w + dwq);
      expect(Math.abs(dw - dwq)).toBeLessThanOrEqual(q / 2);
    }
    expect(tested).toBeGreaterThan(19000);
  });
  it('without the quantum, slow evaporation from deep water rounds away (why it exists)', () => {
    const w = Math.fround(16), e = 3e-7;
    expect(Math.fround(w - e)).toBe(w); // naive store: water unchanged, vapor gained e → water created
  });
});

describe('latitude, temperature, wind (V1)', () => {
  it('temperature is periodic in z with the warm belt at z=0 and the cold belt at NZ/2', () => {
    const u = DEFAULT_CLIMATE_UNIFORMS;
    expect(seaLevelTemp(NZ, u)).toBeCloseTo(seaLevelTemp(0, u), 10);
    expect(seaLevelTemp(0, u) - seaLevelTemp(NZ / 2, u)).toBeCloseTo(2 * CLIMATE.T_AMP, 6);
  });
  it('ice age cools uniformly by ICE_AGE_DT·strength', () => {
    const u = DEFAULT_CLIMATE_UNIFORMS;
    expect(seaLevelTemp(10, u) - seaLevelTemp(10, { ...u, iceAge: 1, iceAgeStrength: 0.5 })).toBeCloseTo(CLIMATE.ICE_AGE_DT * 0.5, 6);
  });
  // Rain belts come from the wind pattern: easterly trades and a converging ITCZ at the warm belt,
  // westerlies at mid latitudes; the pattern must continue smoothly across the z seam.
  it('trade easterlies at the equator, westerlies at mid latitude, mirrored hemispheres', () => {
    expect(windU(0)).toBeLessThan(0);
    expect(windU(NZ / 4)).toBeGreaterThan(0);
    expect(windU(3)).toBeCloseTo(windU(NZ - 3), 12);
    expect(windV(10)).toBeLessThan(0); // north tropics blow toward z=0
    expect(windV(NZ - 10)).toBeGreaterThan(0); // south tropics blow toward z=NZ≡0
    expect(windV(faceZPos(0))).toBeCloseTo(-windV(0.5), 12); // seam face continues the pattern
  });
});

describe('biome classification (T31)', () => {
  it('Whittaker corners', () => {
    expect(classifyBiome(26, 0, 10, 0, 0)).toBe(Biome.DESERT);          // hot, dry
    expect(classifyBiome(26, 0.001, 10, 0, 0)).toBe(Biome.RAINFOREST);  // hot, wet
    expect(classifyBiome(26, 0.0002, 10, 0, 0)).toBe(Biome.SAVANNA);
    expect(classifyBiome(12, 0.0003, 10, 0, 0)).toBe(Biome.TEMPERATE_FOREST);
    expect(classifyBiome(12, 0.0001, 10, 0, 0)).toBe(Biome.GRASSLAND);
    expect(classifyBiome(0, 0.0003, 10, 0, 0)).toBe(Biome.TAIGA);
    expect(classifyBiome(-10, 0.0003, 10, 0, 0)).toBe(Biome.TUNDRA);
    expect(classifyBiome(-10, 0.0003, 30, 0, 0)).toBe(Biome.ALPINE);
    expect(classifyBiome(20, 0.001, 0.5, 0, 0)).toBe(Biome.BEACH);
  });
  it('ice and water override climate', () => {
    expect(classifyBiome(-20, 0, 10, 0, 1)).toBe(Biome.ICE);
    expect(classifyBiome(26, 0.001, -10, 16, 0)).toBe(Biome.OCEAN);
    expect(classifyBiome(-5, 0, -10, 16, 1)).toBe(Biome.ICE); // sea ice
  });
  // Saves, render splats and flora index by biome id: ids are append-only.
  it('keeps persisted ids stable and the veg table complete', () => {
    expect(Biome).toEqual({ OCEAN: 0, ICE: 1, TUNDRA: 2, TAIGA: 3, TEMPERATE_FOREST: 4, GRASSLAND: 5, DESERT: 6,
      SAVANNA: 7, RAINFOREST: 8, BEACH: 9, ALPINE: 10 });
    expect(BIOME_VEG_TARGET.length).toBe(BIOME_COUNT);
    expect(BIOME_VEG_TARGET[Biome.RAINFOREST]).toBeGreaterThan(BIOME_VEG_TARGET[Biome.DESERT]!);
  });
  // Erosion coupling: dense cover protects soil, bare ground erodes at full rate.
  it('veg lowers erodibility: factor 1 when bare, 0.3 at full cover', () => {
    expect(vegErodibilityFactor(0)).toBe(1);
    expect(vegErodibilityFactor(1)).toBeCloseTo(0.3, 12);
  });
});
