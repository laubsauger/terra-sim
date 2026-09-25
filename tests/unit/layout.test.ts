import { describe, it, expect } from 'vitest';
import { packVoxel, voxMat, voxFill, voxAge, voxFlags, Mat, MAT_COUNT, MAT_DENSITY, MAT_ERODIBILITY,
  wrapX, colIdx, voxIdx, NX, NZ, NCOL, ageToCode, codeToAge } from '../../src/sim/layout';

describe('voxel packing', () => {
  it('roundtrips every field at extremes without bleeding into neighbours', () => {
    for (const [m, f, a, fl] of [[0, 0, 0, 0], [255, 255, 255, 255], [12, 1, 254, 0x80], [1, 255, 0, 1]] as const) {
      const v = packVoxel(m, f, a, fl);
      expect(v).toBeGreaterThanOrEqual(0); // unsigned
      expect([voxMat(v), voxFill(v), voxAge(v), voxFlags(v)]).toEqual([m, f, a, fl]);
    }
  });
});

describe('torus indexing (V1)', () => {
  it('wraps negative and overflow coordinates to the opposite edge', () => {
    expect(wrapX(-1)).toBe(NX - 1);
    expect(wrapX(NX)).toBe(0);
    expect(colIdx(-1, -1)).toBe(NX - 1 + (NZ - 1) * NX);
    expect(colIdx(NX + 3, 2)).toBe(colIdx(3, 2));
  });
  it('voxel layers are contiguous so plate shifts copy whole rows', () => {
    expect(voxIdx(0, 1, 0) - voxIdx(0, 0, 0)).toBe(NCOL);
    expect(voxIdx(1, 0, 0) - voxIdx(0, 0, 0)).toBe(1);
  });
});

describe('material table (V21)', () => {
  // Saves store raw ids. Renumbering corrupts every old save, so ids are frozen.
  it('keeps persisted ids stable', () => {
    expect(Mat).toMatchObject({ AIR: 0, BASALT: 1, GRANITE: 3, SEDIMENT: 5, MAGMA: 11, PERIDOTITE: 12 });
  });
  it('has per-material properties for every id', () => {
    expect(Object.keys(Mat).length).toBe(MAT_COUNT);
    expect(MAT_DENSITY.length).toBe(MAT_COUNT);
    expect(MAT_ERODIBILITY.length).toBe(MAT_COUNT);
  });
  it('oceanic crust is denser than continental so it subducts', () => {
    expect(MAT_DENSITY[Mat.BASALT]!).toBeGreaterThan(MAT_DENSITY[Mat.GRANITE]!);
    expect(MAT_DENSITY[Mat.PERIDOTITE]!).toBeGreaterThan(MAT_DENSITY[Mat.BASALT]!);
  });
});

describe('age code', () => {
  it('is monotonic and spans 0..4 Gy within one byte', () => {
    expect(ageToCode(0)).toBe(0);
    expect(ageToCode(4000)).toBeLessThanOrEqual(255);
    expect(ageToCode(4000)).toBeGreaterThan(ageToCode(100));
    expect(Math.abs(codeToAge(ageToCode(200)) - 200) / 200).toBeLessThan(0.05);
  });
});
