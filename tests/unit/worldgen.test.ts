import { beforeAll, describe, expect, it } from 'vitest';
import {
  generateWorld, crustMass, MANTLE_RESERVOIR_FRAC, PLATE_SPEED_MIN, PLATE_SPEED_MAX,
} from '../../src/sim/worldgen';
import {
  NX, NZ, NY, NCOL, Mat, MAT_COUNT, FLAG_CONTINENTAL, Y_SEA_NOMINAL, voxMat, voxFill, voxFlags,
} from '../../src/sim/layout';
import type { WorldData } from '../../src/sim/worldData';

const SEEDS = [1, 2, 3, 42, 1234];
const worlds = new Map<number, WorldData>();
let coldMs = 0;

function hashWorld(w: WorldData): number {
  let h = 0x811c9dc5;
  const mix = (a: Uint32Array) => {
    for (let i = 0; i < a.length; i++) h = Math.imul(h ^ a[i]!, 0x01000193);
  };
  mix(w.vox);
  mix(w.plateId);
  mix(new Uint32Array(w.water.buffer, w.water.byteOffset, w.water.length));
  return h >>> 0;
}

/** Surface height per column as the derived pass defines it: top solid y + fill/255. */
function surface(w: WorldData): Float32Array {
  const s = new Float32Array(NCOL);
  for (let c = 0; c < NCOL; c++) {
    let y = NY - 1;
    while (y > 0 && voxMat(w.vox[c + y * NCOL]!) === Mat.AIR) y--;
    s[c] = y + voxFill(w.vox[c + y * NCOL]!) / 255;
  }
  return s;
}

function topVoxel(w: WorldData, c: number): number {
  let y = NY - 1;
  while (y > 0 && voxMat(w.vox[c + y * NCOL]!) === Mat.AIR) y--;
  return w.vox[c + y * NCOL]!;
}

beforeAll(() => {
  const t = performance.now();
  worlds.set(SEEDS[0]!, generateWorld(SEEDS[0]!)); // cold: includes JIT warm-up
  coldMs = performance.now() - t;
  for (const s of SEEDS.slice(1)) worlds.set(s, generateWorld(s));
}, 60_000);

describe('worldgen determinism (V2)', () => {
  // Saves, replays and GPU-vs-CPU tests all assume a seed fully defines the initial world.
  it('same seed → identical voxels, plates and water; different seeds differ', () => {
    const again = generateWorld(42);
    expect(hashWorld(again)).toBe(hashWorld(worlds.get(42)!));
    expect(again.plates).toEqual(worlds.get(42)!.plates);
    expect(again.mantleReservoir).toBe(worlds.get(42)!.mantleReservoir);
    const hashes = new Set(SEEDS.map((s) => hashWorld(worlds.get(s)!)));
    expect(hashes.size).toBe(SEEDS.length);
  });
});

describe('worldgen performance', () => {
  it('generates the full 256×256×128 grid in < 1.5 s (startup budget)', () => {
    const t = performance.now();
    generateWorld(777);
    const warmMs = performance.now() - t;
    console.log(`worldgen: cold ${coldMs.toFixed(0)} ms, warm ${warmMs.toFixed(0)} ms`);
    expect(coldMs).toBeLessThan(1500);
    expect(warmMs).toBeLessThan(1500);
  });
});

describe('worldgen column structure', () => {
  // Sim passes (derive surfY, erosion, plate shift) assume each column is a solid stack
  // on mantle with a single partially filled top voxel: no floating rock, no buried air.
  it('every column: PERIDOTITE at y=0, contiguous solid, AIR above, only top voxel partial', () => {
    for (const s of [1, 42]) {
      const w = worlds.get(s)!;
      let bad = 0;
      for (let c = 0; c < NCOL; c++) {
        if (voxMat(w.vox[c]!) !== Mat.PERIDOTITE) bad++;
        let top = 0;
        while (top + 1 < NY && voxMat(w.vox[c + (top + 1) * NCOL]!) !== Mat.AIR) top++;
        for (let y = 0; y < NY; y++) {
          const v = w.vox[c + y * NCOL]!;
          const m = voxMat(v);
          if (m >= MAT_COUNT) bad++;
          if (y <= top) {
            if (m === Mat.AIR) bad++;
            const f = voxFill(v);
            if (y < top ? f !== 255 : f < 1) bad++;
          } else if (v !== 0) bad++; // AIR with no fill/age/flags
        }
        if (top >= NY - 1) bad++; // headroom for mountains/lava to grow
      }
      expect(bad).toBe(0);
    }
  });

  it('water is the initial fill to nominal sea level and all fields are finite (V7)', () => {
    const w = worlds.get(3)!;
    const surf = surface(w);
    for (let c = 0; c < NCOL; c++) {
      expect(Number.isFinite(w.crustAge[c]!)).toBe(true);
      const want = Math.max(0, Y_SEA_NOMINAL - surf[c]!);
      if (Math.abs(w.water[c]! - want) > 1e-4) expect(w.water[c]).toBeCloseTo(want, 4);
    }
    expect(Number.isFinite(w.mantleReservoir)).toBe(true);
  });
});

describe('worldgen land/ocean', () => {
  // V8 soak bounds (land 0.15..0.6, ocean ≥ 0.3) must be reachable from the start.
  it('land fraction (surface above nominal sea) ∈ [0.2, 0.5] across seeds', () => {
    for (const s of SEEDS) {
      const surf = surface(worlds.get(s)!);
      let land = 0;
      for (let c = 0; c < NCOL; c++) if (surf[c]! > Y_SEA_NOMINAL) land++;
      const f = land / NCOL;
      expect(f, `seed ${s}`).toBeGreaterThanOrEqual(0.2);
      expect(f, `seed ${s}`).toBeLessThanOrEqual(0.5);
    }
  });

  it('continental crust stands higher than oceanic (buoyancy), and old ocean floor is deeper', () => {
    for (const s of SEEDS) {
      const w = worlds.get(s)!;
      const surf = surface(w);
      let cs = 0, cn = 0, os = 0, on = 0;
      const abyss: [number, number][] = []; // [age, surf]
      for (let c = 0; c < NCOL; c++) {
        if (voxFlags(topVoxel(w, c)) & FLAG_CONTINENTAL) {
          cs += surf[c]!;
          cn++;
        } else {
          os += surf[c]!;
          on++;
          // abyssal plain only (away from shelves): thermal subsidence with age
          if (w.water[c]! > 10) abyss.push([w.crustAge[c]!, surf[c]!]);
        }
      }
      expect(cn / NCOL).toBeGreaterThan(0.25);
      expect(cn / NCOL).toBeLessThan(0.45);
      expect(cs / cn - os / on, `seed ${s}`).toBeGreaterThan(10);
      // youngest vs oldest quartile of abyssal floor
      abyss.sort((a, b) => a[0] - b[0]);
      const q = abyss.length >> 2;
      const meanSurf = (a: [number, number][]) => a.reduce((t, e) => t + e[1], 0) / a.length;
      expect(q).toBeGreaterThan(1000);
      expect(abyss[abyss.length - 1]![0], `seed ${s} oldest floor`).toBeGreaterThan(80);
      expect(meanSurf(abyss.slice(0, q)) - meanSurf(abyss.slice(-q)), `seed ${s}`).toBeGreaterThan(3);
    }
  });

  it('crust stacks: continents ~34-44 layers of granite + strata, oceans ~8-12 of basalt/gabbro', () => {
    // Isostasy (T17) derives elevation from thickness; wrong thickness makes the world jump on tick 1.
    const w = worlds.get(42)!;
    const surf = surface(w);
    const contT: number[] = [];
    const oceT: number[] = [];
    const seen = new Set<number>();
    for (let c = 0; c < NCOL; c++) {
      let t = 0;
      for (let y = 0; y < NY; y++) {
        const m = voxMat(w.vox[c + y * NCOL]!);
        if (m !== Mat.AIR && m !== Mat.PERIDOTITE) t++;
        seen.add(m);
      }
      const cont = (voxFlags(topVoxel(w, c)) & FLAG_CONTINENTAL) !== 0;
      if (cont && surf[c]! > Y_SEA_NOMINAL + 6) contT.push(t); // inland
      if (!cont && w.water[c]! > 12) oceT.push(t); // abyssal
    }
    const med = (a: number[]) => a.sort((x, y) => x - y)[a.length >> 1]!;
    expect(med(contT)).toBeGreaterThanOrEqual(34);
    expect(med(contT)).toBeLessThanOrEqual(44);
    expect(med(oceT)).toBeGreaterThanOrEqual(8);
    expect(med(oceT)).toBeLessThanOrEqual(12);
    for (const m of [Mat.GRANITE, Mat.GNEISS, Mat.GABBRO, Mat.BASALT, Mat.SANDSTONE, Mat.SHALE, Mat.LIMESTONE, Mat.SEDIMENT]) {
      expect(seen.has(m), `material ${m}`).toBe(true);
    }
  });

  it('mantleReservoir starts at 5% of crust mass (budget buffer for V3/V16)', () => {
    const w = worlds.get(2)!;
    expect(w.mantleReservoir / crustMass(w.vox)).toBeCloseTo(MANTLE_RESERVOIR_FRAC, 6);
    expect(Number.isInteger(w.mantleReservoir)).toBe(true); // GPU counters are integer fill units (V3)
    expect(w.mantleReservoir).toBeGreaterThan(0);
  });
});

describe('worldgen torus seam (V1)', () => {
  // A non-periodic field shows as a cliff / straight plate border along x=0 or z=0.
  it('surface step across the wrap edge matches interior neighbour statistics', () => {
    for (const s of SEEDS) {
      const surf = surface(worlds.get(s)!);
      let inSum = 0, inN = 0, inMax = 0, seamSum = 0, seamN = 0, seamMax = 0;
      for (let z = 0; z < NZ; z++) {
        for (let x = 0; x < NX; x++) {
          const h = surf[x + z * NX]!;
          const dx = Math.abs(surf[((x + 1) % NX) + z * NX]! - h);
          const dz = Math.abs(surf[x + ((z + 1) % NZ) * NX]! - h);
          if (x === NX - 1) { seamSum += dx; seamN++; seamMax = Math.max(seamMax, dx); }
          else { inSum += dx; inN++; inMax = Math.max(inMax, dx); }
          if (z === NZ - 1) { seamSum += dz; seamN++; seamMax = Math.max(seamMax, dz); }
          else { inSum += dz; inN++; inMax = Math.max(inMax, dz); }
        }
      }
      const inMean = inSum / inN;
      const seamMean = seamSum / seamN;
      expect(seamMean, `seed ${s}`).toBeLessThan(2 * inMean);
      expect(seamMax, `seed ${s}`).toBeLessThanOrEqual(inMax);
    }
  });

  it('plate borders cross the wrap edge no more often than interior lines', () => {
    for (const s of SEEDS) {
      const pid = worlds.get(s)!.plateId;
      let inDiff = 0, inN = 0, seamDiff = 0, seamN = 0;
      for (let z = 0; z < NZ; z++) {
        for (let x = 0; x < NX; x++) {
          const d = pid[x + z * NX] !== pid[((x + 1) % NX) + z * NX] ? 1 : 0;
          if (x === NX - 1) { seamDiff += d; seamN++; } else { inDiff += d; inN++; }
          const e = pid[x + z * NX] !== pid[x + ((z + 1) % NZ) * NX] ? 1 : 0;
          if (z === NZ - 1) { seamDiff += e; seamN++; } else { inDiff += e; inN++; }
        }
      }
      expect(seamDiff / seamN, `seed ${s}`).toBeLessThanOrEqual(3 * (inDiff / inN) + 0.02);
    }
  });
});

describe('worldgen plates (V6)', () => {
  it('alive count ∈ [3, 12], every plateId references an alive plate', () => {
    for (const s of SEEDS) {
      const w = worlds.get(s)!;
      const alive = w.plates.filter((p) => p.alive);
      expect(alive.length).toBeGreaterThanOrEqual(3);
      expect(alive.length).toBeLessThanOrEqual(12);
      for (let c = 0; c < NCOL; c++) {
        const p = w.plates[w.plateId[c]!];
        if (!p?.alive) expect.fail(`column ${c} references dead plate ${w.plateId[c]}`);
      }
    }
  });

  it('requested plate count is clamped to [3, 12] and honoured exactly within range', () => {
    const count = (n: number) => generateWorld(5, { plates: n }).plates.filter((p) => p.alive).length;
    expect(count(1)).toBe(3);
    expect(count(12)).toBe(12);
    expect(count(40)).toBe(12);
    expect(() => generateWorld(5, { plates: Number.NaN })).toThrow(RangeError);
  });

  it('plates move at 1-4 cells/My (≈2-8 cm/yr at 20 km cells) with zero sub-cell accum', () => {
    // Much slower and tectonics stalls within a 5000 My soak (V8 needs a full Wilson cycle).
    for (const s of SEEDS) {
      for (const p of worlds.get(s)!.plates.filter((q) => q.alive)) {
        const speed = Math.hypot(p.vel[0], p.vel[1]);
        expect(speed).toBeGreaterThanOrEqual(PLATE_SPEED_MIN - 1e-9);
        expect(speed).toBeLessThanOrEqual(PLATE_SPEED_MAX + 1e-9);
        expect(p.accum).toEqual([0, 0]);
        expect(p.age).toBeGreaterThan(0);
      }
    }
  });
});
