// Hand-set biome/veg fields for life tests (life only reads them, so they need not come from the sim).
import { NX, NZ, NCOL } from '../../../src/sim/layout';
import { Biome, BIOME, BIOME_VEG_TARGET, classifyBiome } from '../../../src/sim/biomeModel';
import type { GpuFields } from '../../../src/core/gpu';

/** Veg each biome relaxes toward; ids 11-13 (steppe, shrubland, cold desert) are not in biomeModel yet. */
export const vegTarget = (b: number) => BIOME_VEG_TARGET[b] ?? ({ 11: 0.3, 12: 0.45, 13: 0.08 } as Record<number, number>)[b] ?? 0;

/** Z bands (32 rows each) of the stripes world, land columns only. */
export const STRIPES = [Biome.TEMPERATE_FOREST, Biome.DESERT, Biome.TAIGA, Biome.GRASSLAND, Biome.SAVANNA, Biome.RAINFOREST, Biome.TUNDRA, Biome.ICE];
export const ALPINE_ALT = 22;

function write(fields: GpuFields, biome: Uint32Array, veg: Float32Array): void {
  (fields.cpuArray('biome') as Uint32Array).set(biome);
  (fields.cpuArray('veg') as Float32Array).set(veg);
  fields.markDirty('biome');
  fields.markDirty('veg');
}

/** Crisp test layout: ocean where water ≥ 1, alpine high up, else a biome per z band; veg at target. */
export function paintStripes(fields: GpuFields, surfY: Float32Array, water: Float32Array, sea: number): Uint32Array {
  const biome = new Uint32Array(NCOL), veg = new Float32Array(NCOL);
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const c = x + z * NX;
    const b = water[c]! >= BIOME.OCEAN_MIN ? Biome.OCEAN : surfY[c]! - sea > ALPINE_ALT ? Biome.ALPINE : STRIPES[z >> 5]!;
    biome[c] = b;
    veg[c] = vegTarget(b);
  }
  write(fields, biome, veg);
  return biome;
}

/** Replace one biome by another everywhere (sim evolving), veg to the new target. */
export function repaint(fields: GpuFields, biome: Uint32Array, from: number, to: number): void {
  const veg = new Float32Array(NCOL);
  for (let c = 0; c < NCOL; c++) { if (biome[c] === from) biome[c] = to; veg[c] = vegTarget(biome[c]!); }
  write(fields, biome, veg);
}

/**
 * Plausible-looking biomes for screenshots: temperature from latitude (warm at the z seam, cold
 * mid-block, like climate.ts) and lapse, moisture from distance to water plus noise, then the
 * sim's own Whittaker classifier.
 */
export function paintNatural(fields: GpuFields, surfY: Float32Array, water: Float32Array, sea: number): Uint32Array {
  // distance to water (BFS, torus)
  const dist = new Int32Array(NCOL).fill(-1);
  const q: number[] = [];
  for (let c = 0; c < NCOL; c++) if (water[c]! >= 0.5) { dist[c] = 0; q.push(c); }
  for (let h = 0; h < q.length; h++) {
    const c = q[h]!, x = c % NX, z = (c / NX) | 0;
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const n = ((x + dx + NX) % NX) + ((z + dz + NZ) % NZ) * NX;
      if (dist[n] === -1) { dist[n] = dist[c]! + 1; q.push(n); }
    }
  }
  const biome = new Uint32Array(NCOL), veg = new Float32Array(NCOL);
  const TAU = Math.PI * 2;
  for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
    const c = x + z * NX;
    const alt = surfY[c]! - sea;
    const n1 = Math.sin((TAU * 3 * x) / NX + 1.7 * Math.sin((TAU * 2 * z) / NZ)) * Math.cos((TAU * 4 * z) / NZ + 0.6);
    const n2 = Math.sin((TAU * 7 * x) / NX + (TAU * 5 * z) / NZ + 2.1);
    const t = -6 + 36 * (0.5 + 0.5 * Math.cos((TAU * z) / NZ)) - 0.45 * Math.max(0, alt) + 4 * n1;
    const d = Math.max(0, dist[c]!);
    const m = 1.05 * Math.exp(-d / 28) + 0.3 * n2 + 0.18 * n1 + 0.1;
    const ice = t < -9 && alt > -2 ? 1 : 0;
    let b: number = classifyBiome(t, Math.max(0, m) * BIOME.P_REF, alt, water[c]!, ice);
    // the climate agent's new ids: cool dry grass → steppe, warm dry → shrubland, cold desert
    if (b === Biome.GRASSLAND && t < 10 && m < 0.35) b = 11;
    else if ((b === Biome.GRASSLAND || b === Biome.SAVANNA) && t > 14 && n1 > 0.25) b = 12;
    else if (b === Biome.DESERT && t < 12) b = 13;
    else if (b === Biome.TUNDRA && m < 0.12) b = 13;
    biome[c] = b;
    veg[c] = vegTarget(b);
  }
  write(fields, biome, veg);
  return biome;
}

/** Every land column (water < OCEAN_MIN) one biome, veg at its target: altitude effects in isolation. */
export function paintAll(fields: GpuFields, biome: Uint32Array, water: Float32Array, b: number): void {
  const veg = new Float32Array(NCOL);
  for (let c = 0; c < NCOL; c++) { if (water[c]! < BIOME.OCEAN_MIN) biome[c] = b; veg[c] = vegTarget(biome[c]!); }
  write(fields, biome, veg);
}
