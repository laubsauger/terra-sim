// Stylized per-material look: saturated but tasteful diorama colours. Indexed by Mat id (layout.ts).
import * as THREE from 'three/webgpu';
import { uniformArray } from 'three/tsl';
import { Mat, MAT_COUNT } from '../sim/layout';

interface MatLook {
  color: number;     // sRGB hex
  rough: number;
  band: number;      // strength of fine strata banding on cut faces (0..~0.3)
  emissive?: number; // sRGB hex, multiplied by emissiveGain
  emissiveGain?: number;
}

export const MAT_LOOK: Record<number, MatLook> = {
  [Mat.AIR]:        { color: 0x0c0a09, rough: 1, band: 0 },       // cavities (drained magma chambers) on the cut
  [Mat.BASALT]:     { color: 0x4a5264, rough: 0.8, band: 0.22 },    // dark slate-blue, thin flows
  [Mat.GABBRO]:     { color: 0x46544b, rough: 0.75, band: 0.1 },    // dark green-grey
  [Mat.GRANITE]:    { color: 0xae928a, rough: 0.7, band: 0.04 },    // warm pink-grey
  [Mat.ANDESITE]:   { color: 0x7d7388, rough: 0.8, band: 0.06 },    // violet grey
  [Mat.SEDIMENT]:   { color: 0xcfae7c, rough: 0.95, band: 0.05 },   // tan
  [Mat.SANDSTONE]:  { color: 0xc3905c, rough: 0.9, band: 0.09 },    // muted ochre
  [Mat.SHALE]:      { color: 0x676b72, rough: 0.8, band: 0.1 },     // muted blue-grey slate
  [Mat.LIMESTONE]:  { color: 0xcdbf9f, rough: 0.75, band: 0.06 },   // cream-beige
  [Mat.SCHIST]:     { color: 0x7f8580, rough: 0.55, band: 0.16 },   // silvery banded grey-green
  [Mat.GNEISS]:     { color: 0x979089, rough: 0.6, band: 0.2 },     // banded warm grey
  [Mat.MAGMA]:      { color: 0xc8401a, rough: 0.4, band: 0.1, emissive: 0xff6c14, emissiveGain: 4.5 },  // HDR: incandescent orange, blooms
  [Mat.PERIDOTITE]: { color: 0x3b332c, rough: 0.8, band: 0.04, emissive: 0xff6a20, emissiveGain: 0.65 },  // deep warm grey-brown, glows at depth
};

// Surface / biome placeholder colours (sRGB hex).
export const BIOME = {
  grassLush: 0x4aa843,
  grassDry: 0x9fb84a,
  forest: 0x3f7f3a,
  soil: 0x6b4a33,
  sand: 0xdcc28e,
  snow: 0xf4f7ff,
  wetSand: 0xb89a66,
  waterShallow: 0x2fe0cf,
  waterDeep: 0x0d63a8,
  waterSide: 0x2a86c9,
  mantleGlow: 0xff4a12,
} as const;

const lin = (hex: number) => new THREE.Color(hex); // Color() converts sRGB hex to linear working space

function table<T>(f: (l: MatLook) => T): T[] {
  const out: T[] = [];
  for (let i = 0; i < MAT_COUNT; i++) out.push(f(MAT_LOOK[i]!));
  return out;
}

type V3 = THREE.Node<'vec3'>;
type F = THREE.Node<'float'>;
type Idx = THREE.Node<'uint'> | THREE.Node<'int'>;

/** GPU lookup tables indexed by a mat id node. */
export function createPaletteNodes() {
  const color = uniformArray(table((l) => lin(l.color)), 'color');
  const rough = uniformArray(table((l) => l.rough), 'float');
  const band = uniformArray(table((l) => l.band), 'float');
  const emissive = uniformArray(table((l) => lin(l.emissive ?? 0).multiplyScalar(l.emissiveGain ?? 0)), 'color');
  // uniformArray element typings are untyped strings; pin them once here.
  return {
    color: (m: Idx) => color.element(m) as unknown as V3,
    rough: (m: Idx) => rough.element(m) as unknown as F,
    band: (m: Idx) => band.element(m) as unknown as F,
    emissive: (m: Idx) => emissive.element(m) as unknown as V3,
  };
}
export type PaletteNodes = ReturnType<typeof createPaletteNodes>;

export const biomeColor = (k: keyof typeof BIOME) => lin(BIOME[k]);

// ---- Biome ground look (sim 'biome' ids, src/sim/biomeModel.ts; 11-13 are the climate agent's
// STEPPE, SHRUBLAND, COLD_DESERT). Indexed by id; unknown ids fall back to grassland. ----
interface BiomeLook {
  ground: number;  // bare ground / soil colour (sRGB)
  veg: number;     // vegetation cover colour at veg = 1 (sRGB)
  arid?: number;   // 0..1: steep slopes show layered sandstone mesa / canyon walls
  forest?: number; // 0..1: canopy clumps (dark bumps)
  snow?: number;   // 0..1: permanent snow / ice cover
}
export const BIOME_SLOTS = 16;
const GRASSLAND_LOOK: BiomeLook = { ground: 0x9c8a5a, veg: 0x5aa844 };
export const BIOME_LOOK: Record<number, BiomeLook> = {
  0: { ground: 0xc2b08a, veg: 0xc2b08a },                       // OCEAN (seabed; water shades it)
  1: { ground: 0xeef3fa, veg: 0xeef3fa, snow: 1 },              // ICE
  2: { ground: 0x7d7360, veg: 0x7c8455 },                       // TUNDRA: muted green-brown
  3: { ground: 0x5a5040, veg: 0x22472c, forest: 1 },            // TAIGA: dark conifer green
  4: { ground: 0x6b5a3e, veg: 0x2f6c2a, forest: 0.9 },          // TEMPERATE_FOREST
  5: GRASSLAND_LOOK,                                            // GRASSLAND: lush green
  6: { ground: 0xe2b27a, veg: 0xc2a462, arid: 1 },              // DESERT: warm sand, red rock
  7: { ground: 0xc39a5a, veg: 0xa9a444, arid: 0.4, forest: 0.25 }, // SAVANNA: golden green
  8: { ground: 0x4a3e2a, veg: 0x1d6a2c, forest: 1 },            // RAINFOREST: deep emerald
  9: { ground: 0xcfae78, veg: 0xc4aa74 },                       // BEACH
  10: { ground: 0x8c8578, veg: 0x7a8a4e, snow: 0.35 },          // ALPINE: rock + meadow, patchy snow
  11: { ground: 0xa8954f, veg: 0xc2a64e, arid: 0.5 },           // STEPPE: golden khaki grass (never bare sand)
  12: { ground: 0xa27447, veg: 0x77875a, arid: 0.7 },           // SHRUBLAND: sage over ochre soil
  13: { ground: 0xa9a08c, veg: 0x8f8b70, arid: 0.8 },           // COLD_DESERT: pale grey-tan gravel
};

/** Canyon-wall palette for arid cliffs, indexed by Mat id (sedimentary layers read as red/ochre bands). */
const DESERT_STRATA: Record<number, number> = {
  [Mat.SANDSTONE]: 0xc8683a, [Mat.SHALE]: 0x8c4f45, [Mat.LIMESTONE]: 0xe8cf9e, [Mat.SEDIMENT]: 0xd79a5c,
  [Mat.GRANITE]: 0xb07a60, [Mat.BASALT]: 0x5a4640, [Mat.ANDESITE]: 0x8a5a4a,
};

export function createBiomeNodes() {
  const look = (i: number) => BIOME_LOOK[i] ?? GRASSLAND_LOOK;
  const n = [...Array(BIOME_SLOTS).keys()];
  const ground = uniformArray(n.map((i) => lin(look(i).ground)), 'color');
  const veg = uniformArray(n.map((i) => lin(look(i).veg)), 'color');
  // x arid, y forest, z snow
  // vec4 (not vec3): keeps the uniform array stride unambiguous (16 B per element)
  const flags = uniformArray(n.map((i) => new THREE.Vector4(look(i).arid ?? 0, look(i).forest ?? 0, look(i).snow ?? 0, 0)), 'vec4');
  const strata = uniformArray(table((l) => l).map((_, i) => lin(DESERT_STRATA[i] ?? 0xa86a48)), 'color');
  return {
    ground: (b: Idx) => ground.element(b) as unknown as V3,
    veg: (b: Idx) => veg.element(b) as unknown as V3,
    flags: (b: Idx) => (flags.element(b) as unknown as THREE.Node<'vec4'>).xyz as unknown as V3,
    strata: (m: Idx) => strata.element(m) as unknown as V3,
  };
}
export type BiomeNodes = ReturnType<typeof createBiomeNodes>;
