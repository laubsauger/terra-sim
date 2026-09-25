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
  [Mat.AIR]:        { color: 0xff00ff, rough: 1, band: 0 },
  [Mat.BASALT]:     { color: 0x5a6478, rough: 0.85, band: 0.12 },   // dark slate
  [Mat.GABBRO]:     { color: 0x5d6e66, rough: 0.8, band: 0.06 },    // green-black
  [Mat.GRANITE]:    { color: 0xc49d96, rough: 0.7, band: 0.04 },    // warm pink-grey
  [Mat.ANDESITE]:   { color: 0x7d7388, rough: 0.8, band: 0.06 },    // violet grey
  [Mat.SEDIMENT]:   { color: 0xcfae7c, rough: 0.95, band: 0.05 },   // tan
  [Mat.SANDSTONE]:  { color: 0xdb9a4c, rough: 0.9, band: 0.12 },    // ochre
  [Mat.SHALE]:      { color: 0x6b7d96, rough: 0.8, band: 0.14 },    // blue-grey
  [Mat.LIMESTONE]:  { color: 0xeee0bb, rough: 0.75, band: 0.07 },   // cream
  [Mat.SCHIST]:     { color: 0x8c948f, rough: 0.55, band: 0.22 },   // silvery banded grey-green
  [Mat.GNEISS]:     { color: 0xa9a19c, rough: 0.6, band: 0.3 },     // banded warm grey
  [Mat.MAGMA]:      { color: 0xc8401a, rough: 0.4, band: 0.1, emissive: 0xff5a14, emissiveGain: 2.2 },
  [Mat.PERIDOTITE]: { color: 0x77883a, rough: 0.8, band: 0.05, emissive: 0xff3a0a, emissiveGain: 1.0 },
};

// Surface / biome placeholder colours (sRGB hex).
export const BIOME = {
  grassLush: 0x4fae45,
  grassDry: 0x9fb84a,
  forest: 0x3f7f3a,
  soil: 0x6b4a33,
  sand: 0xecd49a,
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
