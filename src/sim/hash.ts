// State hash for determinism checks (V2, V13). FNV-1a 32-bit over field bytes, in name order.
import type * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';

export function fnv1a(bytes: Uint8Array, h = 0x811c9dc5): number {
  for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]!; h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

export const HASH_FIELDS = ['vox', 'plateId', 'crustAge', 'water', 'sedSusp', 'vapor', 'ice', 'veg', 'magCol', 'arc', 'lava', 'mantle', 'crustTemp', 'magmaCtr', 'counters'] as const;

export async function hashFields(renderer: THREE.WebGPURenderer, fields: GpuFields, names: readonly string[] = HASH_FIELDS): Promise<string> {
  let h = 0x811c9dc5;
  for (const n of names) h = fnv1a(new Uint8Array(await fields.read(renderer, n)), h);
  return h.toString(16).padStart(8, '0');
}
