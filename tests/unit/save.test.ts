// T50 save format (pure half). GPU capture/restore + hash roundtrip live in tests/gpu/save.spec.ts.
import { describe, test, expect } from 'vitest';
import {
  gzip, gunzip, encodeTerra, decodeTerra, readTerraHeader, planRestore, nextRingKey, SaveError,
  TERRA_VERSION, SAVE_EXCLUDE, type FieldData, type FieldRegistry, type TerraHeader,
} from '../../src/core/save';
import type { FieldType } from '../../src/core/gpu';
import { MAT_COUNT } from '../../src/sim/layout';

const header = (over: Partial<TerraHeader> = {}): Omit<TerraHeader, 'fields' | 'rawBytes'> => ({
  app: 'terra-sim', savedAt: 1, matCount: MAT_COUNT, rngState: { state: '0000000000000001', inc: '0000000000000003' },
  params: { seed: 9 }, sim: { tick: 7 }, ...over,
});

function fieldsFixture(): FieldData[] {
  const water = new Float32Array(64).map((_, i) => i * 0.25);
  const vox = new Uint32Array(128).map((_, i) => (i * 2654435761) >>> 0);
  const flux = new Float32Array(16 * 4).fill(-1.5);
  return [
    { name: 'water', type: 'float', count: 64, data: water.buffer },
    { name: 'vox', type: 'uint', count: 128, data: vox.buffer },
    { name: 'flux', type: 'vec4', count: 16, data: flux.buffer },
  ];
}

const registry = (defs: Record<string, [FieldType, number]>): FieldRegistry => ({
  names: () => Object.keys(defs),
  type: (n) => defs[n]![0],
  count: (n) => defs[n]![1],
});

async function patchU16(blob: Blob, offset: number, v: number): Promise<Blob> {
  const b = new Uint8Array(await blob.arrayBuffer());
  new DataView(b.buffer).setUint16(offset, v, true);
  return new Blob([b]);
}

describe('gzip', () => {
  // saves are ~40 MB raw; compression must be lossless and actually shrink the mostly-uniform voxel grid
  test('roundtrip is byte-exact and compresses redundant data', async () => {
    const rand = new Uint8Array(4096).map((_, i) => (i * 131 + 7) ^ (i >> 3));
    const zeros = new Uint8Array(1 << 20);
    const z = await gzip([rand, zeros]);
    expect(z.size).toBeLessThan(zeros.byteLength / 50);
    const back = new Uint8Array(await gunzip(z));
    expect(back.byteLength).toBe(rand.byteLength + zeros.byteLength);
    expect(back.subarray(0, rand.byteLength)).toEqual(rand);
    expect(back.subarray(rand.byteLength).every((x) => x === 0)).toBe(true);
  });

  test('corrupt payload is a SaveError, not a crash', async () => {
    await expect(gunzip(new Blob([new Uint8Array([1, 2, 3, 4, 5])]))).rejects.toBeInstanceOf(SaveError);
  });
});

describe('.terra format', () => {
  test('encode → decode keeps header scalars, JSON and every field byte', async () => {
    const f = fieldsFixture();
    const blob = await encodeTerra({ seed: 0xdeadbeef, geoTime: 123.45, header: header() }, f);
    const d = await decodeTerra(blob);
    expect(d.version).toBe(TERRA_VERSION);
    expect(d.seed).toBe(0xdeadbeef);
    expect(d.geoTime).toBe(123.45);
    expect(d.header.params).toEqual({ seed: 9 });
    expect(d.header.sim).toEqual({ tick: 7 });
    expect(d.header.fields.map((e) => e.name)).toEqual(['water', 'vox', 'flux']);
    for (const [i, e] of d.header.fields.entries()) {
      expect(new Uint8Array(d.payload, e.byteOffset, e.byteLength)).toEqual(new Uint8Array(f[i]!.data));
    }
    const magic = new TextDecoder().decode(new Uint8Array(await blob.slice(0, 4).arrayBuffer()));
    expect(magic).toBe('TERA');
  });

  // V13: a different format version must be refused up front with a message, never partially applied
  test('version mismatch is rejected with a message', async () => {
    const blob = await patchU16(await encodeTerra({ seed: 1, geoTime: 0, header: header() }, fieldsFixture()), 4, TERRA_VERSION + 1);
    await expect(readTerraHeader(blob)).rejects.toThrow(/version .* not supported/);
    await expect(decodeTerra(blob)).rejects.toBeInstanceOf(SaveError);
  });

  test('bad magic and truncation are rejected', async () => {
    const good = await encodeTerra({ seed: 1, geoTime: 0, header: header() }, fieldsFixture());
    const bad = new Uint8Array(await good.arrayBuffer()); bad[0] = 0x58;
    await expect(decodeTerra(new Blob([bad]))).rejects.toThrow(/magic/);
    await expect(decodeTerra(good.slice(0, 30))).rejects.toBeInstanceOf(SaveError);
    await expect(decodeTerra(good.slice(0, good.size - 8))).rejects.toBeInstanceOf(SaveError);
  });

  // V21: ids are append-only, so an older save (fewer materials) is fine but a newer one is not
  test('a save with more materials than this build is rejected', async () => {
    const blob = await encodeTerra({ seed: 1, geoTime: 0, header: header({ matCount: MAT_COUNT + 1 }) }, fieldsFixture());
    await expect(readTerraHeader(blob)).rejects.toThrow(/materials/);
    const older = await encodeTerra({ seed: 1, geoTime: 0, header: header({ matCount: MAT_COUNT - 2 }) }, fieldsFixture());
    await expect(readTerraHeader(older)).resolves.toBeTruthy();
  });

  test('field byte length must match type × count', async () => {
    await expect(encodeTerra({ seed: 1, geoTime: 0, header: header() }, [{ name: 'x', type: 'vec2', count: 4, data: new ArrayBuffer(16) }]))
      .rejects.toThrow(/expected 32/);
  });
});

describe('planRestore', () => {
  const entries = async () => (await decodeTerra(await encodeTerra({ seed: 1, geoTime: 0, header: header() }, fieldsFixture()))).header;

  // a field this build does not know cannot be put anywhere; loading the rest would be a partial load
  test('unknown saved field → reject naming it', async () => {
    const h = await entries();
    expect(() => planRestore(h.fields, registry({ water: ['float', 64], vox: ['uint', 128] }), h.rawBytes)).toThrow(/flux/);
  });

  // how older saves stay loadable when later tasks register new fields (e.g. magma)
  test('registered field missing from the save → listed for zero-fill, rest loads', async () => {
    const h = await entries();
    const plan = planRestore(h.fields, registry({ water: ['float', 64], vox: ['uint', 128], flux: ['vec4', 16], magCol: ['uvec2', 64] }), h.rawBytes);
    expect(plan.missing).toEqual(['magCol']);
    expect(plan.load.map((e) => e.name)).toEqual(['water', 'vox', 'flux']);
  });

  test('scratch fields are neither required nor loaded', async () => {
    const h = await entries();
    const reg = registry({ water: ['float', 64], vox: ['uint', 128], flux: ['vec4', 16], waterTmp: ['vec2', 64] });
    expect(SAVE_EXCLUDE.has('waterTmp')).toBe(true);
    expect(planRestore(h.fields, reg, h.rawBytes).missing).toEqual([]);
    // a field that became scratch since the save is skipped, not an error
    const plan = planRestore(h.fields, reg, h.rawBytes, new Set(['flux']));
    expect(plan.load.map((e) => e.name)).toEqual(['water', 'vox']);
  });

  test('type or size change of a field is rejected (grid or packing changed)', async () => {
    const h = await entries();
    expect(() => planRestore(h.fields, registry({ water: ['float', 32], vox: ['uint', 128], flux: ['vec4', 16] }), h.rawBytes)).toThrow(/water/);
    expect(() => planRestore(h.fields, registry({ water: ['float', 64], vox: ['int', 128], flux: ['vec4', 16] }), h.rawBytes)).toThrow(/vox/);
  });
});

test('slot rings fill free slots first, then overwrite the oldest', () => {
  const m = (key: string, savedAt: number) => ({ key, kind: 'auto' as const, savedAt, tick: 0, geoTime: 0, seed: 0, bytes: 0 });
  expect(nextRingKey('auto-', 3, [])).toBe('auto-0');
  expect(nextRingKey('auto-', 3, [m('auto-0', 5), m('auto-2', 1)])).toBe('auto-1');
  expect(nextRingKey('auto-', 3, [m('auto-0', 5), m('auto-1', 9), m('auto-2', 1)])).toBe('auto-2');
});
