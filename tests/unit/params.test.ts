import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PARAM_DEFS, Params, type ParamDef } from '../../src/core/params';

const DEFS: readonly ParamDef[] = [
  { key: 'a', group: 'Sim', label: 'a', default: 1, min: 0, max: 2, step: 0.1, persist: true, url: 'a' },
  { key: 'n', group: 'Sim', label: 'n', default: 5, min: 0, max: 10, step: 1, persist: true, url: 'n' },
  { key: 'speed', group: 'Sim', label: 'speed', default: 1, min: 0.001, max: 100, persist: false, log: true, url: 'speed' },
  { key: 'flag', group: 'Render', label: 'flag', default: false, persist: false, url: 'ambient' },
  { key: 'q', group: 'Render', label: 'q', default: true, persist: false, url: 'quality', urlEnum: ['low', 'high'] },
];

describe('Params', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it('clamps to range and rounds to step so UI/url/save cannot push the sim out of its tuned envelope', () => {
    const p = new Params(DEFS);
    p.set('a', 5);
    expect(p.get('a')).toBe(2);
    p.set('a', -3);
    expect(p.get('a')).toBe(0);
    p.set('a', 0.349);
    expect(p.get('a')).toBe(0.3); // no float noise like 0.30000000000000004
    p.set('n', 3.6);
    expect(p.get('n')).toBe(4);
    p.set('speed', 0.0001);
    expect(p.get('speed')).toBe(0.001);
  });

  it('rejects wrong types instead of silently coercing', () => {
    const p = new Params(DEFS);
    expect(() => p.set('a', true)).toThrow();
    expect(() => p.set('a', NaN)).toThrow();
    expect(() => p.set('flag', 1)).toThrow();
  });

  it('throws on unknown key (fail loud: typos must not become silent no-ops)', () => {
    const p = new Params(DEFS);
    expect(() => p.get('nope')).toThrow(/unknown key/);
    expect(() => p.set('nope', 1)).toThrow(/unknown key/);
  });

  it('parses URL params by def.url and ignores invalid values', () => {
    const p = new Params(DEFS);
    p.applyURL('?a=1.5&n=7&speed=12.5&ambient=1&quality=low');
    expect(p.get('a')).toBe(1.5);
    expect(p.get('n')).toBe(7);
    expect(p.get('speed')).toBe(12.5);
    expect(p.get('flag')).toBe(true);
    expect(p.get('q')).toBe(false);

    const p2 = new Params(DEFS);
    p2.applyURL('?a=abc&n=2.5&speed=1e9&ambient=maybe&quality=ultra&unknown=3&speed=');
    expect(p2.get('a')).toBe(1);
    expect(p2.get('n')).toBe(5); // integer param rejects fractional
    expect(p2.get('speed')).toBe(1); // out of range ignored, not clamped
    expect(p2.get('flag')).toBe(false);
    expect(p2.get('q')).toBe(true);
    expect(warn).toHaveBeenCalled();
  });

  it('bare boolean url flag (?ambient) turns it on', () => {
    const p = new Params(DEFS);
    p.applyURL('?ambient');
    expect(p.get('flag')).toBe(true);
  });

  it('JSON contains only persisted params and roundtrips', () => {
    const p = new Params(DEFS);
    p.set('a', 0.7);
    p.set('n', 9);
    p.set('speed', 50);
    p.set('flag', true);
    const json = p.toJSON();
    expect(json).toEqual({ a: 0.7, n: 9 });

    const q = new Params(DEFS);
    q.set('speed', 3);
    q.fromJSON(JSON.parse(JSON.stringify(json)) as Record<string, unknown>);
    expect(q.get('a')).toBe(0.7);
    expect(q.get('n')).toBe(9);
    // non-persisted params are not touched by a load
    expect(q.get('speed')).toBe(3);
    expect(q.get('flag')).toBe(false);
  });

  it('fromJSON ignores unknown keys; missing keys fall back to default', () => {
    const p = new Params(DEFS);
    p.set('n', 1);
    p.set('a', 0.2);
    p.fromJSON({ a: 1.9, bogus: 42, speed: 99 });
    expect(p.get('a')).toBe(1.9);
    expect(p.get('n')).toBe(5);
    expect(p.get('speed')).toBe(1);
  });

  it('onChange fires on real changes only, and unsubscribes', () => {
    const p = new Params(DEFS);
    const seen: [string, number | boolean][] = [];
    const off = p.onChange((k, v) => seen.push([k, v]));
    p.set('a', 1.2);
    p.set('a', 1.2); // no change
    p.set('a', 1.21); // rounds to 1.2, no change
    p.set('flag', true);
    expect(seen).toEqual([
      ['a', 1.2],
      ['flag', true],
    ]);
    off();
    p.set('a', 0.5);
    expect(seen).toHaveLength(2);
  });
});

describe('PARAM_DEFS schema', () => {
  it('has unique keys and url names', () => {
    const keys = PARAM_DEFS.map((d) => d.key);
    expect(new Set(keys).size).toBe(keys.length);
    const urls = PARAM_DEFS.flatMap((d) => (d.url ? [d.url] : []));
    expect(new Set(urls).size).toBe(urls.length);
  });

  it('every numeric default lies within [min,max] and on step; types are consistent', () => {
    for (const d of PARAM_DEFS) {
      if (typeof d.default === 'boolean') {
        expect(d.min, d.key).toBeUndefined();
        expect(d.max, d.key).toBeUndefined();
        continue;
      }
      expect(d.min, d.key).toBeDefined();
      expect(d.max, d.key).toBeDefined();
      expect(d.min!, d.key).toBeLessThan(d.max!);
      expect(d.default, d.key).toBeGreaterThanOrEqual(d.min!);
      expect(d.default, d.key).toBeLessThanOrEqual(d.max!);
      if (d.log) expect(d.min!, `${d.key} log needs min>0`).toBeGreaterThan(0);
      // default must survive normalisation unchanged, else the UI would snap it on first touch
      const p = new Params(PARAM_DEFS);
      p.set(d.key, d.default);
      expect(p.get(d.key), d.key).toBe(d.default);
    }
  });

  it('exposes the url params from SPEC §I.url and has no sea-level knob (must emerge from W_total)', () => {
    const urls = PARAM_DEFS.flatMap((d) => (d.url ? [d.url] : []));
    for (const u of ['seed', 'speed', 'ambient', 'quality']) expect(urls).toContain(u);
    expect(PARAM_DEFS.some((d) => /sea/i.test(d.key))).toBe(false);
  });

  it('seed accepts full u32 range via url', () => {
    const p = new Params();
    p.applyURL('?seed=4294967295');
    expect(p.get('seed')).toBe(4294967295);
  });
});
