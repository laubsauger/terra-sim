// Single param schema (V20): UI, URL, save and soak all read PARAM_DEFS.

export type ParamGroup = 'Sim' | 'Tectonics' | 'Climate' | 'Events' | 'Render';
export type ParamValue = number | boolean;

export interface ParamDef {
  key: string;
  group: ParamGroup;
  label: string;
  default: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  persist: boolean;
  /** UI uses a log-scale control (requires min > 0). */
  log?: boolean;
  /** URL param name if exposed. */
  url?: string;
  /** Boolean params only: URL string names for [false, true], e.g. ['low', 'high']. */
  urlEnum?: readonly [string, string];
}

const U32_MAX = 4294967295;

// Sea level is intentionally absent: it must emerge from the water budget (V4).
export const PARAM_DEFS: readonly ParamDef[] = [
  // Sim
  { key: 'seed', group: 'Sim', label: 'seed', default: 1, min: 0, max: U32_MAX, step: 1, persist: true, url: 'seed' },
  { key: 'speed', group: 'Sim', label: 'speed', default: 1, min: 0.001, max: 100, unit: 'My/s', persist: false, log: true, url: 'speed' },
  { key: 'autosaveMinutes', group: 'Sim', label: 'autosave', default: 5, min: 0, max: 60, step: 1, unit: 'min', persist: true },
  // Tectonics
  { key: 'initialPlates', group: 'Tectonics', label: 'initial plates', default: 7, min: 3, max: 12, step: 1, persist: true },
  { key: 'plateSpeed', group: 'Tectonics', label: 'plate speed', default: 1, min: 0, max: 4, step: 0.01, unit: '×', persist: true },
  { key: 'mantleHeat', group: 'Tectonics', label: 'mantle heat', default: 1, min: 0.25, max: 4, step: 0.01, unit: '×', persist: true },
  { key: 'volcanism', group: 'Tectonics', label: 'volcanism', default: 1, min: 0, max: 4, step: 0.01, unit: '×', persist: true },
  // Climate
  { key: 'rainfall', group: 'Climate', label: 'rainfall', default: 1, min: 0, max: 3, step: 0.01, unit: '×', persist: true },
  { key: 'erosionRate', group: 'Climate', label: 'erosion', default: 1, min: 0, max: 4, step: 0.01, unit: '×', persist: true },
  { key: 'thermalErosion', group: 'Climate', label: 'thermal erosion', default: 1, min: 0, max: 4, step: 0.01, unit: '×', persist: true },
  { key: 'iceAgeStrength', group: 'Climate', label: 'ice age strength', default: 1, min: 0, max: 2, step: 0.01, unit: '×', persist: true },
  // Events
  { key: 'eventRate', group: 'Events', label: 'event rate', default: 1, min: 0, max: 5, step: 0.01, unit: '×', persist: true },
  { key: 'meteorRate', group: 'Events', label: 'meteor rate', default: 1, min: 0, max: 5, step: 0.01, unit: '×', persist: true },
  // Render
  { key: 'verticalExaggeration', group: 'Render', label: 'vertical exag.', default: 1.5, min: 1, max: 8, step: 0.1, unit: '×', persist: true },
  { key: 'ambientMode', group: 'Render', label: 'ambient mode', default: false, persist: false, url: 'ambient' },
  { key: 'highQuality', group: 'Render', label: 'high quality', default: true, persist: false, url: 'quality', urlEnum: ['low', 'high'] },
];

function stepDecimals(step: number): number {
  const s = String(step);
  const i = s.indexOf('.');
  return i < 0 ? 0 : s.length - i - 1;
}

export class Params {
  private readonly _defs: readonly ParamDef[];
  private readonly byKey = new Map<string, ParamDef>();
  private readonly values = new Map<string, ParamValue>();
  private readonly listeners = new Set<(key: string, v: ParamValue) => void>();

  constructor(defs: readonly ParamDef[] = PARAM_DEFS) {
    this._defs = defs;
    for (const d of defs) {
      if (this.byKey.has(d.key)) throw new Error(`params: duplicate key '${d.key}'`);
      this.byKey.set(d.key, d);
      this.values.set(d.key, d.default);
    }
  }

  defs(): readonly ParamDef[] {
    return this._defs;
  }

  private def(key: string): ParamDef {
    const d = this.byKey.get(key);
    if (!d) throw new Error(`params: unknown key '${key}'`);
    return d;
  }

  get(key: string): ParamValue {
    this.def(key);
    return this.values.get(key)!;
  }

  /** Clamps to [min,max] and rounds to step. Throws on unknown key or wrong type. */
  set(key: string, v: ParamValue): void {
    const d = this.def(key);
    let nv: ParamValue;
    if (typeof d.default === 'boolean') {
      if (typeof v !== 'boolean') throw new Error(`params: '${key}' expects boolean, got ${typeof v}`);
      nv = v;
    } else {
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`params: '${key}' expects finite number, got ${String(v)}`);
      nv = normalizeNumber(d, v);
    }
    if (this.values.get(key) === nv) return;
    this.values.set(key, nv);
    for (const cb of this.listeners) cb(key, nv);
  }

  onChange(cb: (key: string, v: ParamValue) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Parses a location.search string by def.url. Invalid/out-of-range values are ignored (with a warning). */
  applyURL(search: string): void {
    const sp = new URLSearchParams(search);
    for (const d of this._defs) {
      if (!d.url || !sp.has(d.url)) continue;
      const raw = sp.get(d.url)!.trim();
      const v = parseURLValue(d, raw);
      if (v === undefined) {
        console.warn(`params: ignoring invalid url value ${d.url}=${raw}`);
        continue;
      }
      this.set(d.key, v);
    }
  }

  /** Only persist:true params. */
  toJSON(): Record<string, ParamValue> {
    const o: Record<string, ParamValue> = {};
    for (const d of this._defs) if (d.persist) o[d.key] = this.values.get(d.key)!;
    return o;
  }

  /** Persisted params reset to default then take values from o. Unknown / non-persisted keys ignored. */
  fromJSON(o: Record<string, unknown>): void {
    for (const d of this._defs) {
      if (!d.persist) continue;
      const v = Object.prototype.hasOwnProperty.call(o, d.key) ? o[d.key] : undefined;
      const valid =
        typeof d.default === 'boolean' ? typeof v === 'boolean' : typeof v === 'number' && Number.isFinite(v);
      if (v !== undefined && !valid) console.warn(`params: ignoring invalid saved value ${d.key}=${String(v)}`);
      this.set(d.key, valid ? (v as ParamValue) : d.default);
    }
  }
}

function normalizeNumber(d: ParamDef, v: number): number {
  let x = v;
  if (d.step !== undefined && d.step > 0) {
    x = Number((Math.round(x / d.step) * d.step).toFixed(stepDecimals(d.step)));
  }
  if (d.min !== undefined && x < d.min) x = d.min;
  if (d.max !== undefined && x > d.max) x = d.max;
  return x;
}

function parseURLValue(d: ParamDef, raw: string): ParamValue | undefined {
  if (typeof d.default === 'boolean') {
    const s = raw.toLowerCase();
    if (d.urlEnum) {
      if (s === d.urlEnum[0]) return false;
      if (s === d.urlEnum[1]) return true;
      return undefined;
    }
    if (s === '' || s === '1' || s === 'true') return true; // bare `?ambient` counts as on
    if (s === '0' || s === 'false') return false;
    return undefined;
  }
  if (raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  if (d.step !== undefined && Number.isInteger(d.step) && !Number.isInteger(n)) return undefined;
  if ((d.min !== undefined && n < d.min) || (d.max !== undefined && n > d.max)) return undefined;
  return n;
}
