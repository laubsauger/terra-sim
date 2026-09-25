// §T.50 save/load (V13, V21, I.file, I.idb) + §T.35 autosave rollback (V7).
//
// `.terra` file, little-endian:
//   0  'TERA'          magic, 4 ASCII bytes
//   4  version u16     TERRA_VERSION; any other value is rejected before anything is touched (V13)
//   6  flags u16       0
//   8  seed u32        world seed (params 'seed')
//   12 headerLen u32   UTF-8 byte length of the JSON header
//   16 geoTime f64     My
//   24 header JSON     TerraHeader: rngState, params, sim (controller/CPU state), matCount, field table
//   24+headerLen       gzip(field bytes concatenated in field-table order)
// Field table entries {name, type, count, byteOffset, byteLength} index the *decompressed* payload.
//
// Generic over GpuFields: every registered field except SAVE_EXCLUDE is saved, so fields other passes add
// later are picked up automatically. Only the current ping-pong buffer is stored; load writes it as parity 0.
// Load compatibility: a saved field that is not registered → reject; a registered field missing from the
// save → zero-filled + warning (older saves stay loadable as fields are added). Bump TERRA_VERSION only
// when the meaning of existing bytes changes (grid size, packing, field semantics).
//
// Consistency: capture holds the sim (Sim.hold: runTicks returns 0) across every async read, after
// Sim.settle() has landed the in-flight window readback; that readback travels in the sim state, so the
// window after a load applies exactly what an unsaved run applies. Load validates everything (header,
// field table, sim state) before mutating anything: a rejected file leaves the world untouched.
import type * as THREE from 'three/webgpu';
import type { FieldType, GpuFields } from './gpu';
import type { Params, ParamValue } from './params';
import type { RngState } from './rng';
import type { NanGuard } from './nanGuard';
import type { Sim, SimState } from '../sim/sim';
import { MAT_COUNT } from '../sim/layout';

export const TERRA_MAGIC = 'TERA';
export const TERRA_VERSION = 1;
const PREFIX_BYTES = 24;

/**
 * Scratch fields: fully rewritten before every read within a tick, so their content between ticks is
 * meaningless. Not saved, not NaN-checked. When adding a scratch field, list it here; forgetting only
 * makes saves bigger. Listing a field that is *not* scratch breaks V13 (save.spec hash test catches it
 * for fields that feed the hash). Candidates from the magma work, to add once verified: magColTmp,
 * arcTmp, lavaOut.
 */
export const SAVE_EXCLUDE: ReadonlySet<string> = new Set(['waterTmp', 'tecAct', 'sedRatio', 'talusOut', 'climScratch', 'probe', 'volcanoTmp', 'magFocus']); // magma temps (magColTmp, arcTmp, lavaOut, slabIn) may carry across ticks: saved

const STRIDE: Record<FieldType, number> = { float: 4, uint: 4, int: 4, vec2: 8, vec4: 16, uvec2: 8, uvec4: 16, ivec2: 8 };

export class SaveError extends Error { override name = 'SaveError'; }
export class NanDetected extends Error {
  override name = 'NanDetected';
  constructor(readonly bits: number, readonly fields: string[]) { super(`non-finite values in ${fields.join(', ') || '?'}`); }
}

export interface FieldEntry { name: string; type: FieldType; count: number; byteOffset: number; byteLength: number }

export interface TerraHeader {
  app: 'terra-sim';
  savedAt: number;              // ms since epoch
  /** Materials known to the writer (V21: ids are append-only, so fewer is fine, more is not). */
  matCount: number;
  /** Sim PCG32 state (copy of sim.rng, spec I.file). */
  rngState: RngState;
  params: Record<string, ParamValue>;
  /** Controller / CPU sim state: SimState (sim.ts). */
  sim: unknown;
  fields: FieldEntry[];
  rawBytes: number;
}

export interface TerraFile { version: number; seed: number; geoTime: number; header: TerraHeader; payload: ArrayBuffer }

// ---- gzip (CompressionStream; browsers and Node ≥ 18) -------------------------------------------------

export async function gzip(parts: BlobPart[]): Promise<Blob> {
  return new Response(new Blob(parts).stream().pipeThrough(new CompressionStream('gzip'))).blob();
}

export async function gunzip(data: Blob): Promise<ArrayBuffer> {
  try {
    return await new Response(data.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  } catch (e) {
    throw new SaveError(`save file payload is corrupt (gzip: ${(e as Error).message})`);
  }
}

// ---- format -----------------------------------------------------------------------------------------

export interface FieldData { name: string; type: FieldType; count: number; data: ArrayBuffer }

export async function encodeTerra(meta: { seed: number; geoTime: number; header: Omit<TerraHeader, 'fields' | 'rawBytes'> },
  fields: FieldData[]): Promise<Blob> {
  const table: FieldEntry[] = [];
  let off = 0;
  for (const f of fields) {
    const want = f.count * STRIDE[f.type];
    if (f.data.byteLength !== want) throw new SaveError(`field '${f.name}': ${f.data.byteLength} bytes, expected ${want}`);
    table.push({ name: f.name, type: f.type, count: f.count, byteOffset: off, byteLength: want });
    off += want;
  }
  const header: TerraHeader = { ...meta.header, fields: table, rawBytes: off };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const pre = new DataView(new ArrayBuffer(PREFIX_BYTES));
  for (let i = 0; i < 4; i++) pre.setUint8(i, TERRA_MAGIC.charCodeAt(i));
  pre.setUint16(4, TERRA_VERSION, true);
  pre.setUint16(6, 0, true);
  pre.setUint32(8, meta.seed >>> 0, true);
  pre.setUint32(12, json.byteLength, true);
  pre.setFloat64(16, meta.geoTime, true);
  const body = await gzip(fields.map((f) => f.data));
  return new Blob([pre.buffer, json, body], { type: 'application/octet-stream' });
}

/** Parse and validate prefix + header (no decompression). Throws SaveError with a user-facing message. */
export async function readTerraHeader(blob: Blob): Promise<Omit<TerraFile, 'payload'> & { headerLen: number }> {
  if (blob.size < PREFIX_BYTES) throw new SaveError('not a .terra file (too short)');
  const pre = new DataView(await blob.slice(0, PREFIX_BYTES).arrayBuffer());
  const magic = String.fromCharCode(pre.getUint8(0), pre.getUint8(1), pre.getUint8(2), pre.getUint8(3));
  if (magic !== TERRA_MAGIC) throw new SaveError('not a .terra file (bad magic)');
  const version = pre.getUint16(4, true);
  if (version !== TERRA_VERSION) {
    throw new SaveError(`save format version ${version} is not supported (this build reads version ${TERRA_VERSION}); nothing was loaded`);
  }
  const seed = pre.getUint32(8, true), headerLen = pre.getUint32(12, true), geoTime = pre.getFloat64(16, true);
  if (PREFIX_BYTES + headerLen > blob.size) throw new SaveError('save file truncated (header)');
  let header: TerraHeader;
  try {
    header = JSON.parse(new TextDecoder().decode(await blob.slice(PREFIX_BYTES, PREFIX_BYTES + headerLen).arrayBuffer())) as TerraHeader;
  } catch {
    throw new SaveError('save file header is corrupt (JSON)');
  }
  if (!header || header.app !== 'terra-sim' || !Array.isArray(header.fields) || typeof header.rawBytes !== 'number') {
    throw new SaveError('save file header is incomplete');
  }
  if (typeof header.matCount !== 'number' || header.matCount > MAT_COUNT) {
    throw new SaveError(`save uses ${header.matCount} materials, this build knows ${MAT_COUNT} (V21: newer save?)`);
  }
  if (!Number.isFinite(geoTime)) throw new SaveError('save file geoTime is not finite');
  return { version, seed, geoTime, header, headerLen };
}

export async function decodeTerra(blob: Blob): Promise<TerraFile> {
  const h = await readTerraHeader(blob);
  const payload = await gunzip(blob.slice(PREFIX_BYTES + h.headerLen));
  if (payload.byteLength !== h.header.rawBytes) {
    throw new SaveError(`save file payload is ${payload.byteLength} bytes, header says ${h.header.rawBytes}`);
  }
  return { version: h.version, seed: h.seed, geoTime: h.geoTime, header: h.header, payload };
}

export interface FieldRegistry { names(): string[]; type(name: string): FieldType; count(name: string): number }

export interface RestorePlan { load: FieldEntry[]; missing: string[] }

/**
 * Match a save's field table against the registry. Throws on a saved field that is not registered or that
 * changed type/size. Registered, non-excluded fields absent from the save come back in `missing` (zero-fill).
 * Saved fields that are now excluded (became scratch) are ignored.
 */
export function planRestore(entries: FieldEntry[], reg: FieldRegistry, rawBytes: number,
  exclude: ReadonlySet<string> = SAVE_EXCLUDE): RestorePlan {
  const known = new Set(reg.names());
  const unknown = entries.filter((e) => !known.has(e.name)).map((e) => e.name);
  if (unknown.length) throw new SaveError(`save contains field(s) this build does not have: ${unknown.join(', ')}; nothing was loaded`);
  const load: FieldEntry[] = [];
  const seen = new Set<string>();
  for (const e of entries) {
    if (seen.has(e.name)) throw new SaveError(`save lists field '${e.name}' twice`);
    seen.add(e.name);
    if (exclude.has(e.name)) continue;
    const type = reg.type(e.name), count = reg.count(e.name);
    if (e.type !== type || e.count !== count) {
      throw new SaveError(`field '${e.name}' is ${e.type}×${e.count} in the save but ${type}×${count} here; nothing was loaded`);
    }
    if (e.byteLength !== count * STRIDE[type] || e.byteOffset < 0 || e.byteOffset + e.byteLength > rawBytes) {
      throw new SaveError(`field '${e.name}' has a bad byte range in the save`);
    }
    load.push(e);
  }
  const missing = [...known].filter((n) => !exclude.has(n) && !seen.has(n));
  return { load, missing };
}

export function savedFieldNames(fields: FieldRegistry, exclude: ReadonlySet<string> = SAVE_EXCLUDE): string[] {
  return fields.names().filter((n) => !exclude.has(n));
}

// ---- capture / restore (GPU) ---------------------------------------------------------------------------

export interface WorldRefs {
  renderer: THREE.WebGPURenderer;
  fields: GpuFields;
  sim: Sim;
  params: Params;
  /** Geo clock to set on load (DualClock). Optional in tests. */
  clock?: { geoTime: number };
  /** When given, every capture is NaN-checked in the same held window; a flagged capture throws NanDetected. */
  guard?: NanGuard;
  exclude?: ReadonlySet<string>;
}

export interface Snapshot { blob: Blob; tick: number; geoTime: number; seed: number; savedAt: number }

/** Snapshot the world between ticks. The sim is held from settle() until the last read has landed. */
export async function captureWorld(w: WorldRefs): Promise<Snapshot> {
  const { renderer, fields, sim, params } = w;
  const names = savedFieldNames(fields, w.exclude);
  const release = sim.hold();
  let datas: ArrayBuffer[], state: SimState, bits = 0;
  try {
    await sim.settle();
    const nan = w.guard?.check(renderer);
    datas = await Promise.all(names.map((n) => fields.read(renderer, n)));
    bits = (await nan) ?? 0;
    state = sim.getState();
  } finally {
    release();
  }
  if (bits) throw new NanDetected(bits, w.guard!.decode(bits));
  const seed = params.get('seed') as number;
  const geoTime = sim.geoMy;
  const savedAt = Date.now();
  const blob = await encodeTerra(
    { seed, geoTime, header: { app: 'terra-sim', savedAt, matCount: MAT_COUNT, rngState: state.rng, params: params.toJSON(), sim: state } },
    names.map((n, i) => ({ name: n, type: fields.type(n), count: fields.count(n), data: datas[i]! })),
  );
  return { blob, tick: state.tick, geoTime, seed, savedAt };
}

/**
 * Load a snapshot into the running world. Everything is decoded and validated first; on any error nothing
 * has changed. Returns warnings (e.g. zero-filled fields). Do not overlap with captureWorld (SaveManager
 * serialises its operations).
 */
export async function restoreWorld(w: WorldRefs, blob: Blob): Promise<{ warnings: string[]; tick: number; file: Omit<TerraFile, 'payload'> }> {
  const { renderer, fields, params } = w;
  const sim: Sim = w.sim;
  const file = await decodeTerra(blob);
  const plan = planRestore(file.header.fields, fields, file.header.rawBytes, w.exclude);
  const state = file.header.sim;
  sim.checkState(state);
  if (JSON.stringify(state.rng) !== JSON.stringify(file.header.rngState)) throw new SaveError('save header rngState disagrees with sim state (corrupt)');
  if (typeof file.header.params !== 'object' || file.header.params === null) throw new SaveError('save file params missing');
  const warnings: string[] = [];
  if (plan.missing.length) warnings.push(`fields not in save, zero-filled: ${plan.missing.join(', ')}`);

  // ---- no throws past this point (V13: no partial load) ----
  const release = sim.hold();
  try {
    const touched: string[] = [];
    for (const e of plan.load) {
      const src = new Uint8Array(file.payload, e.byteOffset, e.byteLength);
      writeField(fields, e.name, (dst) => dst.set(src));
      touched.push(e.name);
    }
    for (const n of plan.missing) { writeField(fields, n, (dst) => dst.fill(0)); touched.push(n); }
    sim.loadState(state);
    params.fromJSON(file.header.params);
    if (w.clock) w.clock.geoTime = file.geoTime;
    flushUploads(renderer, fields, touched);
  } finally {
    release();
  }
  for (const m of warnings) console.warn(`load: ${m}`);
  const { payload: _p, ...meta } = file;
  return { warnings, tick: state.tick, file: meta };
}

function buffersOf(fields: GpuFields, name: string) {
  try { return fields.pair<'float'>(name); } catch { return [fields.cur<'float'>(name)]; }
}

/** Write the same bytes into every buffer of a field (ping-pong: both), parity 0, mark for upload. */
function writeField(fields: GpuFields, name: string, fill: (dst: Uint8Array) => void): void {
  const n = buffersOf(fields, name).length as 1 | 2;
  fields.setParity(name, 0);
  for (let i = 0; i < n; i++) {
    const a = fields.cpuArray(name, i as 0 | 1);
    fill(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  }
  fields.markDirty(name);
}

/**
 * Upload marked fields now instead of at their next binding, so a read right after a load (hash, save,
 * probe) sees the loaded data even for fields no kernel has bound yet. Uses three's internal attribute
 * cache (no public API for this in r186); without it the lazy upload at next binding still applies.
 */
function flushUploads(renderer: THREE.WebGPURenderer, fields: GpuFields, names: string[]): void {
  const attrs = (renderer as unknown as { _attributes?: { update(a: object, type: number): void } })._attributes;
  const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } };
  if (!attrs) return;
  const STORAGE = 3; // AttributeType.STORAGE (three/src/renderers/common/Constants.js)
  for (const n of names) for (const b of buffersOf(fields, n)) if (backend.get(b.value).buffer) attrs.update(b.value, STORAGE);
}

// ---- IndexedDB (I.idb) ------------------------------------------------------------------------------

export const IDB_NAME = 'terra-sim';
export const IDB_STORE = 'saves';
export const AUTOSAVE_SLOTS = 3;
export const MANUAL_SLOTS = 8;

export interface SlotMeta { key: string; kind: 'auto' | 'manual'; savedAt: number; tick: number; geoTime: number; seed: number; bytes: number }
export interface SlotRecord extends SlotMeta { blob: Blob }

const req = <T>(r: IDBRequest<T>) => new Promise<T>((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export class SaveStore {
  private constructor(private db: IDBDatabase) {}

  /** Throws when IndexedDB is unavailable (private mode, blocked storage). */
  static async open(): Promise<SaveStore> {
    if (typeof indexedDB === 'undefined') throw new SaveError('IndexedDB unavailable');
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains(IDB_STORE)) r.result.createObjectStore(IDB_STORE, { keyPath: 'key' }); };
    return new SaveStore(await req(r));
  }

  private tx(mode: IDBTransactionMode) { return this.db.transaction(IDB_STORE, mode).objectStore(IDB_STORE); }

  async put(rec: SlotRecord): Promise<void> { await req(this.tx('readwrite').put(rec)); }
  async get(key: string): Promise<SlotRecord | undefined> { return req(this.tx('readonly').get(key)) as Promise<SlotRecord | undefined>; }
  async delete(key: string): Promise<void> { await req(this.tx('readwrite').delete(key)); }
  /** Newest first. */
  async list(): Promise<SlotMeta[]> {
    const all = (await req(this.tx('readonly').getAll())) as SlotRecord[];
    return all.map(({ blob: _b, ...m }) => m).sort((a, b) => b.savedAt - a.savedAt);
  }
}

/** Next slot of a ring: first unused key, else the oldest. */
export function nextRingKey(prefix: string, n: number, existing: SlotMeta[]): string {
  const keys = Array.from({ length: n }, (_, i) => `${prefix}${i}`);
  const have = new Map(existing.map((m) => [m.key, m.savedAt]));
  return keys.find((k) => !have.has(k)) ?? keys.reduce((a, b) => (have.get(a)! <= have.get(b)! ? a : b));
}

// ---- SaveManager: autosave ring, rolling last-good snapshot, NaN guard + rollback (V7) ------------------

export interface NanEvent { at: number; tick: number; fields: string[]; restoredTick: number | null; paused: boolean }

export interface SaveManagerOptions {
  /** Sim stats window in ticks (STATS_WINDOW). */
  windowTicks: number;
  /** Rolling in-memory snapshot every K windows (default 25 = 1000 ticks)… */
  snapshotEveryWindows?: number;
  /** …but at most once per this many real ms (default 30 s; each one reads back the whole world). */
  snapshotMinMs?: number;
  /** NaN check whenever the sim advanced and this many real ms passed (default 250; V7 wants ≤ 1 s). */
  checkMs?: number;
  notice?: (msg: string, level: 'info' | 'warn' | 'error') => void;
  /** Pause the geo clock (NaN recurred after rollback, or nothing to roll back to). */
  pause?: () => void;
}

export interface SaveStatus {
  idb: boolean;
  autosaveMinutes: number;
  lastAutosave: SlotMeta | null;
  lastGoodTick: number | null;
  busy: boolean;
  nanEvents: number;
}

export class SaveManager {
  readonly log: NanEvent[] = [];
  /** Latest verified-finite snapshot in memory (rolling, autosave, manual save or load). At most one. */
  private lastGood: Snapshot | null = null;
  private lastAutosave: SlotMeta | null = null;
  private chain: Promise<unknown> = Promise.resolve();
  private queued = 0;
  private lastAutosaveMs: number;
  private snapWindow = -Infinity;
  private checkedTick = -1;
  private lastCheckMs = 0;
  private lastSnapMs = -Infinity;
  private lastRollbackTo: number | null = null;
  private readonly snapEvery: number;
  private readonly checkMs: number;
  private readonly snapMinMs: number;

  constructor(private w: WorldRefs & { guard: NanGuard }, readonly store: SaveStore | null, private opts: SaveManagerOptions) {
    this.snapEvery = opts.snapshotEveryWindows ?? 25;
    this.checkMs = opts.checkMs ?? 250;
    this.snapMinMs = opts.snapshotMinMs ?? 30_000;
    this.lastAutosaveMs = performance.now();
  }

  status(): SaveStatus {
    return {
      idb: !!this.store, autosaveMinutes: this.w.params.get('autosaveMinutes') as number,
      lastAutosave: this.lastAutosave, lastGoodTick: this.lastGood?.tick ?? null, busy: this.queued > 0, nanEvents: this.log.length,
    };
  }

  /** Serialise every save/load/check; errors are reported, never swallowed silently. */
  private run<T>(label: string, fn: () => Promise<T>): Promise<T> {
    this.queued++;
    const p = this.chain.then(async () => {
      try { return await fn(); } finally { this.queued--; }
    });
    this.chain = p.catch((e: unknown) => {
      console.error(`${label} failed:`, e);
      this.opts.notice?.(`${label} failed: ${(e as Error).message}`, 'error');
    });
    return p;
  }

  /** Resolves when every queued task has finished (tests, shutdown). */
  async idle(): Promise<void> { while (this.queued > 0) await this.chain; }

  /** Call once per frame after sim.runTicks. Schedules at most one background task at a time. */
  frame(now = performance.now()): void {
    if (this.queued > 0) return;
    const tick = this.w.sim.tick;
    const win = Math.floor(tick / this.opts.windowTicks);
    const mins = this.w.params.get('autosaveMinutes') as number;
    if (mins > 0 && now - this.lastAutosaveMs >= mins * 60_000) {
      this.lastAutosaveMs = now;
      void this.run('autosave', () => this.autosave()).catch(() => {});
    } else if (win - this.snapWindow >= this.snapEvery && now - this.lastSnapMs >= this.snapMinMs) {
      this.snapWindow = win; this.lastSnapMs = now;
      void this.run('snapshot', () => this.snapshot()).catch(() => {});
    } else if (tick !== this.checkedTick && now - this.lastCheckMs >= this.checkMs) {
      this.checkedTick = tick; this.lastCheckMs = now;
      void this.check().catch(() => {});
    }
  }

  /** Guard check (queued behind other save tasks); on NaN/Inf roll back. Resolves true when the world was finite. */
  check(): Promise<boolean> {
    return this.run('NaN check', () => this.guarded(async () => {
      const b = await this.w.guard.check(this.w.renderer);
      if (b) throw new NanDetected(b, this.w.guard.decode(b));
    }));
  }

  private async guarded(fn: () => Promise<void>): Promise<boolean> {
    try { await fn(); return true; } catch (e) {
      if (!(e instanceof NanDetected)) throw e;
      await this.rollback(e);
      return false;
    }
  }

  private async capture(): Promise<Snapshot> {
    const s = await captureWorld(this.w); // throws NanDetected → caller rolls back
    this.lastGood = s; // replaces the previous one: ≤ 1 retained snapshot + 1 transient (V10)
    return s;
  }

  private async snapshot(): Promise<void> { await this.guarded(async () => { await this.capture(); }); }

  private async autosave(): Promise<void> {
    await this.guarded(async () => {
      const s = await this.capture();
      if (!this.store) return; // memory-only autosave (lastGood)
      const key = nextRingKey('auto-', AUTOSAVE_SLOTS, (await this.store.list()).filter((m) => m.kind === 'auto'));
      const rec = record(key, 'auto', s);
      await this.store.put(rec);
      const { blob: _b, ...meta } = rec;
      this.lastAutosave = meta;
    });
  }

  private async rollback(e: NanDetected): Promise<void> {
    const ev: NanEvent = { at: Date.now(), tick: this.w.sim.tick, fields: e.fields, restoredTick: null, paused: false };
    this.log.push(ev);
    if (this.log.length > 50) this.log.shift();
    console.error(`NaN guard (V7): ${e.message} at tick ${ev.tick}`);
    const target = await this.rollbackTarget();
    if (!target) {
      this.opts.pause?.(); ev.paused = true;
      this.opts.notice?.(`NaN/Inf in ${e.fields.join(', ')}: no snapshot to roll back to, geo clock paused`, 'error');
      return;
    }
    await restoreWorld(this.w, target.blob);
    ev.restoredTick = target.tick;
    this.lastGood = target;
    this.snapWindow = Math.floor(target.tick / this.opts.windowTicks);
    this.checkedTick = -1;
    // deterministic sim: same state + params reproduce the same NaN; do not loop forever
    if (this.lastRollbackTo === target.tick) {
      this.opts.pause?.(); ev.paused = true;
      this.opts.notice?.(`NaN/Inf recurred after rollback to ${target.geoTime.toFixed(1)} My: geo clock paused (change params, then resume)`, 'error');
    } else {
      this.opts.notice?.(`NaN/Inf in ${e.fields.join(', ')}: rolled back to ${target.geoTime.toFixed(1)} My`, 'warn');
    }
    this.lastRollbackTo = target.tick;
  }

  /** Newest verified snapshot: in-memory last-good or the latest IDB autosave. */
  private async rollbackTarget(): Promise<Snapshot | null> {
    let best = this.lastGood;
    if (this.store) {
      try {
        const auto = (await this.store.list()).find((m) => m.kind === 'auto');
        if (auto && (!best || auto.tick > best.tick)) {
          const r = await this.store.get(auto.key);
          if (r) best = { blob: r.blob, tick: r.tick, geoTime: r.geoTime, seed: r.seed, savedAt: r.savedAt };
        }
      } catch (err) { console.warn('rollback: IDB read failed, using in-memory snapshot', err); }
    }
    return best;
  }

  // ---- user actions (UI, URL, window.terra) ----

  /** Save to a manual slot (default: first free of MANUAL_SLOTS, else overwrite the oldest). */
  saveSlot(key?: string): Promise<SlotMeta> {
    return this.run('save', async () => {
      if (!this.store) throw new SaveError('IndexedDB unavailable: use export file');
      const s = await this.captureOrRollback();
      const k = key ?? nextRingKey('slot-', MANUAL_SLOTS, (await this.store.list()).filter((m) => m.kind === 'manual'));
      const rec = record(k, k.startsWith('auto-') ? 'auto' : 'manual', s);
      await this.store.put(rec);
      const { blob: _b, ...meta } = rec;
      this.opts.notice?.(`saved ${k} (${meta.geoTime.toFixed(1)} My)`, 'info');
      return meta;
    });
  }

  /** Load an IDB slot; 'latest' = newest record of any kind. */
  loadSlot(key: string): Promise<{ warnings: string[] }> {
    return this.run(`load ${key}`, async () => {
      if (!this.store) throw new SaveError('IndexedDB unavailable');
      const k = key === 'latest' ? (await this.store.list())[0]?.key : key;
      const r = k ? await this.store.get(k) : undefined;
      if (!r) throw new SaveError(`no save in slot '${key}'`);
      return this.loadSnapshot({ blob: r.blob, tick: r.tick, geoTime: r.geoTime, seed: r.seed, savedAt: r.savedAt }, k!);
    });
  }

  /** Load a .terra Blob/File (import). */
  loadBlob(blob: Blob, label = 'file'): Promise<{ warnings: string[] }> {
    return this.run(`load ${label}`, async () => {
      const h = await readTerraHeader(blob);
      const tick = (h.header.sim as { tick?: number })?.tick ?? 0;
      return this.loadSnapshot({ blob, tick, geoTime: h.geoTime, seed: h.seed, savedAt: h.header.savedAt }, label);
    });
  }

  private async loadSnapshot(s: Snapshot, label: string): Promise<{ warnings: string[] }> {
    const res = await restoreWorld(this.w, s.blob);
    this.lastGood = s;
    this.lastRollbackTo = null;
    this.snapWindow = Math.floor(res.tick / this.opts.windowTicks);
    this.checkedTick = -1;
    for (const m of res.warnings) this.opts.notice?.(`loaded ${label}: ${m}`, 'warn');
    if (!res.warnings.length) this.opts.notice?.(`loaded ${label} (${s.geoTime.toFixed(1)} My)`, 'info');
    return { warnings: res.warnings };
  }

  /** Capture for export / window.terra.save(). */
  exportFile(): Promise<{ blob: Blob; filename: string }> {
    return this.run('export', async () => {
      const s = await this.captureOrRollback();
      return { blob: s.blob, filename: `terra-${s.seed}-${s.geoTime.toFixed(1)}My.terra` };
    });
  }

  async listSlots(): Promise<SlotMeta[]> { return this.store ? this.store.list() : []; }

  private async captureOrRollback(): Promise<Snapshot> {
    try { return await this.capture(); } catch (e) {
      if (e instanceof NanDetected) { await this.rollback(e); throw new SaveError('world had NaN/Inf and was rolled back; not saved'); }
      throw e;
    }
  }
}

function record(key: string, kind: 'auto' | 'manual', s: Snapshot): SlotRecord {
  return { key, kind, savedAt: s.savedAt, tick: s.tick, geoTime: s.geoTime, seed: s.seed, bytes: s.blob.size, blob: s.blob };
}
