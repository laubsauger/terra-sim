// Sim orchestrator: owns passes, runs geo ticks in SPEC §C pass order, keeps readbacks deterministic.
import type * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { PCG32 } from '../core/rng';
import { createDerivePass } from './derive';
import { Tectonics } from './tectonics';
import { updateKinematics, initPlateMotion } from './kinematics';
import { Lifecycle } from './lifecycle';
import { createHydroPass, type HydroPass } from './hydro';
import { createErosionPass, type ErosionPass } from './erosion';
import { createClimatePass, type ClimatePass } from './climate';
import { createBiomePass, type BiomePass } from './biome';
import { createWorldStats, parseWorldStats, type WorldStats } from './worldStats';
import { CTR_QUAKE } from './fields';
import { WilsonController } from './wilson';
import { Diagenesis } from './diagenesis';
import { OceanLevel } from './oceanLevel';
import { GlacialErosion } from './glacial';
import { GodTools } from './godTools';
import { MantlePass } from './mantle';
import { MagmaPass } from './magma';
import { LavaPass } from './lava';
import { LAVA } from './magmaModel';
import { PLUME, type Plume } from './mantleModel';
import { colIdx, NX, NZ } from './layout';
import { EventScheduler, iceAgeForcing, type GeoEvent, type EventKind } from './events';
import type { WorldData } from './worldData';
// save/load (T50)
import type { RngState } from '../core/rng';
import { MAX_PLATES, type Plate } from './worldData';
import type { LifecycleState } from './lifecycle';
import type { WilsonState } from './wilson';
import type { EventSchedulerState } from './events';

/** Ticks per stats window. Readback issued at window end must land before the next window ends (V2). */
export const STATS_WINDOW = 40;
/** Isostasy runs every N ticks (on tectonics ticks). */
export const ISO_EVERY = 4;
/** Pipe-model substeps per geo tick (quasi-steady flow). */
export const HYDRO_SUBSTEPS = 2;
/** Reservoir debt (fill units) over which collided crust stacking fades from 1 to 0 (≈ 1 layer/col). */
export const RES_GATE_UNITS = 255 * 65536;
/** Erosion runs every N ticks with kGeo scaled by N. */
export const EROSION_EVERY = 2;

export class Sim {
  tick = 0;
  readonly tectonics: Tectonics;
  readonly lifecycle: Lifecycle;
  readonly hydro: HydroPass;
  readonly erosion: ErosionPass;
  readonly climate: ClimatePass;
  readonly biome: BiomePass;
  private worldStats: THREE.ComputeNode;
  private diagenesis: Diagenesis;
  private ocean: OceanLevel;
  private glacial: GlacialErosion;
  readonly god: GodTools;
  readonly mantle: MantlePass;
  readonly magma: MagmaPass;
  readonly lava: LavaPass;
  /** Plate a god-tool split asked for; consumed at the next window end ahead of the Wilson rift. */
  private godSplit: number | undefined;
  stats: WorldStats | null = null;
  /** Last window's earthquake sites (columns) and activity; FX layers sample quakes from these. */
  quakeSites: { subduct: { x: number; z: number; events: number } | null; collide: { x: number; z: number; events: number } | null } = { subduct: null, collide: null };
  readonly wilson = new WilsonController();
  readonly events: EventScheduler;
  /** Handlers for event kinds owned by other passes (magma, god tools). Unhandled kinds are counted, loudly. */
  readonly handlers = new Map<EventKind, (e: GeoEvent) => void>();
  unhandledEvents = 0;
  iceAges: { start: number; durationMy: number; magnitude: number }[] = [];
  private derive;
  private rng: PCG32;
  private pending: Promise<ArrayBuffer> | null = null;
  private pendingResult: ArrayBuffer | null = null;
  private runsInWindow = 0;
  private prevRuns = 0;
  private kGeoBase: number;
  /** Latest counters snapshot (reservoir + plate stats), for UI. */
  lastCounters: Int32Array | null = null;

  constructor(private renderer: THREE.WebGPURenderer, private fields: GpuFields, world: WorldData,
    private params: Params, readonly dtGeo: number) {
    this.derive = createDerivePass(fields);
    this.tectonics = new Tectonics(fields, world.plates);
    initPlateMotion(this.tectonics.plates);
    this.lifecycle = new Lifecycle(fields);
    this.hydro = createHydroPass(fields, params);
    this.climate = createClimatePass(fields, params);
    this.erosion = createErosionPass(fields, params, { seaLevel: this.climate.uniforms.seaLevel });
    this.biome = createBiomePass(fields);
    this.worldStats = createWorldStats(fields);
    this.diagenesis = new Diagenesis(fields);
    this.ocean = new OceanLevel(fields);
    this.glacial = new GlacialErosion(fields);
    this.god = new GodTools(fields);
    this.mantle = new MantlePass(fields, params, world);
    this.magma = new MagmaPass(fields, params);
    this.lava = new LavaPass(fields);
    this.kGeoBase = this.erosion.uniforms.kGeo.value;
    this.erosion.uniforms.kGeo.value = this.kGeoBase * EROSION_EVERY;
    this.rng = new PCG32(world.seed, 0x7ec70);
    // magma scans worldgen chambers, then mantle/crust temperatures start from fresh surfY
    this.magma.init(renderer);
    this.derive.run(renderer);
    this.mantle.init(renderer);
    this.events = new EventScheduler(world.seed);
    this.handlers.set('iceAge', (e) => this.iceAges.push({ start: e.my, durationMy: e.durationMy ?? 20, magnitude: e.magnitude }));
    // god tools + random meteors share handlers (V16); voxel edits need a derive afterwards
    this.handlers.set('uplift', (e) => { this.god.uplift(renderer, e.x, e.z, e.radius ?? 10, e.magnitude); this.derive.run(renderer); });
    this.handlers.set('meteor', (e) => { this.god.meteor(renderer, e.x, e.z, e.radius ?? 4 + 8 * e.magnitude, 1, e.dir); this.derive.run(renderer); });
    this.handlers.set('storm', (e) => { this.god.rainStorm(renderer, e.x, e.z, e.radius ?? 20, e.magnitude); });
    this.handlers.set('split', (e) => { this.godSplit = e.magnitude; });
    // volcanism events: lava drawn from the reservoir (V16); new plumes fixed in the mantle frame
    this.handlers.set('volcano', (e) => { this.lava.inject(renderer, colIdx(e.x, e.z), Math.round(255 * 6 * e.magnitude), LAVA.tBasalt); });
    this.handlers.set('floodBasalt', (e) => {
      for (let i = 0; i < 9; i++) this.lava.inject(renderer, colIdx(e.x + (i % 3) * 3 - 3, e.z + Math.floor(i / 3) * 3 - 3), Math.round(255 * 5 * e.magnitude), LAVA.tBasalt);
    });
    this.handlers.set('hotspot', (e) => {
      const p: Plume = { x: e.x, z: e.z, r: PLUME.radius[0] + (PLUME.radius[1] - PLUME.radius[0]) * Math.min(1, e.magnitude / 1.5), age: 0, life: PLUME.life[0] };
      this.mantle.plumes.plumes.push(p);
    });
    this.derive.run(renderer);
  }

  get geoMy(): number { return this.tick * this.dtGeo; }

  /**
   * Run up to n ticks. Returns ticks actually run: stalls (deterministically) when the previous
   * window's readback has not arrived yet, instead of using stale or missing data.
   */
  runTicks(n: number): number {
    if (this.holds > 0) return 0; // save/rollback in progress (hold())
    let ran = 0;
    for (; ran < n; ran++) {
      const next = this.tick + 1;
      if (next % STATS_WINDOW === 0 && this.pending && !this.pendingResult) break; // stall
      this.step(next);
    }
    return ran;
  }

  private step(t: number): void {
    const r = this.renderer;
    // god-tool events apply at the next tick boundary (deterministic order, same handler path as random ones)
    if (this.events.queue.length) this.dispatch(this.events.drain());
    // 3-4 tectonics + isostasy
    if (this.tectonics.tick(r, t, this.dtGeo, { speedMul: this.params.get('plateSpeed') as number, isoEvery: ISO_EVERY })) {
      this.runsInWindow++;
      this.derive.run(r);
      this.tectonics.flow(r);
      this.lifecycle.clean(r); // stray single cells rejoin the surrounding plate
      this.derive.run(r);
      this.magma.afterTectonics(r);
    }
    // 2 mantle + crust heat, 5 magma, 6 lava; they edit vox in place → derive every tick
    this.mantle.step(r, t, this.dtGeo);
    this.magma.step(r, t, this.dtGeo);
    this.lava.step(r, t, this.dtGeo);
    this.derive.run(r);
    // 7 climate (evap/precip/ice into water), 8 hydrology + erosion (erosion needs fresh surfY)
    this.climate.step(r);
    this.ocean.step(r); // open ocean levels quickly; pipe model handles rivers, lakes, coasts
    this.hydro.step(r, HYDRO_SUBSTEPS);
    if (t % EROSION_EVERY === 0) {
      this.erosion.uniforms.tick.value = t;
      this.erosion.step(r);
      this.glacial.step(r, t, (this.params.get('erosionRate') as number) * EROSION_EVERY);
      this.derive.run(r);
    }
    // 9 diagenesis / metamorphism / aging (mass-neutral material changes)
    this.diagenesis.step(r, t, this.dtGeo);
    // 10 derived: biome + vegetation
    this.biome.step(r);
    this.tick = t;
    if (t % STATS_WINDOW === 0) this.windowEnd();
  }

  private windowEnd(): void {
    const r = this.renderer;
    const plates = this.tectonics.plates;
    // apply stats from the previous window (issued STATS_WINDOW ticks ago)
    if (this.pendingResult) {
      const buf = this.pendingResult;
      this.lastCounters = new Int32Array(buf);
      const life = Lifecycle.parseStats(buf);
      const my = this.geoMy, windowMy = STATS_WINDOW * this.dtGeo;
      this.wilson.update(plates, life, my);
      const tstats = Tectonics.parseStats(buf);
      updateKinematics(plates, tstats, this.prevRuns, windowMy, this.rng, this.wilson.bias, life.contact, life.centroid, [NX, NZ]);

      this.stats = parseWorldStats(buf);
      const q = new Int32Array(buf).subarray(CTR_QUAKE);
      const site = (key: number, n: number) => (n > 0 ? { x: key & 0xff, z: (key >> 8) & 0xff, events: n } : null);
      this.quakeSites = { subduct: site(q[0]!, q[1]!), collide: site(q[2]!, q[3]!) };
      // reservoir snapshot gates crust stacking (see Tectonics.stackGate)
      // full stacking unless the reservoir is in real debt (magma keeps it near 0 in normal operation)
      this.tectonics.stackGate.value = Math.min(1, Math.max(0, 1 + this.lastCounters[0]! / RES_GATE_UNITS));
      if (Number.isFinite(this.stats.seaLevel)) { this.climate.setSeaLevel(this.stats.seaLevel); this.biome.setSeaLevel(this.stats.seaLevel); this.magma.uniforms.seaLevel.value = this.stats.seaLevel; }
      const op = this.lifecycle.decide(plates, life, my, this.rng, this.godSplit ?? this.wilson.riftRequest);
      this.godSplit = undefined;
      if (op) {
        this.lifecycle.apply(r, plates, op, my);
        if (op.kind === 'split') this.wilson.onSplit(my);
      }
      this.runEvents(my, windowMy);
      this.pendingResult = null;
    }
    // snapshot this window, then clear per-window counters (queue order keeps the copy before the clear)
    this.lifecycle.gatherStats(r);
    r.compute(this.worldStats);
    const p = this.fields.read(r, 'counters');
    this.pending = p;
    p.then((b) => { if (this.pending === p) this.pendingResult = b; });
    this.tectonics.clearPlateStats(r);
    this.prevRuns = this.runsInWindow;
    this.runsInWindow = 0;
  }

  private runEvents(my: number, windowMy: number): void {
    const rate = this.params.get('eventRate') as number;
    this.events.roll(my, windowMy, { hotspot: rate, iceAge: rate, floodBasalt: rate, meteor: rate * (this.params.get('meteorRate') as number) });
    this.dispatch(this.events.drain());
    this.iceAges = this.iceAges.filter((a) => my <= a.start + a.durationMy);
    this.climate.setIceAge(iceAgeForcing(this.iceAges, my));
  }

  /**
   * Apply queued god-tool events now (called every frame, so tools work while paused or at slow speeds).
   * Random events are rolled and drained at window ends, so the queue only ever holds user events here.
   */
  flushEvents(): void {
    if (this.events.queue.length) this.dispatch(this.events.drain());
  }

  private dispatch(events: GeoEvent[]): void {
    for (const e of events) {
      const h = this.handlers.get(e.kind);
      if (h) h(e);
      else { if (this.unhandledEvents++ === 0) console.warn(`Sim: no handler for event '${e.kind}' (counted in unhandledEvents)`); }
    }
  }

  /** Reservoir in fill units from the latest snapshot. */
  reservoir(): number { return this.lastCounters?.[0] ?? 0; }
  alivePlates(): number { return this.tectonics.plates.filter((p) => p.alive).length; }

  // ---- save/load (T50, V13) ------------------------------------------------------------------
  // A snapshot = GPU fields (save.ts) + getState(). Taken between ticks while hold() blocks runTicks, after
  // settle() has landed the in-flight window readback; that readback is part of the state (pendingCounters)
  // so the next window end after a load applies exactly what an unsaved run would apply.
  private holds = 0;

  /** Block ticking (runTicks returns 0) until the returned release() is called. Idempotent release. */
  hold(): () => void {
    this.holds++;
    let done = false;
    return () => { if (!done) { done = true; this.holds--; } };
  }

  get held(): boolean { return this.holds > 0; }

  /** Wait for the in-flight window readback so getState() is complete. Call while held. */
  async settle(): Promise<void> {
    const p = this.pending;
    if (!p || this.pendingResult) return;
    const b = await p;
    if (this.pending === p) this.pendingResult = b;
  }

  getState(): SimState {
    if (this.pending && !this.pendingResult) throw new Error('Sim.getState: window readback in flight; hold() and await settle() first');
    const tec = this.tectonics as unknown as { lastRunTick: number };
    return {
      tick: this.tick,
      dtGeo: this.dtGeo,
      plates: this.tectonics.plates.map((p) => ({ ...p, vel: [...p.vel] as [number, number], accum: [...p.accum] as [number, number] })),
      rng: this.rng.getState(),
      lifecycle: this.lifecycle.getState(),
      wilson: this.wilson.getState(),
      events: this.events.getState(),
      iceAges: this.iceAges.map((a) => ({ ...a })),
      unhandledEvents: this.unhandledEvents,
      runsInWindow: this.runsInWindow,
      prevRuns: this.prevRuns,
      tecLastRunTick: tec.lastRunTick,
      godSplit: this.godSplit ?? null,
      plumes: this.mantle.plumes.getState(),
      pendingCounters: this.pendingResult ? [...new Int32Array(this.pendingResult)] : null,
      lastCounters: this.lastCounters ? [...this.lastCounters] : null,
      uniforms: {
        climateSeaLevel: this.climate.uniforms.seaLevel.value,
        stackGate: this.tectonics.stackGate.value,
        iceAge: this.climate.uniforms.iceAge.value,
        biomeSeaLevel: this.biome.uniforms.seaLevel.value,
      },
    };
  }

  /** Throws (message names the bad key) unless s is a SimState this sim can load. Pure: call before mutating anything. */
  checkState(s: unknown): asserts s is SimState {
    const bad = (k: string) => { throw new Error(`Sim state: invalid or missing '${k}'`); };
    if (typeof s !== 'object' || s === null) bad('(root)');
    const o = s as Record<string, unknown>;
    for (const k of ['tick', 'dtGeo', 'unhandledEvents', 'runsInWindow', 'prevRuns', 'tecLastRunTick']) if (typeof o[k] !== 'number' || !Number.isFinite(o[k])) bad(k);
    if (o.dtGeo !== this.dtGeo) throw new Error(`Sim state: dtGeo ${String(o.dtGeo)} ≠ this world's ${this.dtGeo} (V12)`);
    if (!Array.isArray(o.plates) || o.plates.length !== MAX_PLATES) bad('plates');
    for (const k of ['rng', 'lifecycle', 'wilson', 'events', 'uniforms']) if (typeof o[k] !== 'object' || o[k] === null) bad(k);
    if (!Array.isArray(o.iceAges)) bad('iceAges');
    for (const k of ['pendingCounters', 'lastCounters']) if (o[k] !== null && !Array.isArray(o[k])) bad(k);
    if (o.godSplit !== null && typeof o.godSplit !== 'number') bad('godSplit');
    if (typeof o.plumes !== 'object' || o.plumes === null) bad('plumes');
    if (typeof (this.tectonics as unknown as { lastRunTick: unknown }).lastRunTick !== 'number') {
      throw new Error('Sim state: Tectonics.lastRunTick not found (renamed?); update Sim.getState/loadState');
    }
  }

  /** Restore CPU state. GPU fields are restored separately (save.ts); call both while held. */
  loadState(s: SimState): void {
    this.checkState(s);
    const plates = this.tectonics.plates;
    // heading is set explicitly: an absent (dead-plate) heading must clear the target's, not keep it
    s.plates.forEach((p, i) => { Object.assign(plates[i]!, { ...p, vel: [...p.vel], accum: [...p.accum], heading: p.heading }); });
    this.tick = s.tick;
    this.rng = PCG32.fromState(s.rng);
    this.lifecycle.loadState(s.lifecycle);
    this.wilson.loadState(s.wilson);
    this.events.loadState(s.events);
    this.iceAges = s.iceAges.map((a) => ({ ...a }));
    this.unhandledEvents = s.unhandledEvents;
    this.runsInWindow = s.runsInWindow;
    this.prevRuns = s.prevRuns;
    (this.tectonics as unknown as { lastRunTick: number }).lastRunTick = s.tecLastRunTick;
    this.godSplit = s.godSplit ?? undefined;
    this.mantle.plumes.setState(s.plumes);
    // a stale in-flight read (pre-load) is dropped: its then() checks this.pending identity
    this.pending = null;
    this.pendingResult = s.pendingCounters ? Int32Array.from(s.pendingCounters).buffer : null;
    this.lastCounters = s.lastCounters ? Int32Array.from(s.lastCounters) : null;
    this.stats = this.lastCounters ? parseWorldStats(this.lastCounters.buffer as ArrayBuffer) : null;
    this.climate.setSeaLevel(s.uniforms.climateSeaLevel);
    this.magma.uniforms.seaLevel.value = s.uniforms.climateSeaLevel;
    this.climate.uniforms.iceAge.value = s.uniforms.iceAge;
    this.biome.setSeaLevel(s.uniforms.biomeSeaLevel);
    this.tectonics.stackGate.value = s.uniforms.stackGate ?? 1;
  }
}

/** CPU half of a snapshot (T50). New CPU state that a later tick reads must be added here, or V13 breaks. */
export interface SimState {
  tick: number;
  dtGeo: number;
  plates: Plate[];
  rng: RngState;
  lifecycle: LifecycleState;
  wilson: WilsonState;
  events: EventSchedulerState;
  iceAges: { start: number; durationMy: number; magnitude: number }[];
  unhandledEvents: number;
  runsInWindow: number;
  prevRuns: number;
  /** Tectonics.lastRunTick (private there; drives the crust-age dt of the next run). */
  tecLastRunTick: number;
  /** God-tool split request waiting for the next window end. */
  godSplit: number | null;
  /** Mantle plumes + their rng (mantleModel PlumeSet). */
  plumes: { plumes: Plume[]; rng: RngState };
  /** Counters snapshot issued at the last window end, applied at the next one. null before the first window. */
  pendingCounters: number[] | null;
  lastCounters: number[] | null;
  uniforms: { climateSeaLevel: number; iceAge: number; biomeSeaLevel: number; stackGate?: number };
}
