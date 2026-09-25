// Sim orchestrator: owns passes, runs geo ticks in SPEC §C pass order, keeps readbacks deterministic.
import type * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { PCG32 } from '../core/rng';
import { createDerivePass } from './derive';
import { Tectonics } from './tectonics';
import { updateKinematics } from './kinematics';
import { Lifecycle } from './lifecycle';
import { createHydroPass, type HydroPass } from './hydro';
import { createErosionPass, type ErosionPass } from './erosion';
import { createClimatePass, type ClimatePass } from './climate';
import { createBiomePass, type BiomePass } from './biome';
import { createWorldStats, parseWorldStats, type WorldStats } from './worldStats';
import { WilsonController } from './wilson';
import { EventScheduler, iceAgeForcing, type GeoEvent, type EventKind } from './events';
import type { WorldData } from './worldData';

/** Ticks per stats window. Readback issued at window end must land before the next window ends (V2). */
export const STATS_WINDOW = 40;
/** Isostasy runs every N ticks (on tectonics ticks). */
export const ISO_EVERY = 4;
/** Pipe-model substeps per geo tick (quasi-steady flow). */
export const HYDRO_SUBSTEPS = 2;
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
  stats: WorldStats | null = null;
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
    this.lifecycle = new Lifecycle(fields);
    this.hydro = createHydroPass(fields, params);
    this.erosion = createErosionPass(fields, params);
    this.climate = createClimatePass(fields, params);
    this.biome = createBiomePass(fields);
    this.worldStats = createWorldStats(fields);
    this.kGeoBase = this.erosion.uniforms.kGeo.value;
    this.erosion.uniforms.kGeo.value = this.kGeoBase * EROSION_EVERY;
    this.rng = new PCG32(world.seed, 0x7ec70);
    this.events = new EventScheduler(world.seed);
    this.handlers.set('iceAge', (e) => this.iceAges.push({ start: e.my, durationMy: e.durationMy ?? 20, magnitude: e.magnitude }));
    this.derive.run(renderer);
  }

  get geoMy(): number { return this.tick * this.dtGeo; }

  /**
   * Run up to n ticks. Returns ticks actually run: stalls (deterministically) when the previous
   * window's readback has not arrived yet, instead of using stale or missing data.
   */
  runTicks(n: number): number {
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
    // 3-4 tectonics + isostasy
    if (this.tectonics.tick(r, t, this.dtGeo, { speedMul: this.params.get('plateSpeed') as number, isoEvery: ISO_EVERY })) {
      this.runsInWindow++;
      this.derive.run(r);
      this.tectonics.flow(r);
      this.derive.run(r);
    }
    // 7 climate (evap/precip/ice into water), 8 hydrology + erosion (erosion needs fresh surfY)
    this.climate.step(r);
    this.hydro.step(r, HYDRO_SUBSTEPS);
    if (t % EROSION_EVERY === 0) {
      this.erosion.uniforms.tick.value = t;
      this.erosion.step(r);
      this.derive.run(r);
    }
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
      updateKinematics(plates, Tectonics.parseStats(buf), this.prevRuns, windowMy, this.rng, this.wilson.bias, life.contact);
      this.stats = parseWorldStats(buf);
      if (Number.isFinite(this.stats.seaLevel)) { this.climate.setSeaLevel(this.stats.seaLevel); this.biome.setSeaLevel(this.stats.seaLevel); }
      const op = this.lifecycle.decide(plates, life, my, this.rng, this.wilson.riftRequest);
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
    for (const e of this.events.drain()) {
      const h = this.handlers.get(e.kind);
      if (h) h(e);
      else { if (this.unhandledEvents++ === 0) console.warn(`Sim: no handler for event '${e.kind}' (counted in unhandledEvents)`); }
    }
    this.iceAges = this.iceAges.filter((a) => my <= a.start + a.durationMy);
    this.climate.setIceAge(iceAgeForcing(this.iceAges, my));
  }

  /** Reservoir in fill units from the latest snapshot. */
  reservoir(): number { return this.lastCounters?.[0] ?? 0; }
  alivePlates(): number { return this.tectonics.plates.filter((p) => p.alive).length; }
}
