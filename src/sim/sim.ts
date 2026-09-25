// Sim orchestrator: owns passes, runs geo ticks in SPEC §C pass order, keeps readbacks deterministic.
import type * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { Params } from '../core/params';
import { PCG32 } from '../core/rng';
import { createDerivePass } from './derive';
import { Tectonics } from './tectonics';
import { updateKinematics } from './kinematics';
import type { WorldData } from './worldData';

/** Ticks per stats window. Readback issued at window end must land before the next window ends (V2). */
export const STATS_WINDOW = 40;
/** Isostasy runs every N ticks. */
export const ISO_EVERY = 4;

export class Sim {
  tick = 0;
  readonly tectonics: Tectonics;
  private derive;
  private rng: PCG32;
  private pending: Promise<ArrayBuffer> | null = null;
  private pendingResult: ArrayBuffer | null = null;
  private runsInWindow = 0;
  /** Latest counters snapshot (reservoir + plate stats), for UI. */
  lastCounters: Int32Array | null = null;

  constructor(private renderer: THREE.WebGPURenderer, private fields: GpuFields, world: WorldData,
    private params: Params, readonly dtGeo: number) {
    this.derive = createDerivePass(fields);
    this.tectonics = new Tectonics(fields, world.plates);
    this.rng = new PCG32(world.seed, 0x7ec70);
    this.derive.run(renderer);
  }

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
    if (this.tectonics.tick(r, t, this.dtGeo, { speedMul: this.params.get('plateSpeed') as number, isoEvery: ISO_EVERY })) {
      this.runsInWindow++;
      this.derive.run(r);
    }
    this.tick = t;
    if (t % STATS_WINDOW === 0) this.windowEnd();
  }

  private windowEnd(): void {
    // apply stats from the previous window (issued STATS_WINDOW ticks ago)
    if (this.pendingResult) {
      const buf = this.pendingResult;
      this.lastCounters = new Int32Array(buf);
      updateKinematics(this.tectonics.plates, Tectonics.parseStats(buf), this.prevRuns, STATS_WINDOW * this.dtGeo, this.rng);
      this.pendingResult = null;
    }
    // snapshot this window, then clear per-plate counters (queue order keeps the copy before the clear)
    const p = this.fields.read(this.renderer, 'counters');
    this.pending = p;
    p.then((b) => { if (this.pending === p) this.pendingResult = b; });
    this.tectonics.clearPlateStats(this.renderer);
    this.prevRuns = this.runsInWindow;
    this.runsInWindow = 0;
  }
  private prevRuns = 0;

  /** Reservoir in fill units from the latest snapshot. */
  reservoir(): number { return this.lastCounters?.[0] ?? 0; }
}
