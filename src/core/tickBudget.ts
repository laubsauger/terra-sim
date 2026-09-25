// Adaptive ticks-per-frame cap (V9, V22): keep sim GPU time per frame under budget. Speed requests beyond
// what fits show up as effective < requested in the time bar instead of dropping frames.
export class TickBudget {
  private msPerTick = 0.6; // initial guess, refined from measurements
  constructor(readonly budgetMs = 5, readonly minTicks = 1, readonly maxTicks = 64) {}

  /** Feed a measured GPU compute time for a frame that ran `ticks` ticks. */
  observe(gpuComputeMs: number, ticks: number): void {
    if (ticks <= 0 || !Number.isFinite(gpuComputeMs) || gpuComputeMs <= 0) return;
    const sample = gpuComputeMs / ticks;
    this.msPerTick += (sample - this.msPerTick) * 0.1;
  }

  get cap(): number {
    return Math.max(this.minTicks, Math.min(this.maxTicks, Math.floor(this.budgetMs / Math.max(1e-3, this.msPerTick))));
  }
  get estimateMsPerTick(): number { return this.msPerTick; }
}
