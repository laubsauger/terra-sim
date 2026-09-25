// §T.32 Wilson cycle controller (CPU, once per stats window). The only long-term "director" allowed by V5:
// it steers plate headings through supercontinent assembly, tenure, rifting and dispersal. It never tunes
// params. Deterministic: pure function of window stats + its own serialisable state.
import { NX, NZ } from './layout';
import type { Plate } from './worldData';
import type { LifecycleStats } from './lifecycle';
import type { Bias } from './kinematics';

export type WilsonPhase = 'drift' | 'assemble' | 'super' | 'rift' | 'disperse';

export const WILSON = {
  driftMy: 60,        // free drift after dispersal
  assembleMaxMy: 260, // give up assembling after this and call it a supercontinent anyway
  superMy: 80,        // supercontinent tenure (heat builds underneath)
  disperseMy: 90,     // pushed apart after the rift
  /** Assembled when continental mass sits within this RMS torus distance of its centroid (cells). */
  assembledRms: 42,
  assembleStrength: 0.08, // heading bias per My
  disperseStrength: 0.1,
} as const;

export interface WilsonState { phase: WilsonPhase; since: number; cycles: number; rms: number }

const wrapD = (d: number, n: number) => ((d + n / 2) % n + n) % n - n / 2;

/** Area-weighted circular mean of continental plate centroids, and their RMS spread (torus-aware). */
export function continentalCluster(plates: Plate[], s: LifecycleStats): { cx: number; cz: number; rms: number; n: number } {
  let sx = 0, cx = 0, sz = 0, cz = 0, wsum = 0;
  const cont = plates.filter((p) => p.alive && p.continental && s.area[p.id]! > 0);
  for (const p of cont) {
    const w = s.area[p.id]!;
    const [x, z] = s.centroid[p.id]!;
    cx += Math.cos((2 * Math.PI * x) / NX) * w; sx += Math.sin((2 * Math.PI * x) / NX) * w;
    cz += Math.cos((2 * Math.PI * z) / NZ) * w; sz += Math.sin((2 * Math.PI * z) / NZ) * w;
    wsum += w;
  }
  if (wsum === 0) return { cx: 0, cz: 0, rms: 0, n: 0 };
  const mx = ((Math.atan2(sx, cx) / (2 * Math.PI)) * NX + NX) % NX;
  const mz = ((Math.atan2(sz, cz) / (2 * Math.PI)) * NZ + NZ) % NZ;
  let v = 0;
  for (const p of cont) {
    const [x, z] = s.centroid[p.id]!;
    v += (wrapD(x - mx, NX) ** 2 + wrapD(z - mz, NZ) ** 2) * s.area[p.id]!;
  }
  return { cx: mx, cz: mz, rms: Math.sqrt(v / wsum), n: cont.length };
}

export class WilsonController {
  state: WilsonState = { phase: 'drift', since: 0, cycles: 0, rms: 0 };
  private cluster = { cx: 0, cz: 0 };
  private stats: LifecycleStats | null = null;
  /** Set for exactly one window when the controller wants a rift; consumed by Lifecycle.decide. */
  riftRequest: number | undefined;

  /** Advance phase from this window's stats. */
  update(plates: Plate[], s: LifecycleStats, my: number): void {
    this.stats = s;
    const c = continentalCluster(plates, s);
    this.cluster = { cx: c.cx, cz: c.cz };
    this.state.rms = c.rms;
    this.riftRequest = undefined;
    const age = my - this.state.since;
    const go = (phase: WilsonPhase) => { this.state = { ...this.state, phase, since: my }; };
    switch (this.state.phase) {
      case 'drift': if (age >= WILSON.driftMy) go('assemble'); break;
      case 'assemble': if (c.rms <= WILSON.assembledRms || c.n <= 1 || age >= WILSON.assembleMaxMy) go('super'); break;
      case 'super': if (age >= WILSON.superMy) go('rift'); break;
      case 'rift': {
        // rift the largest continental plate; stay in 'rift' until the lifecycle actually splits something
        const big = plates.filter((p) => p.alive && p.continental).sort((a, b) => s.area[b.id]! - s.area[a.id]! || a.id - b.id)[0];
        this.riftRequest = big?.id;
        break;
      }
      case 'disperse': if (age >= WILSON.disperseMy) { this.state.cycles++; go('drift'); } break;
    }
  }

  /** Called by the orchestrator when a split op was applied (rift succeeded). */
  onSplit(my: number): void {
    if (this.state.phase === 'rift') this.state = { ...this.state, phase: 'disperse', since: my };
  }

  /** Heading bias for kinematics: toward the continental centroid while assembling, away while dispersing. */
  bias = (p: Plate): Bias | undefined => {
    const ph = this.state.phase;
    if (!p.continental || !this.stats || (ph !== 'assemble' && ph !== 'disperse' && ph !== 'super')) return undefined;
    const [x, z] = this.stats.centroid[p.id]!;
    const dx = wrapD(this.cluster.cx - x, NX), dz = wrapD(this.cluster.cz - z, NZ);
    if (Math.hypot(dx, dz) < 4) return undefined;
    const toward = Math.atan2(dz, dx);
    if (ph === 'disperse') return { heading: toward + Math.PI, strength: WILSON.disperseStrength };
    return { heading: toward, strength: ph === 'super' ? WILSON.assembleStrength * 0.3 : WILSON.assembleStrength };
  };

  getState(): WilsonState { return { ...this.state }; }
  loadState(s: WilsonState): void { this.state = { ...s }; }
}
