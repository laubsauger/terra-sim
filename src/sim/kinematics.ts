// §T.13 plate kinematics (CPU). Runs once per stats window with counters read back from the GPU.
// Plates that lose many columns at convergent margins are being pulled by their slab and speed up;
// continental plates drag; headings random-walk. The Wilson controller (T32) biases targets later.
import { PCG32 } from '../core/rng';
import type { Plate } from './worldData';
import type { TectonicsStats } from './tectonics';

export const BASE_SPEED = 1.2; // cells per My (≈ 2.4 cm/yr at 20 km cells)
export const MIN_SPEED = 0.4;
export const MAX_SPEED = 3.5;
export const SLAB_GAIN = 1.2;
export const CONT_DRAG = 0.7;
/** Speed factor per unit of continental-collision contact fraction (collisions jam plates). */
export const COLLISION_BRAKE = 2.5;
/** Continental contact cells above which two plates start locking (velocities converge). */
export const LOCK_CONTACT = 8;
/** Time for a fully engaged collision to match plate velocities. */
export const LOCK_MY = 4;
export const RELAX_MY = 10; // velocity relaxes toward target over this time
export const HEADING_SIGMA = 0.15; // rad / sqrt(My)

export interface Bias { heading?: number; strength: number } // per plate, from controller

/**
 * contact[p][q] = continental boundary cells between plates p and q (lifecycle stats), optional.
 * Colliding continents lock: their velocities converge, so collisions decelerate instead of grinding
 * continents into the mantle at full plate speed (B5).
 * Slab measure is normalised by the plate's own speed so faster plates do not pull harder just because
 * they subduct more per window (that feedback saturates every plate at MAX_SPEED).
 */
export function updateKinematics(plates: Plate[], stats: TectonicsStats, runs: number, windowMy: number, rng: PCG32,
  bias?: (p: Plate) => Bias | undefined, contact?: number[][]): void {
  const a = Math.min(1, windowMy / RELAX_MY);
  for (const p of plates) {
    if (!p.alive) continue;
    const area = runs > 0 ? stats.area[p.id]! / runs : 0;
    if (area <= 0) continue;
    const cur = Math.max(0.3, Math.hypot(p.vel[0], p.vel[1]));
    const side = Math.sqrt(area);
    const slab = Math.min(1.5, stats.subducted[p.id]! / (cur * windowMy * side));
    const jam = Math.min(1, (contact?.[p.id]?.reduce((s, v) => s + v, 0) ?? 0) / side);
    let speed = BASE_SPEED * (1 + SLAB_GAIN * slab) * (p.continental ? CONT_DRAG : 1) / (1 + COLLISION_BRAKE * jam);
    speed = Math.min(MAX_SPEED, Math.max(MIN_SPEED, speed));
    let heading = Math.atan2(p.vel[1], p.vel[0]);
    // Box-Muller from PCG32 keeps the walk deterministic (V2)
    const u1 = Math.max(1e-12, rng.nextFloat()), u2 = rng.nextFloat();
    heading += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * HEADING_SIGMA * Math.sqrt(windowMy);
    const b = bias?.(p);
    if (b?.heading !== undefined) {
      const d = Math.atan2(Math.sin(b.heading - heading), Math.cos(b.heading - heading));
      heading += d * Math.min(1, b.strength * windowMy);
    }
    const tx = Math.cos(heading) * speed, tz = Math.sin(heading) * speed;
    p.vel[0] += (tx - p.vel[0]) * a;
    p.vel[1] += (tz - p.vel[1]) * a;
  }
  if (!contact) return;
  for (const p of plates) for (const q of plates) {
    if (!p.alive || !q.alive || q.id <= p.id) continue;
    const c = contact[p.id]![q.id]!;
    if (c < LOCK_CONTACT) continue;
    const ap = stats.area[p.id]! / Math.max(1, runs), aq = stats.area[q.id]! / Math.max(1, runs);
    if (ap <= 0 || aq <= 0) continue;
    const engage = Math.min(1, c / Math.sqrt(Math.min(ap, aq))) * Math.min(1, windowMy / LOCK_MY);
    const mx = (p.vel[0] * ap + q.vel[0] * aq) / (ap + aq), mz = (p.vel[1] * ap + q.vel[1] * aq) / (ap + aq);
    for (const r of [p, q]) { r.vel[0] += (mx - r.vel[0]) * engage; r.vel[1] += (mz - r.vel[1]) * engage; }
  }
}
