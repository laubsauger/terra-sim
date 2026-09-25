// §T.13 plate kinematics (CPU). Runs once per stats window with counters read back from the GPU.
// Plates that lose many columns at convergent margins are being pulled by their slab and speed up;
// continental plates drag; headings random-walk. The Wilson controller (T32) biases targets later.
import { PCG32 } from '../core/rng';
import type { Plate } from './worldData';
import type { TectonicsStats } from './tectonics';

export const BASE_SPEED = 1.6; // cells per My (≈ 3 cm/yr at 20 km cells)
export const MIN_SPEED = 0.5;
export const MAX_SPEED = 4.5;
export const SLAB_GAIN = 2.5;
export const CONT_DRAG = 0.65;
export const RELAX_MY = 10; // velocity relaxes toward target over this time
export const HEADING_SIGMA = 0.15; // rad / sqrt(My)

export interface Bias { heading?: number; strength: number } // per plate, from controller

export function updateKinematics(plates: Plate[], stats: TectonicsStats, runs: number, windowMy: number, rng: PCG32,
  bias?: (p: Plate) => Bias | undefined): void {
  const a = Math.min(1, windowMy / RELAX_MY);
  for (const p of plates) {
    if (!p.alive) continue;
    const area = runs > 0 ? stats.area[p.id]! / runs : 0;
    if (area <= 0) continue;
    const slab = stats.subducted[p.id]! / Math.sqrt(area) / Math.max(1, windowMy);
    let speed = BASE_SPEED * (1 + SLAB_GAIN * slab) * (p.continental ? CONT_DRAG : 1);
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
}
