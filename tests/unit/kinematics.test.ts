import { describe, it, expect } from 'vitest';
import { updateKinematics, MIN_SPEED, MAX_SPEED } from '../../src/sim/kinematics';
import { emptyWorld } from '../../src/sim/worldData';
import { PCG32 } from '../../src/core/rng';

function setup() {
  const w = emptyWorld(1);
  for (const i of [0, 1]) { w.plates[i]!.alive = true; w.plates[i]!.vel = [1.6, 0]; }
  const zeros = () => new Array(16).fill(0);
  const stats = { area: zeros(), subducted: zeros(), created: zeros() };
  stats.area[0] = 10000 * 40; stats.area[1] = 10000 * 40;
  return { plates: w.plates, stats };
}
const speed = (v: [number, number]) => Math.hypot(v[0], v[1]);

describe('plate kinematics', () => {
  // Slab pull is the main plate driver on Earth; subducting plates must end up faster.
  it('subducting plate speeds up relative to a plate without slabs', () => {
    const { plates, stats } = setup();
    stats.subducted[0] = 2000;
    const rng = new PCG32(3);
    for (let i = 0; i < 20; i++) updateKinematics(plates, stats, 40, 2, rng);
    expect(speed(plates[0]!.vel)).toBeGreaterThan(speed(plates[1]!.vel) * 1.5);
  });
  it('keeps speeds within bounds so the world never freezes or explodes', () => {
    const { plates, stats } = setup();
    stats.subducted[0] = 1e9;
    const rng = new PCG32(4);
    for (let i = 0; i < 200; i++) updateKinematics(plates, stats, 40, 2, rng);
    for (const p of plates.slice(0, 2)) {
      expect(speed(p.vel)).toBeLessThanOrEqual(MAX_SPEED + 1e-9);
      expect(speed(p.vel)).toBeGreaterThanOrEqual(MIN_SPEED - 1e-9);
    }
  });
  it('is deterministic for the same rng seed (V2)', () => {
    const a = setup(), b = setup();
    const ra = new PCG32(9), rb = new PCG32(9);
    for (let i = 0; i < 50; i++) { updateKinematics(a.plates, a.stats, 40, 2, ra); updateKinematics(b.plates, b.stats, 40, 2, rb); }
    expect(a.plates[0]!.vel).toEqual(b.plates[0]!.vel);
  });
  it('controller bias steers heading toward target', () => {
    const { plates, stats } = setup();
    const rng = new PCG32(5);
    for (let i = 0; i < 100; i++) updateKinematics(plates, stats, 40, 2, rng, () => ({ heading: Math.PI / 2, strength: 0.5 }));
    const h = Math.atan2(plates[0]!.vel[1], plates[0]!.vel[0]);
    expect(Math.abs(h - Math.PI / 2)).toBeLessThan(0.3);
  });
});
