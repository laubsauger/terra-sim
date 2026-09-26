import { describe, it, expect } from 'vitest';
import { updateKinematics, removeNetMotion, MIN_SPEED, MAX_SPEED } from '../../src/sim/kinematics';
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

describe('collisions brake, never stall or rotate plates (B16)', () => {
  // Averaging colliding plates' velocities stalled the world and swung headings; the brake only slows.
  it('plates in sustained continental contact slow down but keep their headings', () => {
    const { plates, stats } = setup();
    plates[0]!.vel = [1.6, 0]; plates[1]!.vel = [-1.6, 0];
    const contact = Array.from({ length: 16 }, () => new Array(16).fill(0));
    contact[0]![1] = contact[1]![0] = 400;
    const rng = new PCG32(6);
    const free = setup();
    free.plates[0]!.vel = [1.6, 0]; free.plates[1]!.vel = [-1.6, 0];
    const rng2 = new PCG32(6);
    for (let i = 0; i < 30; i++) {
      updateKinematics(plates, stats, 40, 2, rng, undefined, contact);
      updateKinematics(free.plates, free.stats, 40, 2, rng2);
    }
    expect(Math.hypot(...plates[0]!.vel)).toBeLessThan(Math.hypot(...free.plates[0]!.vel) * 0.8);
    expect(Math.cos(Math.atan2(plates[0]!.vel[1], plates[0]!.vel[0]))).toBeGreaterThan(0.8);  // still heading +x
    expect(Math.cos(Math.atan2(plates[1]!.vel[1], plates[1]!.vel[0]))).toBeLessThan(-0.8);   // still heading -x
  });
});

describe('no net motion (user: all plates walk the same way)', () => {
  // A shared drift hides tectonics: plates must move relative to each other, not together.
  it('area-weighted mean plate velocity is zero after an update', () => {
    const { plates, stats } = setup();
    plates[0]!.vel = [2, 0.5]; plates[1]!.vel = [1.5, 0.2];
    stats.area[1] = 10000 * 40 * 3;
    removeNetMotion(plates, stats, 40);
    const a0 = stats.area[0]!, a1 = stats.area[1]!;
    const mx = (plates[0]!.vel[0] * a0 + plates[1]!.vel[0] * a1) / (a0 + a1);
    const mz = (plates[0]!.vel[1] * a0 + plates[1]!.vel[1] * a1) / (a0 + a1);
    expect(Math.abs(mx)).toBeLessThan(1e-9);
    expect(Math.abs(mz)).toBeLessThan(1e-9);
    // relative motion survives
    expect(Math.hypot(plates[0]!.vel[0] - plates[1]!.vel[0], plates[0]!.vel[1] - plates[1]!.vel[1])).toBeGreaterThan(0.1);
  });
});

// User: "continents wander through each other ... they would fold up mountain ranges rather than glide through".
// Continents are too buoyant to subduct: colliding continental plates must (nearly) stop closing, while sliding
// past each other along the suture stays free (blanket braking stalled plates, B16).
describe('continental collision', () => {
  const run = (vA: [number, number], vB: [number, number], contactCells: number) => {
    const { plates, stats } = setup();
    plates[0]!.vel = [...vA]; plates[1]!.vel = [...vB];
    plates[0]!.heading = Math.atan2(vA[1], vA[0]); plates[1]!.heading = Math.atan2(vB[1], vB[0]);
    const contact = Array.from({ length: 16 }, () => new Array(16).fill(0));
    contact[0]![1] = contactCells;
    const centroids: [number, number][] = Array.from({ length: 16 }, () => [0, 0]);
    centroids[0] = [100, 128]; centroids[1] = [156, 128]; // B lies +x of A
    updateKinematics(plates, stats, 40, 2, new PCG32(9), undefined, contact, centroids, [256, 256]);
    return { a: plates[0]!.vel, b: plates[1]!.vel };
  };
  it('head-on continents slow sharply instead of gliding through each other', () => {
    const before = 1.2 - -1.2;
    const { a, b } = run([1.2, 0], [-1.2, 0], 40);
    const closing = a[0] - b[0];
    expect(closing).toBeLessThan(before * 0.5); // per window; compounding keeps a colliding front near stalled
    expect(closing).toBeGreaterThanOrEqual(-1e-6); // stopped, not bounced apart
  });
  it('sliding along the suture is untouched', () => {
    const noC = run([0, 1.2], [0, -1.2], 0), withC = run([0, 1.2], [0, -1.2], 40);
    // only the existing jam speed brake applies (a few %), no approach damping of the slip
    expect(Math.abs(withC.a[1] / noC.a[1] - 1)).toBeLessThan(0.05);
    expect(Math.abs(withC.b[1] / noC.b[1] - 1)).toBeLessThan(0.05);
  });
  it('without continental contact nothing changes', () => {
    const noC = run([1.2, 0], [-1.2, 0], 0);
    expect(noC.a[0] - noC.b[0]).toBeGreaterThan(1.5);
  });
});
