import { describe, it, expect } from 'vitest';
import { WilsonController, WILSON, continentalCluster } from '../../src/sim/wilson';
import { emptyWorld } from '../../src/sim/worldData';
import type { LifecycleStats } from '../../src/sim/lifecycle';

function world(spread: number) {
  const w = emptyWorld(1);
  const s: LifecycleStats = { area: new Array(16).fill(0), centroid: Array.from({ length: 16 }, () => [0, 0] as [number, number]), contact: [] };
  for (let i = 0; i < 4; i++) {
    w.plates[i]!.alive = true; w.plates[i]!.continental = true; w.plates[i]!.vel = [1, 0];
    s.area[i] = 5000;
    s.centroid[i] = [(128 + (i - 1.5) * spread + 256) % 256, 128];
  }
  return { plates: w.plates, s };
}

describe('Wilson cycle controller (T32)', () => {
  // The point of the controller: the world must never settle into one static configuration (V8).
  it('cycles drift → assemble → super → rift → disperse → drift', () => {
    const c = new WilsonController();
    const far = world(60), near = world(10);
    const seen: string[] = [];
    let my = 0;
    for (let i = 0; i < 2000 && c.state.cycles < 1; i++, my += 2) {
      const w = c.state.phase === 'assemble' && my - c.state.since > 40 ? near : far;
      c.update(w.plates, w.s, my);
      if (c.riftRequest !== undefined) c.onSplit(my); // pretend the lifecycle split it
      if (seen[seen.length - 1] !== c.state.phase) seen.push(c.state.phase);
    }
    expect(seen).toEqual(['drift', 'assemble', 'super', 'rift', 'disperse', 'drift']);
    expect(c.state.cycles).toBe(1);
  });
  it('gives up assembling after the timeout so a cycle cannot stall forever', () => {
    const c = new WilsonController();
    const far = world(70);
    let my = 0;
    for (; my < WILSON.driftMy + WILSON.assembleMaxMy + 10; my += 2) c.update(far.plates, far.s, my);
    expect(['super', 'rift']).toContain(c.state.phase);
  });
  it('biases continental plates toward the cluster when assembling and away when dispersing', () => {
    const c = new WilsonController();
    const w = world(60);
    c.state = { phase: 'assemble', since: 0, cycles: 0, rms: 0 };
    c.update(w.plates, w.s, 1);
    const b = c.bias(w.plates[0]!)!; // plate 0 sits left of centre → heading ~0 (east)
    expect(Math.cos(b.heading!)).toBeGreaterThan(0.9);
    c.state = { phase: 'disperse', since: 0, cycles: 0, rms: 0 };
    expect(Math.cos(c.bias(w.plates[0]!)!.heading!)).toBeLessThan(-0.9);
  });
  it('cluster metric is torus-aware (plates straddling the seam are close)', () => {
    const w = world(0);
    w.s.centroid[0] = [2, 128]; w.s.centroid[1] = [254, 128]; w.s.centroid[2] = [1, 128]; w.s.centroid[3] = [255, 128];
    expect(continentalCluster(w.plates, w.s).rms).toBeLessThan(3);
  });
});
