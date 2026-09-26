// Earthquake / tsunami FX model (CPU, no three.js): real-time quake scheduler, magnitude draw, tsunami
// rule and the shared constants of the ring, shake, dust and tsunami layers. FX only (V15): nothing here
// feeds back into the sim. Quakes run on REAL time, not geo time: a plate boundary at sim speed would
// rupture many times per frame, so a quake here stands for "the notable one of the last while".

/** Earthquake site as Sim.quakeSites reports it (columns + boundary events in the last stats window). */
export interface QuakeSite { x: number; z: number; events: number }
export interface QuakeSites { subduct: QuakeSite | null; collide: QuakeSite | null }
export type QuakeKind = 'subduct' | 'collide' | 'impact' | 'debug';
export type Rng = () => number;

export const QUAKE = {
  /** Real seconds: no two scheduled quakes closer than this. */
  MIN_GAP_S: 12,
  /** Real seconds: an active world (activity ≥ ACT_FORCE) always gets its quake by then. */
  MAX_GAP_S: 40,
  /** Mean extra wait (s) after MIN_GAP_S at activity 1; divided by the activity. */
  WAIT_S: 7,
  /** Activity below this never forces a quake at MAX_GAP_S (quiet worlds stay quiet). */
  ACT_FORCE: 0.15,
  /** Boundary events (subduct + 0.7 · collide) for activity 1 − 1/e. */
  ACT_E0: 40,
  /** First scheduled quake no earlier than this after start (s). */
  FIRST_S: 6,
  /** Location jitter around the reported site (cells). */
  JITTER: 3,
  /** Lognormal draw: M = M0 + exp(mu + sigma · N(0,1)), clamped to [M_MIN, max]. */
  M0: 4.7, M_MIN: 5.0,
  SUB: { mu: 0.6, sigma: 0.45, max: 9.3 },
  COL: { mu: 0.35, sigma: 0.42, max: 8.3 },
} as const;

/**
 * Impact scorch (real s): the incandescent floor cools within ~5 s; the charred crater, the sooty ejecta
 * blanket and (big strikes) dark ejecta rays hold, then fade out gradually by SCORCH_S.
 */
export const SCORCH_S = 50;

export const RING = {
  /** Shock ring speed over the ground (cells/s). */
  SPEED: 42,
  /** Max ring radius (cells) = R0 + RM · (M − 5). */
  R0: 26, RM: 24,
  /** Band half-width (cells) = W0 + WM · (M − 5). */
  W0: 3.5, WM: 1.2,
  /** Rumble tint time constant (s): the ground darkens behind the ring, then clears. */
  TINT_S: 2.8,
  /** FX lifetime after the ring has reached its max radius (s). */
  TAIL_S: 3.5,
} as const;

export const SHAKE = {
  /** Screen-relative amplitude (fraction of camera→target distance) at M5, ×GROWTH per magnitude step. */
  A5: 0.0011, GROWTH: 1.75,
  /** Hard cap (fraction of distance): ≈1% of the view, never more. */
  MAX: 0.0085,
  /** Decay time constant (s) = TAU0 + TAUM · (M − 5). */
  TAU0: 0.9, TAUM: 0.45,
  /** Onset (s): the shake swells in instead of snapping. */
  RISE_S: 0.18,
} as const;

export const DUST = {
  POOL_HIGH: 4096, POOL_LOW: 2048,
  /** Particle life after it lifts off (s). */
  LIFE0: 4.5, LIFE1: 8,
} as const;

export const TSUNAMI = {
  /** Minimum epicentre water depth (voxel layers, 'water' field) for a tsunami. */
  MIN_DEPTH: 2,
  /** Minimum magnitude of a subduction (megathrust) quake that spawns one. */
  MIN_MAG: 7.6,
  /** Wave speed = C0 · sqrt(depth / DREF) cells/s (shallow-water, speed ∝ √depth). */
  C0: 13, DREF: 14, DMIN: 0.35,
  /** Run-up speed over land (cells/s) and the max run-up height (layers above sea) per layer of amplitude. */
  C_RUN: 2.2, RUNUP_K: 2.2,
  /** Real seconds a tsunami lives (fades over the last FADE_S). */
  LIFE_S: 30, FADE_S: 9,
  /** Relaxation sweeps per frame (in place; the front moves < 0.3 cells per frame). */
  SWEEPS: 3,
  /** "Not reached" arrival time. */
  INF: 1e5,
} as const;

/** Boundary activity 0..1 from the last window's quake sites. */
export function activity(s: QuakeSites): number {
  const n = (s.subduct?.events ?? 0) + 0.7 * (s.collide?.events ?? 0);
  return n > 0 ? 1 - Math.exp(-n / QUAKE.ACT_E0) : 0;
}

/** Sim speed factor 0..1 (My per real second). Paused → 0: geologic time stands still, so does the ground. */
export function speedFactor(myPerS: number): number {
  if (!(myPerS > 0)) return 0;
  return Math.min(1, Math.max(0.3, 0.5 + 0.25 * Math.log10(myPerS / 0.01)));
}

/** Standard normal (Box-Muller). */
function gauss(rng: Rng): number {
  const u = Math.max(1e-12, rng()), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Lognormal "M5–M9 style" magnitude for a quake at a subduction or collision boundary. */
export function drawMagnitude(kind: 'subduct' | 'collide', rng: Rng): number {
  const p = kind === 'subduct' ? QUAKE.SUB : QUAKE.COL;
  const m = QUAKE.M0 + Math.exp(p.mu + p.sigma * gauss(rng));
  return Math.min(p.max, Math.max(QUAKE.M_MIN, m));
}

export interface QuakePick { kind: 'subduct' | 'collide'; x: number; z: number; mag: number }

/** Site + magnitude: subduction weighted up (megathrusts), collision for mountain quakes. */
export function pickQuake(s: QuakeSites, rng: Rng): QuakePick | null {
  const ws = (s.subduct?.events ?? 0) * 1.3, wc = s.collide?.events ?? 0;
  if (ws + wc <= 0) return null;
  const kind = rng() * (ws + wc) < ws ? 'subduct' : 'collide';
  const site = (kind === 'subduct' ? s.subduct : s.collide)!;
  const j = () => Math.round((rng() * 2 - 1) * QUAKE.JITTER);
  return { kind, x: (site.x + j()) & 255, z: (site.z + j()) & 255, mag: drawMagnitude(kind, rng) };
}

/**
 * Real-time quake scheduler: at most one notable quake per MIN_GAP_S; after that a hazard rate
 * ∝ activity × speed factor (mean extra wait WAIT_S / a), and an active world (a ≥ ACT_FORCE) gets its
 * quake by MAX_GAP_S at the latest. No sites or a paused clock → no quakes.
 */
export class QuakeScheduler {
  sinceLast: number = QUAKE.MIN_GAP_S - QUAKE.FIRST_S;
  constructor(private rng: Rng = Math.random) {}

  /** dt real s; myPerS effective sim speed. Returns the quake to fire now, if any. */
  step(dt: number, sites: QuakeSites, myPerS: number): QuakePick | null {
    this.sinceLast += dt;
    if (this.sinceLast < QUAKE.MIN_GAP_S) return null;
    const a = activity(sites) * speedFactor(myPerS);
    if (a <= 0) return null;
    const force = a >= QUAKE.ACT_FORCE && this.sinceLast >= QUAKE.MAX_GAP_S;
    if (!force && this.rng() >= 1 - Math.exp(-dt * a / QUAKE.WAIT_S)) return null;
    const q = pickQuake(sites, this.rng);
    if (q) this.sinceLast = 0;
    return q;
  }

  /** A quake from elsewhere (meteor, debug) also resets the gap: they never stack up. */
  noteExternal(): void { this.sinceLast = 0; }
}

/** Tsunami amplitude (voxel layers in deep water) for a quake, or 0 for none. */
export function tsunamiAmplitude(kind: QuakeKind, mag: number, waterDepth: number): number {
  if (!(waterDepth > TSUNAMI.MIN_DEPTH)) return 0;
  if (kind === 'impact') return 0.9 + 0.9 * Math.min(1.5, Math.max(0.3, mag)); // mag = meteor magnitude 0.5..1.5
  if (kind === 'collide' || mag < TSUNAMI.MIN_MAG) return 0;
  return 0.8 + 0.85 * (mag - TSUNAMI.MIN_MAG);
}

/** Equivalent moment magnitude of a meteor impact (for ring, shake and dust). */
export const impactMagnitude = (meteorMag: number): number => 6.4 + 1.2 * Math.min(1.5, Math.max(0.3, meteorMag));

export const ringRadius = (mag: number): number => RING.R0 + RING.RM * Math.max(0, mag - 5);
export const ringWidth = (mag: number): number => RING.W0 + RING.WM * Math.max(0, mag - 5);
/** Real seconds a quake's ground FX (ring + tint + dust launch) stays active. */
export const quakeDuration = (mag: number): number => ringRadius(mag) / RING.SPEED + RING.TAIL_S;

/** Real seconds after which a quake's shake is dropped (5 decay constants: < 1 % of its peak). */
export const shakeDuration = (mag: number): number => 5 * (SHAKE.TAU0 + SHAKE.TAUM * Math.max(0, mag - 5)) + SHAKE.RISE_S;

/** Shake amplitude as a fraction of the camera→target distance at time t (s) after the quake. */
export function shakeEnvelope(mag: number, t: number, distToEpicentre: number): number {
  if (t < 0) return 0;
  const a = Math.min(SHAKE.MAX, SHAKE.A5 * SHAKE.GROWTH ** Math.max(0, mag - 5));
  const tau = SHAKE.TAU0 + SHAKE.TAUM * Math.max(0, mag - 5);
  const near = 0.55 + 0.45 * Math.exp(-distToEpicentre / 1.6); // world units; the block is only 4 wide
  return a * near * (1 - Math.exp(-t / SHAKE.RISE_S)) * Math.exp(-t / tau);
}
