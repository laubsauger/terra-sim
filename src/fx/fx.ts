// Earthquakes + tsunamis (FX only, V15: the sim state is never touched). One entry point for main.ts.
//
// Earthquakes are parked (opts.quakes, default off). When on, scheduling runs on REAL time
// (fxModel.QuakeScheduler): at most one notable quake per ~12–40 s, more
// likely with more boundary activity (Sim.quakeSites) and a faster sim; paused → none. Big subduction
// quakes at ocean sites (epicentre 'water' depth > 2) and ocean meteor impacts (Sim.events.log) spawn a
// tsunami. Per quake: decaying camera shake (opt-in: opts.shake / setShake),
// the ground shock ring + rumble tint + dust (quakeFx.ts), the tsunami (tsunami.ts), and an onQuake
// event for audio / camera. Meteors (meteor.ts): meteorStrike() plays the full entry → impact sequence and
// calls back at the impact frame (the god tool enqueues the sim event there, so the crater appears at the
// flash); meteors the sim applied on its own (new 'meteor' entries in the event log) get a short streak
// that lands right away. Every impact also rings the ground, scorches it, shakes and — over the ocean —
// raises a tsunami.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import type { GeoEvent } from '../sim/events';
import { colIdx, Y_SEA_NOMINAL } from '../sim/layout';
import { cellToWorld, voxelToWorldY } from '../render/space';
import {
  QuakeScheduler, tsunamiAmplitude, impactMagnitude, shakeEnvelope, shakeDuration, SHAKE, type QuakeKind, type QuakeSites,
} from './fxModel';
import { createQuakeFx, type QuakeFx } from './quakeFx';
import { createTsunami, type Tsunami } from './tsunami';
import { createMeteorFx, METEOR, type MeteorFx } from './meteor';

/** What the FX read from the sim (a Sim satisfies it). */
export interface FxSimView {
  quakeSites: QuakeSites;
  events: { readonly log: readonly GeoEvent[] };
  stats: { seaLevel: number } | null;
  readonly geoMy: number;
}

export interface FxOptions {
  highQuality?: boolean;
  /** Camera shake + meteor recoil on (default off). */
  shake?: boolean;
  /** Orbit controls: the target moves with the shaken camera (a pure translation, no rotation). */
  controls?: { target: THREE.Vector3 };
  /** Effective sim speed (My per real s), e.g. () => clock.effectiveSpeed. Default: measured from sim.geoMy. */
  speed?: () => number;
  /** Real-time earthquake scheduler on (default false: quakes are parked). triggerQuake and meteors work either way. */
  quakes?: boolean;
  /** Random source of the scheduler (tests). */
  rng?: () => number;
  /** Brief camera recoil away from a meteor impact (default on; only while the shake is on). */
  nudge?: boolean;
}

export interface FxQuake {
  kind: QuakeKind;
  /** Epicentre column. */
  x: number; z: number;
  /** Moment magnitude (meteor impacts: equivalent magnitude). */
  mag: number;
  tsunami: boolean;
  /** 'water' depth at the epicentre (voxel layers). */
  waterDepth: number;
  /** Epicentre in world space (ground or sea surface). */
  position: THREE.Vector3;
  /** Camera → epicentre distance (world units) when it struck. */
  distance: number;
}

export interface Fx {
  /** dt: real seconds since the last frame; ambTime: ambience clock (s). Call once per frame, after camera updates. */
  update(dt: number, ambTime: number): void;
  setEnabled(v: boolean): void;
  setShake(v: boolean): void;
  setHighQuality(v: boolean): void;
  /** Subscribe to quakes and meteor impacts (fires on the impact frame; audio boom, camera POI). Returns an unsubscribe function. */
  onQuake(cb: (q: FxQuake) => void): () => void;
  /** Subscribe to meteor entries: fires when a bolide appears, `entry` s before its impact (audio whoosh). */
  onMeteorEntry(cb: (m: { x: number; z: number; mag: number; entry: number; position: THREE.Vector3 }) => void): () => void;
  /** Debug / god tool: quake at column (x, z). Tsunami rule as for a subduction quake. */
  triggerQuake(x: number, z: number, mag: number): Promise<FxQuake>;
  /**
   * Cinematic meteor strike at column (x, z): travel azimuth `dir` (rad; cells (cos, sin) = (+x, +z)),
   * meteor magnitude (god tool: strength / 12), crater radius in cells (default 4 + 8·magnitude, as the sim).
   * `onImpact` runs on the impact frame (~1.5 s later); enqueue the sim's meteor event there. Resolves at
   * the impact. With the FX disabled, onImpact runs at once.
   */
  meteorStrike(x: number, z: number, dir: number, magnitude: number, onImpact?: () => void, radius?: number): Promise<FxQuake>;
  readonly object: THREE.Group;
  readonly quake: QuakeFx;
  readonly meteor: MeteorFx;
  readonly tsunami: Tsunami;
  readonly stats: { quakes: number; tsunamis: number; active: boolean; speed: number; last: FxQuake | null };
  dispose(): void;
}

interface Shake { t0: number; mag: number; pos: THREE.Vector3; ph: number[] }
interface Nudge { t0: number; from: THREE.Vector3; k: number }


export function createFx(fields: GpuFields, renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera,
  sim: FxSimView, opts: FxOptions = {}): Fx {
  const quake = createQuakeFx(fields, renderer, { highQuality: opts.highQuality ?? true });
  const tsunami = createTsunami(fields);
  const object = new THREE.Group();
  object.name = 'fx';
  const meteor = createMeteorFx(fields, renderer, { highQuality: opts.highQuality ?? true });
  object.add(quake.ground, quake.dust, tsunami.sheet, meteor.object);
  scene.add(object);

  const rng = opts.rng ?? Math.random;
  const scheduler = new QuakeScheduler(rng);
  const listeners = new Set<(q: FxQuake) => void>();
  const entryListeners = new Set<(m: { x: number; z: number; mag: number; entry: number; position: THREE.Vector3 }) => void>();
  const stats = { quakes: 0, tsunamis: 0, active: false, speed: 0, last: null as FxQuake | null };
  let enabled = true;
  let shakeOn = opts.shake ?? false;
  let now = 0;
  let pending = 0;          // quakes waiting for their epicentre probe
  let lastGeo = sim.geoMy;
  let lastSeen: GeoEvent | undefined = sim.events.log[sim.events.log.length - 1]; // the log so far is history
  const shakes: Shake[] = [];
  const nudges: Nudge[] = [];
  /** God-tool strikes whose sim event will show up in the log (skip it there, the sequence is already playing). */
  const expected: { x: number; z: number; until: number }[] = [];
  const applied = new THREE.Vector3();
  const tmp = new THREE.Vector3(), delta = new THREE.Vector3();
  const target = opts.controls?.target;
  const seaLevel = () => { const s = sim.stats?.seaLevel; return Number.isFinite(s) ? s! : Y_SEA_NOMINAL; };

  /** Epicentre probe: two 4-byte reads of the sim's own fields (read-only, V15). */
  async function probe(x: number, z: number): Promise<{ water: number; surf: number }> {
    const off = colIdx(x, z) * 4;
    const [w, s] = await Promise.all([fields.read(renderer, 'water', off, 4), fields.read(renderer, 'surfY', off, 4)]);
    return { water: new Float32Array(w)[0]!, surf: new Float32Array(s)[0]! };
  }

  type Probe = { water: number; surf: number };
  async function fire(kind: QuakeKind, x: number, z: number, mag: number, meteorMag = 0): Promise<FxQuake> {
    x &= 255; z &= 255;
    pending++;
    let p: Probe;
    try { p = await probe(x, z); } finally { pending--; }
    return strikeGround(kind, x, z, mag, meteorMag, p);
  }

  /** Ground ring + dust + shake (+ tsunami) of a quake or impact whose epicentre was probed. */
  function strikeGround(kind: QuakeKind, x: number, z: number, mag: number, meteorMag: number, p: Probe, crater = 0): FxQuake {
    const water = Number.isFinite(p.water) ? Math.max(0, p.water) : 0;
    const amp = tsunamiAmplitude(kind === 'debug' ? 'subduct' : kind, kind === 'impact' ? meteorMag : mag, water);
    const position = new THREE.Vector3(cellToWorld(x), voxelToWorldY(p.surf + water), cellToWorld(z));
    const q: FxQuake = { kind, x, z, mag, tsunami: amp > 0, waterDepth: water, position, distance: camera.position.distanceTo(position) };
    if (!enabled) return q;
    const strength = Math.min(1, Math.max(0.3, (mag - 4.5) / 3.5));
    quake.start(renderer, { x, z, mag, strength }, now, seaLevel(), kind !== 'impact'); // impacts bring their own ejecta
    if (crater > 0 && water < 0.5) quake.scorch(x, z, crater, now);
    shakes.push({ t0: now, mag, pos: position, ph: Array.from({ length: 6 }, () => rng() * Math.PI * 2) });
    if (shakes.length > 3) shakes.shift();
    if (kind === 'impact' && (opts.nudge ?? true)) nudges.splice(0, nudges.length, { t0: now, from: position.clone(), k: Math.min(1.5, meteorMag) });
    if (amp > 0) { tsunami.start(renderer, x, z, amp, seaLevel()); stats.tsunamis++; }
    stats.quakes++;
    stats.last = q;
    for (const cb of listeners) cb(q);
    return q;
  }

  /** Meteor sequence; resolves at the impact. entry = seconds until the impact. */
  async function playMeteor(x: number, z: number, dir: number, mag: number, radius: number, entry: number, onImpact?: () => void): Promise<FxQuake> {
    x &= 255; z &= 255;
    pending++;
    let p: Probe;
    try { p = await probe(x, z); } finally { pending--; }
    const water = Number.isFinite(p.water) ? Math.max(0, p.water) : 0;
    if (!enabled) { onImpact?.(); return strikeGround('impact', x, z, impactMagnitude(mag), mag, p, radius); }
    const point = new THREE.Vector3(cellToWorld(x), voxelToWorldY(p.surf + water), cellToWorld(z));
    scheduler.noteExternal();
    for (const cb of entryListeners) cb({ x, z, mag, entry, position: point });
    return new Promise<FxQuake>((resolve) => {
      meteor.strike({ x, z, dir, mag, radius, point, water }, now, entry, () => {
        onImpact?.();
        resolve(strikeGround('impact', x, z, impactMagnitude(mag), mag, p, radius));
      });
    });
  }

  function applyShake(): void {
    tmp.set(0, 0, 0);
    for (let i = shakes.length - 1; i >= 0; i--) if (now - shakes[i]!.t0 > shakeDuration(shakes[i]!.mag)) shakes.splice(i, 1);
    if (shakeOn && enabled) {
      const dist = target ? camera.position.distanceTo(target) : camera.position.length();
      for (const s of shakes) {
        const t = now - s.t0;
        const e = shakeEnvelope(s.mag, t, s.pos.distanceTo(target ?? s.pos));
        // smooth tremor (two incommensurate 5–9 Hz partials) over a slow 1.7 Hz sway, vertical halved
        const w = (f: number, ph: number) => Math.sin(2 * Math.PI * f * t + ph);
        const a = e * dist;
        tmp.x += a * (0.55 * w(6.3, s.ph[0]!) + 0.3 * w(8.9, s.ph[1]!) + 0.35 * w(1.7, s.ph[2]!));
        tmp.z += a * (0.55 * w(5.7, s.ph[3]!) + 0.3 * w(9.4, s.ph[4]!) + 0.35 * w(1.9, s.ph[5]!));
        tmp.y += a * 0.5 * (0.6 * w(7.4, s.ph[1]! + 1) + 0.4 * w(2.3, s.ph[4]! + 2));
      }
      tmp.clampLength(0, SHAKE.MAX * 1.5 * dist);
      // meteor recoil: the camera is pushed back from the blast by ~1 % of its distance and eases back in
      for (let i = nudges.length - 1; i >= 0; i--) {
        const n = nudges[i]!, t = now - n.t0;
        if (t > 2.5) { nudges.splice(i, 1); continue; }
        const e = (1 - Math.exp(-t / 0.07)) * Math.exp(-t / 0.45) * 0.012 * (0.6 + 0.4 * n.k) * dist;
        tmp.addScaledVector(delta.subVectors(camera.position, n.from).normalize(), e);
      }
    } else nudges.length = 0;
    // move by the change only: the offset never accumulates into the camera or orbit target
    delta.subVectors(tmp, applied);
    camera.position.add(delta);
    target?.add(delta);
    applied.copy(tmp);
  }

  function watchEvents(): void {
    const log = sim.events.log;
    if (!log.length || log[log.length - 1] === lastSeen) return;
    const from = lastSeen ? log.lastIndexOf(lastSeen) + 1 : 0; // the log is a bounded ring: find by identity
    lastSeen = log[log.length - 1];
    for (let i = from; i < log.length; i++) {
      const e = log[i]!;
      if (e.kind !== 'meteor') continue;
      const k = expected.findIndex((m) => Math.abs(m.x - (e.x & 255)) <= 1 && Math.abs(m.z - (e.z & 255)) <= 1 && now <= m.until);
      if (k >= 0) { expected.splice(k, 1); continue; } // our own strike: its sequence is already playing
      // applied by the sim already: a short streak that lands right away (the display eases the crater in)
      void playMeteor(e.x, e.z, e.dir ?? rng() * Math.PI * 2, e.magnitude, e.radius ?? 4 + 8 * e.magnitude, METEOR.ENTRY_SHORT_S);
    }
    for (let i = expected.length - 1; i >= 0; i--) if (now > expected[i]!.until) expected.splice(i, 1);
  }

  return {
    object, quake, tsunami, meteor,
    get stats() { return { ...stats }; },
    update(dt, ambTime) {
      dt = Math.min(Math.max(dt, 0), 0.25);
      now += dt;
      const geo = sim.geoMy;
      const measured = dt > 0 ? Math.max(0, geo - lastGeo) / dt : 0;
      lastGeo = geo;
      stats.speed = opts.speed ? opts.speed() : stats.speed + (measured - stats.speed) * (1 - Math.exp(-dt / 2));
      if (enabled) {
        watchEvents();
        if (opts.quakes && pending === 0) {
          const pick = scheduler.step(dt, sim.quakeSites, stats.speed);
          if (pick) void fire(pick.kind, pick.x, pick.z, pick.mag);
        }
        const g = quake.update(renderer, dt, now);
        const t = tsunami.update(renderer, dt, ambTime);
        const m = meteor.update(renderer, dt, now);
        stats.active = g || t || m || shakes.length > 0;
      }
      applyShake();
    },
    setEnabled(v) {
      enabled = v;
      object.visible = v;
      if (!v) { shakes.length = 0; stats.active = false; applyShake(); }
    },
    setShake(v) { shakeOn = v; if (!v) applyShake(); },
    setHighQuality(v) { quake.setHighQuality(v); meteor.setHighQuality(v); },
    onQuake(cb) { listeners.add(cb); return () => { listeners.delete(cb); }; },
    onMeteorEntry(cb) { entryListeners.add(cb); return () => { entryListeners.delete(cb); }; },
    triggerQuake(x, z, mag) { scheduler.noteExternal(); return fire('debug', x, z, mag); },
    meteorStrike(x, z, dir, magnitude, onImpact, radius) {
      const r = radius ?? 4 + 8 * magnitude;
      const wrapped = () => { expected.push({ x: x & 255, z: z & 255, until: now + 10 }); onImpact?.(); };
      return playMeteor(x, z, dir, magnitude, r, METEOR.ENTRY_S, wrapped);
    },
    dispose() {
      shakes.length = 0; applyShake();
      scene.remove(object);
      quake.dispose(); tsunami.dispose(); meteor.dispose();
      listeners.clear(); entryListeners.clear();
    },
  };
}
