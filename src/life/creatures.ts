// §T.45 creatures: CPU-steered birds (boids), grazing critters and fish schools over a periodically
// read-back column map (surfY, water, biome). Read-only on sim (V15). Instanced draws, one per
// creature type; per-instance data goes through small dynamic instanced attributes each frame.
import * as THREE from 'three/webgpu';
import {
  Fn, float, vec3, uniform, uniformArray, attribute, positionGeometry, normalGeometry, normalLocal, varying,
  sin, cos, select, mix, max, pow, positionWorld, uint, smoothstep,
} from 'three/tsl';
import { BLOCK_SIZE } from '../sim/layout';
import { Biome } from '../sim/biomeModel';
import { voxelToWorldY, AMB_PERIOD } from '../render/space';
import {
  CREATURES as C, GRAZING, type ColumnMap, heightAt, columnAt, waterNear, slopeAt, toCell,
} from './lifeModel';
import { birdGeometry, critterGeometry, fishGeometry } from './geometry';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

const HALF = BLOCK_SIZE / 2;
const TAU = Math.PI * 2;
const lin = (hex: number) => new THREE.Color(hex);

/** Small deterministic PRNG (mulberry32); creatures are cosmetic, not part of the sim hash. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- map helpers ----

/** Bilinear water-surface-or-ground level (voxel y). */
function levelAt(m: ColumnMap, x: number, z: number): number {
  const u = toCell(x), v = toCell(z);
  const u0 = Math.floor(u), v0 = Math.floor(v), fu = u - u0, fv = v - v0;
  const at = (a: number, b: number) => { const c = (((a % 256) + 256) % 256) + (((b % 256) + 256) % 256) * 256; return m.surfY[c]! + m.water[c]!; };
  return (at(u0, v0) * (1 - fu) + at(u0 + 1, v0) * fu) * (1 - fv) + (at(u0, v0 + 1) * (1 - fu) + at(u0 + 1, v0 + 1) * fu) * fv;
}
/** Lowest world y a bird may fly at, at x,z. */
export function birdFloor(m: ColumnMap, x: number, z: number): number {
  return voxelToWorldY(Math.max(levelAt(m, x, z), heightAt(m, x, z))) + C.BIRD_CLEARANCE;
}
export function critterValid(m: ColumnMap, x: number, z: number): boolean {
  const lim = HALF - C.CRITTER_MARGIN;
  if (Math.abs(x) > lim || Math.abs(z) > lim) return false;
  return GRAZING.has(m.biome[columnAt(x, z)]!) && waterNear(m, x, z) <= C.CRITTER_WET && slopeAt(m, x, z) <= C.CRITTER_SLOPE;
}
export function fishValid(m: ColumnMap, x: number, z: number): boolean {
  const lim = HALF - 0.04;
  if (Math.abs(x) > lim || Math.abs(z) > lim) return false;
  const w = m.water[columnAt(x, z)]!;
  return w >= C.FISH_DEPTH_MIN && w <= C.FISH_DEPTH_MAX;
}

// ---- state ----

interface Flock { hx: number; hz: number; alt: number; radius: number; dir: number; timer: number; color: THREE.Color }
interface Bird { x: number; y: number; z: number; vx: number; vy: number; vz: number; flock: number; phase: number; rate: number; amp: number; glide: number; bank: number }
interface Critter { x: number; z: number; yaw: number; walking: number; head: number; phase: number; mode: 0 | 1; timer: number; herd: number; variant: number; turn: number; active: boolean }
interface School { cx: number; cz: number; tx: number; tz: number; spin: number; variant: number; active: boolean }
interface Fish { x: number; z: number; vx: number; vz: number; school: number; ang: number; rad: number; frac: number; phase: number; seed: number }

export interface CreatureSnapshot {
  birds: { x: number; y: number; z: number }[];
  critters: { x: number; z: number }[];
  fish: { x: number; z: number; y: number }[];
}

// Critter looks: 0 white sheep, 1 black sheep, 2 goat, 3 deer.  [body, head, legs, belly]
const CRITTER_COLORS = [[0xf4efe4, 0x2f2826, 0x2f2826, 0xf4efe4], [0x575150, 0x24201f, 0x24201f, 0x575150], [0xe2d2b0, 0xcdb48c, 0x6e5a44, 0xf2ead8], [0xc2803f, 0xb4733a, 0x6b4426, 0xf3e6cf]];
const FISH_COLORS = [0x8fd3ea, 0xff7a2e, 0xffd23a, 0xb9a4ff, 0xe8f4ff];
const BIRD_COLORS = [0xf6f4ee, 0x3b3f4a, 0xd08a5a];

function instanced(base: THREE.BufferGeometry, max: number, names: string[]) {
  const g = new THREE.InstancedBufferGeometry();
  g.index = base.index;
  for (const [k, a] of Object.entries(base.attributes)) g.setAttribute(k, a);
  const attrs = names.map((n) => {
    const a = new THREE.InstancedBufferAttribute(new Float32Array(max * 4), 4);
    a.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute(n, a);
    return a;
  });
  g.instanceCount = 0;
  return { g, attrs };
}

/** Rotate v by yaw about Y (TSL). */
const rotY = (v: V3, c: F, s: F): V3 => vec3(v.x.mul(c).add(v.z.mul(s)), v.y, v.z.mul(c).sub(v.x.mul(s))) as V3;
/** Rotate about X (pitch, +a noses down) and Z (bank). */
const rotX = (v: V3, a: F): V3 => vec3(v.x, v.y.mul(cos(a)).sub(v.z.mul(sin(a))), v.y.mul(sin(a)).add(v.z.mul(cos(a)))) as V3;
const rotZ = (v: V3, a: F): V3 => vec3(v.x.mul(cos(a)).sub(v.y.mul(sin(a))), v.x.mul(sin(a)).add(v.y.mul(cos(a))), v.z) as V3;

export interface Creatures {
  object: THREE.Group;
  setMap(m: ColumnMap): void;
  step(dt: number, ambTime: number): void;
  setQuality(high: boolean): void;
  snapshot(): CreatureSnapshot;
  readonly counts: { birds: number; critters: number; fish: number };
  dispose(): void;
}

export function createCreatures(opts: { highQuality?: boolean; seed?: number } = {}): Creatures {
  const rand = rng(opts.seed ?? 0x5eed);
  const between = (a: number, b: number) => a + (b - a) * rand();
  let high = opts.highQuality ?? true;
  let map: ColumnMap | null = null;
  const uTime = uniform(0);

  const flocks: Flock[] = [];
  const birds: Bird[] = [];
  const critters: Critter[] = [];
  const schools: School[] = [];
  const fish: Fish[] = [];

  const maxBirds = C.BIRD_FLOCKS[0] * C.BIRD_MAX, maxCritters = C.CRITTERS[0];
  const maxFish = C.FISH_SCHOOLS[0] * C.FISH_MAX;

  // ---- meshes ----
  const group = new THREE.Group();
  group.name = 'life-creatures';
  const lp = attribute('lifeP', 'vec4') as unknown as V4;

  // birds: instA (x, y, z, yaw), instB (pitch, bank, flap phase, flap amp), instC (rgb, -)
  const bird = instanced(birdGeometry(), maxBirds, ['instA', 'instB', 'instC']);
  {
    const A = attribute('instA', 'vec4') as unknown as V4, B = attribute('instB', 'vec4') as unknown as V4;
    const Cc = attribute('instC', 'vec4') as unknown as V4;
    const SIZE = 0.024;
    const orient = (v: V3): V3 => rotY(rotX(rotZ(v, B.y), B.x), cos(A.w), sin(A.w));
    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.7, side: THREE.DoubleSide });
    mat.positionNode = Fn(() => {
      const p = positionGeometry.toVar();
      const wing = lp.x.greaterThan(0.5);
      const lift = sin(B.z).mul(lp.y).mul(B.w).mul(1.1);
      p.y.addAssign(select(wing, lift, float(0)));
      normalLocal.assign(orient(normalGeometry as V3));
      return orient(p.mul(SIZE) as V3).add(A.xyz);
    })();
    // darker wing tips, a touch of belly shade
    mat.colorNode = varying(Cc.xyz.mul(mix(float(1), float(0.55), smoothstep(0.45, 0.75, lp.y))), 'vBirdC');
    const mesh = new THREE.Mesh(bird.g, mat);
    mesh.name = 'life-birds';
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    group.add(mesh);
  }

  // critters: instA (x, y, z, yaw), instB (walk phase, walking, head down, variant)
  const crit = instanced(critterGeometry(), maxCritters, ['instA', 'instB']);
  {
    const A = attribute('instA', 'vec4') as unknown as V4, B = attribute('instB', 'vec4') as unknown as V4;
    const SIZE = 0.03;
    const body = uniformArray(CRITTER_COLORS.map((c) => lin(c[0]!)), 'color');
    const head = uniformArray(CRITTER_COLORS.map((c) => lin(c[1]!)), 'color');
    const legs = uniformArray(CRITTER_COLORS.map((c) => lin(c[2]!)), 'color');
    const belly = uniformArray(CRITTER_COLORS.map((c) => lin(c[3]!)), 'color');
    // sheep (variants 0, 1) wear a puffier woolly body
    const wool = select(B.w.lessThan(1.5), vec3(1.35, 1.3, 1.08), vec3(1, 1, 1));
    const puffed = (v: V3): V3 => select(part.lessThan(0.5), v.sub(vec3(0, 0.64, 0)).mul(wool).add(vec3(0, 0.64, 0)), v) as V3;
    const part = lp.x;
    const isLeg = part.greaterThan(1.5), isHead = part.greaterThan(0.5).and(part.lessThan(1.5));
    const legA = sin(B.x.add(select(lp.y.greaterThan(0), float(0), float(Math.PI)))).mul(0.6).mul(B.y);
    const headA = B.z.mul(1.05);
    const pivot = (v: V3, py: number | F, pz: number | F, a: F): V3 => {
      const d = vec3(v.x, v.y.sub(py), v.z.sub(pz));
      return rotX(d as V3, a).add(vec3(0, py, pz)) as V3;
    };
    const local = (v: V3, n: boolean): V3 => {
      const hipped = n ? rotX(v, legA) : pivot(v, 0.48, lp.z, legA);
      const headed = n ? rotX(v, headA) : pivot(v, 0.74, 0.2, headA);
      return select(isLeg, hipped, select(isHead, headed, n ? v : puffed(v))) as V3;
    };
    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 });
    mat.positionNode = Fn(() => {
      const bob = sin(B.x.mul(2)).abs().mul(0.05).mul(B.y);
      const p = local(positionGeometry as V3, false).add(vec3(0, select(isLeg, float(0), bob), 0));
      normalLocal.assign(rotY(local(normalGeometry as V3, true), cos(A.w), sin(A.w)));
      return rotY(p.mul(SIZE) as V3, cos(A.w), sin(A.w)).add(A.xyz);
    })();
    mat.colorNode = varying(Fn(() => {
      const vi = uint(B.w);
      const under = smoothstep(0.58, 0.46, positionGeometry.y);
      const bodyC = mix(body.element(vi) as unknown as V3, belly.element(vi) as unknown as V3, under);
      const c = select(isLeg, legs.element(vi), select(isHead, head.element(vi), bodyC)) as unknown as V3;
      const ao = mix(float(0.55), float(1), smoothstep(0.2, 0.6, positionGeometry.y));
      return c.mul(select(isLeg, float(1), ao));
    })(), 'vCritC');
    const mesh = new THREE.Mesh(crit.g, mat);
    mesh.name = 'life-critters';
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  }

  // fish: instA (x, y, z, yaw), instB (swim phase, seed, size, variant)
  const fishM = instanced(fishGeometry(), maxFish, ['instA', 'instB']);
  {
    const A = attribute('instA', 'vec4') as unknown as V4, B = attribute('instB', 'vec4') as unknown as V4;
    const cols = uniformArray(FISH_COLORS.map(lin), 'color');
    const bend = sin(B.x.sub(lp.x.mul(4))).mul(0.16).mul(lp.x);
    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.3, metalness: 0.35, side: THREE.DoubleSide });
    mat.positionNode = Fn(() => {
      const p = positionGeometry.toVar();
      p.x.addAssign(bend);
      normalLocal.assign(rotY(normalGeometry as V3, cos(A.w), sin(A.w)));
      return rotY(p.mul(B.z) as V3, cos(A.w), sin(A.w)).add(A.xyz);
    })();
    const base = varying(cols.element(uint(B.w)) as unknown as V3, 'vFishC');
    const seed = varying(B.y, 'vFishS');
    mat.colorNode = base.mul(mix(float(1.1), float(0.75), lp.y));
    // shimmer: brief glints as each fish turns its flank; whole cycles per AMB_PERIOD (seamless wrap)
    const glint = pow(max(sin(uTime.mul((TAU * 5400) / AMB_PERIOD).add(seed.mul(40)).add(positionWorld.x.mul(90))), float(0)), float(14));
    mat.emissiveNode = base.mul(glint.mul(1.6).add(0.08));
    const mesh = new THREE.Mesh(fishM.g, mat);
    mesh.name = 'life-fish';
    mesh.frustumCulled = false;
    group.add(mesh);
  }

  // ---- spawning ----
  const randomWhere = (ok: (x: number, z: number) => boolean, tries = 400, cx = 0, cz = 0, r = HALF): [number, number] | null => {
    for (let i = 0; i < tries; i++) {
      const x = cx + (rand() * 2 - 1) * r, z = cz + (rand() * 2 - 1) * r;
      if (Math.abs(x) < HALF && Math.abs(z) < HALF && ok(x, z)) return [x, z];
    }
    return null;
  };
  const landHome = (m: ColumnMap, f: Flock) => {
    const lim = HALF - f.radius - 0.2;
    const p = randomWhere((x, z) => Math.abs(x) < lim && Math.abs(z) < lim && m.biome[columnAt(x, z)] !== Biome.OCEAN && m.water[columnAt(x, z)]! < 0.5, 300)
      ?? [between(-lim, lim), between(-lim, lim)];
    f.hx = p[0]; f.hz = p[1];
  };

  function spawnBirds(m: ColumnMap) {
    flocks.length = 0; birds.length = 0;
    const nf = C.BIRD_FLOCKS[high ? 0 : 1];
    for (let fi = 0; fi < nf; fi++) {
      const f: Flock = { hx: 0, hz: 0, alt: between(0.6, 0.95), radius: between(0.35, 0.65), dir: rand() < 0.5 ? -1 : 1, timer: between(15, 30), color: lin(BIRD_COLORS[fi % BIRD_COLORS.length]!) };
      landHome(m, f);
      flocks.push(f);
      const n = Math.round(between(C.BIRD_MIN, C.BIRD_MAX));
      for (let i = 0; i < n; i++) {
        const a = rand() * TAU;
        const x = f.hx + Math.cos(a) * f.radius, z = f.hz + Math.sin(a) * f.radius;
        birds.push({ x, z, y: Math.max(f.alt, birdFloor(m, x, z) + 0.05), vx: -Math.sin(a) * 0.3 * f.dir, vy: 0, vz: Math.cos(a) * 0.3 * f.dir, flock: fi, phase: rand() * TAU, rate: between(9, 12), amp: 1, glide: rand() * 4, bank: 0 });
      }
    }
  }

  function spawnCritter(m: ColumnMap, c: Critter, near?: [number, number]): void {
    const ok = (x: number, z: number) => critterValid(m, x, z);
    const p = (near && randomWhere(ok, 40, near[0], near[1], 0.08)) || randomWhere(ok, 600);
    if (!p) { c.active = false; return; }
    c.x = p[0]; c.z = p[1]; c.active = true;
    const b = m.biome[columnAt(c.x, c.z)];
    c.variant = b === Biome.SAVANNA ? 3 : rand() < 0.05 ? 1 : rand() < 0.2 ? 2 : 0;
  }
  function spawnCritters(m: ColumnMap) {
    critters.length = 0;
    const n = C.CRITTERS[high ? 0 : 1];
    let herdAt: [number, number] | undefined;
    let herd = -1;
    for (let i = 0; i < n; i++) {
      if (i % 5 === 0) { herd++; herdAt = undefined; }
      const c: Critter = { x: 0, z: 0, yaw: rand() * TAU, walking: 0, head: 1, phase: rand() * TAU, mode: rand() < 0.5 ? 0 : 1, timer: between(1, 6), herd, variant: 0, turn: 0, active: false };
      spawnCritter(m, c, herdAt);
      if (c.active && !herdAt) herdAt = [c.x, c.z];
      critters.push(c);
    }
  }

  function spawnFish(m: ColumnMap) {
    schools.length = 0; fish.length = 0;
    const ns = C.FISH_SCHOOLS[high ? 0 : 1];
    for (let si = 0; si < ns; si++) {
      const p = randomWhere((x, z) => fishValid(m, x, z), 1500);
      const s: School = { cx: p?.[0] ?? 0, cz: p?.[1] ?? 0, tx: p?.[0] ?? 0, tz: p?.[1] ?? 0, spin: (rand() < 0.5 ? -1 : 1) * between(0.6, 1.2), variant: si % FISH_COLORS.length, active: !!p };
      schools.push(s);
      if (!p) continue;
      const n = Math.round(between(C.FISH_MIN, C.FISH_MAX));
      for (let i = 0; i < n; i++) {
        fish.push({ x: s.cx, z: s.cz, vx: 0, vz: 0, school: si, ang: rand() * TAU, rad: between(0.008, 0.035), frac: between(0.3, 0.65), phase: rand() * TAU, seed: rand() });
      }
    }
  }

  // ---- steering ----
  function stepBirds(m: ColumnMap, dt: number) {
    const lim = HALF - C.BIRD_MARGIN;
    const n = birds.length;
    for (const f of flocks) {
      f.timer -= dt;
      if (f.timer < 0) { f.timer = between(15, 30); landHome(m, f); }
    }
    for (let i = 0; i < n; i++) {
      const b = birds[i]!, f = flocks[b.flock]!;
      let ax = 0, ay = 0, az = 0;
      // circle the flock home
      const dx = b.x - f.hx, dz = b.z - f.hz, d = Math.max(1e-4, Math.hypot(dx, dz));
      const tx = (-dz / d) * f.dir, tz = (dx / d) * f.dir;
      const pull = (f.radius - d) / f.radius;
      const speed = 0.3;
      ax += (tx * speed + (dx / d) * pull * 0.4 - b.vx) * 1.4;
      az += (tz * speed + (dz / d) * pull * 0.4 - b.vz) * 1.4;
      // boids: separation, alignment, cohesion with flock mates
      let cx = 0, cy = 0, cz = 0, avx = 0, avz = 0, k = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const o = birds[j]!;
        const ex = b.x - o.x, ey = b.y - o.y, ez = b.z - o.z, e2 = ex * ex + ey * ey + ez * ez;
        if (e2 < 0.05 * 0.05 && e2 > 1e-8) { const s = (0.05 - Math.sqrt(e2)) * 30; ax += ex * s; ay += ey * s; az += ez * s; }
        if (o.flock !== b.flock) continue;
        cx += o.x; cy += o.y; cz += o.z; avx += o.vx; avz += o.vz; k++;
      }
      if (k > 0) {
        ax += (cx / k - b.x) * 0.5 + (avx / k - b.vx) * 0.5;
        az += (cz / k - b.z) * 0.5 + (avz / k - b.vz) * 0.5;
        ay += (cy / k - b.y) * 0.3;
      }
      // soft walls (the hard reflect below is the guarantee)
      const wall = HALF - 0.35;
      if (Math.abs(b.x) > wall) ax -= Math.sign(b.x) * (Math.abs(b.x) - wall) * 6;
      if (Math.abs(b.z) > wall) az -= Math.sign(b.z) * (Math.abs(b.z) - wall) * 6;
      // altitude: hold the flock height, climb over terrain ahead
      const aheadFloor = birdFloor(m, Math.max(-lim, Math.min(lim, b.x + b.vx * 0.6)), Math.max(-lim, Math.min(lim, b.z + b.vz * 0.6)));
      const ty = Math.max(f.alt + 0.04 * Math.sin(b.phase * 0.05 + i), aheadFloor + 0.08);
      ay += (ty - b.y) * 2.2 - b.vy * 1.6;

      b.vx += ax * dt; b.vy += ay * dt; b.vz += az * dt;
      const hs = Math.hypot(b.vx, b.vz), cl = Math.min(0.45, Math.max(0.18, hs));
      if (hs > 1e-6) { b.vx *= cl / hs; b.vz *= cl / hs; }
      b.vy = Math.max(-0.2, Math.min(0.25, b.vy));
      const oldYaw = Math.atan2(b.vx, b.vz);
      b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt;
      // hard bounds: reflect at the block faces, never below terrain + clearance
      if (b.x > lim) { b.x = 2 * lim - b.x; b.vx = -Math.abs(b.vx); } else if (b.x < -lim) { b.x = -2 * lim - b.x; b.vx = Math.abs(b.vx); }
      if (b.z > lim) { b.z = 2 * lim - b.z; b.vz = -Math.abs(b.vz); } else if (b.z < -lim) { b.z = -2 * lim - b.z; b.vz = Math.abs(b.vz); }
      const floor = birdFloor(m, b.x, b.z);
      if (b.y < floor) { b.y = floor; b.vy = Math.max(0, b.vy); }
      if (b.y > C.BIRD_CEIL) { b.y = C.BIRD_CEIL; b.vy = Math.min(0, b.vy); }
      // look: bank into turns, glide now and then
      let dyaw = Math.atan2(b.vx, b.vz) - oldYaw;
      if (dyaw > Math.PI) dyaw -= TAU; else if (dyaw < -Math.PI) dyaw += TAU;
      b.bank += (Math.max(-0.7, Math.min(0.7, -dyaw / Math.max(dt, 1e-3) * 0.35)) - b.bank) * Math.min(1, dt * 4);
      b.glide -= dt;
      if (b.glide < -1.5) b.glide = between(1.5, 5);
      const target = b.glide < 0 || b.vy > 0.05 ? 1 : 0.12;
      b.amp += (target - b.amp) * Math.min(1, dt * 3);
      b.phase += dt * b.rate * (0.4 + 0.6 * b.amp);
    }
  }

  function stepCritters(m: ColumnMap, dt: number) {
    const herdC = new Map<number, [number, number, number]>();
    for (const c of critters) if (c.active) { const h = herdC.get(c.herd) ?? [0, 0, 0]; h[0] += c.x; h[1] += c.z; h[2]++; herdC.set(c.herd, h); }
    for (const c of critters) {
      if (!c.active) continue;
      if (!critterValid(m, c.x, c.z)) { spawnCritter(m, c); if (!c.active) continue; }
      c.timer -= dt;
      if (c.timer < 0) { c.mode = c.mode === 1 ? 0 : 1; c.timer = c.mode === 1 ? between(2, 6) : between(3, 8); }
      if (c.mode === 1) {
        // wander, stay near the herd
        c.turn += (rand() - 0.5) * dt * 3;
        c.turn *= Math.exp(-dt);
        let want = c.yaw + c.turn * dt * 2;
        const h = herdC.get(c.herd);
        if (h && h[2] > 1) {
          const hx = h[0] / h[2] - c.x, hz = h[1] / h[2] - c.z;
          if (Math.hypot(hx, hz) > 0.1) want = Math.atan2(hx, hz);
        }
        let dy = want - c.yaw;
        dy = Math.atan2(Math.sin(dy), Math.cos(dy));
        c.yaw += Math.max(-1.5 * dt, Math.min(1.5 * dt, dy));
        // look ahead; turn toward open ground or stop
        const probe = (yaw: number, d: number) => critterValid(m, c.x + Math.sin(yaw) * d, c.z + Math.cos(yaw) * d);
        if (!probe(c.yaw, 0.03)) {
          const alt = [0.6, -0.6, 1.2, -1.2, 2.0, -2.0, Math.PI].find((o) => probe(c.yaw + o, 0.03));
          if (alt === undefined) { c.mode = 0; c.timer = between(2, 4); } else c.yaw += alt;
        }
        const sp = 0.028 * c.walking;
        const nx = c.x + Math.sin(c.yaw) * sp * dt, nz = c.z + Math.cos(c.yaw) * sp * dt;
        if (critterValid(m, nx, nz)) { c.x = nx; c.z = nz; } else { c.mode = 0; c.timer = between(1, 3); c.yaw += Math.PI * 0.5; }
      }
      const walk = c.mode === 1 ? 1 : 0;
      c.walking += (walk - c.walking) * Math.min(1, dt * 3);
      c.head += ((walk ? 0 : 1) - c.head) * Math.min(1, dt * 2);
      c.phase += dt * TAU * 2.2 * c.walking;
    }
  }

  function stepFish(m: ColumnMap, dt: number) {
    for (const s of schools) {
      if (!s.active) continue;
      if (!fishValid(m, s.cx, s.cz)) {
        const p = randomWhere((x, z) => fishValid(m, x, z), 800);
        if (!p) { s.active = false; continue; }
        s.cx = s.tx = p[0]; s.cz = s.tz = p[1];
      }
      const dx = s.tx - s.cx, dz = s.tz - s.cz, d = Math.hypot(dx, dz);
      if (d < 0.02) {
        const p = randomWhere((x, z) => fishValid(m, x, z), 30, s.cx, s.cz, 0.3);
        if (p) { s.tx = p[0]; s.tz = p[1]; }
      } else {
        const st = Math.min(d, 0.06 * dt);
        const nx = s.cx + (dx / d) * st, nz = s.cz + (dz / d) * st;
        if (fishValid(m, nx, nz)) { s.cx = nx; s.cz = nz; } else { s.tx = s.cx; s.tz = s.cz; }
      }
    }
    for (const f of fish) {
      const s = schools[f.school]!;
      if (!s.active) continue;
      f.ang += dt * s.spin * (1.2 - f.rad * 10);
      const tx = s.cx + Math.cos(f.ang) * f.rad, tz = s.cz + Math.sin(f.ang) * f.rad;
      f.vx += ((tx - f.x) * 6 - f.vx * 2.5) * dt;
      f.vz += ((tz - f.z) * 6 - f.vz * 2.5) * dt;
      const sp = Math.hypot(f.vx, f.vz);
      if (sp > 0.12) { f.vx *= 0.12 / sp; f.vz *= 0.12 / sp; }
      const nx = f.x + f.vx * dt, nz = f.z + f.vz * dt;
      if (fishValid(m, nx, nz)) { f.x = nx; f.z = nz; } else if (!fishValid(m, f.x, f.z)) { f.x = s.cx; f.z = s.cz; f.vx = f.vz = 0; } else { f.vx *= -0.5; f.vz *= -0.5; }
      f.phase += dt * (8 + sp * 120);
    }
  }

  const fishY = (m: ColumnMap, f: Fish) => {
    const c = columnAt(f.x, f.z);
    const s = m.surfY[c]!, w = m.water[c]!;
    return voxelToWorldY(Math.min(s + w - 0.4, Math.max(s + 0.35, s + w * f.frac)));
  };

  // ---- upload ----
  function upload(m: ColumnMap) {
    {
      const [A, B, Cc] = bird.attrs as [THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute];
      birds.forEach((b, i) => {
        const hs = Math.hypot(b.vx, b.vz);
        A.setXYZW(i, b.x, b.y, b.z, Math.atan2(b.vx, b.vz));
        B.setXYZW(i, -Math.atan2(b.vy, hs) * 0.8, b.bank, b.phase, b.amp);
        const col = flocks[b.flock]!.color;
        Cc.setXYZW(i, col.r, col.g, col.b, 0);
      });
      A.needsUpdate = B.needsUpdate = Cc.needsUpdate = true;
      bird.g.instanceCount = birds.length;
    }
    {
      const [A, B] = crit.attrs as [THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute];
      let n = 0;
      for (const c of critters) {
        if (!c.active) continue;
        A.setXYZW(n, c.x, voxelToWorldY(heightAt(m, c.x, c.z)) - 0.002, c.z, c.yaw);
        B.setXYZW(n, c.phase, c.walking, c.head, c.variant);
        n++;
      }
      A.needsUpdate = B.needsUpdate = true;
      crit.g.instanceCount = n;
    }
    {
      const [A, B] = fishM.attrs as [THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute];
      let n = 0;
      for (const f of fish) {
        const s = schools[f.school]!;
        if (!s.active) continue;
        A.setXYZW(n, f.x, fishY(m, f), f.z, Math.atan2(f.vx, f.vz) + Math.sin(f.phase * 0.5) * 0.15);
        B.setXYZW(n, f.phase, f.seed, 0.019 * (0.8 + 0.4 * f.seed), s.variant);
        n++;
      }
      A.needsUpdate = B.needsUpdate = true;
      fishM.g.instanceCount = n;
    }
  }

  let spawned = false;
  const respawnAll = (m: ColumnMap) => { spawnBirds(m); spawnCritters(m); spawnFish(m); spawned = true; };

  return {
    object: group,
    get counts() {
      return { birds: birds.length, critters: critters.filter((c) => c.active).length, fish: fish.filter((f) => schools[f.school]!.active).length };
    },
    setMap(m) {
      map = m;
      if (!spawned) { respawnAll(m); return; }
      for (const c of critters) if (!c.active || !critterValid(m, c.x, c.z)) spawnCritter(m, c);
      for (const s of schools) if (!s.active) {
        const p = randomWhere((x, z) => fishValid(m, x, z), 800);
        if (p) { s.active = true; s.cx = s.tx = p[0]; s.cz = s.tz = p[1]; for (const f of fish) if (f.school === schools.indexOf(s)) { f.x = s.cx; f.z = s.cz; } }
      }
    },
    step(dt, ambTime) {
      uTime.value = ((ambTime % AMB_PERIOD) + AMB_PERIOD) % AMB_PERIOD;
      if (!map) return;
      // fixed small substeps: steering is stable at any frame rate
      const n = Math.min(8, Math.max(1, Math.ceil(dt / (1 / 60))));
      const h = Math.min(dt, 0.25) / n;
      for (let i = 0; i < n; i++) { stepBirds(map, h); stepCritters(map, h); stepFish(map, h); }
      upload(map);
    },
    setQuality(v) { if (v === high) return; high = v; if (map) respawnAll(map); },
    snapshot() {
      const m = map;
      return {
        birds: birds.map((b) => ({ x: b.x, y: b.y, z: b.z })),
        critters: critters.filter((c) => c.active).map((c) => ({ x: c.x, z: c.z })),
        fish: m ? fish.filter((f) => schools[f.school]!.active).map((f) => ({ x: f.x, z: f.z, y: fishY(m, f) })) : [],
      };
    },
    dispose() {
      for (const o of group.children as THREE.Mesh[]) { o.geometry.dispose(); (o.material as THREE.Material).dispose(); }
      group.removeFromParent();
    },
  };
}
