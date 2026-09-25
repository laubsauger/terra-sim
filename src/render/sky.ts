// §T.39 sky + day/night. CPU time-of-day model drives shared uniforms (sun/moon direction, sky
// colours); `skyColor(dir)` is the one TSL sky used by the backdrop, the floor haze and water
// reflections, so everything reflects the same sky. Time of day comes from ambTime only (V17).
import * as THREE from 'three/webgpu';
import {
  Fn, uniform, vec3, float, dot, max, mix, pow, saturate, smoothstep, hash, floor, normalize, step,
} from 'three/tsl';

type V3 = THREE.Node<'vec3'>;

/** Default look: late golden hour, sun ~14° high. */
export const GOLDEN_HOUR = 0.705;

/** Horizontal compass of the sun path (world xz). Sunset lands at the camera's left in the hero view. */
const WEST = new THREE.Vector2(-0.75, 0.66).normalize();
const SOUTH = new THREE.Vector2(-0.66, -0.75).normalize(); // noon sun comes from behind the hero view
const MAX_ELEV = THREE.MathUtils.degToRad(58);

interface Key { e: number; zen: number; hor: number; gnd: number; sun: number; sunI: number; hemi: number }
// Keyframes by sun elevation (degrees). Colours are sRGB hex, interpolated in linear space.
const KEYS: Key[] = [
  { e: -20, zen: 0x03060f, hor: 0x0b1226, gnd: 0x07090f, sun: 0x9fb4ff, sunI: 0.0, hemi: 0.1 },
  { e: -6, zen: 0x0e1838, hor: 0x3a3558, gnd: 0x0f1018, sun: 0xff6a3a, sunI: 0.0, hemi: 0.35 },
  { e: 0, zen: 0x2c4478, hor: 0xe07a52, gnd: 0x2a2224, sun: 0xff6a2a, sunI: 0.8, hemi: 0.55 },
  { e: 6, zen: 0x3e62a6, hor: 0xf4a26a, gnd: 0x3a3028, sun: 0xff9a4c, sunI: 2.6, hemi: 0.75 },
  { e: 14, zen: 0x3b64b0, hor: 0xf2a676, gnd: 0x453a30, sun: 0xffc285, sunI: 4.6, hemi: 1.0 },
  { e: 30, zen: 0x3f78cc, hor: 0xc4d8ee, gnd: 0x4a443c, sun: 0xfff0dc, sunI: 4.0, hemi: 1.0 },
  { e: 90, zen: 0x356fcc, hor: 0xcfe2f6, gnd: 0x4c4640, sun: 0xffffff, sunI: 4.2, hemi: 1.05 },
];

const lin = (hex: number) => new THREE.Color(hex);

export interface SkyState {
  tod: number;
  sunDir: THREE.Vector3;   // towards the sun
  moonDir: THREE.Vector3;  // towards the moon
  sunElevDeg: number;
  /** 0 day … 1 full night. */
  night: number;
  zenith: THREE.Color; horizon: THREE.Color; ground: THREE.Color; sunColor: THREE.Color;
  sunIntensity: number; hemiIntensity: number;
}

/** Pure CPU time-of-day model. tod ∈ [0,1): 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset. */
export function skyAt(tod: number): SkyState {
  const t = ((tod % 1) + 1) % 1;
  const a = (t - 0.25) * Math.PI * 2; // 0 at sunrise, π/2 noon, π sunset
  const elev = Math.asin(Math.sin(a) * Math.sin(MAX_ELEV));
  // horizontal: east (-WEST) → south → west over the day, continuing round at night.
  const hx = -Math.cos(a) * WEST.x + Math.sin(a) * SOUTH.x;
  const hz = -Math.cos(a) * WEST.y + Math.sin(a) * SOUTH.y;
  const ce = Math.cos(elev);
  const sunDir = new THREE.Vector3(hx * ce, Math.sin(elev), hz * ce).normalize();
  // Moon: roughly opposite, slightly off so it is not a mirror of the sun path.
  const moonDir = new THREE.Vector3(-sunDir.x * 0.9 + 0.25, -sunDir.y, -sunDir.z * 0.9 - 0.2).normalize();
  const e = THREE.MathUtils.radToDeg(elev);
  let i = 0;
  while (i < KEYS.length - 2 && e > KEYS[i + 1]!.e) i++;
  const k0 = KEYS[i]!, k1 = KEYS[i + 1]!;
  const f = THREE.MathUtils.clamp((e - k0.e) / (k1.e - k0.e), 0, 1);
  const c = (h0: number, h1: number) => lin(h0).lerp(lin(h1), f);
  return {
    tod: t, sunDir, moonDir, sunElevDeg: e,
    night: THREE.MathUtils.clamp((-e - 2) / 12, 0, 1),
    zenith: c(k0.zen, k1.zen), horizon: c(k0.hor, k1.hor), ground: c(k0.gnd, k1.gnd), sunColor: c(k0.sun, k1.sun),
    sunIntensity: THREE.MathUtils.lerp(k0.sunI, k1.sunI, f),
    hemiIntensity: THREE.MathUtils.lerp(k0.hemi, k1.hemi, f),
  };
}

/** Shared sky uniforms, updated by applySky(). */
export const skyU = {
  sunDir: uniform(new THREE.Vector3(0, 1, 0)),
  moonDir: uniform(new THREE.Vector3(0, -1, 0)),
  sunColor: uniform(new THREE.Color(1, 1, 1)),
  zenith: uniform(new THREE.Color()),
  horizon: uniform(new THREE.Color()),
  ground: uniform(new THREE.Color()),
  night: uniform(0),
  sunIntensity: uniform(1),
};

export function applySky(s: SkyState): void {
  skyU.sunDir.value.copy(s.sunDir);
  skyU.moonDir.value.copy(s.moonDir);
  skyU.sunColor.value.copy(s.sunColor);
  skyU.zenith.value.copy(s.zenith);
  skyU.horizon.value.copy(s.horizon);
  skyU.ground.value.copy(s.ground);
  skyU.night.value = s.night;
  skyU.sunIntensity.value = s.sunIntensity;
}

/**
 * Sky radiance (linear HDR) in world direction `dir` (normalized). `discs` adds the sun and moon
 * discs + stars (backdrop), without them it is the smooth sky for reflections and haze.
 */
export const skyColor = (dir: V3, discs = true): V3 => Fn(() => {
  const up = dir.y;
  // Three-stop gradient: warm horizon → pale cyan-blue band → zenith. A straight horizon→zenith mix
  // goes lavender at golden hour (and so do the water reflections).
  const mid = (skyU.zenith as unknown as V3).mul(vec3(1.45, 1.75, 1.55)).add(skyU.horizon.mul(0.12));
  const sky = mix(mix(skyU.horizon, mid, smoothstep(0.0, 0.22, up)), skyU.zenith, smoothstep(0.18, 0.9, up));
  const below = mix(skyU.horizon, skyU.ground, saturate(up.negate().mul(4)));
  let col = mix(below, sky, step(0, up)) as V3;
  // Mie-ish halo around the sun, stronger near the horizon (golden-hour glow).
  const cs = max(dot(dir, skyU.sunDir), 0).toVar();
  const lowSun = float(1).sub(saturate(skyU.sunDir.y.mul(2.5)));
  col = col.add(skyU.sunColor.mul(pow(cs, 6).mul(0.35).add(pow(cs, 48).mul(0.6)).mul(lowSun.mul(0.8).add(0.3))
    .mul(saturate(skyU.sunDir.y.add(0.12).mul(6)))));
  if (discs) {
    // Sun: HDR disc (blooms into a soft star) + tight corona, cut by the horizon.
    const aboveH = saturate(up.mul(40).add(0.5));
    const disc = smoothstep(0.99925, 0.9996, cs).mul(aboveH);
    col = col.add(skyU.sunColor.mul(disc.mul(skyU.sunIntensity).mul(5).add(pow(cs, 900).mul(skyU.sunIntensity).mul(1.5).mul(aboveH))));
    // Moon: disc with a terminator facing the sun (phase from the sun-moon angle), cool halo.
    const cm = max(dot(dir, skyU.moonDir), 0).toVar();
    const inDisc = smoothstep(0.99935, 0.99955, cm).mul(aboveH);
    const offs = dir.sub(skyU.moonDir.mul(dot(dir, skyU.moonDir)));            // tangent-plane offset
    const toSun = normalize(skyU.sunDir.sub(skyU.moonDir.mul(dot(skyU.sunDir, skyU.moonDir))).add(vec3(1e-5)));
    const r = 0.034;                                                              // ≈ acos(0.99942)
    const phase = dot(skyU.sunDir, skyU.moonDir).negate();                        // 1 full … -1 new
    const lit = smoothstep(-0.12, 0.12, dot(offs, toSun).div(r).add(phase));
    const moonVis = float(0.35).add(skyU.night.mul(0.65));                        // pale by day, bright at night
    col = col.add(vec3(0.82, 0.86, 1.0).mul(inDisc.mul(mix(0.06, 1.0, lit)).mul(moonVis).mul(2.6)))
      .add(vec3(0.3, 0.4, 0.7).mul(pow(cm, 80).mul(0.3).mul(skyU.night)));
    // Stars: hashed cells on the direction sphere, twinkle-free (ambient stays calm).
    const cell = floor(normalize(dir).mul(260));
    const hsh = hash(cell.x.add(cell.y.mul(157.1)).add(cell.z.mul(311.7)));
    const star = smoothstep(0.9975, 1.0, hsh).mul(skyU.night).mul(saturate(up.mul(6)));
    col = col.add(vec3(star.mul(2.2)));
  }
  // Keep it finite and non-negative for bloom / tone mapping.
  return max(col, vec3(0));
})() as V3;
