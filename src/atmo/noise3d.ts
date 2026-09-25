// Tileable 3D cloud noise, generated once on the CPU at init (fixed size, V10).
// R = Perlin-Worley (billowy base shape), G/B/A = inverted Worley at 4/8/16 cells per tile
// (cauliflower erosion octaves). All channels tile with the texture (RepeatWrapping).
import * as THREE from 'three/webgpu';

const hash3 = (x: number, y: number, z: number, s: number) => {
  let h = Math.imul(x, 0x8da6b343) ^ Math.imul(y, 0xd8163841) ^ Math.imul(z, 0xcb1ab31f) ^ Math.imul(s, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
};
const wrap = (i: number, p: number) => ((i % p) + p) % p;

/** Inverted Worley F1 (1 at feature points), period `p` cells over the unit tile. */
function worley(x: number, y: number, z: number, p: number, seed: number): number {
  const fx = x * p, fy = y * p, fz = z * p;
  const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
  let d = 9;
  for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const cx = ix + dx, cy = iy + dy, cz = iz + dz;
    const wx = wrap(cx, p), wy = wrap(cy, p), wz = wrap(cz, p);
    const px = cx + hash3(wx, wy, wz, seed), py = cy + hash3(wx, wy, wz, seed + 1), pz = cz + hash3(wx, wy, wz, seed + 2);
    const e = (px - fx) ** 2 + (py - fy) ** 2 + (pz - fz) ** 2;
    if (e < d) d = e;
  }
  return 1 - Math.min(1, Math.sqrt(d));
}

/** Tileable gradient noise in [-1, 1], period `p`. */
function perlin(x: number, y: number, z: number, p: number, seed: number): number {
  const fx = x * p, fy = y * p, fz = z * p;
  const ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
  const tx = fx - ix, ty = fy - iy, tz = fz - iz;
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const g = (cx: number, cy: number, cz: number, dx: number, dy: number, dz: number) => {
    const wx = wrap(cx, p), wy = wrap(cy, p), wz = wrap(cz, p);
    const a = hash3(wx, wy, wz, seed) * Math.PI * 2, b = Math.acos(2 * hash3(wx, wy, wz, seed + 7) - 1);
    return Math.sin(b) * Math.cos(a) * dx + Math.sin(b) * Math.sin(a) * dy + Math.cos(b) * dz;
  };
  const u = fade(tx), v = fade(ty), w = fade(tz);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const x00 = lerp(g(ix, iy, iz, tx, ty, tz), g(ix + 1, iy, iz, tx - 1, ty, tz), u);
  const x10 = lerp(g(ix, iy + 1, iz, tx, ty - 1, tz), g(ix + 1, iy + 1, iz, tx - 1, ty - 1, tz), u);
  const x01 = lerp(g(ix, iy, iz + 1, tx, ty, tz - 1), g(ix + 1, iy, iz + 1, tx - 1, ty, tz - 1), u);
  const x11 = lerp(g(ix, iy + 1, iz + 1, tx, ty - 1, tz - 1), g(ix + 1, iy + 1, iz + 1, tx - 1, ty - 1, tz - 1), u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 1.6;
}

const remap = (v: number, a: number, b: number, c: number, d: number) => c + ((v - a) / (b - a)) * (d - c);
const sat = (v: number) => Math.min(1, Math.max(0, v));

let cached: THREE.Data3DTexture | null = null;

/** Shared 3D noise (size³ RGBA8). */
export function cloudNoiseTexture(size = 32): THREE.Data3DTexture {
  if (cached) return cached;
  const data = new Uint8Array(size * size * size * 4);
  for (let z = 0; z < size; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const px = x / size, py = y / size, pz = z / size;
    const pf = perlin(px, py, pz, 4, 11) * 0.625 + perlin(px, py, pz, 8, 12) * 0.25 + perlin(px, py, pz, 16, 13) * 0.125;
    const w4 = worley(px, py, pz, 4, 21), w8 = worley(px, py, pz, 8, 22), w16 = worley(px, py, pz, 16, 23);
    const wf = w4 * 0.625 + w8 * 0.25 + w16 * 0.125;
    const pw = sat(remap(pf * 0.5 + 0.5, wf - 1, 1, 0, 1));
    const o = ((z * size + y) * size + x) * 4;
    data[o] = Math.round(pw * 255);
    data[o + 1] = Math.round(sat(w4 * 0.625 + w8 * 0.25 + w16 * 0.125) * 255);
    data[o + 2] = Math.round(sat(w8 * 0.625 + w16 * 0.375) * 255);
    data[o + 3] = Math.round(sat(w16) * 255);
  }
  const tex = new THREE.Data3DTexture(data, size, size, size);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  tex.name = 'cloudNoise';
  cached = tex;
  return tex;
}
