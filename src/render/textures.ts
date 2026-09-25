// Small tileable look textures generated once on the CPU at init (V10: fixed allocation, no per-frame work).
// Texture lookups replace per-pixel 3D noise in the hot fragment paths (terrain, water, sides).
import * as THREE from 'three/webgpu';
import { PeriodicNoise2D } from '../core/noise';
import { PCG32 } from '../core/rng';

const TAU = Math.PI * 2;

function makeTex(data: Uint8Array, size: number, srgb = false): THREE.DataTexture {
  const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = 4;
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}

/**
 * Wave normals: sum of sines with integer wave vectors (exactly tileable), steepness falling with
 * frequency. RG = slope dH/du, dH/dv encoded 0.5 ± , B = height 0..1, A = crest (foam) mask.
 */
function waveTexture(size: number, rng: PCG32): THREE.DataTexture {
  const waves: { kx: number; kz: number; a: number; ph: number }[] = [];
  for (let i = 0; i < 40; i++) {
    const k = 2 + Math.floor(rng.nextFloat() * 14);
    const ang = rng.nextFloat() * TAU;
    const kx = Math.round(Math.cos(ang) * k), kz = Math.round(Math.sin(ang) * k);
    if (kx === 0 && kz === 0) continue;
    waves.push({ kx, kz, a: 1 / Math.pow(Math.hypot(kx, kz), 1.35), ph: rng.nextFloat() * TAU });
  }
  const h = new Float32Array(size * size), gx = new Float32Array(size * size), gz = new Float32Array(size * size);
  let hMin = Infinity, hMax = -Infinity, gMax = 0;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = x / size, v = y / size;
    let s = 0, dx = 0, dz = 0;
    for (const w of waves) {
      const p = TAU * (w.kx * u + w.kz * v) + w.ph;
      // sharpened crests: (1 - |sin|) style trochoid approximation
      const sn = Math.sin(p), cs = Math.cos(p);
      s += w.a * sn; dx += w.a * cs * TAU * w.kx; dz += w.a * cs * TAU * w.kz;
    }
    const i = x + y * size;
    h[i] = s; gx[i] = dx; gz[i] = dz;
    hMin = Math.min(hMin, s); hMax = Math.max(hMax, s); gMax = Math.max(gMax, Math.abs(dx), Math.abs(dz));
  }
  const d = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const hn = (h[i]! - hMin) / (hMax - hMin);
    d[i * 4] = Math.round(127.5 + 127 * gx[i]! / gMax);
    d[i * 4 + 1] = Math.round(127.5 + 127 * gz[i]! / gMax);
    d[i * 4 + 2] = Math.round(255 * hn);
    d[i * 4 + 3] = Math.round(255 * Math.max(0, Math.min(1, (hn - 0.62) / 0.3)));
  }
  return makeTex(d, size);
}

/**
 * Caustics: tileable Worley F2-F1 ridges (bright lines on cell borders). Two channels with
 * different cell counts so shaders can min() two scrolled layers into a moving caustic web.
 */
function causticTexture(size: number, rng: PCG32): THREE.DataTexture {
  const d = new Uint8Array(size * size * 4);
  const chan = (cells: number, c: number) => {
    const px = new Float32Array(cells * cells), pz = new Float32Array(cells * cells);
    for (let i = 0; i < cells * cells; i++) { px[i] = rng.nextFloat(); pz[i] = rng.nextFloat(); }
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const u = (x / size) * cells, v = (y / size) * cells;
      const cu = Math.floor(u), cv = Math.floor(v);
      let f1 = 9, f2 = 9;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        const gx = cu + i, gz = cv + j;
        const k = ((gx % cells) + cells) % cells + (((gz % cells) + cells) % cells) * cells;
        const dx = gx + px[k]! - u, dz = gz + pz[k]! - v;
        const dd = Math.sqrt(dx * dx + dz * dz);
        if (dd < f1) { f2 = f1; f1 = dd; } else if (dd < f2) f2 = dd;
      }
      const ridge = Math.pow(Math.max(0, 1 - (f2 - f1) / 0.22), 3);
      d[(x + y * size) * 4 + c] = Math.round(255 * ridge);
    }
  };
  chan(7, 0); chan(11, 1);
  for (let i = 0; i < size * size; i++) { d[i * 4 + 2] = 0; d[i * 4 + 3] = 255; }
  return makeTex(d, size);
}

/**
 * Detail noise, all channels tileable fbm in 0..1: R broad (patches), G mid, B fine grain,
 * A ridged (rock cracks / strata micro detail).
 */
function detailTexture(size: number, seed: number): THREE.DataTexture {
  const n = [0, 1, 2, 3].map((k) => new PeriodicNoise2D(new PCG32(seed, 900 + k)));
  const f: ((u: number, v: number) => number)[] = [
    (u, v) => n[0]!.fbm(u, v, 4, 4, 4, 0.5),
    (u, v) => n[1]!.fbm(u, v, 16, 16, 3, 0.5),
    (u, v) => n[2]!.fbm(u, v, 64, 64, 2, 0.5),
    (u, v) => n[3]!.ridged(u, v, 12, 12, 4, 0.5),
  ];
  const raw = new Float32Array(size * size * 4);
  const lo = [Infinity, Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity, -Infinity];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    for (let c = 0; c < 4; c++) {
      const val = f[c]!(x / size, y / size);
      raw[(x + y * size) * 4 + c] = val;
      lo[c] = Math.min(lo[c]!, val); hi[c] = Math.max(hi[c]!, val);
    }
  }
  // Stretch each channel to the full 0..1 range.
  const d = new Uint8Array(size * size * 4);
  for (let i = 0; i < raw.length; i++) d[i] = Math.round(255 * (raw[i]! - lo[i & 3]!) / (hi[i & 3]! - lo[i & 3]!));
  return makeTex(d, size);
}

export interface LookTextures { waves: THREE.DataTexture; caustics: THREE.DataTexture; detail: THREE.DataTexture; dispose(): void }

let shared: LookTextures | null = null;

/** Built once per page (≈ 2 MB total), shared by every material. */
export function lookTextures(): LookTextures {
  if (shared) return shared;
  const rng = new PCG32(0x1ee7, 7);
  const t = { waves: waveTexture(256, rng), caustics: causticTexture(256, rng), detail: detailTexture(256, 0xd37a) };
  shared = { ...t, dispose() { t.waves.dispose(); t.caustics.dispose(); t.detail.dispose(); shared = null; } };
  return shared;
}
