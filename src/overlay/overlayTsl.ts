// Overlay colour science (GPU half): value scales, LUT lookup, contour lines, and the inverse of the post
// tone map so on-screen overlay colours match the legend.
import * as THREE from 'three/webgpu';
import { float, int, vec3, clamp, floor, sqrt, log, log2, max, min, mix, smoothstep, fwidth, abs, dot, uniformArray, Fn } from 'three/tsl';
import { LUT_N, bakeRamp, type Ramp, type Scale } from './colormaps';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/** Scale spec → t in 0..1 (mirror of colormaps.scaleT). */
export function tScale(s: Scale, v: F): F {
  let t: F;
  switch (s.kind) {
    case 'linear': t = v.sub(s.v0).div(s.v1 - s.v0); break;
    case 'sqrt': t = sqrt(max(v.sub(s.v0).div(s.v1 - s.v0), 0)); break;
    case 'log': t = log2(max(v, 1e-30).div(s.v0)).div(Math.log2(s.v1 / s.v0)); break;
    case 'log1p': t = log(max(v, 0).div(s.s).add(1)).div(Math.log1p(s.v1 / s.s)); break;
    case 'split': {
      const lo = v.sub(s.lo).div(s.mid - s.lo).mul(0.5);
      const hi = v.sub(s.mid).div(s.hi - s.mid).mul(0.5).add(0.5);
      t = mix(lo, hi, v.greaterThanEqual(s.mid).toFloat()) as F;
      break;
    }
  }
  return clamp(t, 0, 1) as F;
}

/** Baked OKLab ramp as a uniform LUT; lookup(t) interpolates linearly between entries. */
export function rampLut(r: Ramp) {
  const lut = uniformArray(bakeRamp(r), 'color');
  return (t: F): V3 => {
    const x = clamp(t, 0, 1).mul(LUT_N - 1);
    const i0 = min(floor(x), LUT_N - 2);
    const e = (i: THREE.Node<'int'>) => lut.element(i) as unknown as V3;
    return mix(e(int(i0)), e(int(i0).add(1)), x.sub(i0)) as unknown as V3;
  };
}

/**
 * Anti-aliased iso-line strength (0..1) where `t` crosses any of `levels`, ~widthPx wide on screen.
 * Screen-space derivatives: call only in uniform control flow (no Discard / If before it).
 */
export function isoLines(t: F, levels: number[], widthPx: number): F {
  const fw = max(fwidth(t), 1e-6);
  let m: F = float(0);
  for (const l of levels) m = max(m, float(1).sub(smoothstep(widthPx * 0.5 - 0.5, widthPx * 0.5 + 0.5, abs(t.sub(l)).div(fw)))) as F;
  return m;
}

// ACES filmic (three.js ACESFilmicToneMapping: exposure/0.6, input mat, RRT+ODT fit, output mat), inverted.
// Rows of the true matrices (three lists them transposed for GLSL's column-major constructor).
const ACES_IN = [[0.59719, 0.35458, 0.04823], [0.076, 0.90834, 0.01566], [0.0284, 0.13383, 0.83777]];
const ACES_OUT = [[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]];
function inv3(m: number[][]): number[][] {
  const M = new THREE.Matrix3().set(...(m.flat() as [number, number, number, number, number, number, number, number, number])).invert();
  const e = M.elements; // column-major
  return [[e[0]!, e[3]!, e[6]!], [e[1]!, e[4]!, e[7]!], [e[2]!, e[5]!, e[8]!]];
}
const IN_INV = inv3(ACES_IN), OUT_INV = inv3(ACES_OUT);
const mulRows = (m: number[][], c: V3): V3 => vec3(dot(vec3(...(m[0] as [number, number, number])), c),
  dot(vec3(...(m[1] as [number, number, number])), c), dot(vec3(...(m[2] as [number, number, number])), c)) as V3;

/** Largest HDR luminance we emit: keeps light overlay colours under the bloom threshold (post.ts, 1.1). */
const HDR_MAX_LUM = 1.0;

/**
 * Display-linear colour (what the legend shows) → scene-linear HDR that the post chain's ACES + exposure maps
 * back to it. The grade's mild S-curve / vignette on top are left alone.
 */
export const untonemap = Fn(([c, exposure]: [V3, F]) => {
  const y = clamp(mulRows(OUT_INV, c), 0, 0.985).toVar();
  // RRTAndODTFit y = (v(v+0.0245786) − 0.000090537) / (v(0.983729v + 0.432951) + 0.238081), solved for v ≥ 0
  const A = y.mul(0.983729).sub(1), B = y.mul(0.432951).sub(0.0245786), C = y.mul(0.238081).add(0.000090537);
  const v = B.negate().sub(sqrt(max(B.mul(B).sub(A.mul(C).mul(4)), 0))).div(A.mul(2));
  const x = max(mulRows(IN_INV, max(v, vec3(0)) as V3).mul(0.6).div(exposure), vec3(0)).toVar();
  const lum = dot(x, vec3(0.2126, 0.7152, 0.0722));
  return x.mul(min(float(1), float(HDR_MAX_LUM).div(max(lum, 1e-6))));
}) as unknown as (c: V3, exposure: F) => V3;
