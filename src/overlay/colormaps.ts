// Overlay colour science (CPU half): OKLab ramps baked into small LUTs, categorical palettes, value scales.
// Everything here is display colour (what the legend shows); overlayTsl.ts undoes the post tone map on the
// GPU so the screen shows the same colour (legend swatch == map colour).
import * as THREE from 'three/webgpu';

// ---- colour conversions ---------------------------------------------------------------------------
const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
type RGB = [number, number, number];

export function hexToLinear(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [toLin(((n >> 16) & 255) / 255), toLin(((n >> 8) & 255) / 255), toLin((n & 255) / 255)];
}
export function linearToHex([r, g, b]: RGB): string {
  return '#' + [r, g, b].map((v) => Math.round(Math.min(1, Math.max(0, toSrgb(v))) * 255).toString(16).padStart(2, '0')).join('');
}
function linToOklab([r, g, b]: RGB): RGB {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function oklabToLin([L, a, b]: RGB): RGB {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
}

/** A ramp: sRGB hex stops at positions 0..1 (evenly spaced when `at` is omitted), interpolated in OKLab. */
export interface Ramp { stops: string[]; at?: number[] }

export function sampleRamp(r: Ramp, t: number): RGB {
  const n = r.stops.length;
  const at = r.at ?? r.stops.map((_, i) => i / (n - 1));
  const u = Math.min(1, Math.max(0, t));
  let i = 0;
  while (i < n - 2 && u > at[i + 1]!) i++;
  const f = (u - at[i]!) / Math.max(1e-9, at[i + 1]! - at[i]!);
  const a = linToOklab(hexToLinear(r.stops[i]!)), b = linToOklab(hexToLinear(r.stops[i + 1]!));
  return oklabToLin([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
}

/** Entries per baked ramp LUT (linear interpolation between entries on the GPU). */
export const LUT_N = 48;
/** Linear-sRGB display colours, LUT_N entries, for uniformArray. */
export function bakeRamp(r: Ramp): THREE.Color[] {
  return Array.from({ length: LUT_N }, (_, i) => new THREE.Color().setRGB(...sampleRamp(r, i / (LUT_N - 1)), THREE.LinearSRGBColorSpace));
}
/** CSS linear-gradient for the legend bar (same OKLab interpolation as the LUT). */
export function rampCss(r: Ramp, steps = 24): string {
  const parts: string[] = [];
  for (let i = 0; i <= steps; i++) parts.push(`${linearToHex(sampleRamp(r, i / steps))} ${((i / steps) * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

// ---- value scales: one spec, a JS and a TSL evaluation (overlayTsl.ts) ---------------------------
export type Scale =
  | { kind: 'linear'; v0: number; v1: number }
  | { kind: 'sqrt'; v0: number; v1: number }
  | { kind: 'log'; v0: number; v1: number }            // log10(v/v0) / log10(v1/v0)
  | { kind: 'log1p'; s: number; v1: number }            // ln(1 + v/s) / ln(1 + v1/s): 0 stays 0
  | { kind: 'split'; lo: number; mid: number; hi: number }; // lo..mid → 0..0.5, mid..hi → 0.5..1

export function scaleT(s: Scale, v: number): number {
  let t: number;
  switch (s.kind) {
    case 'linear': t = (v - s.v0) / (s.v1 - s.v0); break;
    case 'sqrt': t = Math.sqrt(Math.max(0, (v - s.v0) / (s.v1 - s.v0))); break;
    case 'log': t = Math.log10(Math.max(v, 1e-30) / s.v0) / Math.log10(s.v1 / s.v0); break;
    case 'log1p': t = Math.log1p(Math.max(0, v) / s.s) / Math.log1p(s.v1 / s.s); break;
    case 'split': t = v < s.mid ? 0.5 * (v - s.lo) / (s.mid - s.lo) : 0.5 + 0.5 * (v - s.mid) / (s.hi - s.mid); break;
  }
  return Math.min(1, Math.max(0, t));
}

// ---- palettes (display sRGB). Validated with the dataviz validator, see overlays report. -----------
// Sequential / diverging ramps: monotone OKLab lightness per arm, tuned to the warm diorama grade.
export const RAMPS = {
  /** young (hot, bright gold) → old (deep indigo). */
  age: { stops: ['#ecd08a', '#efa262', '#d86f68', '#a8538a', '#6a4690', '#343a6e'] },
  /** diverging at 0 °C: ice blue ← warm grey → terracotta. */
  temp: { stops: ['#2c4f8c', '#4f7fbf', '#93b5da', '#d8d3c8', '#eab27f', '#d9704b', '#a8392f'], at: [0, 0.2, 0.38, 0.5, 0.64, 0.82, 1] },
  /** heat: dark aubergine → ember red → amber (dark→light on the dark map: hot glows). */
  heat: { stops: ['#231c38', '#4d2459', '#8c2f5a', '#c44b43', '#e8843a', '#f6c46a', '#eed9a0'] },
  /** river lines: small streams mid blue → great rivers pale bright aqua. */
  flow: { stops: ['#2d6fae', '#3a8fca', '#5db4dc', '#94d3e6', '#b9e0e8'] },
  /** moisture: sand → sage → teal → deep blue. */
  moist: { stops: ['#e8cf96', '#c7c47d', '#8db283', '#4f9a8f', '#2f7896', '#2b4f86'] },
  /** hypsometric: abyss → shelf | lowland green → tan → umber → rock → snow. */
  elev: {
    stops: ['#172a55', '#23508d', '#3f83bb', '#8cc5dc', '#5d9b5c', '#a4b86b', '#dcc585', '#b98a5a', '#8e7361', '#dcd8d0'],
    at: [0, 0.2, 0.38, 0.499, 0.501, 0.6, 0.72, 0.84, 0.93, 1],
  },
  /** crust thickness change: thinning slate blue ← grey → thickening umber. */
  thick: { stops: ['#56679a', '#737c93', '#7d7b77', '#958066', '#a8714a'] },
} satisfies Record<string, Ramp>;

/** Plate fill colours by plate id (colour follows the plate, never its rank). */
export const PLATE_COLORS = ['#6298cf', '#ad6057', '#439c7a', '#8c71b2', '#b48c46', '#0b87a0', '#b96e8c', '#698d4c',
  '#7b90d2', '#a96542', '#289d91', '#9d6ba4', '#a29546', '#3481ad', '#be6f75', '#4d9262'];

/** Plate boundary classes (display sRGB). Shared with the activity overlay so red always means "converging". */
export const BOUNDARY = {
  convergent: '#f0584e',
  divergent: '#6fcadb',
  transform: '#f5cd4a',
} as const;
export const ACTIVITY = {
  subduction: BOUNDARY.convergent,
  ridge: BOUNDARY.divergent,
  collision: '#f39a3c',
} as const;

/** Rivers overlay fills: dry land tint, lakes, dimmed open sea. */
export const RIVER = { land: '#1f232b', lake: '#9fcbe3', sea: '#262a31' } as const;

/**
 * Colour for a plate split off `parent` (its k-th child): same family, visibly different step:
 * OKLCH lightness ±0.075 and hue ±24° (alternating per child, wider for later children).
 */
export function childColor(parent: string, k: number): string {
  const [L, a, b] = linToOklab(hexToLinear(parent));
  const C = Math.hypot(a, b), h = Math.atan2(b, a);
  const sgn = k % 2 ? -1 : 1, step = 1 + Math.floor(k / 2) * 0.6;
  const L2 = Math.min(0.72, Math.max(0.5, L + sgn * 0.075 * step));
  const h2 = h + sgn * (24 * Math.PI / 180) * step;
  return linearToHex(oklabToLin([L2, C * Math.cos(h2), C * Math.sin(h2)]));
}

/** Biome colours by Biome id (biomeModel.ts). Names in defs.ts. */
export const BIOME_COLORS = [
  '#2d5487', // ocean
  '#d9e2e6', // ice
  '#8fa596', // tundra
  '#3d6b5c', // taiga
  '#4f9046', // temperate forest
  '#a8c565', // grassland
  '#ebc56b', // desert
  '#d19a42', // savanna
  '#1f6b3a', // rainforest
  '#e6d6a8', // beach
  '#8d8ba6', // alpine
  '#c1be83', // steppe
  '#9d7a48', // shrubland
  '#c3a996', // cold desert
];
