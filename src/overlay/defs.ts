// Overlay catalogue (§I.overlays, keys 1-9): what each overlay shows, how values map to colour, and what the
// legend says. Pure data + formatting; the GPU side lives in prep.ts (per-column values) and sheet.ts (colour).
import { RAMPS, BIOME_COLORS, BOUNDARY, ACTIVITY, RIVER, type Ramp, type Scale } from './colormaps';

/** Plate speeds: sim cells/My → cm/yr at the kinematics convention of 20 km cells (kinematics.ts BASE_SPEED). */
export const CM_PER_YR_PER_CELL_MY = 2;

export interface Swatch { name: string; color: string; line?: boolean }
export interface RampLegend {
  ramp: Ramp; scale: Scale; ticks: number[]; unit: string; fmt?: (v: number) => string;
  /** Iso-line levels drawn on the map (default: the ticks); `bold` is drawn stronger (0 °C, sea level). */
  contours?: number[]; bold?: number;
  /** Optional caption above the bar (overlays with two bars). */
  caption?: string;
}

export type OverlayId = 'plates' | 'age' | 'temp' | 'heat' | 'activity' | 'flow' | 'moist' | 'biome' | 'elev';

export interface OverlayDef {
  key: number;
  id: OverlayId;
  title: string;
  /** One plain-language line under the title. */
  blurb: string;
  /** Main colour bar (continuous overlays). */
  bar?: RampLegend;
  /** Secondary bar (cut faces for heat, thickness change for activity). */
  bar2?: RampLegend;
  swatches?: Swatch[];
  /** Fraction of the terrain relief shading kept in the overlay colour. */
  relief: number;
  /** Real seconds of temporal smoothing of the value (removes per-tick flicker); 0 = none. */
  smoothS: number;
  /** Low values fade toward the terrain (rivers, activity): alpha at t = 0. */
  fadeLow?: number;
}

const fmtNum = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toPrecision(2)}k` : `${+v.toPrecision(2)}`);
const SUP: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
/** 1e-3 → 10⁻³ (ticks are exact powers of ten). */
const pow10 = (v: number) => {
  if (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) > 1e-6) return fmtNum(v);
  const e = Math.round(Math.log10(v));
  return e === 0 ? '1' : `10${[...String(e)].map((ch) => SUP[ch]).join('')}`;
};

export const BIOME_NAMES = ['Ocean', 'Ice', 'Tundra', 'Taiga', 'Temperate forest', 'Grassland', 'Desert', 'Savanna',
  'Rainforest', 'Beach', 'Alpine', 'Steppe', 'Shrubland', 'Cold desert'];

/** Heat flow scale (mW/m²): normal ~65, plumes ~200, over sinking slabs ~20. */
const HEAT_SCALE: Scale = { kind: 'sqrt', v0: 20, v1: 220 };

export const OVERLAYS: OverlayDef[] = [
  {
    key: 1, id: 'plates', title: 'Tectonic plates',
    blurb: 'Each colour is one plate; arrows show which way it drifts and how fast.',
    swatches: [
      { name: 'convergent: plates collide', color: BOUNDARY.convergent, line: true },
      { name: 'divergent: plates pull apart', color: BOUNDARY.divergent, line: true },
      { name: 'transform: plates slide past', color: BOUNDARY.transform, line: true },
    ],
    relief: 0.55, smoothS: 0,
  },
  {
    key: 2, id: 'age', title: 'Crust age',
    blurb: 'Where plates pull apart new crust forms; age grows away from ridges. Continents are the old cores.',
    bar: { ramp: RAMPS.age, scale: { kind: 'log1p', s: 20, v1: 3500 }, ticks: [0, 20, 100, 300, 1000, 3500], unit: 'My', fmt: fmtNum, contours: [20, 100, 300, 1000] },
    relief: 0.5, smoothS: 0.3,
  },
  {
    key: 3, id: 'temp', title: 'Surface temperature',
    blurb: 'Warm tropics, cold poles and mountain tops; the bold line is freezing (0 °C).',
    bar: { ramp: RAMPS.temp, scale: { kind: 'split', lo: -30, mid: 0, hi: 35 }, ticks: [-30, -20, -10, 0, 10, 20, 30], unit: '°C', fmt: (v) => `${v}`,
      contours: [-20, -10, 10, 20, 30], bold: 0 },
    relief: 0.5, smoothS: 0.4,
  },
  {
    key: 4, id: 'heat', title: 'Mantle heat flow',
    blurb: 'Heat rising out of the mantle: hot plumes glow, sinking cold slabs go dark. Cut faces show crust temperature.',
    bar: { ramp: RAMPS.heat, scale: HEAT_SCALE, ticks: [20, 40, 65, 100, 150, 220], unit: 'mW/m²', fmt: fmtNum, contours: [40, 100, 150] },
    bar2: { caption: 'cut faces · crust temperature', ramp: RAMPS.heat, scale: { kind: 'linear', v0: 0, v1: 1300 }, ticks: [0, 400, 800, 1300], unit: '°C', fmt: fmtNum },
    relief: 0.45, smoothS: 0.4,
  },
  {
    key: 5, id: 'activity', title: 'Tectonic activity',
    blurb: 'Where the crust is being made, destroyed or piled up right now; tint shows it thickening or thinning.',
    swatches: [
      { name: 'subduction: sea floor sinks', color: ACTIVITY.subduction },
      { name: 'ridge: new ocean crust', color: ACTIVITY.ridge },
      { name: 'collision: crust piles up', color: ACTIVITY.collision },
    ],
    bar2: { caption: 'crust thickness change', ramp: RAMPS.thick, scale: { kind: 'split', lo: -1, mid: 0, hi: 1 }, ticks: [-1, -0.5, 0, 0.5, 1], unit: 'layers/My',
      fmt: (v) => (v > 0 ? `+${v}` : `${v}`) },
    relief: 0.6, smoothS: 0,
  },
  {
    key: 6, id: 'flow', title: 'Rivers & lakes',
    blurb: 'Rivers gather downhill into ever wider, brighter trunks; lakes are pale fills, the sea is dimmed.',
    bar: { ramp: RAMPS.flow, scale: { kind: 'log', v0: 1e-3, v1: 0.5 }, ticks: [1e-3, 1e-2, 1e-1, 0.5], unit: 'discharge · cells³/step', fmt: pow10, contours: [] },
    swatches: [
      { name: 'lake', color: RIVER.lake },
      { name: 'open sea (dimmed)', color: RIVER.sea },
    ],
    relief: 0.8, smoothS: 0.3,
  },
  {
    key: 7, id: 'moist', title: 'Precipitation',
    blurb: 'Long-term rain and snow; 1 = wet tropics. Mountains wring out moist air, leaving dry rain shadows.',
    bar: { ramp: RAMPS.moist, scale: { kind: 'log', v0: 0.03, v1: 5 }, ticks: [0.03, 0.1, 0.3, 1, 3], unit: '× tropical rain', fmt: fmtNum, contours: [0.1, 0.3, 1, 3] },
    relief: 0.5, smoothS: 0.4,
  },
  {
    key: 8, id: 'biome', title: 'Biomes',
    blurb: 'Life zones set by long-term temperature and rainfall.',
    swatches: BIOME_NAMES.map((name, i) => ({ name, color: BIOME_COLORS[i]! })),
    relief: 0.4, smoothS: 0,
  },
  {
    key: 9, id: 'elev', title: 'Elevation & bathymetry',
    blurb: 'Height above or depth below today’s sea level; the bold line is the coast.',
    bar: { ramp: RAMPS.elev, scale: { kind: 'split', lo: -25, mid: 0, hi: 35 }, ticks: [-25, -15, -5, 0, 10, 20, 35], unit: 'layers', fmt: (v) => (v > 0 ? `+${v}` : `${v}`),
      contours: [-20, -15, -10, -5, 5, 10, 15, 20, 25, 30], bold: 0 },
    relief: 0.85, smoothS: 0,
  },
];

export const overlayByKey = (k: number) => OVERLAYS.find((d) => d.key === k);

// ---- hover readout: one column's prep values → text --------------------------------------------------
export interface ReadoutCtx {
  plates: readonly { id: number; alive: boolean; vel: [number, number] }[];
  /** ovA / ovB values of the hovered column (prep.ts layout). */
  a: Float32Array; b: Float32Array;
}

export function plateSpeedText(vel: [number, number]): string {
  const s = Math.hypot(vel[0], vel[1]) * CM_PER_YR_PER_CELL_MY;
  return `${s < 0.95 ? s.toFixed(2) : s.toFixed(1)} cm/yr`;
}

export function readout(def: OverlayDef, c: ReadoutCtx): string {
  const { a, b } = c;
  switch (def.id) {
    case 'plates': {
      const id = Math.round(a[0]!);
      const p = c.plates[id];
      const edge = a[1]! > 0.5 ? (a[2]! > 0.4 ? ' · convergent edge' : a[2]! < -0.4 ? ' · divergent edge' : ' · transform edge') : '';
      return p ? `Plate ${id} · ${plateSpeedText(p.vel)}${edge}` : `Plate ${id}`;
    }
    case 'age': return `${fmtNum(a[1]!)} My old`;
    case 'temp': return `${a[1]!.toFixed(1)} °C`;
    case 'heat': return `${a[1]!.toFixed(0)} mW/m²`;
    case 'activity': {
      const tags = [a[1]! > 0.15 ? 'subduction' : '', a[2]! > 0.15 ? 'ridge' : '', a[3]! > 0.15 ? 'collision' : ''].filter(Boolean);
      const r = b[1]!;
      return `${tags.length ? tags.join(' + ') : 'quiet'} · ${r >= 0 ? '+' : ''}${r.toFixed(2)} layers/My`;
    }
    case 'flow': return a[2]! > 0.5 ? 'open sea' : a[3]! > 0.5 ? 'lake' : `${b[0]! < 1e-5 ? 'dry' : `${b[0]!.toPrecision(2)} cells³/step`}`;
    case 'moist': return `${a[1]!.toFixed(2)} × tropical rain`;
    case 'biome': return BIOME_NAMES[Math.round(a[0]!)] ?? `biome ${a[0]}`;
    case 'elev': { const e = a[1]!; return `${e >= 0 ? '+' : ''}${e.toFixed(1)} layers ${e >= 0 ? 'above' : 'below'} sea level`; }
  }
}
