// Overlay sheets: translucent grids draped over terrain / water top, coloured per pixel from the prep buffers
// (bilinear across columns → smooth), with relief shading, iso-lines at the legend ticks, crisp anti-aliased
// plate boundaries, and the post tone map undone so the screen matches the legend.
//  - data overlay sheet (keys 1-9): prep buffer (or plate buffers T/TN) + render columns (≤ 3 storage, V23)
//  - Tectonics layer: boundary lines only, no fill; T + TN + render columns (3)
import * as THREE from 'three/webgpu';
import {
  Fn, float, int, uint, vec2, vec3, vec4, positionGeometry, positionWorld, varying, uniformArray, floor, fract, max, min, mix,
  smoothstep, clamp, abs, fwidth, dot, normalize, step, sin, exp, sign, length,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { CELL, NY, VOXEL_H } from '../sim/layout';
import { tColIdx } from '../sim/tslLayout';
import { CX, CZ, CY } from '../sim/mantleModel';
import { columnSampler, tWorldToCell, tWorldToColumn, tWorldY, tVoxelY, vertEx, HALF, Y_RENDER_BOTTOM, ambTime } from '../render/space';
import { skyU } from '../render/sky';
import { hexToLinear, scaleT, BIOME_COLORS, BOUNDARY, ACTIVITY, RIVER } from './colormaps';
import { rampLut, isoLines, untonemap, tScale } from './overlayTsl';
import type { OverlayDef, RampLegend } from './defs';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

const col3 = (hex: string): V3 => vec3(...hexToLinear(hex)) as V3;
const linColor = (h: string) => new THREE.Color().setRGB(...hexToLinear(h), THREE.LinearSRGBColorSpace);
type Palette = ReturnType<typeof uniformArray>;
const palette = (hexes: string[]): Palette => uniformArray(hexes.map(linColor), 'color');
/** Ink for bold iso-lines and boundary casings (display colour). */
const INK = '#15171f';
/** World-space light for relief shading (fixed, map-style from upper left of the hero view). */
const LIGHT = new THREE.Vector3(-0.55, 0.75, -0.35).normalize();

export interface SheetUniforms {
  opacity: THREE.UniformNode<'float', number>;
  exposure: THREE.UniformNode<'float', number>;
  /** Tectonics layer strength 0..1 (eased on/off). */
  tecOpacity: THREE.UniformNode<'float', number>;
  /** Arrow scale 0..1 (arrows grow in / shrink out with the layer). */
  arrowFade: THREE.UniformNode<'float', number>;
}

export interface SheetBuffers {
  /** Data overlay prep buffer. */
  ovA: StorageNode<'vec4'>;
  /** Plate boundary buffers (prep.ts). */
  T: StorageNode<'vec4'>;
  TN: StorageNode<'vec4'>;
  /** Plate colours by plate id (linear display colours, updated from the CPU lineage table). */
  plateColors: THREE.Color[];
}

/** Contour levels of a legend in t-space. */
const levelsT = (b: RampLegend) => (b.contours ?? b.ticks).map((v) => scaleT(b.scale, v)).filter((t) => t > 1e-4 && t < 1 - 1e-4);

/** Sheet position (top of terrain or water) and a smooth relief normal, shared by every overlay material. */
export function drape(fields: GpuFields) {
  const cs = columnSampler(fields);
  const topAt = (u: F, v: F) => { const c = cs.corners(u, v); return max(cs.height(c), cs.level(c).level); };
  const u = tWorldToCell(positionGeometry.x), v = tWorldToCell(positionGeometry.z);
  const positionNode = Fn(() => {
    const c = cs.corners(u, v);
    const h = cs.height(c), lv = cs.level(c);
    const wet = step(0.001, lv.wet).mul(step(h, lv.level));
    // above the water swell (≤ ~0.006 world) at sea, just above the ground on land
    return vec3(positionGeometry.x, tWorldY(max(h, lv.level)).add(mix(0.004, 0.011, wet)), positionGeometry.z);
  })();
  const normal = varying(Fn(() => {
    const k = vertEx.mul(VOXEL_H / (2 * CELL));
    const dx = topAt(u.add(1), v).sub(topAt(u.sub(1), v)).mul(k);
    const dz = topAt(u, v.add(1)).sub(topAt(u, v.sub(1))).mul(k);
    return normalize(vec3(dx.negate(), 1, dz.negate()));
  })(), 'vOvNormal');
  return { positionNode, normal };
}

/** Bilinear corners of the prep buffer at the fragment. */
function prepCorners(ov: StorageNode<'vec4'>) {
  const u = tWorldToCell(positionWorld.x), v = tWorldToCell(positionWorld.z);
  const u0 = floor(u), v0 = floor(v), fu = fract(u).toVar(), fv = fract(v).toVar();
  const xi = int(u0).toVar(), zi = int(v0).toVar();
  return ([[0, 0], [1, 0], [0, 1], [1, 1]] as const).map(([dx, dz]) => ({
    w: (dx ? fu : float(1).sub(fu)).mul(dz ? fv : float(1).sub(fv)).toVar() as F,
    a: (ov.element(tColIdx(xi.add(dx), zi.add(dz))) as unknown as V4).toVar() as V4,
  }));
}
const wsum = (c: ReturnType<typeof prepCorners>, f: (k: ReturnType<typeof prepCorners>[number]) => THREE.Node) =>
  c.slice(1).reduce((s, k) => s.add(k.w.mul(f(k) as F)), c[0]!.w.mul(f(c[0]!) as F)) as THREE.Node;

/**
 * Categorical field in the prep x channel: the locally dominant id wins (its weighted share f ≥ ½ on its own
 * side of the marching-squares contour), edges anti-aliased in screen space. Returns the fill colour and the
 * pixel distance to the id boundary.
 */
function categorical(c: ReturnType<typeof prepCorners>, pal: Palette) {
  const ids = c.map((k) => floor(k.a.x.add(0.5)).toVar());
  const share = ids.map((id) => c.reduce((s: F, k, j) => s.add(k.w.mul(abs(ids[j]!.sub(id)).lessThan(0.5).toFloat())) as F, float(0)).toVar());
  const best = share[0]!.toVar(), bid = ids[0]!.toVar();
  for (let k = 1; k < 4; k++) {
    const take = share[k]!.greaterThan(best).toFloat();
    best.assign(mix(best, share[k]!, take)); bid.assign(mix(bid, ids[k]!, take));
  }
  const at = (id: F) => pal.element(int(clamp(id, 0, 15))) as unknown as V3;
  const avg = c.reduce((s: V3, k, j) => s.add(at(ids[j]!).mul(k.w)) as V3, vec3(0) as V3);
  const g = best.mul(2).sub(1);
  const dist = abs(g).div(max(fwidth(g), 1e-5)).toVar(); // pixels to the boundary (∞ inside a region)
  return { color: mix(avg, at(bid), smoothstep(0, 1, dist)) as V3, dist };
}

/** Relief shading factor from the drape normal (1 on flat ground). */
const relief = (n: V3, amount: number): F => {
  const L = vec3(LIGHT.x, LIGHT.y, LIGHT.z);
  const s = clamp(dot(normalize(n), L).sub(LIGHT.y).mul(1.6).add(1), 0.5, 1.35);
  return mix(float(1), s, amount) as F;
};

/** Boundary class at the fragment from the plate buffers: weights of convergent / divergent / transform + line colour. */
function boundaryClass(cT: ReturnType<typeof prepCorners>) {
  const wb = (wsum(cT, (k) => k.a.y) as F).toVar();
  const cls = (wsum(cT, (k) => k.a.y.mul(k.a.z)) as F).div(max(wb, 1e-4)).toVar();
  const conv = smoothstep(0.3, 0.45, cls).toVar(), div = smoothstep(0.3, 0.45, cls.negate()).toVar();
  const color = mix(mix(col3(BOUNDARY.transform), col3(BOUNDARY.convergent), conv), col3(BOUNDARY.divergent), div) as V3;
  return { conv, div, color, wb };
}

export function createSheetMaterial(fields: GpuFields, B: SheetBuffers, def: OverlayDef, U: SheetUniforms): THREE.MeshBasicNodeMaterial {
  const { positionNode, normal } = drape(fields);
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  m.fog = false;
  m.positionNode = positionNode;
  const ink = col3(INK);

  m.colorNode = Fn(() => {
    let col: V3;
    let alpha: F = U.opacity;
    if (def.id === 'flow') {
      // river network on land: lines ∝ log discharge, lakes as flat fills, the sea dimmed
      const c = prepCorners(B.ovA);
      const r = (wsum(c, (k) => k.a.x) as F).toVar(), rm = (wsum(c, (k) => k.a.y) as F).toVar();
      const sea = smoothstep(0.35, 0.65, wsum(c, (k) => k.a.z) as F), lake = smoothstep(0.35, 0.65, wsum(c, (k) => k.a.w) as F);
      // r / rmax ≈ 1 − distance to the river centreline (cells); half width grows with discharge
      const g = r.div(max(rm, 1e-3)).sub(float(1).sub(mix(0.22, 0.95, rm))).toVar();
      const line = smoothstep(-0.5, 0.5, g.div(max(fwidth(g), 1e-5))).mul(smoothstep(0.0, 0.18, rm));
      col = mix(col3(RIVER.land), rampLut(def.bar!.ramp)(rm), line) as V3;
      alpha = mix(U.opacity.mul(0.4), U.opacity, line) as F;
      col = mix(col, col3(RIVER.lake), lake) as V3;
      alpha = mix(alpha, U.opacity.mul(0.9), lake) as F;
      col = mix(col, col3(RIVER.sea), sea) as V3;
      alpha = mix(alpha, U.opacity.mul(0.55), sea) as F;
    } else if (def.bar) {
      const c = prepCorners(B.ovA);
      const b = def.bar;
      const t = (wsum(c, (k) => k.a.x) as F).toVar();
      col = rampLut(b.ramp)(t);
      const iso = isoLines(t, levelsT(b), 1.1);
      col = mix(col, col.mul(0.5), iso.mul(0.55)) as V3;
      if (b.bold !== undefined) col = mix(col, ink, isoLines(t, [scaleT(b.scale, b.bold)], 2.2).mul(0.85)) as V3;
      if (def.fadeLow !== undefined) alpha = alpha.mul(mix(def.fadeLow, 1, smoothstep(0.0, 0.45, t))) as F;
    } else if (def.id === 'plates') {
      const c = prepCorners(B.T);
      const cat = categorical(c, uniformArray(B.plateColors, 'color'));
      col = cat.color;
      const bc = boundaryClass(c);
      const line = float(1).sub(smoothstep(1.0, 2.0, cat.dist));
      const casing = float(1).sub(smoothstep(2.4, 3.6, cat.dist));
      col = mix(mix(col, ink, casing.mul(0.7)), bc.color, line) as V3;
    } else if (def.id === 'biome') {
      col = categorical(prepCorners(B.ovA), palette(BIOME_COLORS)).color;
    } else {
      // activity: thickness-change tint underneath, glowing activity marks on top
      const c = prepCorners(B.ovA);
      const b2 = def.bar2!;
      col = rampLut(b2.ramp)(wsum(c, (k) => k.a.x) as F);
      const mark = (ch: 'y' | 'z' | 'w', hex: string) => {
        const s = wsum(c, (k) => k.a[ch]) as F;
        col = mix(col, col3(hex), smoothstep(0.08, 0.75, s).mul(0.9)) as V3; // fresh events solid, fading ones translucent
      };
      mark('z', ACTIVITY.ridge); mark('w', ACTIVITY.collision); mark('y', ACTIVITY.subduction);
    }
    col = col.mul(relief(normal, def.relief)) as V3;
    return vec4(untonemap(col, U.exposure), alpha);
  })();
  return m;
}

/**
 * Tectonics layer (game UI, key T): plate boundaries only, as thin softly glowing lines draped on the surface,
 * coloured by type, with a gentle animation of the relative motion: ridges send ripples outward, trenches draw
 * them in, transforms slide along the line. No fill: the diorama stays fully visible. Dimmed a little at night.
 */
export function createTectonicsMaterial(fields: GpuFields, B: SheetBuffers, U: SheetUniforms): THREE.MeshBasicNodeMaterial {
  const { positionNode, normal } = drape(fields);
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  m.fog = false;
  m.positionNode = positionNode;
  m.colorNode = Fn(() => {
    const cT = prepCorners(B.T), cN = prepCorners(B.TN);
    // lines climbing cliff faces read as noise: fade them on steep ground
    const flat = mix(float(0.2), float(1), smoothstep(0.35, 0.7, normalize(normal).y));
    // pixel distance to the id contour (same construction as the Plates overlay)
    const ids = cT.map((k) => floor(k.a.x.add(0.5)).toVar());
    const share = ids.map((id) => cT.reduce((s: F, k, j) => s.add(k.w.mul(abs(ids[j]!.sub(id)).lessThan(0.5).toFloat())) as F, float(0)));
    const best = share.slice(1).reduce((a: F, b) => max(a, b) as F, share[0]!).toVar();
    const gC = best.mul(2).sub(1);
    const dist = abs(gC).div(max(fwidth(gC), 1e-5)).toVar();
    const bc = boundaryClass(cT);
    const trans = float(1).sub(bc.conv).sub(bc.div).max(0);
    // along-line coordinate from the interpolated boundary normal (world units)
    const nrm = cN.slice(1).reduce((s: THREE.Node<'vec2'>, k, j) => s.add(k.a.xy.mul(k.w.mul(cT[j + 1]!.a.y))) as THREE.Node<'vec2'>,
      cN[0]!.a.xy.mul(cN[0]!.w.mul(cT[0]!.a.y)) as THREE.Node<'vec2'>);
    const n = nrm.div(max(length(nrm), 1e-5));
    const along = dot(positionWorld.xz, vec2(n.y.negate(), n.x));
    const shear = (wsum(cN.map((k, j) => ({ w: k.w.mul(cT[j]!.a.y) as F, a: k.a })), (k) => k.a.z) as F).div(max(bc.wb, 1e-4));
    const t = ambTime;
    const slide = sin(along.mul(Math.PI * 2 / 0.09).sub(t.mul(2.2).mul(sign(shear)))).mul(0.5).add(0.5);
    const ripple = sin(dist.mul(0.55).add(t.mul(3.2).mul(bc.conv.sub(bc.div)))).mul(0.5).add(0.5);
    const pulse = mix(ripple, slide, trans);
    const core = float(1).sub(smoothstep(0.7, 1.5, dist));
    const halo = exp(dist.negate().div(3.2)).mul(0.5).mul(pulse.mul(0.55).add(0.45));
    const casing = float(1).sub(smoothstep(1.5, 2.6, dist)).sub(core).max(0).mul(0.3);
    const lineCol = bc.color.mul(core.mul(pulse.mul(0.18).add(0.9)).add(halo).add(0.0001)).div(core.add(halo).add(0.0001));
    const col = mix(lineCol, col3(INK), casing.div(core.add(halo).add(casing).add(1e-4)));
    const night = mix(float(1), 0.7, skyU.night);
    const a = min(core.add(halo).add(casing), 1).mul(U.tecOpacity).mul(night).mul(flat);
    return vec4(untonemap(col as V3, U.exposure), a);
  })();
  return m;
}

// ---- cut faces: crust temperature cross-section (heat overlay) ----------------------------------------------
/** Four vertical quads just outside the rock faces; y in voxel units (mapped by positionNode, follows vertEx). */
function faceGeometry(): THREE.BufferGeometry {
  const H = NY, B = Y_RENDER_BOTTOM, E = HALF + 0.0015;
  const quads = [
    [[E, B, HALF], [E, B, -HALF], [E, H, -HALF], [E, H, HALF]],
    [[-E, B, -HALF], [-E, B, HALF], [-E, H, HALF], [-E, H, -HALF]],
    [[-HALF, B, E], [HALF, B, E], [HALF, H, E], [-HALF, H, E]],
    [[HALF, B, -E], [-HALF, B, -E], [-HALF, H, -E], [HALF, H, -E]],
  ];
  const pos: number[] = [], idx: number[] = [];
  for (const q of quads) {
    const base = pos.length / 3;
    for (const c of q) pos.push(...c);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

/** Heat overlay on the cut faces: 'crustTemp' (half-res, pair[0] current between mantle steps) + render columns. */
export function createFaceMesh(fields: GpuFields, bar: RampLegend, U: SheetUniforms): THREE.Mesh | null {
  if (!fields.names().includes('crustTemp')) return null;
  const crustT = fields.pair('crustTemp')[0];
  const cs = columnSampler(fields);
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.FrontSide });
  m.fog = false;
  m.positionNode = vec3(positionGeometry.x, tWorldY(positionGeometry.y), positionGeometry.z);
  m.colorNode = Fn(() => {
    const p = positionWorld;
    const inX = clamp(p.x, -HALF + 1e-4, HALF - 1e-4), inZ = clamp(p.z, -HALF + 1e-4, HALF - 1e-4);
    const vy = tVoxelY(p.y);
    const surf = cs.height(cs.corners(tWorldToCell(inX), tWorldToCell(inZ)));
    // face frame: along-face coordinate + the edge column (branch-free, see space.ts BRANCH-FREE NOTE)
    const onX = step(HALF, abs(p.x));
    const along = mix(tWorldToCell(inX), tWorldToCell(inZ), onX);
    const edge = mix(float(tWorldToColumn(inZ)), float(tWorldToColumn(inX)), onX);
    const uh = along.mul(0.5).sub(0.25), yh = vy.mul(0.5).sub(0.25);
    const eh = int(edge.mul(0.5));
    const u0 = floor(uh), y0 = floor(yh), fu = fract(uh), fy = fract(yh);
    const at = (du: number, dy: number) => {
      const a = int(clamp(u0.add(du), 0, CX - 1));
      const cx = int(mix(float(a), float(eh), onX)), cz = int(mix(float(eh), float(a), onX));
      const cy = int(clamp(y0.add(dy), 0, CY - 1));
      return crustT.element(uint(cx.add(cz.mul(CX)).add(cy.mul(CX * CZ)))) as unknown as F;
    };
    const T = mix(mix(at(0, 0), at(1, 0), fu), mix(at(0, 1), at(1, 1), fu), fy);
    const t = tScale(bar.scale, T).toVar();
    let col = rampLut(bar.ramp)(t);
    col = mix(col, col.mul(0.55), isoLines(t, [200, 400, 600, 800, 1000, 1200].map((v) => scaleT(bar.scale, v)), 1.1).mul(0.6)) as V3;
    // below the silhouette only (alpha, not Discard: keeps derivatives in uniform control flow)
    const inside = step(vy, surf).mul(step(Y_RENDER_BOTTOM, vy));
    return vec4(untonemap(col, U.exposure), U.opacity.mul(0.92).mul(inside));
  })();
  const mesh = new THREE.Mesh(faceGeometry(), m);
  mesh.name = 'overlay-faces';
  mesh.frustumCulled = false;
  mesh.renderOrder = 11;
  return mesh;
}
