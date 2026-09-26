// T11 side cut render: 4 vertical faces + bottom at the block edges. Each fragment reads the voxels
// behind it straight from the 'vox' storage buffer. Above the column surface → discard, so the face
// silhouette follows the terrain exactly (same bilinear surfY as the terrain mesh edge).
// Look: polished "cake slice": strata are blended between neighbouring columns/layers with a soft
// step (organic lines instead of voxel stairs), per-material sheen, micro-normal grooves, glowing
// magma and a hot, slowly convecting mantle (emissive > 1 → bloom). The water column on the faces
// is glass-like: refracted scene behind, Beer-Lambert tint by thickness, a meniscus line.
// The shading is one shared set (createFaceShading) for any vertical cut quad: the outer faces here and
// the slice caps (src/slice) render the cross-section identically. With the Tectonics layer on
// (setFaceTectonics) plate boundaries run down the cut: classified like the map lines, dipping under
// the higher plate at convergent boundaries, each plate's crust faintly tinted with its colour.
import * as THREE from 'three/webgpu';
import {
  Fn, If, vec2, vec3, vec4, float, int, uint, uniform, uniformArray, positionGeometry, positionWorld, normalGeometry, Discard,
  mix, smoothstep, saturate, sin, cos, exp, floor, fract, min, max, pow, hash, abs, clamp, texture, normalize, step,
  transformNormalToView, dot, screenUV, positionView, viewportSharedTexture,
} from 'three/tsl';
import { NX, NY, CELL, VOXEL_H, Mat, Y_SEA_NOMINAL } from '../sim/layout';
import { tMat, tFill } from '../sim/tslLayout';
import type { GpuFields } from '../core/gpu';
import { HALF, tWorldY, tVoxelY, tWorldToCell, tWorldToColumn, columnSampler, voxReader, tTopVoxel, ambTime, Y_RENDER_BOTTOM, vertEx, advect, displayBuffers } from './space';
import { tColIdx, uMin } from '../sim/tslLayout';
import { viewDirWorld } from './space';
import { createPaletteNodes, biomeColor } from './palette';
import { lookTextures } from './textures';
import { WATER_SCATTER, sceneViewZ, waterTransmit, swell, seabedGlowK } from './water';
import { skyColor, skyU } from './sky';
import { CX, CZ, CY } from '../sim/mantleModel';
import { MAX_PLATES } from '../sim/worldData';
import { BOUNDARY, PLATE_COLORS } from '../overlay/colormaps';
import { shadowProxy } from './lighting';
import { cloudShadowAt } from '../atmo/atmosphere';

type F = THREE.Node<'float'>;
type I = THREE.Node<'int'>;
type U = THREE.Node<'uint'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

/** Voxel layers above the rendered bottom where the mantle glows (narrow, deep, subtle). */
const HOT_TOP = Y_RENDER_BOTTOM + 14;

/** sRGB hex → linear display colour (the overlay's convention). */
const linHex = (h: string) => new THREE.Color().setStyle(h, THREE.SRGBColorSpace);

/**
 * Plate boundaries on the cut faces, shared by every face material. Driven by the overlay module
 * (setFaceTectonics each frame while the Tectonics layer is visible); 0 opacity = off, no cost.
 */
const faceTec = {
  opacity: uniform(0),
  vel: uniformArray([...Array(MAX_PLATES)].map(() => new THREE.Vector4()), 'vec4'),
  colors: uniformArray([...Array(MAX_PLATES)].map((_, i) => linHex(PLATE_COLORS[i % PLATE_COLORS.length]!)), 'color'),
};
/**
 * Tectonics layer on the cut faces: `opacity` 0..1 (the layer's eased fade), plate velocities
 * (vx, vz, alive, ·) in cells/My by plate id and plate colours (linear), as the overlay module keeps them.
 */
export function setFaceTectonics(opacity: number, plateVel?: readonly THREE.Vector4[], plateColors?: readonly THREE.Color[]): void {
  faceTec.opacity.value = opacity;
  const v = faceTec.vel.array as THREE.Vector4[], c = faceTec.colors.array as THREE.Color[];
  if (plateVel) for (let i = 0; i < Math.min(MAX_PLATES, plateVel.length); i++) v[i]!.copy(plateVel[i]!);
  if (plateColors) for (let i = 0; i < Math.min(MAX_PLATES, plateColors.length); i++) c[i]!.copy(plateColors[i]!);
}
/** Cells of horizontal offset per voxel layer of depth for a subducting slab (≈ 30° dip). */
const SLAB_DIP = 0.55;

/** Five quads; y in voxel units (Y_RENDER_BOTTOM..NY), mapped to world by positionNode so vertEx applies. */
function createSideGeometry(): THREE.BufferGeometry {
  const H = NY, B = Y_RENDER_BOTTOM;
  // [corner a, corner b, corner c, corner d] counter-clockwise seen from outside, + outward normal
  const quads: [number[][], number[]][] = [
    [[[HALF, B, HALF], [HALF, B, -HALF], [HALF, H, -HALF], [HALF, H, HALF]], [1, 0, 0]],
    [[[-HALF, B, -HALF], [-HALF, B, HALF], [-HALF, H, HALF], [-HALF, H, -HALF]], [-1, 0, 0]],
    [[[-HALF, B, HALF], [HALF, B, HALF], [HALF, H, HALF], [-HALF, H, HALF]], [0, 0, 1]],
    [[[HALF, B, -HALF], [-HALF, B, -HALF], [-HALF, H, -HALF], [HALF, H, -HALF]], [0, 0, -1]],
    [[[-HALF, B, -HALF], [HALF, B, -HALF], [HALF, B, HALF], [-HALF, B, HALF]], [0, -1, 0]],
  ];
  const pos: number[] = [], nrm: number[] = [], idx: number[] = [];
  for (const [corners, n] of quads) {
    const base = pos.length / 3;
    for (const c of corners) { pos.push(...c); nrm.push(...n); }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setIndex(idx);
  return g;
}

/**
 * Cut-face shading for vertical quads whose vertices hold world x, z and y in voxel units (see
 * createSideGeometry / createFaceQuad) and an outward normal along ±x or ±z. The quad may sit anywhere
 * on the grid: the section shows the column just inside it.
 */
export interface FaceShading {
  rock: THREE.MeshStandardNodeMaterial;
  water: THREE.MeshStandardNodeMaterial;
  positionNode: THREE.Node;
  /** Shadow-pass mask: solid below the column surface. */
  shadowMask: THREE.Node;
  dispose(): void;
}

export function createFaceShading(fields: GpuFields, opts: { seaLevel?: number } = {}): FaceShading {
  const S = columnSampler(fields);
  const vox = voxReader(fields);
  const pal = createPaletteNodes();
  const { bioG, bioV, cols } = displayBuffers(fields);
  const gridIdx = (u: F, v: F) => tColIdx(int(floor(u.add(0.5))), int(floor(v.add(0.5))));
  const bioSmGrid = (u: F, v: F) => ({
    ground: bioG.element(gridIdx(u, v)) as unknown as THREE.Node<'vec4'>,
    veg: bioV.element(gridIdx(u, v)) as unknown as THREE.Node<'vec4'>,
  });
  const vegGrid = (u: F, v: F) => (cols.element(gridIdx(u, v)) as unknown as THREE.Node<'vec4'>).w;
  const crustT = fields.names().includes('crustTemp') ? fields.pair('crustTemp')[0] : null;
  const tex = lookTextures();
  const seaLevel = uniform(opts.seaLevel ?? Y_SEA_NOMINAL);
  const positionNode = vec3(positionGeometry.x, tWorldY(positionGeometry.y), positionGeometry.z);

  const p = positionWorld;
  const vy = tVoxelY(p.y).toVar('sideVy');
  const surfH = Fn(() => S.height(S.corners(tWorldToCell(p.x), tWorldToCell(p.z)))).once()().toVar('sideSurfH');
  const depth = surfH.sub(vy).toVar('sideDepth'); // voxel layers below the silhouette
  // Face frame: `along` runs horizontally in the face, the perpendicular column is the edge column.
  // Branch-free throughout (mix by 0/1 masks, no select): see BRANCH-FREE NOTE in space.ts.
  const onX = step(0.5, abs(normalGeometry.x));
  const along = mix(p.x, p.z, onX).toVar('sideAlong');
  const tangent = mix(vec3(1, 0, 0), vec3(0, 0, 1), onX);
  const pick = (ifX: I, ifZ: I): I => int(mix(float(ifZ), float(ifX), onX)) as I;
  // Plate advection at the face (display, see space.ts): content slides along the face with the
  // along-offset; the cut itself stays on the grid edge column.
  const advF = advect(fields);
  const faceOff = Fn(() => {
    const cx = tWorldToCell(p.x), cz = tWorldToCell(p.z);
    const [ax, az] = advF(cx, cz);
    return vec2(cx.sub(ax), cz.sub(az));
  }).once()().toVar('sideAdv');
  // Only the along-face component is applied. Rounding the perpendicular offset switched the face to
  // the neighbouring column (or, across the torus seam, the far side of the block) whenever it crossed
  // ±0.5, which read as the faces alternating between two cross-sections.
  const oAlong = mix(faceOff.x, faceOff.y, onX);
  // the column just inside the face (outer faces: the edge column; slice caps: the column at the cut)
  const edgeCol = pick(tWorldToColumn(p.x.sub(normalGeometry.x.mul(CELL * 0.5))), tWorldToColumn(p.z.sub(normalGeometry.z.mul(CELL * 0.5)))).toVar('sideEdge');

  // Organic warp so layer boundaries wander instead of following the voxel grid.
  const warpT = texture(tex.detail, vec2(along.mul(0.55), vy.mul(0.018))).toVar('sideWarp');
  const u = tWorldToCell(along).sub(oAlong).add(warpT.r.sub(0.5).mul(1.6)).toVar('sideU');
  const u0 = floor(u);
  // Vertical display lag of the centre column (isostasy / uplift): layers slide instead of jumping.
  const uN = int(floor(u.add(0.5)));
  const cShift = S.shiftAt(pick(edgeCol, uN), pick(uN, edgeCol));
  const yF = vy.add(warpT.g.sub(0.5).mul(2.2)).sub(0.5).sub(cShift).toVar('sideYF');

  // 5×3 voxel samples (columns × layers) around the nearest cell, each clamped to its column's top
  // solid voxel, material ids packed 4 bits each into floats (exact below 2^24).
  // Discard above the silhouette happens here (the Fn is shared by colour and emissive).
  const uc = floor(u.add(0.5)), yc = floor(yF.add(0.5));
  const tu = u.sub(uc).toVar('sideTu'), ty = yF.sub(yc).toVar('sideTy'); // -0.5..0.5 within the centre cell
  const mats = Fn(() => {
    Discard(vy.greaterThan(surfH));
    const packed = [float(0), float(0), float(0), float(0)] as F[];
    let k = 0;
    for (const da of [-2, -1, 0, 1, 2]) {
      const a = int(uc.add(da)); // wraps in tColIdx (V1), so advected content continues across
      const x = pick(edgeCol, a), z = pick(a, edgeCol);
      const topV = tTopVoxel(S.surfAt(x, z));
      for (const dy of [-1, 0, 1]) {
        const y = int(min(yc.add(dy), float(topV)));
        const v = vox(x, y, z).toVar();
        // drained chamber (MAGMA with fill 0) reads as a dark cavity (AIR), not glowing magma
        const m = tMat(v);
        const id = float(m).mul(float(float(tFill(v)).greaterThan(0).or(m.notEqual(uint(Mat.MAGMA)))));
        packed[k >> 2] = packed[k >> 2]!.add(id.mul(16 ** (k & 3)));
        k++;
      }
    }
    return vec4(packed[0]!, packed[1]!, packed[2]!, packed[3]!);
  }).once()().toVar('sideMats');
  const pk = [mats.x, mats.y, mats.z, mats.w];
  const mf = [...Array(15).keys()].map((k) => float(int(pk[k >> 2]!).shiftRight(int(4 * (k & 3))).bitAnd(int(15))).toVar(`sideM${k}`));
  const mi = mf.map((m) => int(m));
  // Quadratic B-spline weights over the 3 samples per axis (smooth, sum to 1).
  const bs = (t: F) => [float(0.5).mul(float(0.5).sub(t).pow(2)), float(0.75).sub(t.mul(t)), float(0.5).mul(float(0.5).add(t).pow(2))];
  // Along the face a broad, nearly flat kernel over 5 columns (the local majority wins: 1-2 column
  // streaks and stray voxels from folding vanish), vertically a B-spline (layers stay crisp).
  const tent = (d: number) => max(float(0), float(1).sub(abs(tu.negate().add(d)).div(2.6)));
  const wu = [-2, -1, 0, 1, 2].map((d) => tent(d));
  const wy = bs(ty);
  const bil = [...Array(15).keys()].map((k) => wu[Math.floor(k / 3)]!.mul(wy[k % 3]!).toVar(`sideBil${k}`));
  // Mode filter: each sample's material scores the summed weight of all samples sharing it;
  // sharpening that score keeps the local majority (single stray voxels and 1-column streaks from
  // folding vanish) with smooth, slanted boundaries instead of voxel stairs.
  const same = (i: number, j: number) => (i === j ? float(1) : float(mf[i]!.equal(mf[j]!)));
  const q = bil.map((w, i) => {
    const W = bil.reduce((acc, wj, j) => acc.add(wj.mul(same(i, j))), float(0) as F);
    return w.mul(pow(W, 5)).add(1e-7);
  });
  const qSum = q.reduce((a, b) => a.add(b), float(0) as F);
  const wts = q.map((w, i) => w.div(qSum).toVar(`sideW${i}`));
  const blendF = (f: (m: I) => F): F => wts.reduce((acc, w, i) => acc.add(f(mi[i]!).mul(w)), float(0) as F);
  const blendV = (f: (m: I) => V3): V3 => wts.reduce((acc, w, i) => acc.add(f(mi[i]!).mul(w)), vec3(0) as V3);
  const isMat = (id: number) => (m: I) => float(m.equal(int(id))) as F;

  const baseCol = blendV(pal.color).toVar('sideBase');
  const band = blendF(pal.band).toVar('sideBand');
  const periW = blendF(isMat(Mat.PERIDOTITE)).toVar('sidePeri');
  const magW = blendF(isMat(Mat.MAGMA)).toVar('sideMag');
  const a0 = int(clamp(u0, 0, NX - 1));
  const colWater = S.waterAt(pick(edgeCol, a0), pick(a0, edgeCol)).toVar('sideColWater');

  // Plate boundaries down the cut (Tectonics layer). Along the face on the advected grid (no warp: a
  // clean line), a boundary sits half-way between two edge columns of different plates (each side two
  // columns of the same plate, so single-cell noise draws nothing). Classified from the plates'
  // relative motion along the face like the map lines; at convergent boundaries the line dips under
  // the higher (overriding) plate as a stylised slab, elsewhere it runs straight down.
  // tecLine: (line colour, line alpha); tecTint: (plate colour, 1) of the column's plate.
  const pidPair = fields.names().includes('plateId') ? fields.pair<'uint'>('plateId') : null;
  const pidParity = uniform(0, 'uint').onRenderUpdate(() => (pidPair ? fields.parity('plateId') : 0));
  const pid = (a: I): U => {
    const idx = tColIdx(pick(edgeCol, a), pick(a, edgeCol)).toVar(); // var before the If (BRANCH-FREE NOTE)
    const v = uint(0).toVar();
    if (pidPair) If(pidParity.equal(uint(0)), () => { v.assign(pidPair[0].element(idx)); }).Else(() => { v.assign(pidPair[1].element(idx)); });
    return v;
  };
  const uMin4 = (id: U) => uMin(id, uint(MAX_PLATES - 1));
  const tecLine = Fn(() => {
    const line = vec4(0).toVar();
    If(faceTec.opacity.greaterThan(0.001), () => {
      const uB = tWorldToCell(along).sub(oAlong).toVar();
      const d = max(depth, 0).toVar();
      const tan = vec2(float(1).sub(onX), onX); // +u in world xz
      const velOf = (id: U) => (faceTec.vel.element(uMin4(id)) as unknown as V4).xy;
      const lc = (h: string) => { const c = linHex(h); return vec3(c.r, c.g, c.b); };
      const cConv = lc(BOUNDARY.convergent), cDiv = lc(BOUNDARY.divergent), cTr = lc(BOUNDARY.transform);
      for (const sgn of [0, 1, -1]) {
        const ub = uB.sub(d.mul(SLAB_DIP * sgn));
        const j = floor(ub);
        const ja = int(j);
        const pA = pid(ja), pB = pid(ja.add(1));
        const stable = float(pid(ja.sub(1)).equal(pA).and(pid(ja.add(2)).equal(pB)));
        const isB = float(pA.notEqual(pB)).mul(stable);
        const rel = velOf(pA).sub(velOf(pB));
        const closing = dot(rel, tan).div(max(rel.length(), 1e-4));
        const conv = smoothstep(0.3, 0.45, closing), div = smoothstep(0.3, 0.45, closing.negate());
        const col = mix(mix(cTr, cConv, conv), cDiv, div);
        let w: F;
        if (sgn === 0) w = float(1).sub(conv).mul(float(1).sub(smoothstep(40, 90, d)));
        else {
          // dips toward +u (sgn 1) when the plate at +u stands higher: it overrides, the other goes under
          const hA = S.surfAt(pick(edgeCol, ja), pick(ja, edgeCol)), hB = S.surfAt(pick(edgeCol, ja.add(1)), pick(ja.add(1), edgeCol));
          const over = sgn > 0 ? step(hA, hB) : float(1).sub(step(hA, hB));
          w = conv.mul(over).mul(float(1).sub(smoothstep(50, 80, d)));
        }
        const dist = abs(ub.sub(j).sub(0.5));
        const a = float(1).sub(smoothstep(0.18, 0.5, dist)).mul(isB).mul(w);
        line.assign(vec4(mix(line.rgb, col, a), max(line.a, a)));
      }
    });
    return line;
  }).once()().toVar('sideTecLine');
  const tecTint = Fn(() => {
    const tint = vec3(0).toVar();
    If(faceTec.opacity.greaterThan(0.001), () => {
      const own = pid(int(floor(tWorldToCell(along).sub(oAlong).add(0.5))));
      tint.assign(faceTec.colors.element(uMin4(own)) as unknown as V3);
    });
    return tint;
  }).once()().toVar('sideTecTint');

  // Strata grooves: wavy fine bands; their phase also tilts the normal (micro relief on the face).
  const bandPhase = vy.add(warpT.b.sub(0.5).mul(5)).mul(2.2).toVar('sideBandPhase');
  const fineN = texture(tex.detail, vec2(along.mul(0.25), vy.mul(0.1))).toVar('sideFine');
  const grainT = texture(tex.detail, vec2(along.mul(1.6), vy.mul(0.05))).toVar('sideGrain');

  // Slow convection plumes rising through the mantle (ambTime: 0.0025·3600 = 9 texture periods).
  const flow = texture(tex.detail, vec2(along.mul(0.35), vy.mul(0.01).sub(ambTime.mul(0.0025)))).r.toVar('sideFlow');

  const color = Fn(() => {
    const b1 = sin(bandPhase);
    const b2 = fineN.g.sub(0.5).mul(2);
    const jit = hash(floor(vy).add(mf[7]!.mul(31.7)));
    let col = baseCol.mul(float(1).add(band.mul(b1.mul(0.55).add(b2.mul(0.7))))).mul(jit.mul(0.12).add(0.94)) as V3;
    // Grain speckle, stronger in crystalline rock.
    col = col.mul(grainT.b.sub(0.5).mul(0.16).add(1));
    // Darken gently with depth (deep crust reads as "inside").
    col = col.mul(mix(1.0, 0.8, saturate(depth.div(70))));
    // Diorama cake rim on dry land: grass lip over a soil band; snow when high; wet sediment under water.
    const warp = warpT.r.sub(0.5).mul(2);
    const dry = float(1).sub(step(0.01, colWater));
    const alt = surfH.sub(seaLevel);
    const lip = float(0.6).add(warp.mul(0.25));
    const soil = float(2.4).add(warp.mul(0.9));
    // Turf lip follows the column's biome cover (sand over deserts, dark green under taiga…).
    // grid-space lookups (u / edgeCol are already advected): read the display directly
    const bcx = mix(u, float(edgeCol), onX), bcz = mix(float(edgeCol), u, onX);
    const bs = bioSmGrid(bcx, bcz);
    const vegL = vegGrid(bcx, bcz);
    const bioLip = mix(bs.ground.rgb, bs.veg.rgb, smoothstep(0.0, 0.35, max(vegL, 0).sub(0.1)));
    const procLip = mix(biomeColor('grassLush'), vec3(0.2, 0.45, 0.08), grainT.g);
    const grassLip = mix(procLip, bioLip.mul(grainT.g.mul(0.15).add(0.92)), step(0, vegL));
    const topCol = mix(grassLip, biomeColor('snow'), smoothstep(27, 32, alt.add(warp.mul(4))));
    const soilCol = mix(biomeColor('soil'), vec3(0.24, 0.13, 0.07), grainT.r);
    const inLip = step(depth, lip), inSoil = step(depth, soil);
    const rimLand = mix(mix(soilCol, col, smoothstep(soil.sub(0.8), soil, depth)), topCol, inLip);
    const rimWet = mix(biomeColor('sand'), col, smoothstep(0.6, 1.6, depth.add(warp.mul(0.4))));
    const landRim = dry.mul(step(1.2, alt)); // dry and above the beach
    col = mix(col, mix(rimWet, rimLand, landRim), inSoil);
    // Soft occlusion under the turf lip and where the slice meets the tray.
    const under = mix(mix(0.78, 1.0, smoothstep(soil, soil.add(5), depth)), float(1), inSoil);
    col = col.mul(under).mul(mix(0.55, 1.0, smoothstep(Y_RENDER_BOTTOM, Y_RENDER_BOTTOM + 5, vy)));
    // Peridotite: deep warm grey-brown rock with broad mottling and faint rising convection
    // streaks, a gentle warm gradient with depth and a narrow hot band at the very bottom.
    const streak = flow.sub(0.5).mul(0.35).add(warpT.b.sub(0.5).mul(0.3)).add(1);
    const warm = smoothstep(Y_RENDER_BOTTOM + 45, Y_RENDER_BOTTOM, vy);
    col = mix(col, col.mul(streak).mul(mix(vec3(1), vec3(1.18, 0.92, 0.8), warm)), periW);
    const hot = pow(smoothstep(HOT_TOP, Y_RENDER_BOTTOM, vy), 1.5).mul(periW);
    col = mix(col, col.mul(vec3(1.3, 0.55, 0.3)), hot.mul(0.7));
    // Tectonics layer: faint plate-colour tint of the crust, dark casing under the boundary lines
    col = mix(col, col.mul(tecTint.mul(1.8)), faceTec.opacity.mul(0.3).mul(float(1).sub(periW)));
    return mix(col, col.mul(0.3), tecLine.a.mul(faceTec.opacity).mul(0.6));
  }).once()().toVar('sideColor');

  // Heat in cross-section from the sim's half-res 'crustTemp' (°C; pair[0] is current between mantle
  // steps): isotherms bulge up under spreading ridges and plumes and halo magma chambers, so the
  // cut reads "the crust is splitting here". Glow starts dull red ~800 °C, orange-yellow at 1300.
  const thermal = Fn(() => {
    if (!crustT) return vec3(0);
    const uh = u.mul(0.5).sub(0.25), yh = vy.mul(0.5).sub(0.25);
    const eh = int(float(edgeCol).mul(0.5));
    const u0h = floor(uh), y0h = floor(yh), fuh = fract(uh), fyh = fract(yh);
    const at = (du: number, dy: number) => {
      const a = int(clamp(u0h.add(du), 0, CX - 1));
      const cx = pick(eh, a), cz = pick(a, eh);
      const cy = int(clamp(y0h.add(dy), 0, CY - 1));
      return crustT.element(uint(cx.add(cz.mul(CX)).add(cy.mul(CX * CZ)))) as unknown as F;
    };
    const T = mix(mix(at(0, 0), at(1, 0), fuh), mix(at(0, 1), at(1, 1), fuh), fyh);
    const g = smoothstep(800, 1320, T);
    const ramp = mix(vec3(0.5, 0.06, 0.01), mix(vec3(1.0, 0.3, 0.05), vec3(1.0, 0.6, 0.18), smoothstep(0.6, 1.0, g)), smoothstep(0.0, 0.5, g));
    return ramp.mul(g.mul(g)).mul(flow.mul(0.5).add(0.75)).mul(1.5);
  })();
  const emissive = Fn(() => {
    const hot = pow(smoothstep(HOT_TOP, Y_RENDER_BOTTOM, vy), 1.6);
    const periGlow = mix(float(1), hot.mul(flow.mul(0.9).add(0.3)), periW);
    // Magma: bright cells with darker cooling veins.
    const veins = mix(0.35, 1.0, smoothstep(0.25, 0.6, fineN.a)).mul(flow.mul(0.4).add(0.8));
    const magGlow = mix(float(1), veins, magW);
    // Display fill so faces turned away from the sun still read as geology, not a black void
    // (sky-tinted, so it fades with the night).
    const fill = color.mul(skyU.zenith.mul(0.35).add(0.07));
    const tec = tecLine.rgb.mul(tecLine.a.mul(faceTec.opacity).mul(1.3)); // boundary lines read on unlit faces too
    return blendV(pal.emissive).mul(periGlow).mul(magGlow).add(fill).add(thermal).add(tec);
  })();

  // Micro normal: groove tilt from the band phase + grain relief, all in the face plane.
  const faceN = Fn(() => {
    const groove = cos(bandPhase).mul(band).mul(0.9);
    const g = vec2(grainT.a.sub(0.5), fineN.a.sub(0.5)).mul(0.35);
    const lip = smoothstep(0.0, 0.8, depth); // turf lip stays smooth
    return normalize(normalGeometry.add(vec3(0, groove.add(g.y), 0).mul(lip)).add(tangent.mul(g.x.mul(lip))));
  })();

  const rockMat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  rockMat.positionNode = positionNode;
  rockMat.colorNode = color;
  rockMat.normalNode = transformNormalToView(faceN);
  // Polished cut: glossier than the terrain, per-material, with grain breaking up the sheen.
  rockMat.roughnessNode = blendF(pal.rough).mul(mix(0.55, 0.85, grainT.b)).mul(mix(1.0, 0.85, magW));
  rockMat.emissiveNode = emissive;
  rockMat.receivedShadowNode = Fn(([s]: [F]) => s.mul(cloudShadowAt(positionWorld))) as unknown as () => THREE.Node;
  // Shadow proxy mask with the same silhouette as the colour pass (discard above the column surface).
  const shadowMask = Fn(() => {
    const y = tVoxelY(positionWorld.y);
    return y.lessThanEqual(S.height(S.corners(tWorldToCell(positionWorld.x), tWorldToCell(positionWorld.z))));
  })();

  // ---- water on the faces: glass-like column between surfY and the wet water level ----
  const wInfo = Fn(() => {
    const q = positionWorld;
    const wy = tVoxelY(q.y).toVar();
    const c = S.corners(tWorldToCell(q.x), tWorldToCell(q.z));
    const wl = S.level(c);
    // same swell as the water sheet, so the meniscus rides the waves at the cut
    const lvl = wl.level.add(swell(q.xz, wl.level.sub(S.height(c))).x.div(vertEx.mul(VOXEL_H))).toVar();
    Discard(wy.lessThan(S.height(c)).or(wy.greaterThan(lvl)).or(wl.wet.lessThan(0.001)));
    return lvl.sub(wy);
  }).once()().toVar('sideWaterDepth');
  const wd = wInfo;
  const waterMat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.04, transparent: true, depthWrite: false });
  waterMat.positionNode = positionNode;
  {
    const fragZ = positionView.z;
    const rayScale = positionView.length().div(fragZ.negate().max(1e-4));
    // Thickness to whatever is behind the glass (seabed, or deep into the block), capped.
    const thick = min(fragZ.sub(sceneViewZ(screenUV)).max(0).mul(rayScale), 0.9).toVar('glassThick');
    const V = viewDirWorld;
    // seafloor magma glow behind the glass passes neutrally, as through the sea sheet (water.ts)
    const glowK = seabedGlowK(fields, positionWorld.add(V.mul(thick)));
    const T = mix(waterTransmit(thick.add(CELL * 0.5)), vec3(exp(thick.mul(-3.0))), glowK).toVar('glassT');
    const fres = float(0.02).add(pow(float(1).sub(saturate(dot(normalGeometry, V.negate()))), 5).mul(0.98));
    const refl = skyColor(V.sub(normalGeometry.mul(dot(V, normalGeometry).mul(2))) as V3, false);
    // Meniscus: bright line right at the surface, a thin darker line just below it.
    const menBright = float(1).sub(smoothstep(0.0, 0.35, wd));
    const menDark = smoothstep(0.35, 0.6, wd).mul(float(1).sub(smoothstep(0.6, 1.1, wd)));
    const scatter = vec3(WATER_SCATTER.r, WATER_SCATTER.g, WATER_SCATTER.b).mul(float(1).sub(T.g)).mul(1.15);
    waterMat.colorNode = scatter.add(vec3(menBright.mul(0.5)));
    waterMat.emissiveNode = viewportSharedTexture(screenUV).rgb.mul(T).mul(float(1).sub(fres)).mul(float(1).sub(menDark.mul(0.35)))
      .add(refl.mul(fres)).add(vec3(0.7, 0.9, 1.0).mul(menBright.mul(0.35)));
  }
  return { rock: rockMat, water: waterMat, positionNode, shadowMask, dispose() { rockMat.dispose(); waterMat.dispose(); } };
}

/** Rock + glass water meshes (and the shadow proxy) of one face geometry drawn with `shading`. */
function faceMeshes(geo: THREE.BufferGeometry, shading: FaceShading, name: string) {
  const rock = new THREE.Mesh(geo, shading.rock);
  rock.name = name;
  rock.frustumCulled = false;
  rock.receiveShadow = true;
  const proxy = shadowProxy(geo, shading.positionNode, shading.shadowMask);
  rock.add(proxy);
  const water = new THREE.Mesh(geo, shading.water);
  water.name = `${name}-water`;
  water.frustumCulled = false;
  water.renderOrder = 3;
  water.receiveShadow = true;
  return { rock, water, proxy };
}

export function createSides(fields: GpuFields, opts: { seaLevel?: number } = {}): { object: THREE.Object3D; dispose(): void } {
  const shading = createFaceShading(fields, opts);
  const geo = createSideGeometry();
  const { rock, water, proxy } = faceMeshes(geo, shading, 'sides');
  const group = new THREE.Group();
  group.name = 'sides-group';
  group.add(rock, water);
  return {
    object: group,
    dispose() { geo.dispose(); shading.dispose(); (proxy.material as THREE.Material).dispose(); },
  };
}

/**
 * One movable vertical cut quad with the face shading (slice caps): `set` places it on the plane
 * x = const (normal ±x) or z = const (normal ±z) between two horizontal endpoints, full slice height.
 */
export function createFaceQuad(shading: FaceShading, name: string) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(12), nrm = new Float32Array(12);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  const { rock, water, proxy } = faceMeshes(geo, shading, name);
  const group = new THREE.Group();
  group.name = `${name}-group`;
  group.add(rock, water);
  return {
    object: group,
    /** a → b along the bottom edge, counter-clockwise seen from the side the normal (nx, nz) points to. */
    set(ax: number, az: number, bx: number, bz: number, nx: number, nz: number) {
      const B = Y_RENDER_BOTTOM, H = NY;
      pos.set([ax, B, az, bx, B, bz, bx, H, bz, ax, H, az]);
      for (let k = 0; k < 4; k++) nrm.set([nx, 0, nz], k * 3);
      geo.attributes.position!.needsUpdate = true;
      geo.attributes.normal!.needsUpdate = true;
    },
    dispose() { geo.dispose(); (proxy.material as THREE.Material).dispose(); },
  };
}
