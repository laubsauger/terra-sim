// T10 terrain render phase 1: heightfield mesh displaced on the GPU by `surfY`.
// Vertex heights and normals read the storage buffers directly (no 3D texture copy); reads wrap (V1),
// so interpolation across the cut edge uses the far side's columns and the world stays seamless.
// Look (§T.37/§T.40 pulled forward): texture-driven detail (no per-pixel 3D noise), rock detail
// normals, lush grass with colour variation, forest canopy bumps, snow glints, shore wetness and
// animated caustics on the seabed. Shadows follow the displacement (positionNode is used by the
// shadow pass too).
import * as THREE from 'three/webgpu';
import {
  Fn, vec2, vec3, vec4, float, uniform, varying, positionGeometry, positionWorld, transformNormalToView,
  mix, smoothstep, saturate, normalize, exp, uint, texture, max, min, dot, hash, floor, reflect,
  color, step, cos, int, fract, sin, abs, cameraPosition, length,
} from 'three/tsl';
import { NX, NZ, CELL, VOXEL_H, Y_SEA_NOMINAL, Mat } from '../sim/layout';
import { tMat } from '../sim/tslLayout';
import type { GpuFields } from '../core/gpu';
import { HALF, vertEx, tWorldY, tWorldToCell, columnSampler, voxReader, tTopVoxel, ambTime, viewDirWorld, tVoxelY, heatSampler, biomeSampler, seamGlow, volcanoSampler } from './space';
import { createPaletteNodes, createBiomeNodes, biomeColor } from './palette';
import { lookTextures } from './textures';
import { skyU } from './sky';
import { shadowProxy } from './lighting';
import { cloudShadowAt } from '../atmo/atmosphere';
import { tMeadowDensity } from '../life/flora';

type F = THREE.Node<'float'>;
type V3 = THREE.Node<'vec3'>;

/**
 * Stylised blackbody ramp for lava °C (linear HDR): dull red ~650, deep orange ~900-1000, orange-yellow
 * ~1150, yellow ~1250. Green stays low through the body and the intensity moderate: the ACES tone map
 * bends bright orange toward yellow-white, so only a hot open core should read yellow.
 */
export const blackbody = (T: F): V3 => {
  const t = smoothstep(600, 1250, T);
  const c = mix(mix(vec3(0.7, 0.08, 0.0), vec3(1.0, 0.28, 0.035), smoothstep(0.0, 0.35, t)),
    mix(vec3(1.0, 0.5, 0.1), vec3(1.0, 0.75, 0.35), smoothstep(0.85, 1.0, t)), smoothstep(0.6, 0.9, t));
  return c.mul(mix(float(1.2), float(4.5), t.mul(t))) as V3;
};

/** Magma orange (linear), the hue of rift and ridge glow on land and seen through the sea (water.ts). */
export const MAGMA_ORANGE = [1.0, 0.3, 0.045] as const;

/**
 * Slow patchy intensity along spreading seams (0.45..1.15): two world-space noise layers drifting in
 * different directions on ambTime, so patches brighten and dim over a few seconds, soft, no strobing.
 * Speeds are k/AMB_PERIOD multiples (seamless wrap).
 */
export function seamPulse(xz: THREE.Node<'vec2'>): F {
  const tex = lookTextures();
  const a = texture(tex.detail, xz.mul(0.55).add(vec2(ambTime.mul(0.02), ambTime.mul(-0.0125)))).r;
  const b = texture(tex.detail, xz.mul(0.9).add(vec2(ambTime.mul(-0.015), ambTime.mul(0.02)))).g;
  return mix(float(0.45), float(1.15), smoothstep(0.3, 0.7, a.mul(0.6).add(b.mul(0.4)))) as F;
}

/** 1 where h > threshold (sparse selection helper). */
const step01 = (h: F, t: number): F => smoothstep(t, t + 0.001, h) as F;

/**
 * Column grid: one vertex per column centre plus an outer ring on the block edges, so the mesh
 * reaches exactly the cut faces. Edge vertices sample surfY at the edge coordinate (half-way to
 * the wrapped neighbour); sides.ts uses the same bilinear sample for its silhouette.
 * Only x,z are meaningful; y comes from positionNode.
 */
export function createColumnGrid(): THREE.BufferGeometry {
  const W = NX + 2, D = NZ + 2;
  const pos = new Float32Array(W * D * 3);
  const nrm = new Float32Array(W * D * 3);
  const coord = (i: number) => Math.min(HALF, Math.max(-HALF, (i - 0.5) * CELL - HALF));
  for (let k = 0; k < D; k++) {
    for (let i = 0; i < W; i++) {
      const o = (i + k * W) * 3;
      pos[o] = coord(i); pos[o + 2] = coord(k);
      nrm[o + 1] = 1;
    }
  }
  const idx = new Uint32Array((W - 1) * (D - 1) * 6);
  let n = 0;
  for (let k = 0; k < D - 1; k++) {
    for (let i = 0; i < W - 1; i++) {
      const a = i + k * W, b = a + 1, c = a + W, d = c + 1;
      idx[n++] = a; idx[n++] = c; idx[n++] = b;
      idx[n++] = b; idx[n++] = c; idx[n++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

export interface TerrainOptions {
  /** Voxel-y used for altitude tints (beach, snow). Placeholder until the derived sea level exists. */
  seaLevel?: number;
  snowLine?: number; // voxels above sea level
}

/**
 * Animated caustic web on a horizontal plane at world xz, projected along the sun.
 * Two scrolled Worley-ridge layers min()'d together; returns 0..~1.
 */
/** Seabed caustic strength (albedo boost). Tunable at runtime. */
export const causticGain = uniform(0.4);

export function causticsAt(p: V3, depth: F): F {
  const tex = lookTextures();
  const off = skyU.sunDir.xz.mul(depth.mul(VOXEL_H * 0.6)); // shift with depth along the light
  const q = p.xz.add(off);
  const t = ambTime;
  // scroll speeds are k / AMB_PERIOD multiples (seamless wrap): 0.01 = 36/3600
  // larger cells + a blurrier mip keep the web soft instead of wiry
  const a = texture(tex.caustics, q.mul(0.9).add(vec2(t.mul(0.01), t.mul(0.0075)))).level(float(1.5)).r;
  const b = texture(tex.caustics, q.mul(1.15).add(vec2(t.mul(-0.0075), t.mul(0.01)))).level(float(1.5)).g;
  return smoothstep(0.05, 0.6, min(a, b)) as F;
}

export function createTerrain(fields: GpuFields, opts: TerrainOptions = {}): { object: THREE.Mesh; dispose(): void } {
  const S = columnSampler(fields);
  const vox = voxReader(fields);
  const pal = createPaletteNodes();
  const bio = createBiomeNodes();
  const tex = lookTextures();
  const seaLevel = uniform(opts.seaLevel ?? Y_SEA_NOMINAL);
  const snowLine = uniform(opts.snowLine ?? 30);

  // ---- vertex ----
  const gu = tWorldToCell(positionGeometry.x), gv = tWorldToCell(positionGeometry.z);
  const heightAt = (du: number, dv: number) => S.height(S.corners(gu.add(du), gv.add(dv)));

  const hVary = varying(Fn(() => heightAt(0, 0))(), 'vTerrH');
  // xyz = normal (wrapped finite differences, V1), w = broad cavity term (Laplacian over ±3 cells) for fake AO.
  const nVary = varying(Fn(() => {
    const k = vertEx.mul(VOXEL_H);
    const hl = heightAt(-1, 0), hr = heightAt(1, 0), hd = heightAt(0, -1), hu = heightAt(0, 1);
    const n = normalize(vec3(hl.sub(hr).mul(k), 2 * CELL, hd.sub(hu).mul(k)));
    const lap = heightAt(-3, 0).add(heightAt(3, 0)).add(heightAt(0, -3)).add(heightAt(0, 3)).sub(heightAt(0, 0).mul(4));
    return vec4(n, lap);
  })(), 'vTerrN');

  const positionNode = Fn(() => vec3(positionGeometry.x, tWorldY(heightAt(0, 0)), positionGeometry.z))();

  // ---- fragment ----
  const p = positionWorld;
  // detail texture taps (tileable): r broad patches, g mid, b fine grain, a ridged cracks
  const dBroad = texture(tex.detail, p.xz.mul(0.22)).toVar('dBroad');
  const dMid = texture(tex.detail, p.xz.mul(1.1)).toVar('dMid');
  const dFine = texture(tex.detail, p.xz.mul(4.5)).toVar('dFine');

  const h = hVary;
  const nLo = dBroad.r.sub(0.5).mul(2).toVar('nLo'); // -1..1 broad patches
  const nMid = dMid.g.sub(0.5).mul(2).toVar('nMid');
  const nHi = dFine.b.sub(0.5).mul(2).toVar('nHi');
  const alt = h.sub(seaLevel).toVar('terrAlt');

  // Cold band: SPEC climate latitude term cos(2πz/Z) (warm at z=0, cold mid-block) lowers the snow line.
  const cold = float(0.5).sub(cos(tWorldToCell(p.z).mul(Math.PI * 2 / NZ)).mul(0.5));
  const snowLineEff = snowLine.sub(cold.mul(12)).toVar('snowLineEff');

  // Biome display colours (eased in time, advected with the plates; colours blend, ids never do).
  // bioCover = (ground covered by its vegetation) rgb, w = veg; bioFlags = (arid, forest, -, has biome data).
  const bioS = biomeSampler(fields)(tWorldToCell(p.x), tWorldToCell(p.z));
  const vegAmt = Fn(() => {
    const c = S.corners(tWorldToCell(p.x), tWorldToCell(p.z));
    return c.slice(1).reduce((a, k) => a.add(k.veg.mul(k.w)), c[0]!.veg.mul(c[0]!.w) as F);
  }).once()().toVar('terrVeg');
  const hasVeg = step(0, vegAmt);
  // patchy cover: veg thins out in noise-shaped holes instead of a uniform fade
  const coverT = smoothstep(0.0, 0.35, max(vegAmt, 0).add(nMid.mul(0.25)).sub(0.1));
  const bioCover = vec4(mix(bioS.ground.rgb, bioS.veg.rgb, coverT), max(vegAmt, 0)).toVar('terrBioCover');
  const bioFlags = vec4(bioS.ground.w, bioS.veg.w, 0, hasVeg).toVar('terrBioFlags');
  const arid = bioFlags.x.mul(bioFlags.w), hasBio = bioFlags.w;

  // Volcano display: x vent activity, y lava channel memory (both eased, advected).
  const volcS = volcanoSampler(fields)(tWorldToCell(p.x), tWorldToCell(p.z)).toVar('terrVolc');
  // Heat summary (renderHeat): x crust age My, y lava layers, z lava °C, w sim 'ice' cover.
  const heat = heatSampler(fields)(tWorldToCell(p.x), tWorldToCell(p.z)).toVar('terrHeat');
  const lavaMask = smoothstep(0.02, 0.35, heat.y).toVar('terrLava');
  const slopeAt = float(1).sub(nVary.xyz.normalize().y);

  // Surface class masks, shared by colour, roughness, normal and emissive:
  // x rock, y snow, z forest, w underwater depth (voxel units) or -1 on land.
  const masks = Fn(() => {
    const n = nVary.xyz.normalize();
    const slope = float(1).sub(n.y);
    // arid ground breaks into bare cliffs at gentler slopes (mesas, canyon walls)
    const rockAmt = smoothstep(mix(0.2, 0.1, arid), mix(0.36, 0.24, arid), slope.add(nHi.mul(0.03)).add(nLo.mul(0.05)));
    // Snow from the sim: 'ice' cover, biome ICE/ALPINE, plus a dusting on steep high peaks. The
    // altitude-only snow line is the fallback when there is no climate data.
    const simSnow = smoothstep(0.1, 0.6, heat.w.add(nHi.mul(0.08)).add(nMid.mul(0.1)));
    const peak = smoothstep(snowLineEff.add(2), snowLineEff.add(8), alt.add(nLo.mul(3))).mul(smoothstep(0.12, 0.3, slope));
    const altSnow = smoothstep(snowLineEff.sub(3), snowLineEff.add(2), alt.add(nLo.mul(4)).add(nHi.mul(1.2)));
    const snowAmt = mix(altSnow.mul(float(1).sub(arid.mul(0.7))), max(simSnow, peak.mul(0.8)), hasBio)
      .mul(float(1).sub(smoothstep(0.45, 0.72, slope.add(nHi.mul(0.05)))));
    const clumps = smoothstep(0.08, 0.28, nMid.mul(0.6).add(nLo.mul(0.55)));
    const procForest = clumps.mul(float(1).sub(saturate(alt.sub(4).div(14)))).mul(smoothstep(1.2, 3.0, alt));
    const bioForest = bioFlags.y.mul(smoothstep(0.3, 0.7, bioCover.w)).mul(smoothstep(-0.2, 0.3, nMid.mul(0.7).add(nLo.mul(0.4)).add(0.2)));
    const forestAmt = mix(procForest, bioForest, hasBio).mul(float(1).sub(rockAmt));
    const wl = S.level(S.corners(tWorldToCell(p.x), tWorldToCell(p.z)));
    // Underwater only where the corners are (nearly) all wet: near the coast the smoothed land can
    // dip under the neighbours' water level without a real water sheet over it.
    const depth = wl.level.sub(h).mul(smoothstep(0.5, 0.95, wl.wet));
    const under = wl.wet.greaterThan(0.5).and(depth.greaterThan(0.0));
    return vec4(rockAmt, snowAmt, forestAmt, mix(float(-1), depth, float(under)));
  }).once()().toVar('terrMask');
  const rockAmt = masks.x, snowAmt = masks.y, forestAmt = masks.z, uwDepth = masks.w;

  // out = (rgb albedo, roughness)
  const out0 = Fn(() => {
    const c = S.corners(tWorldToCell(p.x), tWorldToCell(p.z));
    // Top voxel material of each corner column, bilinear. Steep faces show bedrock: loose SEDIMENT
    // does not hold on cliffs, so they use the voxel 2 below. Bedrock only *tints* humid rock
    // (strata colours belong on the cut faces, not as zebra stripes on mountains).
    let top = vec3(0) as V3;
    let bedTint = vec3(0) as V3;
    let rockR = float(0) as F;
    for (const k of c) {
      const ty = tTopVoxel(k.raw);
      const m = tMat(vox(k.x, ty, k.z));
      const mb = tMat(vox(k.x, ty.sub(2), k.z));
      const bed = m.add(mb.sub(m).mul(uint(m.equal(uint(Mat.SEDIMENT))))); // branch-free select
      top = top.add(pal.color(m).mul(k.w));
      bedTint = bedTint.add(pal.color(bed).mul(k.w));
      rockR = rockR.add(pal.rough(bed).mul(k.w));
    }
    const n = nVary.xyz.normalize();
    const slope = float(1).sub(n.y);
    const cavity = nVary.w;

    // Procedural fallback (no biome fields): lush lowlands → alpine meadow, from altitude.
    const lush = mix(biomeColor('grassLush'), vec3(0.16, 0.4, 0.06), saturate(nMid.mul(0.6).add(0.35)).mul(0.55));
    const dry = mix(biomeColor('grassDry'), vec3(0.5, 0.45, 0.16), saturate(nLo.mul(0.5).add(0.2)));
    const alpine = mix(vec3(0.33, 0.4, 0.13), vec3(0.42, 0.38, 0.2), saturate(nMid.mul(0.5).add(0.5)));
    let proc = mix(lush, dry, saturate(alt.div(26).add(nLo.mul(0.3)).sub(0.15))) as V3;
    proc = mix(proc, alpine, smoothstep(snowLineEff.mul(0.4), snowLineEff.mul(0.85), alt.add(nLo.mul(3))));
    // Biome ground cover with a little tonal variation so fields of one biome stay alive, and the
    // life module's meadow patch field: greener/darker under dense drifts, straw on bare patches.
    const meadow = smoothstep(0.15, 2.5, tMeadowDensity(p.x, p.z)).mul(max(vegAmt, 0));
    const meadowTint = mix(vec3(1.12, 1.06, 0.82), vec3(0.78, 0.95, 0.72), meadow);
    const cover = bioCover.rgb.mul(nMid.mul(0.07).add(1)).mul(nLo.mul(0.06).add(1)).mul(mix(vec3(1), meadowTint, float(1).sub(forestAmt)));
    const ground = mix(proc, cover, hasBio);
    const canopy = dFine.a; // clumps of trees
    const forestCol = mix(vec3(0.035, 0.12, 0.045), vec3(0.11, 0.26, 0.07), canopy);
    const vegCol = mix(ground, mix(forestCol, cover.mul(mix(0.55, 0.9, canopy)), hasBio), forestAmt);
    // Humid rock: warm grey-brown, crevices from the ridged detail, faint bedrock tint.
    const rockBase = mix(vec3(0.21, 0.17, 0.14), vec3(0.34, 0.3, 0.26), dMid.a);
    const rockCol = mix(rockBase, bedTint.mul(0.8), 0.18).mul(nHi.mul(0.1).add(1)).mul(mix(0.75, 1.1, dFine.a));
    // Arid cliffs: the real near-surface layers at this height (SANDSTONE/SHALE/LIMESTONE → red,
    // ochre, cream bands), each layer with a lit top ledge and a shadowed lip.
    // Layer colour blended over the four corner columns at this height (no brick seams between cells).
    const vyF = tVoxelY(p.y).sub(0.3);
    let strataCol = vec3(0) as V3;
    for (const k of c) {
      const cy = int(min(floor(vyF), float(tTopVoxel(k.raw))));
      strataCol = strataCol.add(bio.strata(tMat(vox(k.x, cy, k.z))).mul(k.w));
    }
    const ledge = mix(0.74, 1.08, smoothstep(0.05, 0.35, fract(vyF))).mul(mix(1.0, 0.86, smoothstep(0.85, 1.0, fract(vyF))));
    const canyon = strataCol.mul(ledge).mul(dFine.b.sub(0.5).mul(0.14).add(1));
    const cliff = mix(rockCol, canyon, arid);
    const scree = mix(mix(vec3(0.42, 0.37, 0.31), vec3(0.55, 0.5, 0.43), dFine.b), canyon.mul(1.15), arid.mul(0.6));
    const screeAmt = smoothstep(0.08, 0.2, slope).mul(float(1).sub(rockAmt)).mul(smoothstep(0.5, 4, cavity))
      .mul(smoothstep(4, 9, alt));
    let land = mix(vegCol, scree, screeAmt) as V3;
    land = mix(land, cliff, rockAmt);
    // Beach band: narrow when the biome field already marks beaches.
    const sandTop = mix(float(2.2), float(1.1), hasBio);
    const sandAmt = float(1).sub(smoothstep(sandTop.mul(0.25), sandTop, alt.add(nHi.mul(0.4)).add(nLo.mul(0.5))))
      .mul(float(1).sub(rockAmt.mul(0.6))).toVar();
    land = mix(land, color(biomeColor('sand')).mul(nMid.mul(0.05).add(1)), sandAmt);
    land = mix(land, color(biomeColor('snow')).mul(mix(0.92, 1.0, dFine.b)), snowAmt);
    // Swash: a thin foam front runs up the beach and slides back on ambTime; sand it reached stays wet.
    const surge = sin(ambTime.mul(0.9).add(dMid.b.mul(5))).mul(0.5).add(0.5).mul(sin(ambTime.mul(0.37).add(1.3)).mul(0.25).add(0.75));
    const reach = mix(0.1, 0.75, surge);
    const altS = alt.add(nHi.mul(0.08));
    const front = smoothstep(reach.sub(0.12), reach, altS).mul(float(1).sub(smoothstep(reach, reach.add(0.04), altS)));
    // Wet shore band just above the waterline: darker, glossier (up to the highest swash reach).
    const wetBand = float(1).sub(smoothstep(0.0, 0.9, altS)).mul(sandAmt).toVar();
    land = mix(land, biomeColor('wetSand'), wetBand.mul(0.65));
    land = mix(land, vec3(0.85, 0.88, 0.9), front.mul(sandAmt).mul(step(0, altS)).mul(0.8));

    // Seabed: the water shader does absorption, so the bed keeps its own colour; bright sand in the
    // shallows (luminous turquoise lagoons), darker sediment deeper.
    // Fresh ridge basalt is near-black blue; sediment settles with crust age (bright sand only on
    // old, shallow shelves), so a raised young crest is dark, not a pale stripe.
    const sedAge = smoothstep(0.5, 8.0, heat.x);
    const bedOld = mix(mix(top, vec3(0.3, 0.26, 0.2), 0.5), color(biomeColor('sand')).mul(1.1), smoothstep(7, 0.5, uwDepth));
    const seabed = mix(vec3(0.05, 0.06, 0.08), bedOld, sedAge)
      .mul(nHi.mul(0.08).add(1)).mul(nLo.mul(0.06).add(1));

    // Soft fake AO from curvature (SSGI/GTAO add real AO on top in high quality).
    const ao = float(1).sub(saturate(cavity.mul(0.05)).mul(0.3)).add(saturate(cavity.mul(-0.05)).mul(0.06));
    const underF = step(0, uwDepth);
    land = mix(vec3(dot(land, vec3(0.2126, 0.7152, 0.0722))), land, 0.85) as V3; // ACES saturates: pre-soften
    const col = mix(land, seabed, underF).mul(ao);
    const landR = mix(mix(float(0.92), mix(rockR.mul(0.8), float(0.85), arid), rockAmt), float(0.5), snowAmt)
      .mul(float(1).sub(wetBand.mul(0.6)));
    return vec4(col, mix(landR, float(0.7), underF));
  }).once()().toVar('terrShade');

  // Detail normal: finite differences of the ridged channel (crisp cracks on rock), canopy bumps in
  // forests, soft ripple in grass. 3 extra taps on one texture.
  const bumpN = Fn(() => {
    const uv = p.xz.mul(2.6);
    const e = 1 / 256;
    const h0 = texture(tex.detail, uv).a;
    const hx = texture(tex.detail, uv.add(vec2(e, 0))).a;
    const hz = texture(tex.detail, uv.add(vec2(0, e))).a;
    const g = vec2(hx.sub(h0), hz.sub(h0)).mul(256 * 2.6 * 0.004);
    const canopyG = vec2(dFine.a.sub(0.5), dFine.b.sub(0.5));
    const k = rockAmt.mul(1.6).add(forestAmt.mul(0.15)).add(0.08).mul(float(1).sub(snowAmt.mul(0.8)));
    const n = nVary.xyz.normalize();
    return normalize(n.add(vec3(g.x.negate(), 0, g.y.negate()).mul(k)).add(vec3(canopyG.x, 0, canopyG.y).mul(forestAmt.mul(0.35))));
  })();

  // Seabed caustics: albedo boost so the sun shadow still gates them; fade in from the shoreline and out with depth.
  const caus = Fn(() => {
    const d = max(uwDepth, 0);
    // subtle, soft and shallow-only: fade in off the shoreline, gone by ~10 layers depth
    const k = step(0, uwDepth).mul(smoothstep(0.4, 1.8, d)).mul(exp(d.mul(-0.22)))
      // distance-aware: the web is a close-up detail; from the hero distance it would tile into noise
      .mul(float(1).div(positionWorld.sub(cameraPosition).length().mul(0.35).add(1)).mul(2.2).min(1));
    return causticsAt(p, d).mul(k).mul(saturate(skyU.sunDir.y.mul(5)));
  })();

  // Snow glints: sparse hashed facets that flash when their jittered mirror direction meets the sun.
  const glint = Fn(() => {
    const cell = floor(p.mul(900));
    const hsh = hash(cell.x.add(cell.y.mul(57.3)).add(cell.z.mul(113.9)));
    const r = reflect(viewDirWorld, bumpN);
    const jitter = vec3(hash(hsh.add(1.7)), hash(hsh.add(4.1)), hash(hsh.add(7.3))).sub(0.5).mul(0.5);
    const facing = max(dot(normalize(r.add(jitter)), skyU.sunDir), 0);
    return smoothstep(0.985, 0.998, facing).mul(step01(hsh, 0.7)).mul(snowAmt).mul(skyU.sunIntensity).mul(3.5);
  })();

  // Spreading ridges and lava (§T.38 pulled forward). renderHeat: x crust age My, y lava layers,
  // z lava °C. Young crust (< ~1 My) glows along the seam through flickering fissures, on land and on
  // the seafloor; seen through the sea, water.ts passes this light neutrally (dimmed, not turned green).
  // Lava: blackbody glow broken by cooling crust.
  const glow = Fn(() => {
    const crack = texture(tex.detail, p.xz.mul(1.7).add(vec2(0, ambTime.mul(0.0025)))).a;
    const pulse = seamPulse(p.xz);
    const rg = seamGlow(volcS.zw, pulse.mul(crack.mul(0.4).add(0.6))); // fissures break up the land core
    // Subaerial (rift) glow only on truly dry ground: the bilinear age also sees young seafloor
    // next to a coast, which must not light the beach.
    const wet = S.level(S.corners(tWorldToCell(p.x), tWorldToCell(p.z))).wet;
    const subaerial = float(1).sub(step(0, uwDepth)).mul(float(1).sub(smoothstep(0.0, 0.15, wet)));
    const submerged = step(0, uwDepth);
    // magma-orange core → dim halo of the same hue (kept moderate: little bloom)
    const ridgeCol = vec3(...MAGMA_ORANGE).mul(rg.x.mul(1.6).add(rg.y.mul(0.8)));
    // Seafloor seam: same field without the fissure pattern, a little dimmer; water.ts lets this light
    // through the sea (see there).
    const rgSea = seamGlow(volcS.zw, pulse);
    const seaCol = vec3(...MAGMA_ORANGE).mul(rgSea.x.mul(1.4).add(rgSea.y));
    const ridgeGlowC = ridgeCol.mul(subaerial).add(seaCol.mul(submerged));
    // Lava as molten rock (display-eased depth / temperature / channel memory): black crust plates
    // drifting downhill (two-phase flow map along the slope), glowing cracks between them, an
    // incandescent open core where the flow is thick, hot and in a live channel, blackbody colour
    // by temperature, a faint heat shimmer. Never a smooth glaze.
    const T = heat.z;
    // thin films count too: most flows are a fraction of a layer deep, and a hot one must read as lava
    const lavaAmt = smoothstep(0.005, 0.15, heat.y);
    const hot = smoothstep(800, 1150, T);
    const coreness = smoothstep(0.05, 0.6, heat.y).mul(hot).mul(volcS.y.mul(0.5).add(0.5));
    const nrm = nVary.xyz.normalize();
    const flow = nrm.xz.div(max(length(nrm.xz), 1e-3)).mul(smoothstep(0.0, 0.08, length(nrm.xz)));
    const ph0 = fract(ambTime.mul(0.15)), ph1 = fract(ambTime.mul(0.15).add(0.5)); // 540 periods / AMB_PERIOD
    const uvL = p.xz.mul(6.0);
    const plA = texture(tex.detail, uvL.sub(flow.mul(ph0.mul(0.35)))).a;
    const plB = texture(tex.detail, uvL.sub(flow.mul(ph1.mul(0.35))).add(0.37)).a;
    const plates = mix(plA, plB, abs(ph0.mul(2).sub(1)));
    const shimmer = texture(tex.detail, p.xz.mul(14).add(vec2(ambTime.mul(0.02), 0))).b.sub(0.5).mul(0.06);
    const crust = smoothstep(0.42, 0.6, plates.add(shimmer).add(float(1).sub(coreness).mul(0.35)));
    const cracks = float(1).sub(smoothstep(0.012, 0.07, abs(plates.sub(0.5))));
    const pulseL = sin(ambTime.mul(1.1).add(dMid.r.mul(9))).mul(0.08).add(0.92);
    // a hot flow glows through its crust too (dull, never black): visible in daylight from the hero view
    const open = max(float(1).sub(crust).add(crust.mul(cracks).mul(0.6)), hot.mul(0.6));
    // colour temperature: crust and cracks stay deep orange (≤ 1000 °C), only the open core of a hot
    // live channel shows its real temperature (orange-yellow)
    const lavaGlow = blackbody(mix(min(T, 1000), T, coreness.mul(float(1).sub(crust)))).mul(open.mul(lavaAmt).mul(pulseL).mul(smoothstep(550, 800, T)));
    // Summit crater: glowing lava lake in the concave top while the vent is active, pulsing rim embers.
    const act = volcS.x;
    // vent columns are single cells: the bilinear activity already gives a small round crater spot;
    // concave tops (carved craters) glow a little wider
    // (low threshold: the bilinear spot spreads over the neighbouring cells, so the vent reads from afar)
    const lake = smoothstep(0.08, 0.5, act).mul(smoothstep(-2.0, 1.5, nVary.w).mul(0.5).add(0.5));
    const rimPulse = sin(ambTime.mul(2.2)).mul(0.3).add(0.7);
    // a deep orange lake (not white-hot: it is a small spot and would clip to yellow-white)
    const craterGlow = blackbody(float(1000)).mul(lake.mul(open.mul(0.5).add(0.5)).mul(1.2))
      .add(blackbody(float(850)).mul(act.mul(float(1).sub(lake)).mul(rimPulse).mul(smoothstep(0.05, 0.25, slopeAt)).mul(0.25)));
    return ridgeGlowC.add(lavaGlow).add(craterGlow);
  })();
  // Crust albedo: black basalt under lava; old channels stay dark basalt ribbons after the flow stops.
  // Fresh basalt along spreading seams, a little wider than the crack: the glow sits on dark rock, and
  // where the sea passes the glow light neutrally (water.ts) it shows basalt, never a pale sand band.
  const lavaCrust = max(max(lavaMask.mul(0.97), volcS.y.mul(float(1).sub(lavaMask)).mul(0.75)),
    max(smoothstep(0.1, 0.5, volcS.z), smoothstep(0.1, 0.5, volcS.w).mul(0.7)).mul(0.9));

  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  mat.positionNode = positionNode;
  mat.normalNode = transformNormalToView(bumpN);
  mat.colorNode = mix(out0.rgb.mul(caus.mul(causticGain).add(1)), vec3(0.035, 0.03, 0.03), lavaCrust);
  mat.roughnessNode = mix(out0.a, float(0.88), lavaMask);
  mat.emissiveNode = skyU.sunColor.mul(glint).add(glow);
  // cloud shadows from the atmosphere module (1 until it runs)
  mat.receivedShadowNode = Fn(([s]: [F]) => s.mul(cloudShadowAt(positionWorld))) as unknown as () => THREE.Node;

  const mesh = new THREE.Mesh(createColumnGrid(), mat);
  mesh.name = 'terrain';
  mesh.frustumCulled = false; // displaced on GPU; CPU bounds are meaningless
  mesh.receiveShadow = true;
  // Casts through a proxy with the same displacement, so shadows follow the relief without the
  // shadow pass paying for the full surface shader.
  const proxy = shadowProxy(mesh.geometry, positionNode);
  mesh.add(proxy);
  return {
    object: mesh,
    dispose() { mesh.geometry.dispose(); mat.dispose(); (proxy.material as THREE.Material).dispose(); },
  };
}

