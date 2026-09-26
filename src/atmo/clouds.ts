// §T.41 clouds: noise-based raymarched cloud volume over the diorama (Nubis-style, stylised),
// following the sim's vapor/precip.
//
// Pipeline (per frame; reads sim fields only, V15):
//   1. summary  (SUM² threads): 4×4-column averages of cloud cover (relative humidity + precip),
//               precip, surfTemp and the max ground/water level.
//   2. weather  (RES² → rgba16f 2D texture): the summary blurred heavily (Gaussian σ ≈ 9 columns),
//               times a domain-warped low-frequency coverage noise that drifts with the steering wind
//               (each wind band translates its noise rigidly on ambTime, V17, bands cross-fade where they meet, so the latitude shear never smears clouds into streaks; the noise also evolves slowly). Organic
//               cells with gaps, never grid-aligned; coverage stays clumpy and ≤ ~35 % of the sky.
//               r = coverage, g = storm (cumulonimbus), b = stratus (cold, calm), a = cloud base (world y).
//   3. density  (VX·VY·VX → rgba16f 3D texture): coverage × height profile (flat base at the
//               condensation level, rounded cumulus tops, flat stratus, tall storm towers whose anvils
//               spread downwind), eroded by Perlin-Worley in the drifting noise frame; sparse cirrus
//               streaks near the slab top; faded to 0 at every slab boundary (no visible volume).
//   4. shadow   (RES² → 2D texture): transmittance of the slab along the key light; cloudShadowAt()
//               projects any world point onto it (terrain / water can multiply their sun term).
//   5. march    (half res high / third res low, before the scene pass): full-screen raymarch through
//               the density with per-pixel, per-frame jitter; 2-octave detail Worley erosion; a short
//               light march toward the key light per sample (4 steps) + sky visibility upward;
//               Beer-Lambert + powder, dual-lobe HG (silver lining), sky ambient, darker bases; fades
//               out near the camera; strides through empty space. Writes premultiplied colour + the
//               cloud's front distance, blended with the history reprojected at that depth (temporal
//               accumulation; TRAA smooths further in the high tier).
//   6. composite (scene pass, full res): a back-faced box over the slab samples the march target and
//               hides clouds whose front lies behind the opaque scene depth (sceneViewZ): mountains and
//               the plinth occlude correctly. Fully clear pixels are discarded.
// Storage buffers bound: summary 6, cells 2, others ≤ 1 (+ storage textures) (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, Loop, Break, float, int, uint, vec2, vec3, vec4, uvec2, uvec3, uniform, uniformArray, instanceIndex,
  instancedArray, exp, mix, smoothstep, saturate, max, min, floor, fract, abs, length, normalize, dot, pow,
  sign, textureStore, texture, texture3D, positionWorld, cameraPosition, cameraViewMatrix, step,
  screenUV, uv, property, outputStruct,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NX, NZ, CELL, BLOCK_SIZE } from '../sim/layout';
import { tColIdx } from '../sim/tslLayout';
import { CLIMATE } from '../sim/climateModel';
import { tWorldY, sceneViewZ } from '../render/shared';
import { skyU } from '../render/sky';
import { ATMO, HALF, AMB_PERIOD, BANDS, bandSpeeds, cirrusSpeed } from './atmoModel';
import { atmoSeaLevel, tWind, tWindHF, tKeyColor, fxMRT, tCutSoft } from './atmoTsl';
import { cloudNoiseTexture } from './noise3d';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

// ---------------------------------------------------------------------------------------------
// Cloud shadow export (module level, so render materials can reference it before/without atmo).
// ---------------------------------------------------------------------------------------------

/**
 * Cloud shadow texture over the block (x,z ∈ [-HALF, HALF] → uv [0,1]) at the slab bottom.
 * r = direct-light factor along the key light (1 clear … 1 - SHADOW_MAX under thick cloud),
 * g = cloud optical depth, scaled 0..1. Written by the atmosphere every frame.
 */
export const cloudShadowTexture = new THREE.StorageTexture(ATMO.SHADOW_RES, ATMO.SHADOW_RES);
cloudShadowTexture.generateMipmaps = false;
cloudShadowTexture.wrapS = cloudShadowTexture.wrapT = THREE.ClampToEdgeWrapping;
cloudShadowTexture.name = 'cloudShadow';

/** Uniforms behind cloudShadowAt (driven by the atmosphere; `on` stays 0 until it has written the texture). */
export const cloudShadowU = {
  on: uniform(0),
  /** Direction towards the shadow-casting key light (sun, or moon at night), as lighting.ts places it. */
  keyDir: uniform(new THREE.Vector3(0, 1, 0)),
  /** World y of the slab bottom, where the texture lives. */
  planeY: uniform(0.3),
};

/**
 * Direct-light multiplier (0..1) at a world position from the clouds above it, projected along the
 * key light. Multiply the sun's received shadow with it, e.g.
 *   material.receivedShadowNode = Fn(([s]) => s.mul(cloudShadowAt(positionWorld)));
 */
export const cloudShadowAt = (p: V3): F => Fn(() => {
  const k = cloudShadowU.keyDir;
  const t = max(cloudShadowU.planeY.sub(p.y), 0).div(max(k.y, 0.08));
  const q = p.xz.add(k.xz.mul(t));
  const uv = q.add(HALF).div(BLOCK_SIZE);
  const inside = step(0, uv.x).mul(step(uv.x, 1)).mul(step(0, uv.y)).mul(step(uv.y, 1));
  const l = texture(cloudShadowTexture, uv).r;
  return mix(float(1), l, cloudShadowU.on.mul(inside));
})() as F;

// ---------------------------------------------------------------------------------------------

export interface Clouds {
  object: THREE.Mesh;
  /** Climate summary (SUM² vec4: cover, precip, surfTemp °C, max ground/water level voxel-y). */
  summary: StorageNode<'vec4'>;
  /** Weather probe grid (CELLS² vec4: coverage, storm, base world y, ground world y) for lightning + tests. */
  cells: StorageNode<'vec4'>;
  weather: THREE.StorageTexture;
  density: THREE.Storage3DTexture;
  /** Bilinear summary lookup at world x/z (TSL, for other atmo kernels). */
  sampleSummary(x: F, z: F): V4;
  /** Weather map at world x/z (TSL, compute or vertex): (coverage, storm, stratus, base world y). */
  weatherAt(x: F, z: F): V4;
  uniforms: {
    time: THREE.UniformNode<'float', number>;
    yLo: THREE.UniformNode<'float', number>;
    yHi: THREE.UniformNode<'float', number>;
    coverage: THREE.UniformNode<'float', number>;
    flashPos: THREE.UniformNode<'vec3', THREE.Vector3>;
    flash: THREE.UniformNode<'float', number>;
    fade: THREE.UniformNode<'float', number>;
  };
  /**
   * Cloud occlusion for transparent FX drawn after the cloud composite: multiply alpha by this.
   * Depth-aware: 1 in front of the cloud's front, (1 - cloud opacity) behind it. `centre` = world position.
   */
  occlusion(centre: V3): F;
  /** Volcanic plumes drawn in the volume: vent (x, y, z), strength 0..1.5, wind dir (x, z), drift length, umbrella y. */
  setPlumes(list: { x: number; y: number; z: number; s: number; dx: number; dz: number; len: number; top: number }[]): void;
  /** Run summary → weather → density → shadow, then the reduced-resolution raymarch for `camera`. */
  compute(renderer: THREE.WebGPURenderer, camera: THREE.Camera): void;
  setHighQuality(v: boolean): void;
  dispose(): void;
}

const S = ATMO.SUM;
const STEP = NX / S;
const W = ATMO.WEATHER_RES;
const VX = ATMO.VX, VY = ATMO.VY;
const NC = ATMO.CELLS;

export function createClouds(fields: GpuFields, opts: { highQuality: boolean; renderer: THREE.WebGPURenderer }): Clouds {
  const vapor = fields.cur('vapor'), precip = fields.cur('precip'), surfTemp = fields.cur('surfTemp');
  const surfY = fields.cur('surfY'), water = fields.cur('water');
  const summary = instancedArray(S * S, 'vec4') as unknown as StorageNode<'vec4'>;
  summary.setName('atmoSummary');
  const cells = instancedArray(NC * NC, 'vec4') as unknown as StorageNode<'vec4'>;
  cells.setName('atmoCells');
  const noise = cloudNoiseTexture();

  const weather = new THREE.StorageTexture(W, W);
  weather.type = THREE.HalfFloatType;
  weather.generateMipmaps = false;
  weather.minFilter = weather.magFilter = THREE.LinearFilter;
  weather.wrapS = weather.wrapT = THREE.RepeatWrapping; // torus: off-block lookups (anvils) wrap instead of smearing edge texels
  weather.name = 'cloudWeather';
  const mk3 = (x: number, y: number, z: number, type: THREE.TextureDataType, name: string) => {
    const t = new THREE.Storage3DTexture(x, y, z);
    t.name = name; t.type = type; t.format = THREE.RGBAFormat;
    t.generateMipmaps = false;
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    return t;
  };
  const density = mk3(VX, VY, VX, THREE.HalfFloatType, 'cloudDensity');

  const u = {
    time: uniform(0),
    yLo: uniform(0.3),
    yHi: uniform(1.1),
    coverage: uniform(ATMO.COVERAGE),
    flashPos: uniform(new THREE.Vector3()),
    flash: uniform(0),
    /** 0 hidden … 1 normal (overlays dim the clouds). */
    fade: uniform(1),
  };
  const baseTile = ATMO.DETAIL_TILE * 4;
  const cirrusV = cirrusSpeed();
  const lvl0 = int(0);

  // ---- wind drift: noise translates rigidly with its wind band (no shear streaks), bands cross-fade ----
  const bands = bandSpeeds();
  const bandU = uniformArray(bands.map((b) => b.u), 'float');
  const bandV = uniformArray(bands.map((b) => b.v), 'float');
  const DT = ATMO.DRIFT_TILE;
  const bandFlow = (z: F): { o0: V2; o1: V2; w1: F } => {
    const b = z.add(HALF).div(CELL).sub(0.5).div(NZ / BANDS);
    const i0f = floor(b.add(0.5));
    const f = b.sub(i0f);
    const i0 = int(i0f).add(int(BANDS * 4)).mod(int(BANDS));
    const i1 = int(i0f.add(sign(f))).add(int(BANDS * 4)).mod(int(BANDS));
    const o = (i: THREE.Node<'int'>) => vec2(
      fract((bandU.element(i) as unknown as F).mul(u.time).div(DT)).mul(DT),
      fract((bandV.element(i) as unknown as F).mul(u.time).div(DT)).mul(DT)) as V2;
    return { o0: o(i0), o1: o(i1), w1: smoothstep(0.3, 0.5, abs(f)).mul(0.5) as F };
  };
  /** Towers lean downwind: the height-dependent part of the wind over TILT_S seconds (bounded). */
  const tilt = (z: F, y: F): V2 => tWind(z, tWindHF(y)).sub(tWind(z, float(ATMO.CLOUD_HF))).mul(ATMO.TILT_S) as V2;
  const evolve = u.time.mul(ATMO.EVOLVE / AMB_PERIOD);

  // ---- 1. climate summary ----
  const summaryK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(S * S)), () => { Return(); });
    const sx = int(instanceIndex.mod(uint(S))).mul(STEP), sz = int(instanceIndex.div(uint(S))).mul(STEP);
    const cov = float(0).toVar(), pr = float(0).toVar(), tp = float(0).toVar(), gmax = float(-1e9).toVar();
    for (let dz = 0; dz < STEP; dz++) for (let dx = 0; dx < STEP; dx++) {
      const c = tColIdx(sx.add(int(dx)), sz.add(int(dz)));
      const T = (surfTemp.element(c) as unknown as F).toVar();
      const q = vapor.element(c) as unknown as F;
      const p = (precip.element(c) as unknown as F).toVar();
      const rh = q.div(exp(T.mul(CLIMATE.CAP_K)).mul(CLIMATE.CAP0));
      cov.addAssign(min(smoothstep(ATMO.RH0, ATMO.RH1, rh).mul(0.8).add(smoothstep(ATMO.PRECIP_CLOUD, ATMO.STORM0, p).mul(0.7)), 1));
      pr.addAssign(p);
      tp.addAssign(T);
      gmax.assign(max(gmax, (surfY.element(c) as unknown as F).add(max(water.element(c) as unknown as F, 0))));
    }
    const n = 1 / (STEP * STEP);
    summary.element(instanceIndex).assign(vec4(cov.mul(n), pr.mul(n), tp.mul(n), gmax));
  })().compute(S * S);

  /** Bilinear summary sample at world x,z, wrapping on the torus (V1). */
  const sampleSummary = (x: F, z: F): V4 => {
    // summary cell (a,b) centres on column 4a + 1.5
    const su = x.add(HALF).div(CELL).sub(0.5).sub((STEP - 1) / 2).div(STEP);
    const sv = z.add(HALF).div(CELL).sub(0.5).sub((STEP - 1) / 2).div(STEP);
    const u0 = floor(su), v0 = floor(sv);
    const fu = su.sub(u0), fv = sv.sub(v0);
    const at = (a: F, b: F) => summary.element(
      uint(int(a).add(int(S * 4)).mod(int(S)).add(int(b).add(int(S * 4)).mod(int(S)).mul(int(S))))) as unknown as V4;
    const r0 = mix(at(u0, v0), at(u0.add(1), v0), fu);
    const r1 = mix(at(u0, v0.add(1)), at(u0.add(1), v0.add(1)), fu);
    return mix(r0, r1, fv) as V4;
  };
  const worldUV = (x: F, z: F) => vec2(x.add(HALF).div(BLOCK_SIZE), z.add(HALF).div(BLOCK_SIZE));
  const weatherAt = (x: F, z: F): V4 => texture(weather, worldUV(x, z)).level(lvl0) as unknown as V4;

  // ---- 2. weather map ----
  const cell = STEP * CELL; // one summary cell in world units
  const taps: [number, number, number][] = [];
  let wsum = 0;
  for (let dz = -3; dz <= 3; dz++) for (let dx = -3; dx <= 3; dx++) {
    const w = Math.exp(-(dx * dx + dz * dz) / (2 * ATMO.BLUR_SIGMA ** 2));
    taps.push([dx, dz, w]); wsum += w;
  }
  const covNoise = (p: V2, slice: number): F => {
    // domain warp: a second, larger noise bends the coverage field so cells are organic
    const wv = texture3D(noise, vec3(p.div(ATMO.COV_TILE * 2), evolve.mul(0.5).add(slice + 0.29))).level(lvl0);
    const q = p.add(vec2(wv.g, wv.b).sub(0.5).mul(ATMO.WARP * ATMO.COV_TILE * 2));
    const n1 = texture3D(noise, vec3(q.div(ATMO.COV_TILE), evolve.add(slice))).level(lvl0);
    const n2 = texture3D(noise, vec3(q.div(ATMO.COV_TILE / 2), evolve.mul(2).add(slice + 0.5))).level(lvl0);
    // the Perlin-Worley channel spans ~0.57..0.8 (10-95 %): stretch it to 0..1 so the coverage threshold means what it says
    return saturate(n1.r.mul(0.6).add(n2.r.mul(0.4)).sub(0.6).div(0.18)) as F;
  };
  const weatherK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(W * W)), () => { Return(); });
    const tx = instanceIndex.mod(uint(W)), ty = instanceIndex.div(uint(W));
    const x = float(tx).add(0.5).div(W).mul(BLOCK_SIZE).sub(HALF).toVar();
    const z = float(ty).add(0.5).div(W).mul(BLOCK_SIZE).sub(HALF).toVar();
    const bl = vec4(0).toVar();
    const gmax = float(-1e9).toVar();
    for (const [dx, dz, w] of taps) {
      const s = sampleSummary(x.add(dx * cell), z.add(dz * cell)).toVar();
      bl.addAssign(s.mul(w / wsum));
      if (Math.abs(dx) <= 1 && Math.abs(dz) <= 1) gmax.assign(max(gmax, s.w));
    }
    const fl = bandFlow(z);
    const p = vec2(x, z);
    const cn = mix(covNoise(p.sub(fl.o0), 0.13), covNoise(p.sub(fl.o1), 0.13), fl.w1);
    // near the cut faces the coverage itself tapers: clouds shrink away organically instead of being sliced
    const edgeW = smoothstep(0.05, 0.4, float(HALF).sub(max(abs(x), abs(z))));
    const covT = saturate(bl.x.mul(u.coverage)).mul(edgeW);
    // coverage: only noise peaks above 1 - cover survive → clumpy cells with clear gaps; soft rim
    const cov = smoothstep(float(1).sub(covT), float(1.45).sub(covT), cn).mul(step(0.01, covT));
    const storm = smoothstep(ATMO.STORM0, ATMO.STORM1, bl.y).mul(smoothstep(0.25, 0.7, cov));
    const stratus = smoothstep(8, -4, bl.z).mul(float(1).sub(storm));
    const base = tWorldY(max(gmax.add(ATMO.BASE_CLEAR), atmoSeaLevel.add(ATMO.BASE_ABOVE_SEA)));
    textureStore(weather, uvec2(tx, ty), vec4(cov, storm, stratus, base)).toWriteOnly();
  })().compute(W * W);

  const cellsK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NC * NC)), () => { Return(); });
    const x = float(instanceIndex.mod(uint(NC))).add(0.5).div(NC).mul(BLOCK_SIZE).sub(HALF);
    const z = float(instanceIndex.div(uint(NC))).add(0.5).div(NC).mul(BLOCK_SIZE).sub(HALF);
    const w = weatherAt(x, z);
    cells.element(instanceIndex).assign(vec4(w.x, w.y, w.w, tWorldY(sampleSummary(x, z).w)));
  })().compute(NC * NC);

  // ---- 3. density volume ----
  const vox = (ix: THREE.Node<'uint'>, iy: THREE.Node<'uint'>, iz: THREE.Node<'uint'>, nx: number, ny: number) => vec3(
    float(ix).add(0.5).div(nx).mul(BLOCK_SIZE).sub(HALF),
    mix(u.yLo, u.yHi, float(iy).add(0.5).div(ny)),
    float(iz).add(0.5).div(nx).mul(BLOCK_SIZE).sub(HALF));
  // ---- volcanic plumes in the volume: bent column + umbrella spreading downwind (CPU-fed, ≤ PLUMES) ----
  const NP = ATMO.PLUMES;
  const plA = uniformArray(Array.from({ length: NP }, () => new THREE.Vector4()), 'vec4'); // vent x, y, z, strength
  const plB = uniformArray(Array.from({ length: NP }, () => new THREE.Vector4()), 'vec4'); // wind dir x, z, drift length, umbrella y
  const plumeShape = (p: V3, billow: F): F => {
    let acc: F = float(0);
    for (let i = 0; i < NP; i++) {
      const A = plA.element(i) as unknown as V4, B = plB.element(i) as unknown as V4;
      const rel = p.sub(A.xyz);
      const along = rel.x.mul(B.x).add(rel.z.mul(B.y)), cross = rel.x.mul(B.y).sub(rel.z.mul(B.x));
      const hh = max(B.w.sub(A.y), 0.05);
      const hN = saturate(rel.y.div(hh));
      // column: rises from the vent, leaning downwind ever more as it climbs, widening with height
      const rC = hN.mul(0.06).add(0.022);
      const dC = vec2(along.sub(B.z.mul(0.22).mul(hN.mul(hN))), cross).length();
      const col = smoothstep(rC, rC.mul(0.3), dC).mul(step(0, rel.y)).mul(smoothstep(1.08, 0.9, rel.y.div(hh)));
      // umbrella: a lens at neutral buoyancy, widening and thinning downwind, a little upwind spread
      const aN = saturate(along.add(0.06).div(max(B.z, 0.1)));
      const wdt = aN.mul(0.3).add(0.09), thk = aN.mul(0.04).add(0.08);
      const yc = B.w.add(aN.mul(0.03));
      // billow noise heaves the umbrella's top and base: cauliflower lumps, not a lid
      const yb = p.y.sub(yc).sub(billow.sub(0.5).mul(thk).mul(1.8));
      const umb = smoothstep(wdt, wdt.mul(0.2), abs(cross)).mul(smoothstep(thk, thk.mul(0.1), abs(yb)))
        .mul(smoothstep(-0.1, 0.02, along)).mul(float(1).sub(smoothstep(0.55, 1.0, aN))).mul(mix(float(1), float(0.35), aN));
      // mushroom head where the column meets the umbrella
      const head = vec3(along.sub(B.z.mul(0.22)), p.y.sub(B.w).mul(1.4), cross).length();
      const cap = smoothstep(A.w.mul(0.07).add(0.06), 0.02, head.sub(billow.sub(0.5).mul(0.05)));
      acc = max(acc, max(max(col, umb), cap).mul(min(A.w, 1.2))) as F;
    }
    return acc;
  };
  const densityK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(VX * VY * VX)), () => { Return(); });
    const ix = instanceIndex.mod(uint(VX)), iz = instanceIndex.div(uint(VX)).mod(uint(VX)), iy = instanceIndex.div(uint(VX * VX));
    const p = vox(ix, iy, iz, VX, VY).toVar();
    const w = weatherAt(p.x, p.z).toVar();
    const cov = w.x, storm = w.y, strat = w.z, base = w.w;
    // anvil source: storm tops spread downwind at the upper wind (sample the storm field upwind)
    const up = tWind(p.z, float(0.95));
    const ws = weatherAt(p.x.sub(up.x.mul(2.2)), p.z.sub(up.y.mul(2.2))).toVar();
    const cum = float(0).toVar(), cir = float(0).toVar(), ash = float(0).toVar();
    const ashShape = plumeShape(p, float(0.5)).toVar();
    // most of the slab is clear air: skip the noise there (early out keeps the pass cheap)
    If(cov.greaterThan(0.002).or(ws.y.greaterThan(0.01)).or(ashShape.greaterThan(0.0005)), () => {
      // base shape noise in the drifting frame (second band only where two bands meet), Perlin-Worley
      const fl = bandFlow(p.z);
      const pt = p.xz.sub(tilt(p.z, p.y)).toVar();
      const nA = texture3D(noise, vec3(pt.sub(fl.o0), p.y).xzy.div(baseTile)).level(lvl0);
      const nA2 = texture3D(noise, vec3(pt.sub(fl.o0), p.y).xzy.div(baseTile / 2).add(0.37)).level(lvl0);
      const bn = nA.r.mul(0.62).add(nA2.r.mul(0.38)).toVar();
      If(fl.w1.greaterThan(0.001), () => {
        const nB = texture3D(noise, vec3(pt.sub(fl.o1), p.y).xzy.div(baseTile)).level(lvl0);
        const nB2 = texture3D(noise, vec3(pt.sub(fl.o1), p.y).xzy.div(baseTile / 2).add(0.37)).level(lvl0);
        bn.assign(mix(bn, nB.r.mul(0.62).add(nB2.r.mul(0.38)), fl.w1));
      });
      // heaps: each coverage cell becomes a dome — tallest where coverage peaks, thin at its rim, the top
      // surface pushed in and out by the 3D billow noise (cauliflower bulges); flat base at the
      // condensation level. Fair cumulus / flat stratus / towering cumulonimbus by type.
      const Hmax = mix(mix(float(0.3), float(0.07), strat), float(0.78), storm);
      const top = Hmax.mul(pow(cov, 0.6)).mul(mix(float(0.5), float(1.35), bn)).mul(mix(float(1), float(0.7), strat));
      // bases undulate per cloud (thin cells float higher, big ones sag) and are ragged, not a ruler line
      const hb = p.y.sub(base).sub(bn.sub(0.5).mul(0.1)).sub(float(0.5).sub(cov).mul(0.06));
      const hl = hb.div(max(top, 1e-3));
      const shell = smoothstep(1.0, 0.5, hl).mul(smoothstep(0.0, top.mul(0.25).add(0.015), hb)).mul(step(0.002, cov));
      // Nubis-style carve: the shell only decides how much of the billow noise becomes cloud → lobes
      const body = saturate(bn.mul(0.85).add(0.15).sub(float(1).sub(shell)).div(max(shell, 0.08)).mul(1.35));
      const aTop = ws.w.add(float(0.78).mul(ws.y));
      const anvil = ws.y.mul(smoothstep(0.1, 0.5, ws.x)).mul(smoothstep(aTop.sub(0.16), aTop.sub(0.1), p.y)).mul(smoothstep(aTop.add(0.02), aTop.sub(0.04), p.y))
        .mul(mix(float(0.5), float(1.1), bn));
      cum.assign(max(body, anvil));
      // volcanic ash: carved by a finer, slowly churning billow noise (plume-sized lumps: the cloud-scale
      // noise is wider than the umbrella is thick, which left it a smooth pill / sausage)
      If(ashShape.greaterThan(0.0005), () => {
        const qa = p.add(vec3(0, u.time.mul(-0.006), 0)).xzy;
        const na = texture3D(noise, qa.div(ATMO.DETAIL_TILE)).level(lvl0), na2 = texture3D(noise, qa.div(ATMO.DETAIL_TILE * 0.5).add(0.61)).level(lvl0);
        const ba = na.r.mul(0.6).add(na2.g.mul(0.4));
        const shp = plumeShape(p, ba);
        ash.assign(saturate(shp.mul(1.3).sub(float(1).sub(ba).mul(1.05)).mul(1.9)).mul(min(shp.mul(2), 1)));
      });
    });
    // cirrus: sparse streaks near the slab top, stretched along the jet, uniform drift (no shear)
    const cy = saturate(float(1).sub(abs(p.y.sub(u.yHi.sub(0.1)).div(0.045)))).toVar();
    If(cy.greaterThan(0), () => {
      const cx = p.x.sub(u.time.mul(cirrusV));
      const cn = texture3D(noise, vec3(cx.div(ATMO.COV_TILE), float(0.37), p.z.div(baseTile * 0.45))).level(lvl0);
      const cmask = smoothstep(1 - ATMO.CIRRUS_COVER, 1 - ATMO.CIRRUS_COVER + 0.25, cn.r.mul(0.7).add(cn.b.mul(0.3)));
      cir.assign(cmask.mul(cy).mul(ATMO.CIRRUS_DENS));
    });
    // no visible volume: everything fades to 0 at the slab faces
    const edge = smoothstep(0, 0.035, float(HALF).sub(max(abs(p.x), abs(p.z))))
      .mul(smoothstep(u.yHi, u.yHi.sub(0.06), p.y)).mul(smoothstep(u.yLo, u.yLo.add(0.02), p.y));
    const tot = max(max(cum, cir), ash);
    const ashF = ash.div(max(tot, 1e-4));
    textureStore(density, uvec3(ix, iy, iz), vec4(tot.mul(edge), storm.mul(step(cir, cum)).mul(float(1).sub(ashF)), ashF, 1)).toWriteOnly();
  })().compute(VX * VY * VX);

  const uvw = (p: V3) => vec3(p.x.add(HALF).div(BLOCK_SIZE), p.y.sub(u.yLo).div(u.yHi.sub(u.yLo)), p.z.add(HALF).div(BLOCK_SIZE));

  // ---- 5. shadow texture ----
  const R = ATMO.SHADOW_RES;
  const shadowK = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(R * R)), () => { Return(); });
    const tx = instanceIndex.mod(uint(R)), ty = instanceIndex.div(uint(R));
    const key = cloudShadowU.keyDir;
    const x = float(tx).add(0.5).div(R).mul(BLOCK_SIZE).sub(HALF);
    const z = float(ty).add(0.5).div(R).mul(BLOCK_SIZE).sub(HALF);
    const len = min(u.yHi.sub(u.yLo).div(max(key.y, 0.08)), 3);
    const od = float(0).toVar();
    const NS = 12;
    for (let k = 0; k < NS; k++) {
      const p = vec3(x, u.yLo, z).add(key.mul(len.mul((k + 0.5) / NS)));
      od.addAssign(texture3D(density, uvw(p)).level(lvl0).r.mul(tCutSoft(p)));
    }
    const tau = od.mul(len.div(NS)).mul(ATMO.SIGMA * 0.5);
    const lightF = float(1).sub(float(1).sub(exp(tau.negate())).mul(ATMO.SHADOW_MAX));
    textureStore(cloudShadowTexture, uvec2(tx, ty), vec4(lightF, saturate(tau.mul(0.3)), 0, 1)).toWriteOnly();
  })().compute(R * R);

  // ---- 6. raymarch at reduced resolution (own camera uniforms; runs before the scene pass) ----
  const cam = {
    pos: uniform(new THREE.Vector3()),
    invProj: uniform(new THREE.Matrix4()),
    world: uniform(new THREE.Matrix4()),
    view: uniform(new THREE.Matrix4()),
    prevViewProj: uniform(new THREE.Matrix4()),
    blend: uniform(1),
    frame: uniform(0),
    res: uniform(new THREE.Vector2(1, 1)),
  };
  const steps = uniform(opts.highQuality ? ATMO.STEPS_HIGH : ATMO.STEPS_LOW);
  const stepIn = uniform(opts.highQuality ? ATMO.STEP_IN_HIGH : ATMO.STEP_IN_LOW);
  const mkRT = () => {
    const rt = new THREE.RenderTarget(1, 1, { count: 2, type: THREE.HalfFloatType, depthBuffer: false });
    for (const t of rt.textures) { t.minFilter = t.magFilter = THREE.LinearFilter; t.generateMipmaps = false; }
    rt.textures[0]!.name = 'cloudColor'; rt.textures[1]!.name = 'cloudFront';
    return rt;
  };
  const rts = [mkRT(), mkRT()];
  const histColor = texture(rts[1]!.textures[0]!), histFront = texture(rts[1]!.textures[1]!);
  const hg = (c: F, g: number) => float(1 - g * g).div(pow(float(1 + g * g).sub(c.mul(2 * g)), 1.5));
  const colField = property('vec4'), frontField = property('vec4');
  const marchFn = Fn(() => {
    const suv = uv();
    const ndc = vec4(suv.x.mul(2).sub(1), float(1).sub(suv.y.mul(2)), 1, 1);
    const pv = cam.invProj.mul(ndc);
    const ro = cam.pos;
    const rd = normalize(cam.world.mul(vec4(pv.xyz.div(pv.w), 0)).xyz).toVar();
    const bmin = vec3(-HALF, u.yLo, -HALF), bmax = vec3(HALF, u.yHi, HALF);
    const inv = vec3(1).div(rd.add(sign(rd).mul(1e-6)).add(vec3(1e-7)));
    const ta = bmin.sub(ro).mul(inv), tb = bmax.sub(ro).mul(inv);
    const tn = min(ta, tb), tf = max(ta, tb);
    const t0 = max(max(max(tn.x, tn.y), tn.z), float(ATMO.NEAR_FADE0)).toVar();
    const t1 = min(min(tf.x, tf.y), tf.z).toVar();
    const T = float(1).toVar();
    const acc = vec3(0).toVar();
    const front = t0.add(t1).mul(0.5).toVar();
    const hit = float(0).toVar();
    If(t1.greaterThan(t0), () => {
      // fine fixed steps inside cloud (the density has sharp domes; coarse steps slice them into contour
      // bands), long strides through empty air; the iteration budget bounds the cost
      const dt = float(stepIn).toVar();
      const dtEmpty = max(t1.sub(t0).div(steps.mul(0.6)), dt.mul(2.5)).toVar();
      // blue-ish per-pixel jitter, decorrelated per frame (accumulated below and by TRAA)
      const px = suv.mul(cam.res);
      const ign = fract(float(52.9829189).mul(fract(px.x.mul(0.06711056).add(px.y.mul(0.00583715)))));
      const jit = fract(ign.add(cam.frame.mul(0.618034)));
      const key = cloudShadowU.keyDir;
      const cosT = dot(rd, key);
      // dual-lobe phase: strong forward lobe = silver lining toward the sun, soft back lobe
      const phase = mix(hg(cosT, 0.7), hg(cosT, -0.2), 0.4).mul(0.9).add(0.25);
      const keyCol = tKeyColor();
      const skyTop = skyU.zenith, skyLow = mix(skyU.horizon, skyU.zenith, 0.35);
      const t = t0.add(jit.mul(dtEmpty)).toVar();
      Loop({ start: int(0), end: int(steps), type: 'int', condition: '<' }, () => {
        const p = ro.add(rd.mul(t)).toVar();
        const c = uvw(p);
        const v = texture3D(density, c).level(lvl0).toVar();
        const stepLen = dt.toVar();
        If(v.r.greaterThan(0.004), () => {
          // detail erosion (2 Worley octaves) in the same drifting frame as the base shape
          const fl = bandFlow(p.z);
          const qd = vec3(p.xz.sub(fl.o0), p.y).xzy.toVar();
          const da = texture3D(noise, qd.div(ATMO.DETAIL_TILE)).level(lvl0);
          // curl-ish distortion: the coarse octave bends the fine one → wispy, swirled edges
          const qw = qd.add(vec3(da.g.sub(0.5), da.b.sub(0.5), da.r.sub(0.5)).mul(0.03)).add(vec3(0, u.time.mul(0.004), 0));
          const dh = texture3D(noise, qw.div(ATMO.DETAIL_TILE * 0.5)).level(lvl0);
          const fbm = da.g.mul(0.5).add(da.b.mul(0.28)).add(dh.b.mul(0.17)).add(dh.a.mul(0.05)).toVar();
          If(fl.w1.greaterThan(0.001), () => {
            const db = texture3D(noise, vec3(p.xz.sub(fl.o1), p.y).xzy.div(ATMO.DETAIL_TILE)).level(lvl0);
            fbm.assign(mix(fbm, db.g.mul(0.5).add(db.b.mul(0.28)).add(dh.b.mul(0.17)).add(dh.a.mul(0.05)), fl.w1));
          });
          // erode wispy edges, dense cores barely (cauliflower edges, soft interiors)
          // near the cut faces the detail noise eats the cloud (ragged edge, never a planar slice)
          const nearEdge = float(1).sub(smoothstep(0.03, 0.16, float(HALF).sub(max(abs(p.x), abs(p.z)))));
          // (ash is eroded through its dense core as well: lumpy plume, never a smooth solid lens)
          const ero = float(1).sub(fbm).mul(0.62).mul(mix(float(1).sub(smoothstep(0.12, 0.7, v.r)).mul(0.78).add(0.22), float(0.85), v.b)).add(nearEdge.mul(0.8));
          const d = saturate(v.r.sub(ero).div(max(float(1).sub(ero), 0.05)))
            .mul(smoothstep(ATMO.NEAR_FADE0, ATMO.NEAR_FADE1, t)).mul(tCutSoft(p));
          const sigma = d.mul(ATMO.SIGMA).mul(v.b.mul(0.4).add(1));
          // light march toward the key light (Beer), plus sky visibility straight up (darker bases)
          const od = float(0).toVar(), odUp = float(0).toVar();
          let tt = 0;
          for (const ds of [0.022, 0.045, 0.08, 0.13]) {
            tt += ds;
            od.addAssign(texture3D(density, uvw(p.add(key.mul(tt - ds * 0.5)))).level(lvl0).r.mul(ds));
          }
          od.addAssign(v.r.mul(0.01));
          odUp.addAssign(texture3D(density, uvw(p.add(vec3(0, 0.06, 0)))).level(lvl0).r.mul(0.13));
          const sOd = od.mul(ATMO.SIGMA);
          const L = vec3(exp(sOd.negate()), exp(odUp.mul(ATMO.SIGMA * 0.35).negate()), exp(sOd.mul(-0.22)));
          const storm = v.g;
          const powder = float(1).sub(exp(sigma.mul(-0.08))).mul(0.6).add(0.4);
          // warm key light with Beer + powder, a little multiple scattering in the shade
          // lobe shading: density rising toward the light = a crease in shadow, falling = a sunlit bulge
          const dk = texture3D(density, uvw(p.add(key.mul(0.018)))).level(lvl0).r;
          const lobe = saturate(v.r.sub(dk).mul(7).add(0.55));
          const cavity = mix(float(0.72), float(1.06), fbm);
          const sun = keyCol.mul(L.x.mul(0.95).add(L.z.mul(0.12)).mul(phase).mul(powder).mul(lobe.mul(0.8).add(0.35)).mul(cavity).mul(0.8));
          // sky ambient: bright tops, cool lavender bases
          const amb = mix(skyLow.mul(vec3(0.5, 0.56, 0.95)), skyTop, L.y).mul(mix(float(0.22), float(0.7), L.y)).mul(0.5);
          // ash: grey-brown, a touch denser; storms: dark grey
          const albedo = mix(vec3(mix(float(1), float(0.4), storm)), vec3(0.34, 0.3, 0.27), v.b);
          const flash = vec3(0.75, 0.8, 1.0).mul(u.flash.mul(exp(length(p.sub(u.flashPos)).mul(-7))).mul(20));
          const col = sun.add(amb).mul(albedo).add(flash);
          const Ts = exp(sigma.mul(dt).negate());
          acc.addAssign(col.mul(T).mul(float(1).sub(Ts)));
          T.mulAssign(Ts);
          If(hit.lessThan(0.5).and(T.lessThan(0.9)), () => { front.assign(t); hit.assign(1); });
        }).Else(() => { stepLen.assign(dtEmpty); }); // empty space: stride
        t.addAssign(stepLen);
        If(T.lessThan(0.02).or(t.greaterThan(t1)), () => { Break(); });
      });
    });
    const cur = vec4(acc, float(1).sub(T));
    // temporal accumulation: reproject the history at the cloud's front depth
    const wp = ro.add(rd.mul(front));
    const pc = cam.prevViewProj.mul(vec4(wp, 1));
    const puv = vec2(pc.x.div(pc.w).mul(0.5).add(0.5), float(0.5).sub(pc.y.div(pc.w).mul(0.5)));
    const valid = step(0, puv.x).mul(step(puv.x, 1)).mul(step(0, puv.y)).mul(step(puv.y, 1)).mul(step(0, pc.w));
    const hc = histColor.sample(puv);
    const hf = histFront.sample(puv).x;
    // reject history on disocclusion (front depth jumped) as well as off-screen
    const agree = step(abs(hf.sub(front)), front.mul(0.12).add(0.02));
    const k = mix(float(1), mix(float(0.7), cam.blend, agree), valid);
    colField.assign(mix(hc, cur, k));
    frontField.assign(vec4(mix(hf, front, max(k, float(1).sub(hit.mul(0.8)))), 0, 0, 1));
    return vec4(0);
  });
  const marchMat = new THREE.NodeMaterial();
  marchMat.name = 'cloudMarch';
  marchMat.colorNode = marchFn();
  marchMat.outputNode = outputStruct(colField, frontField);
  const quad = new THREE.QuadMesh(marchMat);
  quad.name = 'cloudMarch';

  // ---- 7. depth-aware composite in the scene pass (full res, cheap) ----
  const curColor = texture(rts[0]!.textures[0]!), curFront = texture(rts[0]!.textures[1]!);
  const mat = new THREE.MeshBasicNodeMaterial();
  mat.name = 'clouds';
  mat.transparent = true;
  mat.depthWrite = false;
  mat.depthTest = false;
  mat.side = THREE.BackSide;
  mat.fog = false;
  fxMRT(mat, opts.renderer);
  mat.alphaTest = 0.003;
  const composite = Fn(() => {
    const c = curColor.sample(screenUV).toVar();
    const fr = curFront.sample(screenUV).x;
    const rd = normalize(positionWorld.sub(cameraPosition));
    const rdV = cameraViewMatrix.mul(vec4(rd, 0)).xyz;
    const tScene = sceneViewZ(screenUV).div(min(rdV.z, -1e-4));
    // terrain / plinth in front of the cloud's front hides it (soft over a short range)
    const vis = smoothstep(fr.sub(0.04), fr.add(0.01), tScene);
    const a = c.a.mul(vis).mul(u.fade);
    return vec4(c.rgb.div(max(c.a, 1e-4)), a);
  })();
  mat.colorNode = vec4(composite.rgb, 1);
  mat.opacityNode = composite.a;

  const geo = new THREE.BoxGeometry(BLOCK_SIZE, 1, BLOCK_SIZE);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'clouds';
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.renderOrder = 6; // transparents ≥ 4 (shared.ts); plumes and rain draw after it (7) with clouds.occlusion

  const size = new THREE.Vector2();
  const prevVP = new THREE.Matrix4(), vp = new THREE.Matrix4();
  const lastPos = new THREE.Vector3(), lastDir = new THREE.Vector3(), curDir = new THREE.Vector3();
  let havePrev = false;
  const rstate = {} as ReturnType<typeof THREE.RendererUtils.resetRendererState>;
  function renderMarch(renderer: THREE.WebGPURenderer, camera: THREE.Camera): void {
    renderer.getDrawingBufferSize(size);
    const scale = hq ? ATMO.RES_HIGH : ATMO.RES_LOW;
    const w = Math.max(1, Math.round(size.x * scale)), h = Math.max(1, Math.round(size.y * scale));
    if (rts[0]!.width !== w || rts[0]!.height !== h) { rts[0]!.setSize(w, h); rts[1]!.setSize(w, h); havePrev = false; }
    cam.res.value.set(w, h);
    // ping-pong: write rts[0], read history from rts[1]
    rts.reverse();
    curColor.value = rts[0]!.textures[0]!; curFront.value = rts[0]!.textures[1]!;
    histColor.value = rts[1]!.textures[0]!; histFront.value = rts[1]!.textures[1]!;
    camera.updateMatrixWorld();
    cam.pos.value.setFromMatrixPosition(camera.matrixWorld);
    cam.invProj.value.copy(camera.projectionMatrixInverse);
    cam.world.value.copy(camera.matrixWorld);
    vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // camera cut (big jump between frames): start the history over
    if (havePrev && (lastPos.distanceTo(cam.pos.value) > 0.25 || lastDir.dot(curDir.set(0, 0, -1).transformDirection(camera.matrixWorld)) < 0.995)) havePrev = false;
    lastPos.copy(cam.pos.value); lastDir.set(0, 0, -1).transformDirection(camera.matrixWorld);
    cam.prevViewProj.value.copy(havePrev ? prevVP : vp);
    cam.blend.value = havePrev ? ATMO.TEMPORAL : 1;
    cam.frame.value = (cam.frame.value + 1) % 64;
    prevVP.copy(vp);
    havePrev = true;
    THREE.RendererUtils.resetRendererState(renderer, rstate);
    renderer.setRenderTarget(rts[0]!);
    quad.render(renderer);
    THREE.RendererUtils.restoreRendererState(renderer, rstate);
  }

  let hq = opts.highQuality;
  let frame = 0;
  return {
    object: mesh, summary, cells, weather, density, sampleSummary, weatherAt,
    occlusion: (c: V3): F => {
      const col = curColor.sample(screenUV), fr = curFront.sample(screenUV).x;
      const behind = smoothstep(fr.sub(0.03), fr.add(0.03), length(c.sub(cameraPosition)));
      return float(1).sub(behind.mul(col.a).mul(u.fade)) as F;
    },
    uniforms: u,
    compute(renderer, camera) {
      mesh.position.y = (u.yLo.value + u.yHi.value) / 2;
      mesh.scale.y = u.yHi.value - u.yLo.value;
      // low tier: the volume passes are time-sliced over three frames (clouds drift < 1 voxel meanwhile)
      frame++;
      const slot = frame % 3;
      // high tier: weather + density on even frames, shadow on odd ones
      if (hq ? frame % 2 === 0 : slot === 0) { renderer.compute(summaryK); renderer.compute(weatherK); }
      if (hq ? frame % 2 === 0 : slot === 1) renderer.compute(densityK);
      if (hq ? frame % 2 === 1 : slot === 2) renderer.compute(shadowK);
      if (frame % 15 === 1) renderer.compute(cellsK);
      cloudShadowU.on.value = u.fade.value;
      renderMarch(renderer, camera);
    },
    setPlumes(list) {
      for (let i = 0; i < NP; i++) {
        const q = list[i];
        (plA.array[i] as THREE.Vector4).set(q?.x ?? 0, q?.y ?? -10, q?.z ?? 0, q?.s ?? 0);
        (plB.array[i] as THREE.Vector4).set(q?.dx ?? 1, q?.dz ?? 0, q?.len ?? 0.5, q?.top ?? -9);
      }
    },
    setHighQuality(v) {
      hq = v;
      steps.value = v ? ATMO.STEPS_HIGH : ATMO.STEPS_LOW;
      stepIn.value = v ? ATMO.STEP_IN_HIGH : ATMO.STEP_IN_LOW;
    },
    dispose() {
      cloudShadowU.on.value = 0;
      geo.dispose(); mat.dispose(); marchMat.dispose(); quad.dispose(); for (const rt of rts) rt.dispose();
      density.dispose(); weather.dispose();
    },
  };
}
