// Tsunami FX: an expanding ring wave on the water surface. A per-column arrival-time field T (real s after
// the quake) is grown by in-place eikonal relaxation (Godunov upwind update, a few sweeps per frame, well
// ahead of the visible front) with shallow-water speed c = C0 · sqrt(depth / DREF): the front slows over
// shelves, refracts around islands and bunches up at coasts; low coastal land within the run-up height
// is reached at a slow run-up speed. The wave itself is a separate transparent sheet over the water
// level (columnSampler, same swell as the water) raised by the crest profile p(t − T): a leading crest
// with a foam line, trough, a smaller second crest; Green's-law shoaling makes it pile up in the
// shallows, where it breaks into foam; on land it surges up as a white sheet, recedes and leaves dark
// wet sand that dries off. Reads display/sim fields only (V15). Storage buffers: kernels 2, sheet 3 (V23).
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, int, uint, vec2, vec3, vec4, uniform, instanceIndex, instancedArray, exp, mix, smoothstep, saturate,
  max, min, floor, sqrt, pow, step, abs, length, normalize, texture, varying, positionGeometry, positionWorld,
  transformNormalToView,
} from 'three/tsl';
import type { GpuFields, StorageNode } from '../core/gpu';
import { NCOL, CELL, VOXEL_H } from '../sim/layout';
import { tColIdx, tColXZ } from '../sim/tslLayout';
import { columnSampler, displayBuffers, tWorldToCell, tWorldY, vertEx } from '../render/space';
import { createColumnGrid } from '../render/terrain';
import { swell, WATER_SHALLOW, WATER_SCATTER } from '../render/water';
import { lookTextures } from '../render/textures';
import { TSUNAMI } from './fxModel';
import { tTorusD, type F } from './fxTsl';

type V4 = THREE.Node<'vec4'>;
type I = THREE.Node<'int'>;

/** Spreading reference radius (cells): amplitude ∝ sqrt(R_SPREAD / r) beyond it. */
const R_SPREAD = 9;
/** Crest profile: (centre s, width s, weight) — leading crest, trough, second crest. */
const PROFILE: [number, number, number][] = [[0.38, 0.24, 1], [1.1, 0.36, -0.55], [1.9, 0.32, 0.28]];

export interface Tsunami {
  sheet: THREE.Mesh;
  /** Arrival time per column (s after the quake; ≥ TSUNAMI.INF/2 = not reached). */
  T: StorageNode<'float'>;
  readonly active: boolean;
  /** Seconds since the current tsunami started (−1 when idle). */
  readonly age: number;
  /** x, z epicentre (cells), a0 deep-water amplitude (voxel layers). */
  start(renderer: THREE.WebGPURenderer, x: number, z: number, a0: number, seaLevel: number): void;
  update(renderer: THREE.WebGPURenderer, dt: number, ambTime: number): boolean;
  dispose(): void;
}

export function createTsunami(fields: GpuFields): Tsunami {
  const S = columnSampler(fields);
  const { cols } = displayBuffers(fields);
  const T = instancedArray(NCOL, 'float') as unknown as StorageNode<'float'>;
  T.setName('tsunamiT');
  (T.value.array as Float32Array).fill(TSUNAMI.INF);
  const u = {
    epi: uniform(new THREE.Vector4()), // (ex, ez, source radius cells, a0 layers)
    time: uniform(0), fade: uniform(0), sea: uniform(76), amb: uniform(0),
  };
  const speedAt = (depth: F): F => sqrt(max(depth, TSUNAMI.DMIN).div(TSUNAMI.DREF)).mul(TSUNAMI.C0);
  const spreadAt = (dx: F, dz: F): F => sqrt(float(R_SPREAD).div(max(length(vec2(dx, dz)), R_SPREAD)));

  // ---- arrival-time field ----
  const initK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(i);
    const d = length(vec2(tTorusD(float(x), u.epi.x), tTorusD(float(z), u.epi.y)));
    const w = (cols.element(i) as unknown as V4).z;
    const src = step(d, u.epi.z).mul(step(0.05, w));
    T.element(i).assign(mix(float(TSUNAMI.INF), d.div(speedAt(w)), src));
  })().compute(NCOL);

  const relaxK = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(i);
    const c = (cols.element(i) as unknown as V4).toVar();
    const wet = step(0.05, c.z);
    // land: run-up up to RUNUP_K × the local deep-water amplitude above sea level, slowing as it climbs
    const alt = c.y.sub(u.sea);
    const spread = spreadAt(tTorusD(float(x), u.epi.x), tTorusD(float(z), u.epi.y));
    const runup = spread.mul(u.epi.w).mul(TSUNAMI.RUNUP_K);
    const climb = saturate(float(1).sub(alt.div(max(runup, 1e-3))));
    const reach = max(wet, step(alt, runup).mul(step(0.01, climb)));
    If(reach.greaterThan(0.5), () => {
      const sp = mix(max(climb, 0.15).mul(TSUNAMI.C_RUN), speedAt(c.z), wet);
      const h = float(1).div(sp); // seconds per cell
      const at = (dx: number, dz: number) => T.element(tColIdx((x as I).add(dx), (z as I).add(dz))) as unknown as F;
      const a = min(at(-1, 0), at(1, 0)), b = min(at(0, -1), at(0, 1));
      const lo = min(a, b), dab = abs(a.sub(b));
      // Godunov upwind eikonal update: one-sided if the other axis is far behind, else the 2D quadratic
      const two = a.add(b).add(sqrt(max(h.mul(h).mul(2).sub(dab.mul(dab)), 0))).mul(0.5);
      const tn = mix(two, lo.add(h), step(h, dab));
      const cur = T.element(i) as unknown as F;
      If(tn.lessThan(cur), () => { T.element(i).assign(tn); });
    });
  })().compute(NCOL);
  const sweeps = Array.from({ length: TSUNAMI.SWEEPS }, () => relaxK);

  // ---- sheet ----
  const gu = tWorldToCell(positionGeometry.x), gv = tWorldToCell(positionGeometry.z);
  // Sheet vertices sit on column centres (createColumnGrid), so T reads are exact grid reads.
  const Tat = (du: number, dv: number): F => {
    const xi = int(floor(gu.add(du + 0.5))), zi = int(floor(gv.add(dv + 0.5)));
    return min(T.element(tColIdx(xi, zi)) as unknown as F, TSUNAMI.INF) as F;
  };
  // Ground normal from the display heights of the neighbour columns (grid reads; the sheet's vertices sit on
  // column centres): 4 loads instead of 4 advected bilinear samples.
  const hsAt = (du: number, dv: number): F => (cols.element(tColIdx(int(floor(gu.add(du + 0.5))), int(floor(gv.add(dv + 0.5))))) as unknown as V4).x as F;
  const profile = (s: F) => {
    let p: F = float(0), dp: F = float(0);
    for (const [c0, w, k] of PROFILE) {
      const q = s.sub(c0).div(w);
      const g = exp(q.mul(q).negate()).mul(k);
      p = p.add(g);
      dp = dp.add(g.mul(q).mul(-2 / w));
    }
    return { p, dp };
  };
  // Vertex graph, built once and shared by positionNode and the varyings (all vertex stage).
  const vert = () => {
    const c = S.corners(gu, gv);
    const hgt = S.height(c).toVar();
    const lv = S.level(c);
    const depth = lv.level.sub(hgt).toVar();
    const t0 = Tat(0, 0).toVar();
    const s = u.time.sub(t0).toVar();
    const dx = tTorusD(gu, u.epi.x), dz = tTorusD(gv, u.epi.y);
    const spread = spreadAt(dx, dz);
    const shoal = min(pow(float(TSUNAMI.DREF).div(max(depth, 0.6)), 0.25), 2.2);
    const amp = u.epi.w.mul(spread).mul(shoal).mul(u.fade).toVar();
    const { p, dp } = profile(s);
    const hw = amp.mul(p).toVar(); // layers
    // front direction from the T gradient (one-sided next to unreached columns), slowness from depth
    const fin = (t: F) => step(t, TSUNAMI.INF * 0.5);
    const l = Tat(-1, 0), r = Tat(1, 0), d = Tat(0, -1), up = Tat(0, 1);
    const gx = mix(mix(float(0), t0.sub(l), fin(l)), mix(r.sub(t0), r.sub(l).mul(0.5), fin(l)), fin(r));
    const gz = mix(mix(float(0), t0.sub(d), fin(d)), mix(up.sub(t0), up.sub(d).mul(0.5), fin(d)), fin(up));
    const dir = normalize(vec2(gx, gz).add(vec2(1e-5, 0)));
    const slow = float(1).div(speedAt(max(depth, 0)).mul(CELL)); // s per world unit
    const kY = vertEx.mul(VOXEL_H);
    const dhdw = amp.mul(dp).mul(slow).negate().mul(kY); // world slope along the travel direction
    const nWave = normalize(vec3(dir.x.mul(dhdw).negate(), 1, dir.y.mul(dhdw).negate()));
    // land run-up: inundation depth surges up and drains back
    const alt = hgt.sub(u.sea);
    const runup = spread.mul(u.epi.w).mul(TSUNAMI.RUNUP_K).mul(u.fade);
    const env = smoothstep(0.0, 0.45, s).mul(float(1).sub(smoothstep(0.9, 3.2, s)));
    const reached = step(t0, TSUNAMI.INF * 0.5);
    const dI = max(runup.sub(alt), 0).mul(env).mul(reached).mul(0.6);
    const k = vertEx.mul(VOXEL_H);
    const hl = hsAt(-1, 0), hr = hsAt(1, 0), hd = hsAt(0, -1), hu = hsAt(0, 1);
    const nGround = normalize(vec3(hl.sub(hr).mul(k), 2 * CELL, hd.sub(hu).mul(k)));
    const isWet = step(0.05, depth);
    const sw = swell(positionGeometry.xz, max(depth, 0));
    const yW = tWorldY(lv.level.add(max(hw, 0)).add(0.15)).add(sw.x);
    const yL = tWorldY(hgt.add(dI).add(0.2));
    return {
      pos: vec3(positionGeometry.x, mix(yL, yW, isWet), positionGeometry.z),
      a: vec4(s, hw, amp, depth),
      b: vec4(mix(nGround, nWave, isWet), dI),
      c: vec4(isWet, step(alt, runup).mul(reached), reached, 0),
    };
  };
  const V = vert();
  const vA = varying(V.a, 'vTsuA'), vB = varying(V.b, 'vTsuB'), vC = varying(V.c, 'vTsuC');

  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0, transparent: true, depthWrite: false });
  mat.name = 'tsunami';
  mat.positionNode = V.pos;
  mat.alphaTest = 0.004;
  const tex = lookTextures();
  const shade = (() => {
    const s = vA.x, hw = vA.y, amp = max(vA.z, 1e-3), depth = vA.w;
    const wetCol = vC.x, flooded = vC.y, reached = vC.z;
    const n = texture(tex.detail, positionWorld.xz.mul(5.5).add(vec2(u.amb.mul(0.02), 0))).b;
    const n2 = texture(tex.detail, positionWorld.xz.mul(13).add(vec2(0, u.amb.mul(-0.03)))).g;
    const rel = hw.div(amp);
    const shallow = smoothstep(4.5, 0.8, depth);
    const nb = texture(tex.detail, positionWorld.xz.mul(1.7)).r; // broad breakup along the crest
    // water: glassy raised crest (lit turquoise face), dark trough, broken foam line on the crest top
    // that turns into a breaking white front in the shallows
    const body = smoothstep(0.06, 0.55, hw).mul(0.85);
    const trough = smoothstep(0.04, 0.5, hw.negate()).mul(0.5);
    const crestLine = smoothstep(nb.mul(0.3).add(0.42), 0.9, rel).mul(shallow.mul(0.45).add(0.55));
    const breaking = shallow.mul(smoothstep(0.2, 0.6, rel)).mul(step(s, 0.6));
    const foamW = saturate(crestLine.add(breaking).mul(n.mul(0.8).add(0.5)).sub(n2.mul(0.3)).mul(1.3)).mul(step(0.04, hw));
    const face = saturate(rel).mul(0.7).add(shallow.mul(0.3));
    const crestCol = mix(vec3(WATER_SCATTER.r, WATER_SCATTER.g, WATER_SCATTER.b).mul(2.2), vec3(WATER_SHALLOW.r, WATER_SHALLOW.g, WATER_SHALLOW.b).mul(1.15), face);
    const wCol = mix(mix(crestCol, vec3(0.005, 0.025, 0.06), step(hw, 0)), vec3(0.93, 0.96, 0.98), foamW);
    const wA = max(max(body, trough), foamW);
    // land: white surge sheet, then wet dark sand drying off
    const surge = smoothstep(0.02, 0.2, vB.w);
    const foamL = surge.mul(n.mul(0.6).add(0.6)).min(1);
    const wetSand = flooded.mul(smoothstep(0.3, 1.4, s)).mul(float(1).sub(smoothstep(5, 20, s))).mul(0.62);
    const lCol = mix(vec3(0.045, 0.036, 0.03), vec3(0.92, 0.95, 0.97), step(0.02, foamL.mul(surge)));
    const lA = max(foamL.mul(surge).mul(0.95), wetSand);
    const col = mix(lCol, wCol, wetCol);
    const a = mix(lA, wA, wetCol).mul(reached);
    return { col: vec4(col, a), foam: mix(foamL.mul(surge), foamW, wetCol) };
  })();
  mat.colorNode = vec4(shade.col.rgb, 1);
  mat.opacityNode = shade.col.a;
  mat.roughnessNode = mix(float(0.08), float(0.85), shade.foam);
  mat.emissiveNode = vec3(0.9, 0.95, 1).mul(shade.foam).mul(0.12); // foam flash stays readable in shade
  mat.normalNode = transformNormalToView(normalize(vB.xyz));
  const sheet = new THREE.Mesh(createColumnGrid(), mat);
  sheet.name = 'tsunami';
  sheet.frustumCulled = false;
  sheet.receiveShadow = true;
  sheet.renderOrder = 4; // after the water sheet (2) and its glass faces (3)
  sheet.visible = false;

  let age = -1;
  return {
    sheet, T,
    get active() { return age >= 0; },
    get age() { return age; },
    start(renderer, x, z, a0, seaLevel) {
      age = 0;
      u.epi.value.set(x, z, 3 + 2.5 * a0, a0);
      u.sea.value = seaLevel;
      u.time.value = 0; u.fade.value = 1;
      renderer.compute(initK);
      renderer.compute([...sweeps, ...sweeps, ...sweeps]);
      sheet.visible = true;
    },
    update(renderer, dt, ambTime) {
      if (age < 0) return false;
      age += dt;
      if (age > TSUNAMI.LIFE_S) { age = -1; sheet.visible = false; return false; }
      u.time.value = age;
      u.amb.value = ambTime % 3600;
      u.fade.value = Math.min(1, (TSUNAMI.LIFE_S - age) / TSUNAMI.FADE_S);
      renderer.compute(sweeps);
      return true;
    },
    dispose() { sheet.geometry.dispose(); mat.dispose(); },
  };
}
