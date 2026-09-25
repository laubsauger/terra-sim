// §T.45 flora: GPU-scattered instanced plants per biome, read-only on sim fields (V15).
//
// Slots: two fixed jittered grids over the block (lifeModel.ts). A refresh kernel (every ~2 s) reads
// biome/veg/surfY/water at each slot and sets its target species + scale; the per-slot hash is
// fixed, so plants never jump between refreshes, and density ∝ veg only adds/removes slots at the
// threshold. Each slot keeps (voxelY, prevScale, targetScale, species); the shader shows
// mix(prev, target, blend) with blend ramping 0 → 1 over GROW_S after each refresh, so biome change
// grows/shrinks plants instead of popping them. A slot changing species first shrinks to 0 under
// its old species, then regrows as the new one.
// The same kernel compacts live slots into one list per mesh kind with atomics and writes the
// instanceCount of an indirect draw, so only live plants are drawn: 7 draw calls (+ shadow).
// Storage buffers: refresh 6 (renderCols, biome, veg, state, args, lists), vertex 2 (V23 ≤ 8).
// Terrain height/water come through space.ts columnSampler (render column summary: smoothed and raw
// surfY, water), so plant bases sit on the terrain as rendered; wet/slope culls use the raw values.
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, uint, int, vec3, vec4, uniform, uniformArray, instancedArray, storage, instanceIndex,
  atomicAdd, atomicStore, attribute, positionGeometry, normalGeometry, normalLocal, varying, mix,
  smoothstep, clamp, floor, max, min, sin, cos, select, pow,
} from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NX, NZ, CELL, BLOCK_SIZE, Y_SEA_NOMINAL } from '../sim/layout';
import { tColIdx } from '../sim/tslLayout';
import { columnSampler, updateRenderColumns, tWorldToCell, tWorldY, AMB_PERIOD } from '../render/space';
import {
  FLORA, FLORA_RULES, GRID_L, GRID_S, NSLOT, NSLOT_L, JITTER, EDGE_MARGIN, H, KIND_COUNT, KIND_SET, KIND_BASE, KIND_NAMES,
  LIST_SIZE, SPECIES_COUNT, SPECIES_KIND, SPECIES_SIZE, SPECIES_COLOR, SPECIES_TRUNK, PETALS, Kind, Sp,
  LIFE_BIOMES, TREELINE as T, TREELINE_BIOME,
} from './lifeModel';
import {
  treeGeometry, pineGeometry, cactusGeometry, acaciaGeometry, shrubGeometry, grassGeometry, flowerGeometry, cypressGeometry,
} from './geometry';

type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;
// TSL typings are loose around mixed int/float ops; builders below go through these casts.
const asF = (n: unknown) => n as F;
const asU = (n: unknown) => n as U;

const HALF = BLOCK_SIZE / 2;
const TAU = Math.PI * 2;

/** TSL lowbias32, bit-identical to lifeModel.hashU. */
const tHashU = (x: U): U => {
  const a = x.bitXor(x.shiftRight(uint(16))).mul(uint(0x7feb352d)).toVar();
  const b = a.bitXor(a.shiftRight(uint(15))).mul(uint(0x846ca68b)).toVar();
  return b.bitXor(b.shiftRight(uint(16)));
};
/** lifeModel.slotRand. */
const tSlotRand = (local: U, set: U, ch: number): F =>
  asF(float(tHashU(local.mul(uint(16)).add(uint(ch)).add(set.mul(uint(0x1000000)))).shiftRight(uint(8))).mul(1 / 16777216));

/** World x,z of a slot from its set-local index (lifeModel.slotXZ). */
function tSlotXZ(local: U, set: U, grid: U, spacing: F): { x: F; z: F } {
  const j = local.div(grid).toVar();
  const i = local.sub(j.mul(grid));
  const x = float(i).add(0.5).add(tSlotRand(local, set, H.JX).sub(0.5).mul(JITTER)).mul(spacing).sub(HALF);
  const z = float(j).add(0.5).add(tSlotRand(local, set, H.JZ).sub(0.5).mul(JITTER)).mul(spacing).sub(HALF);
  return { x: asF(x), z: asF(z) };
}

const lin = (hex: number) => new THREE.Color(hex);

export interface Flora {
  object: THREE.Group;
  /** Re-evaluate every slot from the current fields. snap: jump straight to the targets (tests, load). */
  refresh(renderer: THREE.WebGPURenderer, snap?: boolean): void;
  /** Advance the grow ramp and wind clock. */
  frame(dt: number, ambTime: number): void;
  setQuality(high: boolean): void;
  /** Emergent sea level (voxel y), e.g. sim.stats.seaLevel. Picked up at the next refresh. */
  setSeaLevel(y: number): void;
  /** Test/debug handles: per-slot state (vec4), per-kind lists, indirect args (5 u32 per kind). */
  buffers: { state: THREE.BufferAttribute; lists: THREE.BufferAttribute; args: THREE.IndirectStorageBufferAttribute };
  meshes: THREE.Mesh[];
  kernels: { clear: THREE.ComputeNode; refresh: THREE.ComputeNode };
  dispose(): void;
}

export function createFlora(fields: GpuFields, opts: { highQuality?: boolean } = {}): Flora {
  const S = columnSampler(fields);
  const biome = fields.cur<'uint'>('biome');
  const veg = fields.cur('veg');

  const state = instancedArray(NSLOT, 'vec4');
  state.setName('lifeFloraState');
  const lists = instancedArray(LIST_SIZE, 'uint');
  lists.setName('lifeFloraLists');

  const geos = [treeGeometry(), pineGeometry(), cactusGeometry(), acaciaGeometry(), shrubGeometry(), grassGeometry(), flowerGeometry(), cypressGeometry()];
  const argsArr = new Uint32Array(KIND_COUNT * 5);
  geos.forEach((g, k) => { argsArr[k * 5] = g.index!.count; });
  const argsAttr = new THREE.IndirectStorageBufferAttribute(argsArr, 1);
  const args = storage(argsAttr, 'uint', KIND_COUNT * 5).toAtomic();

  // ---- tables ----
  const rules = uniformArray(FLORA_RULES.flatMap(([l, s]) => [l, s].map((r) => new THREE.Vector4(r.density, r.a, r.b, r.pB))), 'vec4');
  const spKind = uniformArray(Array.from({ length: SPECIES_COUNT }, (_, s) => Math.max(0, SPECIES_KIND[s]!)), 'uint');
  const spBase = uniformArray(Array.from({ length: SPECIES_COUNT }, (_, s) => KIND_BASE[Math.max(0, SPECIES_KIND[s]!)]!), 'uint');
  const spSize = uniformArray([...SPECIES_SIZE], 'float');
  const spColor = uniformArray(SPECIES_COLOR.map(lin), 'color');
  const spTrunk = uniformArray(SPECIES_TRUNK.map(lin), 'color');
  const petals = uniformArray(PETALS.map(lin), 'color');
  const treeBiome = uniformArray([...TREELINE_BIOME], 'float');

  const uQuality = uniform(opts.highQuality === false ? FLORA.LOW_QUALITY : 1);
  const uBlendNow = uniform(1); // ramp value at the moment of a refresh (compute)
  const uSnap = uniform(0, 'uint');
  const uBlend = uniform(1);    // ramp value for drawing
  const uTime = uniform(0);     // wrapped ambience clock (wind)
  const uSeaLevel = uniform(Y_SEA_NOMINAL); // emergent sea level (voxel y) for the altitude zonation

  // ---- refresh + compact kernel ----
  const clearArgs = Fn(() => {
    If(instanceIndex.lessThan(uint(KIND_COUNT)), () => { atomicStore(args.element(instanceIndex.mul(uint(5)).add(uint(1))), uint(0)); });
  })().compute(KIND_COUNT);

  const refreshKernel = Fn(() => {
    const s = instanceIndex;
    If(s.greaterThanEqual(uint(NSLOT)), () => { Return(); });
    const isSmall = s.greaterThanEqual(uint(NSLOT_L));
    const set = select(isSmall, uint(1), uint(0)).toVar();
    const local = select(isSmall, s.sub(uint(NSLOT_L)), s).toVar();
    const grid = select(isSmall, uint(GRID_S), uint(GRID_L));
    const spacing = asF(select(isSmall, float(BLOCK_SIZE / GRID_S), float(BLOCK_SIZE / GRID_L)));
    const { x, z } = tSlotXZ(local, set, grid, spacing);
    const px = x.toVar(), pz = z.toVar();
    const margin = select(isSmall, float(EDGE_MARGIN[1]!), float(EDGE_MARGIN[0]!));
    const inside = px.abs().lessThanEqual(float(HALF).sub(margin)).and(pz.abs().lessThanEqual(float(HALF).sub(margin)));

    const cx = int(clamp(floor(px.add(HALF).div(CELL)), 0, NX - 1)).toVar();
    const cz = int(clamp(floor(pz.add(HALF).div(CELL)), 0, NZ - 1)).toVar();
    const ci = tColIdx(cx, cz).toVar();
    const c = S.corners(tWorldToCell(px), tWorldToCell(pz));
    const h = S.height(c).toVar(); // render height (smoothed surfY, space.ts); culls below use raw fields
    const wet = max(max(c[0]!.water, c[1]!.water), max(c[2]!.water, c[3]!.water));
    const gx = S.surfAt(cx.add(1), cz).sub(S.surfAt(cx.sub(1), cz)).mul(0.5);
    const gz = S.surfAt(cx, cz.add(1)).sub(S.surfAt(cx, cz.sub(1))).mul(0.5);
    const slope = gx.mul(gx).add(gz.mul(gz)).sqrt();

    const b = asU(biome.element(ci));
    const bi = select(b.lessThan(uint(LIFE_BIOMES)), b, uint(0)).toVar();
    const rule = (rules.element(bi.mul(uint(2)).add(set)) as unknown as V4).toVar();
    const v = asF(veg.element(ci)).toVar();
    // altitude zonation (lifeModel TREELINE, mirrors floraDecide)
    const alt = asF(S.surfAt(cx, cz)).sub(uSeaLevel).toVar();
    const bare = float(1).sub(smoothstep(T.BARE_A0, T.BARE_A1, alt));
    const line = asF(treeBiome.element(bi)).mul(smoothstep(T.TREE_A0, T.TREE_A1, alt)).toVar();
    const isLarge = isSmall.not();
    const p = select(isLarge, v.mul(rule.x).mul(float(1).sub(line)), v.mul(rule.x.add(line.mul(T.LINE_BOOST)))).mul(bare);
    const occ = tSlotRand(local, set, H.OCC).lessThan(p)
      .and(tSlotRand(local, set, H.QUALITY).lessThan(uQuality))
      .and(wet.lessThanEqual(FLORA.WET_MAX)).and(slope.lessThanEqual(FLORA.SLOPE_MAX)).and(inside);
    const pick0 = select(tSlotRand(local, set, H.SPECIES).lessThan(rule.w), rule.z, rule.y).toVar();
    const broad = pick0.equal(float(Sp.BROADLEAF)).or(pick0.equal(float(Sp.BROADLEAF_RAIN)));
    const toPine = broad.and(tSlotRand(local, set, H.ALT).lessThan(smoothstep(T.PINE_A0, T.PINE_A1, alt)));
    const lineSp = select(tSlotRand(local, set, H.ALT2).lessThan(0.45), float(Sp.SHRUB_TUNDRA), float(Sp.GRASS_ALPINE));
    const toLine = tSlotRand(local, set, H.ALT).lessThan(line.mul(T.LINE_SHARE));
    const pick = select(isLarge, select(toPine, float(Sp.PINE), pick0), select(toLine, lineSp, pick0));
    const sp = select(occ, uint(pick), uint(0)).toVar();
    const stunt = select(isLarge, float(1).sub(line.mul(T.STUNT)), float(1));
    const tScale = select(occ, tSlotRand(local, set, H.SCALE).mul(0.5).add(0.75).mul(v.mul(0.4).add(0.6)).mul(stunt), float(0));

    const st = (state.element(s) as unknown as V4).toVar();
    const oldSp = uint(st.w).toVar();
    const cur = mix(st.y, st.z, uBlendNow);
    const switchOK = sp.equal(oldSp).or(cur.lessThan(FLORA.SWITCH_MIN)).or(uSnap.equal(uint(1)));
    const newSp = select(switchOK, sp, oldSp).toVar();
    const target = select(switchOK, tScale, float(0)).toVar();
    const prev = select(newSp.equal(oldSp), cur, float(0));
    const prevS = select(uSnap.equal(uint(1)), target, prev).toVar();
    state.element(s).assign(vec4(h, prevS, target, float(newSp)));

    If(newSp.notEqual(uint(0)).and(max(prevS, target).greaterThan(0)), () => {
      const kind = asU(spKind.element(newSp));
      const idx = uint(0).toVar();
      idx.assign(atomicAdd(args.element(kind.mul(uint(5)).add(uint(1))), uint(1)));
      lists.element(asU(spBase.element(newSp)).add(idx)).assign(s);
    });
  })().compute(NSLOT);

  // ---- meshes ----
  const group = new THREE.Group();
  group.name = 'life-flora';
  const lp = attribute('lifeP', 'vec4') as unknown as V4;
  const wind1 = (TAU * 1260) / AMB_PERIOD; // 0.35 Hz, whole cycles per AMB_PERIOD (seamless wrap)
  const wind2 = (TAU * 2268) / AMB_PERIOD; // 0.63 Hz flutter

  const meshes = geos.map((geo, k) => {
    const set = KIND_SET[k]!;
    const grid = set === 0 ? GRID_L : GRID_S;
    const slot = asU(lists.element(uint(KIND_BASE[k]!).add(instanceIndex))).toVar();
    const st = (state.element(slot) as unknown as V4).toVar();
    const local = slot.sub(uint(set === 0 ? 0 : NSLOT_L)).toVar();
    const setU = uint(set);
    const { x, z } = tSlotXZ(local, setU, uint(grid), float(BLOCK_SIZE / grid));
    const sp = uint(st.w).toVar();
    const r = (ch: number) => tSlotRand(local, setU, ch);
    const size = asF(spSize.element(sp)).mul(mix(st.y, st.z, uBlend)).toVar();
    const yaw = r(H.YAW).mul(TAU).toVar();
    const cy = cos(yaw), sy = sin(yaw);
    const tall = r(H.SQUASH).mul(0.3).add(0.85);
    const wide = r(H.TINT).mul(0.2).add(0.9);
    const h01 = lp.x, part = lp.y, flex = lp.z;

    const positionNode = Fn(() => {
      const p = positionGeometry;
      const lx = p.x.mul(wide).mul(size), ly = p.y.mul(tall).mul(size), lz = p.z.mul(wide).mul(size);
      const wx = x.toVar(), wz = z.toVar();
      const ph = uTime.mul(wind1).add(wx.mul(3.1)).add(wz.mul(2.3));
      const bend = flex.mul(h01).mul(h01).mul(size).mul(0.1);
      const gust = sin(ph).add(sin(uTime.mul(wind2).add(wx.mul(7.0)).sub(wz.mul(5.0))).mul(0.35));
      const sx = gust.mul(bend), sz = cos(ph.mul(0.8).add(1.3)).mul(bend).mul(0.5);
      const base = tWorldY(st.x.sub(FLORA.SINK));
      // rotated normal into normalLocal (not a normalNode) so lighting AND the shadow normal bias use it
      const n = normalGeometry;
      normalLocal.assign(vec3(n.x.mul(cy).sub(n.z.mul(sy)), n.y, n.x.mul(sy).add(n.z.mul(cy))));
      return vec3(wx.add(lx.mul(cy)).sub(lz.mul(sy)).add(sx), base.add(ly), wz.add(lx.mul(sy)).add(lz.mul(cy)).add(sz));
    })();

    // Albedo per vertex: species colour, base→top gradient, contact AO at the ground, slot tint.
    const colorV = varying(Fn(() => {
      const fol = spColor.element(sp) as unknown as V3;
      const trunk = spTrunk.element(sp) as unknown as V3;
      const petal = select(sp.equal(uint(Sp.CACTUS)), vec3(lin(0xff7fa8).r, lin(0xff7fa8).g, lin(0xff7fa8).b),
        petals.element(uint(min(floor(r(H.PETAL).mul(PETALS.length)), PETALS.length - 1))) as unknown as V3);
      const isTrunk = part.lessThan(0.5), isPetal = part.greaterThan(1.5).and(part.lessThan(2.5)), isBlade = part.greaterThan(2.5);
      // ~4% of temperate broadleaf trees wear autumn colours
      const autumn = sp.equal(uint(Sp.BROADLEAF)).and(r(H.PETAL).greaterThan(0.96));
      const autumnCol = mix(vec3(lin(0xe0782a).r, lin(0xe0782a).g, lin(0xe0782a).b), vec3(lin(0xd9b13b).r, lin(0xd9b13b).g, lin(0xd9b13b).b), r(H.SCALE));
      const folC = select(autumn, autumnCol, fol);
      const col = select(isTrunk, trunk, select(isPetal, petal, folC)).toVar();
      const grad = select(isTrunk, mix(float(0.7), float(1.0), h01),
        select(isBlade, mix(float(0.42), float(1.18), pow(h01, float(0.8))), mix(float(0.6), float(1.12), h01)));
      const ao = mix(float(0.42), float(1), smoothstep(0, 0.2, h01));
      const tint = r(H.TINT).mul(0.26).add(0.87);
      const warm = mix(vec3(0.9, 1.0, 1.08), vec3(1.12, 1.04, 0.74), r(H.PETAL));
      return col.mul(select(isPetal, float(1), grad.mul(ao))).mul(tint).mul(select(isPetal, vec3(1), warm));
    })(), 'vLifeC');

    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0 });
    mat.positionNode = positionNode;
    mat.colorNode = colorV;
    if (k === Kind.GRASS || k === Kind.FLOWER) mat.side = THREE.DoubleSide;
    geo.setIndirect(argsAttr, k * 20);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = `life-${KIND_NAMES[k]}`;
    mesh.frustumCulled = false;
    mesh.castShadow = k !== Kind.GRASS && k !== Kind.FLOWER;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  });

  let sinceRefresh: number = FLORA.GROW_S;
  const ease = (t: number) => { const u = Math.min(1, Math.max(0, t)); return u * u * (3 - 2 * u); };

  return {
    object: group,
    meshes,
    kernels: { clear: clearArgs, refresh: refreshKernel },
    buffers: { state: state.value, lists: lists.value, args: argsAttr },
    refresh(renderer, snap = false) {
      uBlendNow.value = ease(sinceRefresh / FLORA.GROW_S);
      uSnap.value = snap ? 1 : 0;
      updateRenderColumns(renderer, fields); // plant bases sit on the rendered (smoothed) terrain
      renderer.compute(clearArgs);
      renderer.compute(refreshKernel);
      sinceRefresh = snap ? FLORA.GROW_S : 0;
      uBlend.value = ease(sinceRefresh / FLORA.GROW_S);
    },
    frame(dt, ambTime) {
      sinceRefresh += dt;
      uBlend.value = ease(sinceRefresh / FLORA.GROW_S);
      uTime.value = ((ambTime % AMB_PERIOD) + AMB_PERIOD) % AMB_PERIOD;
    },
    setQuality(high) { uQuality.value = high ? 1 : FLORA.LOW_QUALITY; },
    setSeaLevel(y) { if (Number.isFinite(y)) uSeaLevel.value = y; },
    dispose() {
      for (const m of meshes) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); }
      group.removeFromParent();
    },
  };
}
