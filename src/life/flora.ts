// §T.45 flora: GPU-scattered instanced plants per biome, read-only on sim fields (V15).
//
// Slots: three fixed jittered grids over the block (lifeModel.ts): large (trees), small (shrubs,
// tufts, rocks), fine (ground cover). A refresh kernel (every ~2 s) reads biome/veg/terrain at
// each slot and sets its target species + scale; the per-slot hash is fixed, so plants never jump
// between refreshes. Patch noise clumps density into drifts, flower patches and bare spots.
// Each slot keeps (voxelY, prevScale, targetScale, species + moisture tint in the fraction); the
// shader shows mix(prev, target, blend) with blend ramping 0 → 1 over GROW_S after each refresh, so
// biome change grows/shrinks plants instead of popping them. A slot changing species first shrinks
// to 0 under its old species, then regrows as the new one.
// Large + small slots are compacted into per-kind lists at refresh (atomics → indirect draw args).
// Fine ground cover is compacted every frame by a cull kernel: camera frustum + distance thinning
// (surviving far tufts widen to keep the coverage), so only what is seen near the camera is drawn.
// Storage buffers: refresh 6 (renderCols, biome, veg, state, args, lists), cull 3, vertex 2 (V23 ≤ 8).
// Terrain height/water come through space.ts columnSampler (render column summary: smoothed and raw
// surfY, water), so plant bases sit on the terrain as rendered; wet/slope culls use the raw values.
import * as THREE from 'three/webgpu';
import {
  Fn, If, Return, float, uint, int, vec3, vec4, uniform, uniformArray, instancedArray, storage, instanceIndex,
  atomicAdd, atomicStore, atomicLoad, attribute, positionGeometry, normalGeometry, normalLocal, varying, mix,
  smoothstep, clamp, floor, fract, max, min, sin, cos, select, pow, exp, log, sqrt, cameraPosition, transformNormalToView, uv,
} from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NX, NZ, NCOL, CELL, BLOCK_SIZE, Y_SEA_NOMINAL } from '../sim/layout';
import { tColIdx, tColXZ } from '../sim/tslLayout';
import { columnSampler, updateRenderColumns, tWorldToCell, tWorldY, AMB_PERIOD } from '../render/space';
import {
  FLORA, FLORA_RULES, GRID_F, NSLOT, NSLOT_L, NSLOT_LS, NSLOT_F, SET_GRID, SET_BASE, JITTER, EDGE_MARGIN, H,
  KIND_COUNT, KIND_SET, KIND_BASE, KIND_NAMES, LIST_SIZE, SPECIES_COUNT, SPECIES_KIND, SPECIES_SIZE, SPECIES_COLOR,
  SPECIES_TRUNK, SPECIES_FLEX, SPECIES_LEAN, PETAL_PALETTES, TUSSOCK_TONES, Kind, Sp, KIND_FOOT, KIND_BLOB,
  LIFE_BIOMES, TREELINE as T, TREELINE_BIOME, PATCH, SALT,
} from './lifeModel';
import {
  treeGeometry, pineGeometry, cactusGeometry, acaciaGeometry, shrubGeometry, grassGeometry, flowerGeometry, cypressGeometry,
  rockGeometry, tussockGeometry, hedgeGeometry, fineShortGeometry, fineTallGeometry, seedHeadGeometry, cloverGeometry,
  fineFlowerGeometry, blobGeometry,
} from './geometry';

type F = THREE.Node<'float'>;
type U = THREE.Node<'uint'>;
type I = THREE.Node<'int'>;
type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;
// TSL typings are loose around mixed int/float ops; builders below go through these casts.
const asF = (n: unknown) => n as F;
const asU = (n: unknown) => n as U;

const HALF = BLOCK_SIZE / 2;
const TAU = Math.PI * 2;
const LAT_OFF = 4096; // lifeModel latticeRand

/** TSL lowbias32, bit-identical to lifeModel.hashU. */
const tHashU = (x: U): U => {
  const a = x.bitXor(x.shiftRight(uint(16))).mul(uint(0x7feb352d)).toVar();
  const b = a.bitXor(a.shiftRight(uint(15))).mul(uint(0x846ca68b)).toVar();
  return b.bitXor(b.shiftRight(uint(16)));
};
const tU01 = (h: U): F => asF(float(h.shiftRight(uint(8))).mul(1 / 16777216));
/** lifeModel.slotRand. */
const tSlotRand = (local: U, set: U, ch: number): F =>
  tU01(tHashU(local.mul(uint(16)).add(uint(ch)).add(set.mul(uint(0x1000000)))));

/** lifeModel.latticeRand. */
const tLattice = (ix: I, iz: I, salt: number): F =>
  tU01(tHashU(uint(ix.add(LAT_OFF)).mul(uint(73856093)).bitXor(uint(iz.add(LAT_OFF)).mul(uint(19349663))).bitXor(uint(salt))));
/** lifeModel.valueNoise. */
function tValueNoise(x: F, z: F, cell: number, salt: number): F {
  const u = x.div(cell).toVar(), v = z.div(cell).toVar();
  const i = int(floor(u)).toVar(), j = int(floor(v)).toVar();
  const fu0 = fract(u), fv0 = fract(v);
  const fu = fu0.mul(fu0).mul(float(3).sub(fu0.mul(2))).toVar(), fv = fv0.mul(fv0).mul(float(3).sub(fv0.mul(2))).toVar();
  const a = tLattice(i, j, salt).toVar(), b = tLattice(i.add(1), j, salt), c = tLattice(i, j.add(1), salt).toVar(), d = tLattice(i.add(1), j.add(1), salt);
  const ab = a.add(b.sub(a).mul(fu)).toVar();
  const cd = c.add(d.sub(c).mul(fu));
  return asF(ab.add(cd.sub(ab).mul(fv)));
}
const tPatchNoise = (x: F, z: F): F =>
  asF(tValueNoise(x, z, PATCH.CELL_A, SALT.PATCH_A).mul(0.65).add(tValueNoise(x, z, PATCH.CELL_B, SALT.PATCH_B).mul(0.35)));
const tPatchMul = (n: F): F => { const t = smoothstep(PATCH.N0, PATCH.N1, n); return asF(t.mul(t).mul(PATCH.M_SPAN).add(PATCH.M_MIN)); };
const tFlowerMul = (x: F, z: F): F =>
  asF(smoothstep(PATCH.F0, PATCH.F1, tValueNoise(x, z, PATCH.CELL_FLOWER, SALT.FLOWER)).mul(PATCH.F_SPAN).add(PATCH.F_MIN));

/**
 * Meadow density field 0..~4.75 at world x,z (same patch field the fine ground cover follows).
 * Exported for the terrain: a ground-cover tint under dense meadows stays in step with the plants.
 */
export const tMeadowDensity = (x: F, z: F): F => tPatchMul(tPatchNoise(x, z));

/** World x,z of a slot from its set-local index (lifeModel.slotXZ). */
function tSlotXZ(local: U, set: U, grid: U, spacing: F): { x: F; z: F } {
  const j = local.div(grid).toVar();
  const i = local.sub(j.mul(grid));
  const x = float(i).add(0.5).add(tSlotRand(local, set, H.JX).sub(0.5).mul(JITTER)).mul(spacing).sub(HALF);
  const z = float(j).add(0.5).add(tSlotRand(local, set, H.JZ).sub(0.5).mul(JITTER)).mul(spacing).sub(HALF);
  return { x: asF(x), z: asF(z) };
}

/** Fine ground cover keep fraction at camera distance d (lifeModel FLORA.LOD_*). */
const tLodKeep = (d: F): F =>
  asF(clamp(float(1).sub(d.sub(FLORA.LOD_D0).div(FLORA.LOD_D1 - FLORA.LOD_D0).mul(1 - FLORA.LOD_MIN)), FLORA.LOD_MIN, 1));

const lin = (hex: number) => new THREE.Color(hex);
const col3 = (hex: number) => { const c = lin(hex); return vec3(c.r, c.g, c.b); };

// Per mesh kind: double sided, casts shadow, random tilt (rad, shear), lognormal size sigma, moisture tint strength.
const KIND_LOOK: Record<number, { ds: boolean; shadow: boolean; tilt: number; sigma: number; moist: number; wide?: number }> = {
  [Kind.TREE]: { ds: false, shadow: true, tilt: 0.04, sigma: 0.14, moist: 0.25 },
  [Kind.PINE]: { ds: false, shadow: true, tilt: 0.03, sigma: 0.14, moist: 0.15 },
  [Kind.CACTUS]: { ds: false, shadow: true, tilt: 0.05, sigma: 0.18, moist: 0 },
  [Kind.ACACIA]: { ds: false, shadow: true, tilt: 0.04, sigma: 0.15, moist: 0.3 },
  [Kind.SHRUB]: { ds: false, shadow: true, tilt: 0.1, sigma: 0.28, moist: 0.5 },
  [Kind.GRASS]: { ds: true, shadow: false, tilt: 0.2, sigma: 0.3, moist: 1 },
  [Kind.FLOWER]: { ds: true, shadow: false, tilt: 0.15, sigma: 0.25, moist: 0.6 },
  [Kind.CYPRESS]: { ds: false, shadow: true, tilt: 0.03, sigma: 0.15, moist: 0.15 },
  [Kind.ROCK]: { ds: false, shadow: true, tilt: 0.45, sigma: 0.4, moist: 0 },
  [Kind.TUSSOCK]: { ds: true, shadow: true, tilt: 0.12, sigma: 0.3, moist: 0.6 },
  [Kind.HEDGE]: { ds: false, shadow: true, tilt: 0.02, sigma: 0.15, moist: 0.3 },
  [Kind.F_SHORT]: { ds: true, shadow: false, tilt: 0.22, sigma: 0.32, moist: 1, wide: 1.45 },
  [Kind.F_TALL]: { ds: true, shadow: false, tilt: 0.18, sigma: 0.3, moist: 0.9, wide: 1.45 },
  [Kind.F_SEED]: { ds: true, shadow: false, tilt: 0.2, sigma: 0.3, moist: 0.8, wide: 1.45 },
  [Kind.F_CLOVER]: { ds: true, shadow: false, tilt: 0.25, sigma: 0.3, moist: 0.8, wide: 1.45 },
  [Kind.F_FLOWER]: { ds: true, shadow: false, tilt: 0.18, sigma: 0.25, moist: 0.6 },
};

export interface Flora {
  object: THREE.Group;
  /** Re-evaluate every slot from the current fields. snap: jump straight to the targets (tests, load). */
  refresh(renderer: THREE.WebGPURenderer, snap?: boolean): void;
  /** Per frame: grow ramp, wind clock, and the fine ground-cover cull against the camera. */
  frame(renderer: THREE.WebGPURenderer, dt: number, ambTime: number): void;
  /** Camera for fine ground-cover culling / thinning; null keeps every live fine slot (tests). */
  setCamera(camera: THREE.Camera | null): void;
  /** Re-run only the fine ground-cover compaction (after setCamera, tests). */
  cullFine(renderer: THREE.WebGPURenderer): void;
  setQuality(high: boolean): void;
  /** Emergent sea level (voxel y), e.g. sim.stats.seaLevel. Picked up at the next refresh. */
  setSeaLevel(y: number): void;
  /** Test/debug handles: per-slot state (vec4), per-kind lists, indirect args (5 u32 per kind). */
  buffers: { state: THREE.BufferAttribute; lists: THREE.BufferAttribute; args: THREE.IndirectStorageBufferAttribute;
    /** Render height per column (voxel y), refreshed with each flora refresh. */
    renderH: THREE.BufferAttribute };
  meshes: THREE.Mesh[];
  /** Contact-shadow blob meshes (one per large/small kind). */
  blobs: THREE.Mesh[];
  kernels: { clear: THREE.ComputeNode; refresh: THREE.ComputeNode; clearFine: THREE.ComputeNode; cull: THREE.ComputeNode };
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

  const geos: THREE.BufferGeometry[] = [];
  geos[Kind.TREE] = treeGeometry(); geos[Kind.PINE] = pineGeometry(); geos[Kind.CACTUS] = cactusGeometry();
  geos[Kind.ACACIA] = acaciaGeometry(); geos[Kind.SHRUB] = shrubGeometry(); geos[Kind.GRASS] = grassGeometry();
  geos[Kind.FLOWER] = flowerGeometry(); geos[Kind.CYPRESS] = cypressGeometry(); geos[Kind.ROCK] = rockGeometry();
  geos[Kind.TUSSOCK] = tussockGeometry(); geos[Kind.HEDGE] = hedgeGeometry(); geos[Kind.F_SHORT] = fineShortGeometry();
  geos[Kind.F_TALL] = fineTallGeometry(); geos[Kind.F_SEED] = seedHeadGeometry(); geos[Kind.F_CLOVER] = cloverGeometry();
  geos[Kind.F_FLOWER] = fineFlowerGeometry();
  // indirect args: [0, KIND_COUNT) plants, [KIND_COUNT, 2·KIND_COUNT) their contact-shadow blobs
  // (same instance count, copied after each refresh; blob index count)
  const blobGeo = blobGeometry();
  const argsArr = new Uint32Array(KIND_COUNT * 2 * 5);
  geos.forEach((g, k) => { argsArr[k * 5] = g.index!.count; argsArr[(KIND_COUNT + k) * 5] = blobGeo.index!.count; });
  const argsAttr = new THREE.IndirectStorageBufferAttribute(argsArr, 1);
  const args = storage(argsAttr, 'uint', KIND_COUNT * 2 * 5).toAtomic();

  // ---- tables ----
  // rules: 3 vec4 per (biome, set): (density, t1, t2, t3), (sp a, b, c, d), (flower, pF, -, -)
  const rules = uniformArray(FLORA_RULES.flatMap((sets) => sets.flatMap((r) => [
    new THREE.Vector4(r.density, r.t[0], r.t[1], r.t[2]), new THREE.Vector4(...r.sp), new THREE.Vector4(r.flower, r.pF, 0, 0),
  ])), 'vec4');
  const spKind = uniformArray(Array.from({ length: SPECIES_COUNT }, (_, s) => Math.max(0, SPECIES_KIND[s]!)), 'uint');
  const spBase = uniformArray(Array.from({ length: SPECIES_COUNT }, (_, s) => KIND_BASE[Math.max(0, SPECIES_KIND[s]!)]!), 'uint');
  const spSize = uniformArray([...SPECIES_SIZE], 'float');
  const spFlex = uniformArray([...SPECIES_FLEX], 'float');
  const spLean = uniformArray([...SPECIES_LEAN], 'float');
  const spColor = uniformArray(SPECIES_COLOR.map(lin), 'color');
  const spTrunk = uniformArray(SPECIES_TRUNK.map(lin), 'color');
  const petals = uniformArray(PETAL_PALETTES.flatMap((p) => p.map(lin)), 'color');
  const tussock = uniformArray(TUSSOCK_TONES.map(lin), 'color');
  const treeBiome = uniformArray([...TREELINE_BIOME], 'float');
  // base contact radius per species in columns (lifeModel.baseSink without slope and scale)
  const spFoot = uniformArray(Array.from({ length: SPECIES_COUNT }, (_, s) =>
    s === 0 ? 0 : (SPECIES_SIZE[s]! * KIND_FOOT[SPECIES_KIND[s]!]! * 1.2) / CELL), 'float');

  const uQuality = uniform(opts.highQuality === false ? FLORA.LOW_QUALITY : 1);
  const uBlendNow = uniform(1); // ramp value at the moment of a refresh (compute)
  const uSnap = uniform(0, 'uint');
  const uBlend = uniform(1);    // ramp value for drawing
  const uTime = uniform(0);     // wrapped ambience clock (wind)
  const uSeaLevel = uniform(Y_SEA_NOMINAL); // emergent sea level (voxel y) for the altitude zonation
  const uCull = uniform(0, 'uint'); // 1: fine ground cover culled/thinned against the camera
  const uCamPos = uniform(new THREE.Vector3());
  const uViewProj = uniform(new THREE.Matrix4());

  // ---- refresh + compact kernel (large + small sets; fine set state only) ----
  const clearArgs = Fn(() => {
    If(instanceIndex.lessThan(uint(KIND_COUNT)), () => { atomicStore(args.element(instanceIndex.mul(uint(5)).add(uint(1))), uint(0)); });
  })().compute(KIND_COUNT);

  const refreshKernel = Fn(() => {
    const s = instanceIndex;
    If(s.greaterThanEqual(uint(NSLOT)), () => { Return(); });
    const isS = s.greaterThanEqual(uint(NSLOT_L)).and(s.lessThan(uint(NSLOT_LS)));
    const isF = s.greaterThanEqual(uint(NSLOT_LS));
    const isL = s.lessThan(uint(NSLOT_L));
    const set = select(isF, uint(2), select(isS, uint(1), uint(0))).toVar();
    const local = s.sub(select(isF, uint(NSLOT_LS), select(isS, uint(NSLOT_L), uint(0)))).toVar();
    const grid = select(isF, uint(SET_GRID[2]!), select(isS, uint(SET_GRID[1]!), uint(SET_GRID[0]!)));
    const spacing = asF(select(isF, float(BLOCK_SIZE / SET_GRID[2]!), select(isS, float(BLOCK_SIZE / SET_GRID[1]!), float(BLOCK_SIZE / SET_GRID[0]!))));
    const { x, z } = tSlotXZ(local, set, grid, spacing);
    const px = x.toVar(), pz = z.toVar();
    const margin = select(isF, float(EDGE_MARGIN[2]!), select(isS, float(EDGE_MARGIN[1]!), float(EDGE_MARGIN[0]!)));
    const inside = px.abs().lessThanEqual(float(HALF).sub(margin)).and(pz.abs().lessThanEqual(float(HALF).sub(margin)));
    const r = (ch: number) => tSlotRand(local, set, ch);

    const cx = int(clamp(floor(px.add(HALF).div(CELL)), 0, NX - 1)).toVar();
    const cz = int(clamp(floor(pz.add(HALF).div(CELL)), 0, NZ - 1)).toVar();
    const ci = tColIdx(cx, cz).toVar();
    const cu = tWorldToCell(px).toVar(), cv = tWorldToCell(pz).toVar();
    const c = S.corners(cu, cv);
    // render height as the terrain mesh draws it (smoothed surfY at column centres, linear per
    // triangle, lifeModel.heightAt); culls below use the raw fields
    const fu = fract(cu), fv = fract(cv);
    const hA = c[0]!.surf, hB = c[1]!.surf, hC = c[2]!.surf, hD = c[3]!.surf;
    const h = select(fu.add(fv).lessThanEqual(1),
      hA.add(fu.mul(hB.sub(hA))).add(fv.mul(hC.sub(hA))),
      hD.add(float(1).sub(fu).mul(hC.sub(hD))).add(float(1).sub(fv).mul(hB.sub(hD)))).toVar();
    const wet = max(max(c[0]!.water, c[1]!.water), max(c[2]!.water, c[3]!.water)).toVar();
    const sC = asF(S.surfAt(cx, cz)).toVar();
    const sL = asF(S.surfAt(cx.sub(1), cz)), sR = asF(S.surfAt(cx.add(1), cz));
    const sD = asF(S.surfAt(cx, cz.sub(1))), sU = asF(S.surfAt(cx, cz.add(1)));
    const gx = sR.sub(sL).mul(0.5), gz = sU.sub(sD).mul(0.5);
    const slope = gx.mul(gx).add(gz.mul(gz)).sqrt().toVar();
    const hollow = sL.add(sR).add(sD).add(sU).mul(0.25).sub(sC); // > 0 in hollows, < 0 on ridges

    const b = asU(biome.element(ci));
    const bi = select(b.lessThan(uint(LIFE_BIOMES)), b, uint(0)).toVar();
    const ri = bi.mul(uint(3)).add(set).mul(uint(3)).toVar();
    const r0 = (rules.element(ri) as unknown as V4).toVar();              // density, t1..t3
    const r1 = (rules.element(ri.add(uint(1))) as unknown as V4).toVar(); // species a..d
    const r2 = (rules.element(ri.add(uint(2))) as unknown as V4).toVar(); // flower, pF
    const v = asF(veg.element(ci)).toVar();
    // altitude zonation (lifeModel TREELINE) and patch clumping (PATCH), mirrors floraDecide
    const alt = sC.sub(uSeaLevel).toVar();
    const bare = float(1).sub(smoothstep(T.BARE_A0, T.BARE_A1, alt));
    const line = asF(treeBiome.element(bi)).mul(smoothstep(T.TREE_A0, T.TREE_A1, alt)).toVar();
    const pn = tPatchNoise(px, pz).toVar();
    const pm = tPatchMul(pn).toVar();
    const patch = select(isL, pm.sub(1).mul(PATCH.LARGE_MIX).add(1), pm);
    const pBase = select(isL, v.mul(r0.x).mul(float(1).sub(line)),
      select(isS, v.mul(r0.x.add(line.mul(T.LINE_BOOST))), v.mul(r0.x)));
    const p = pBase.mul(bare).mul(patch);
    const slopeMax = select(isF, float(FLORA.SLOPE_MAX_SET[2]!), select(isS, float(FLORA.SLOPE_MAX_SET[1]!), float(FLORA.SLOPE_MAX_SET[0]!)));
    const reed = isF.and(wet.greaterThan(FLORA.WET_MAX)).and(wet.lessThanEqual(FLORA.REED_MAX)).and(r(H.ALT).lessThan(FLORA.REED_P)).toVar();
    const occ = r(H.OCC).lessThan(p)
      .and(r(H.QUALITY).lessThan(uQuality))
      .and(wet.lessThanEqual(FLORA.WET_MAX).or(reed)).and(slope.lessThanEqual(slopeMax)).and(inside);
    const rs = r(H.SPECIES).toVar();
    const pick0 = select(rs.lessThan(r0.y), r1.x, select(rs.lessThan(r0.z), r1.y, select(rs.lessThan(r0.w), r1.z, r1.w)));
    const flowerOK = r2.x.greaterThan(0.5).and(r(H.FLOWER).lessThan(r2.y.mul(tFlowerMul(px, pz))));
    const pick1 = select(flowerOK, r2.x, pick0).toVar();
    const broad = pick1.equal(float(Sp.BROADLEAF)).or(pick1.equal(float(Sp.BROADLEAF_RAIN)));
    const toPine = broad.and(r(H.ALT).lessThan(smoothstep(T.PINE_A0, T.PINE_A1, alt)));
    const lineSp = select(r(H.ALT2).lessThan(0.45), float(Sp.SHRUB_TUNDRA), float(Sp.GRASS_ALPINE));
    const toLine = r(H.ALT).lessThan(line.mul(T.LINE_SHARE));
    const pick2 = select(isL, select(toPine, float(Sp.PINE), pick1), select(isS.and(toLine), lineSp, pick1));
    const pick = select(reed, float(Sp.REED), pick2);
    const sp = select(occ, uint(pick), uint(0)).toVar();
    const stunt = select(isL, float(1).sub(line.mul(T.STUNT)), float(1));
    const tScale = select(occ, r(H.SCALE).mul(0.5).add(0.75).mul(v.mul(0.4).add(0.6)).mul(stunt), float(0));
    // moisture tint 0 (straw, dry ridge) .. 1 (lush, wet hollow); packed into the species fraction
    const tint = clamp(v.mul(0.55).add(0.3).add(clamp(hollow.mul(0.35), -0.3, 0.3)).add(pn.sub(0.5).mul(0.6)), 0, 0.99);

    const st = (state.element(s) as unknown as V4).toVar();
    const oldSp = uint(st.w).toVar();
    const cur = mix(st.y, st.z, uBlendNow);
    const switchOK = sp.equal(oldSp).or(cur.lessThan(FLORA.SWITCH_MIN)).or(uSnap.equal(uint(1)));
    const newSp = select(switchOK, sp, oldSp).toVar();
    const target = select(switchOK, tScale, float(0)).toVar();
    const prev = select(newSp.equal(oldSp), cur, float(0));
    const prevS = select(uSnap.equal(uint(1)), target, prev).toVar();
    const packed = select(newSp.equal(uint(0)), float(0), float(newSp).add(tint));
    // on slopes the base sinks so its downhill edge touches the ground (lifeModel.baseSink)
    const sink = slope.mul(asF(spFoot.element(newSp))).mul(target);
    state.element(s).assign(vec4(h.sub(sink), prevS, target, packed));

    If(isF.not().and(newSp.notEqual(uint(0))).and(max(prevS, target).greaterThan(0)), () => {
      const kind = asU(spKind.element(newSp));
      const idx = uint(0).toVar();
      idx.assign(atomicAdd(args.element(kind.mul(uint(5)).add(uint(1))), uint(1)));
      lists.element(asU(spBase.element(newSp)).add(idx)).assign(s);
    });
  })().compute(NSLOT);

  // Render height per column (what the terrain mesh is displaced by, whatever space.ts smooths),
  // copied into a life-owned buffer so CPU creatures and tests read the terrain as drawn.
  const renderH = instancedArray(NCOL, 'float');
  renderH.setName('lifeRenderH');
  const renderHKernel = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const { x, z } = tColXZ(instanceIndex);
    renderH.element(instanceIndex).assign(S.corners(float(x), float(z))[0]!.surf);
  })().compute(NCOL);

  // blob instance counts = their plants' (large + small kinds only; fine kinds have no blob)
  const copyBlobArgs = Fn(() => {
    If(instanceIndex.lessThan(uint(KIND_COUNT)), () => {
      atomicStore(args.element(instanceIndex.add(uint(KIND_COUNT)).mul(uint(5)).add(uint(1))), atomicLoad(args.element(instanceIndex.mul(uint(5)).add(uint(1)))));
    });
  })().compute(KIND_COUNT);

  /** Render height (voxel y) at world x,z as the terrain mesh draws it (linear per triangle). */
  const tRenderHeight = (wx: F, wz: F): F => {
    const u = tWorldToCell(wx).toVar(), v = tWorldToCell(wz).toVar();
    const c = S.corners(u, v);
    const fu = fract(u), fv = fract(v);
    const hA = c[0]!.surf, hB = c[1]!.surf, hC = c[2]!.surf, hD = c[3]!.surf;
    return asF(select(fu.add(fv).lessThanEqual(1),
      hA.add(fu.mul(hB.sub(hA))).add(fv.mul(hC.sub(hA))),
      hD.add(float(1).sub(fu).mul(hC.sub(hD))).add(float(1).sub(fv).mul(hB.sub(hD)))));
  };

  // ---- per-frame fine ground-cover cull + compaction ----
  const fineKinds = KIND_SET.map((st, k) => (st === 2 ? k : -1)).filter((k) => k >= 0);
  const fineArr = uniformArray(fineKinds, 'uint');
  const clearFine = Fn(() => {
    If(instanceIndex.lessThan(uint(fineKinds.length)), () => {
      atomicStore(args.element(asU(fineArr.element(instanceIndex)).mul(uint(5)).add(uint(1))), uint(0));
    });
  })().compute(fineKinds.length);
  const cullKernel = Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(uint(NSLOT_F)), () => { Return(); });
    const s = i.add(uint(NSLOT_LS)).toVar();
    const st = (state.element(s) as unknown as V4).toVar();
    const sp = uint(st.w).toVar();
    If(sp.equal(uint(0)).or(max(st.y, st.z).lessThanEqual(0)), () => { Return(); });
    const { x, z } = tSlotXZ(i, uint(2), uint(GRID_F), float(BLOCK_SIZE / GRID_F));
    const pos = vec3(x, tWorldY(st.x), z).toVar();
    const keep = tLodKeep(pos.sub(uCamPos).length());
    const clip = uViewProj.mul(vec4(pos, 1)).toVar();
    const lim = clip.w.mul(1.08).add(0.06);
    const seen = clip.w.greaterThan(0).and(clip.x.abs().lessThan(lim)).and(clip.y.abs().lessThan(lim));
    const ok = uCull.equal(uint(0)).or(seen.and(tSlotRand(i, uint(2), H.LOD).lessThan(keep)));
    If(ok, () => {
      const kind = asU(spKind.element(sp));
      const idx = uint(0).toVar();
      idx.assign(atomicAdd(args.element(kind.mul(uint(5)).add(uint(1))), uint(1)));
      lists.element(asU(spBase.element(sp)).add(idx)).assign(s);
    });
  })().compute(NSLOT_F);

  // ---- meshes ----
  const group = new THREE.Group();
  group.name = 'life-flora';
  const lp = attribute('lifeP', 'vec4') as unknown as V4;
  const wind1 = (TAU * 1260) / AMB_PERIOD; // 0.35 Hz, whole cycles per AMB_PERIOD (seamless wrap)
  const wind2 = (TAU * 2268) / AMB_PERIOD; // 0.63 Hz flutter
  const gustW = (TAU * 2005) / AMB_PERIOD; // gust bands travel ~0.5 world units/s downwind
  const WD = new THREE.Vector2(1, 0.35).normalize();

  const blobs: THREE.Mesh[] = [];
  const meshes = geos.map((geo, k) => {
    const set = KIND_SET[k]!;
    const look = KIND_LOOK[k]!;
    const grid = SET_GRID[set]!;
    const slot = asU(lists.element(uint(KIND_BASE[k]!).add(instanceIndex))).toVar();
    const st = (state.element(slot) as unknown as V4).toVar();
    const local = slot.sub(uint(SET_BASE[set]!)).toVar();
    const setU = uint(set);
    const { x, z } = tSlotXZ(local, setU, uint(grid), float(BLOCK_SIZE / grid));
    const sp = uint(st.w).toVar();
    const moist = fract(st.w).div(0.99).toVar();
    const r = (ch: number) => tSlotRand(local, setU, ch);
    // lognormal-ish size variance: logistic(u) ≈ N(0, 1.8²) scaled by sigma
    const u = clamp(r(H.LOGN), 0.002, 0.998);
    const logn = clamp(exp(log(u.div(float(1).sub(u))).mul(0.55 * look.sigma)), 0.5, 2.2);
    // rocks sit half-buried by a random amount (in their own height units)
    const rockBury = k === Kind.ROCK ? r(H.TILT).mul(0.3).add(0.05) : float(0);
    const baseY = tWorldY(st.x.sub(FLORA.SINK_SET[set]!)).toVar();
    // fine ground cover: thinned with distance by the cull kernel; survivors widen to keep coverage
    const lodW = set === 2
      ? select(uCull.equal(uint(1)), float(1).div(sqrt(tLodKeep(vec3(x, baseY, z).sub(cameraPosition).length()))), float(1))
      : float(1);
    const size = asF(spSize.element(sp)).mul(mix(st.y, st.z, uBlend)).mul(logn).toVar();
    const yaw = r(H.YAW).mul(TAU).toVar();
    const cy = cos(yaw), sy = sin(yaw);
    const tall = r(H.SQUASH).mul(0.3).add(0.85);
    // fine tufts spread wider than tall (model-railway flock); survivors of distance thinning widen more
    const wide = r(H.TINT).mul(0.2).add(0.9).mul(look.wide ?? 1).mul(lodW);
    const tiltA = r(H.TILT).mul(look.tilt), tiltD = yaw.add(1.7);
    const h01 = lp.x, part = lp.y, flex = lp.z;

    const positionNode = Fn(() => {
      const p = positionGeometry;
      const lx = p.x.mul(wide).mul(size), ly = p.y.mul(tall).mul(size), lz = p.z.mul(wide).mul(size);
      const wx = x.toVar(), wz = z.toVar();
      // wind: gentle sway + gust bands travelling downwind across the meadow, + static windswept lean
      const ph = uTime.mul(wind1).add(wx.mul(3.1)).add(wz.mul(2.3));
      const along = wx.mul(WD.x).add(wz.mul(WD.y)), across = wz.mul(WD.x).sub(wx.mul(WD.y));
      const gust = pow(sin(along.mul(7).sub(uTime.mul(gustW)).add(sin(across.mul(2.3).add(along.mul(1.1))).mul(0.9))).mul(0.5).add(0.5), float(5));
      const sway = sin(ph).add(sin(uTime.mul(wind2).add(wx.mul(7.0)).sub(wz.mul(5.0))).mul(0.35));
      const bendK = flex.mul(asF(spFlex.element(sp))).mul(h01).mul(h01).mul(size).toVar();
      const push = gust.mul(0.26).add(sway.mul(0.05)).mul(bendK).add(asF(spLean.element(sp)).mul(h01).mul(h01).mul(size));
      const flutter = cos(ph.mul(0.8).add(1.3)).mul(bendK).mul(0.04);
      // random tilt as a shear along the tilt direction (cheap, keeps the base planted)
      const shear = ly.mul(tiltA);
      const ox = lx.mul(cy).sub(lz.mul(sy)).add(shear.mul(cos(tiltD)));
      const oz = lx.mul(sy).add(lz.mul(cy)).add(shear.mul(sin(tiltD)));
      // rotated normal into normalLocal (not a normalNode) so lighting AND the shadow normal bias use it
      const n = normalGeometry;
      normalLocal.assign(vec3(n.x.mul(cy).sub(n.z.mul(sy)), n.y, n.x.mul(sy).add(n.z.mul(cy))));
      return vec3(wx.add(ox).add(push.mul(WD.x)).sub(flutter.mul(WD.y)), baseY.add(ly).sub(rockBury.mul(size)), wz.add(oz).add(push.mul(WD.y)).add(flutter.mul(WD.x)));
    })();

    // Albedo per vertex: species colour, moisture drift, base→top gradient, contact AO, slot jitter.
    const colorV = varying(Fn(() => {
      const fol0 = spColor.element(sp) as unknown as V3;
      const trunk = spTrunk.element(sp) as unknown as V3;
      const tussockCol = tussock.element(uint(min(floor(r(H.PETAL).mul(3)), 2))) as unknown as V3;
      // wildflower palette per patch cell, colour within it per slot
      const pc = tLattice(int(floor(x.div(PATCH.CELL_PALETTE))), int(floor(z.div(PATCH.CELL_PALETTE))), SALT.PALETTE);
      const palI = uint(min(floor(pc.mul(PETAL_PALETTES.length)), PETAL_PALETTES.length - 1));
      const petal = petals.element(palI.mul(uint(3)).add(uint(min(floor(r(H.PETAL).mul(3)), 2)))) as unknown as V3;
      const accent = select(sp.equal(uint(Sp.CACTUS)), col3(0xff7fa8), select(uint(k).equal(uint(Kind.ROCK)), trunk, petal));
      const isTrunk = part.lessThan(0.5), isPetal = part.greaterThan(1.5).and(part.lessThan(2.5)), isBlade = part.greaterThan(2.5);
      // ~4% of temperate broadleaf trees wear autumn colours
      const autumn = sp.equal(uint(Sp.BROADLEAF)).and(r(H.PETAL).greaterThan(0.96));
      const folA = select(autumn, mix(col3(0xe0782a), col3(0xd9b13b), r(H.SCALE)), select(sp.equal(uint(Sp.TUSSOCK)), tussockCol, fol0));
      // moisture drift: straw on dry ridges, lush green in wet hollows
      const drift = mix(vec3(1.25, 1.08, 0.6), vec3(0.82, 1.07, 0.84), moist);
      const fol = folA.mul(mix(vec3(1), drift, look.moist));
      // blades: roots darken toward the ground (the species' dark root/trunk tone) so tufts blend in
      const rooted = mix(fol, trunk.mul(0.72), float(1).sub(smoothstep(0, 0.5, h01)).mul(0.7));
      const col = select(isTrunk, select(uint(k).equal(uint(Kind.ROCK)), fol0, trunk), select(isPetal, accent, select(isBlade, rooted, fol))).toVar();
      // blades: soft base→tip gradient (dark bases read as sticks at diorama scale), lighter contact shade
      const grad = select(isTrunk, mix(float(0.7), float(1.0), h01),
        select(isBlade, mix(float(0.68), float(1.14), pow(h01, float(0.8))), mix(float(0.6), float(1.12), h01)));
      const ao = k === Kind.ROCK ? mix(float(0.36), float(1), smoothstep(0.0, 0.55, h01))
        : select(isBlade, mix(float(0.72), float(1), smoothstep(0, 0.3, h01)), mix(float(0.42), float(1), smoothstep(0, 0.2, h01)));
      const value = r(H.TINT).mul(0.32).add(0.84);
      const hue = mix(vec3(0.9, 1.0, 1.1), vec3(1.14, 1.04, 0.72), r(H.PETAL));
      return col.mul(select(isPetal, float(1), grad.mul(ao))).mul(value).mul(select(isPetal, vec3(1), hue));
    })(), 'vLifeC');

    const mat = new THREE.MeshStandardNodeMaterial({ roughness: k === Kind.ROCK ? 0.95 : 0.85, metalness: 0 });
    mat.positionNode = positionNode;
    mat.colorNode = colorV;
    if (look.ds) {
      // Double-sided blades: an explicit view normal skips three's back-face negation, so both
      // sides light like the (mostly up-facing) blade normal instead of the back going dark.
      mat.side = THREE.DoubleSide;
      const n = normalGeometry;
      mat.normalNode = transformNormalToView(varying(vec3(n.x.mul(cy).sub(n.z.mul(sy)), n.y, n.x.mul(sy).add(n.z.mul(cy))), 'vLifeN')).normalize();
    }
    geo.setIndirect(argsAttr, k * 20);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = `life-${KIND_NAMES[k]}`;
    mesh.frustumCulled = false;
    mesh.castShadow = look.shadow;
    mesh.receiveShadow = true;
    group.add(mesh);

    // contact shadow: a soft dark disc hugging the terrain under the plant (grounds it in any light)
    const [bx, bz, bOpacity] = KIND_BLOB[k]!;
    if (bOpacity > 0) {
      const buv = uv().mul(2).sub(1) as unknown as THREE.Node<'vec2'>;
      const bmat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
      bmat.positionNode = Fn(() => {
        const lx = buv.x.mul(bx).mul(size).mul(wide), lz = buv.y.mul(bz).mul(size).mul(wide);
        const wx = x.add(lx.mul(cy)).sub(lz.mul(sy)).toVar(), wz = z.add(lx.mul(sy)).add(lz.mul(cy)).toVar();
        // lifted 0.15 voxel over the terrain as drawn; the disc conforms per vertex (5×5)
        return vec3(wx, tWorldY(tRenderHeight(wx, wz).add(0.15)), wz);
      })();
      const bd = buv.length();
      bmat.colorNode = vec3(0.025, 0.035, 0.015);
      // plateau then soft edge: the dark core sits under the plant, the visible rim must still read
      // (WGSL smoothstep is undefined for low ≥ high, so write the falling edge as 1 − smoothstep)
      bmat.opacityNode = float(1).sub(smoothstep(0.35, 1, bd)).mul(bOpacity);
      bmat.polygonOffset = true;
      bmat.polygonOffsetFactor = -1;
      bmat.polygonOffsetUnits = -4;
      const bgeo = blobGeo.clone();
      bgeo.setIndirect(argsAttr, (KIND_COUNT + k) * 20);
      const blob = new THREE.Mesh(bgeo, bmat);
      blob.name = `life-${KIND_NAMES[k]}-contact`;
      blob.frustumCulled = false;
      blob.renderOrder = 1; // after the opaque pass and before the water sheet (renderOrder 2)
      group.add(blob);
      blobs.push(blob);
    }
    return mesh;
  });

  let sinceRefresh: number = FLORA.GROW_S;
  let camera: THREE.Camera | null = null;
  const vp = new THREE.Matrix4();
  const ease = (t: number) => { const u = Math.min(1, Math.max(0, t)); return u * u * (3 - 2 * u); };
  const cullFine = (renderer: THREE.WebGPURenderer) => {
    if (camera) {
      camera.updateMatrixWorld();
      vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      uViewProj.value.copy(vp);
      camera.getWorldPosition(uCamPos.value);
      uCull.value = 1;
    } else uCull.value = 0;
    renderer.compute(clearFine);
    renderer.compute(cullKernel);
  };

  return {
    object: group,
    meshes, blobs,
    kernels: { clear: clearArgs, refresh: refreshKernel, clearFine, cull: cullKernel },
    buffers: { state: state.value, lists: lists.value, args: argsAttr, renderH: renderH.value },
    refresh(renderer, snap = false) {
      uBlendNow.value = ease(sinceRefresh / FLORA.GROW_S);
      uSnap.value = snap ? 1 : 0;
      updateRenderColumns(renderer, fields); // plant bases sit on the rendered (smoothed) terrain
      renderer.compute(renderHKernel);
      renderer.compute(clearArgs);
      renderer.compute(refreshKernel);
      renderer.compute(copyBlobArgs);
      cullFine(renderer);
      sinceRefresh = snap ? FLORA.GROW_S : 0;
      uBlend.value = ease(sinceRefresh / FLORA.GROW_S);
    },
    frame(renderer, dt, ambTime) {
      sinceRefresh += dt;
      uBlend.value = ease(sinceRefresh / FLORA.GROW_S);
      uTime.value = ((ambTime % AMB_PERIOD) + AMB_PERIOD) % AMB_PERIOD;
      cullFine(renderer);
    },
    setCamera(c) { camera = c; },
    cullFine,
    setQuality(high) { uQuality.value = high ? 1 : FLORA.LOW_QUALITY; },
    setSeaLevel(y) { if (Number.isFinite(y)) uSeaLevel.value = y; },
    dispose() {
      for (const m of [...meshes, ...blobs]) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); }
      group.removeFromParent();
    },
  };
}
