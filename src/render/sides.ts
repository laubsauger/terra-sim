// T11 side cut render: 4 vertical faces + bottom at the block edges. Each fragment reads the voxel
// behind it straight from the 'vox' storage buffer. Above the column surface → discard, so the face
// silhouette follows the terrain exactly (same bilinear surfY as the terrain mesh edge).
import * as THREE from 'three/webgpu';
import {
  Fn, vec3, vec4, float, int, uniform, positionGeometry, positionWorld, Discard,
  mix, smoothstep, saturate, sin, floor, min, pow, hash, mx_noise_float, mx_noise_vec3,
} from 'three/tsl';
import { NY, CELL, Mat, Y_MANTLE_TOP, Y_SEA_NOMINAL } from '../sim/layout';
import { tMat } from '../sim/tslLayout';
import type { GpuFields } from '../core/gpu';
import { HALF, tWorldY, tVoxelY, tWorldToCell, tWorldToColumn, columnSampler, voxReader, tTopVoxel } from './space';
import { createPaletteNodes, biomeColor } from './palette';

/** Five quads; y in voxel units (0..NY), mapped to world by positionNode so vertEx applies. */
function createSideGeometry(): THREE.BufferGeometry {
  const H = NY;
  // [corner a, corner b, corner c, corner d] counter-clockwise seen from outside, + outward normal
  const quads: [number[][], number[]][] = [
    [[[HALF, 0, HALF], [HALF, 0, -HALF], [HALF, H, -HALF], [HALF, H, HALF]], [1, 0, 0]],
    [[[-HALF, 0, -HALF], [-HALF, 0, HALF], [-HALF, H, HALF], [-HALF, H, -HALF]], [-1, 0, 0]],
    [[[-HALF, 0, HALF], [HALF, 0, HALF], [HALF, H, HALF], [-HALF, H, HALF]], [0, 0, 1]],
    [[[HALF, 0, -HALF], [-HALF, 0, -HALF], [-HALF, H, -HALF], [HALF, H, -HALF]], [0, 0, -1]],
    [[[-HALF, 0, -HALF], [HALF, 0, -HALF], [HALF, 0, HALF], [-HALF, 0, HALF]], [0, -1, 0]],
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

export function createSides(fields: GpuFields, opts: { seaLevel?: number } = {}): { object: THREE.Object3D; dispose(): void } {
  const S = columnSampler(fields);
  const vox = voxReader(fields);
  const pal = createPaletteNodes();
  const seaLevel = uniform(opts.seaLevel ?? Y_SEA_NOMINAL);
  const geo = createSideGeometry();
  const positionNode = vec3(positionGeometry.x, tWorldY(positionGeometry.y), positionGeometry.z);

  // ---- rock faces ----
  // info = (mat, voxel y, depth below silhouette, column water); computed once, used by colour + emissive.
  const info = Fn(() => {
    const p = positionWorld;
    const vy = tVoxelY(p.y).toVar();
    const surfH = S.height(S.corners(tWorldToCell(p.x), tWorldToCell(p.z))).toVar();
    Discard(vy.greaterThan(surfH));
    // Jitter the lookup by ~half a voxel so layer boundaries read as organic wavy lines, not voxel stairs.
    const j = mx_noise_vec3(vec3(p.x.mul(12), vy.mul(0.3), p.z.mul(12)));
    const cx = tWorldToColumn(p.x.add(j.x.mul(CELL * 0.35))), cz = tWorldToColumn(p.z.add(j.z.mul(CELL * 0.35)));
    // Interpolated silhouette can sit above this column's own top voxel: show its top solid layer there.
    const y = int(min(floor(vy.add(j.y.mul(0.55))), float(tTopVoxel(S.surfAt(cx, cz)))));
    const m = tMat(vox(cx, y, cz));
    return vec4(m.toFloat(), vy, surfH.sub(vy), S.waterAt(cx, cz));
  }).once()().toVar('sideInfo');

  const mId = int(info.x);
  const vy = info.y, depth = info.z, colWater = info.w;
  const p = positionWorld;

  const color = Fn(() => {
    const base = pal.color(mId);
    const band = pal.band(mId);
    // Strata: wavy fine bands + streaks + per-layer jitter so stacked layers of one rock still read.
    const warp = mx_noise_float(vec3(p.x.mul(9), vy.mul(0.08), p.z.mul(9))).toVar();
    const b1 = sin(vy.add(warp.mul(2.5)).mul(2.2));
    const b2 = mx_noise_float(vec3(p.x.mul(4), vy.mul(1.7), p.z.mul(4)));
    const jit = hash(floor(vy).add(mId.toFloat().mul(31.7)));
    let col = base.mul(float(1).add(band.mul(b1.mul(0.55).add(b2.mul(0.6))))).mul(jit.mul(0.1).add(0.95));
    // Grain speckle, stronger in crystalline rock.
    col = col.mul(mx_noise_float(vec3(p.x.mul(160), vy.mul(3), p.z.mul(160))).mul(0.06).add(1));
    // Darken gently with depth (deep crust reads as "inside").
    col = col.mul(mix(1.0, 0.76, saturate(depth.div(70))));
    // Diorama cake rim on dry land: grass lip over a soil band; snow when high; wet sediment under water.
    const dry = colWater.lessThan(0.01);
    const alt = vy.add(depth).sub(seaLevel);
    const lip = float(0.55).add(warp.mul(0.25));
    const soil = float(2.2).add(warp.mul(0.9));
    const topCol = mix(biomeColor('grassLush'), biomeColor('snow'), smoothstep(27, 32, alt.add(warp.mul(4))));
    const rimLand = depth.lessThan(lip).select(topCol, mix(biomeColor('soil'), col, smoothstep(soil.sub(0.8), soil, depth)));
    const rimWet = mix(biomeColor('sand'), col, smoothstep(0.6, 1.6, depth.add(warp.mul(0.4))));
    const beach = alt.lessThan(1.2);
    col = depth.lessThan(soil).select(dry.and(beach.not()).select(rimLand, rimWet), col);
    // Peridotite: olive in the mantle top, heating to a deep glow towards the bottom.
    const hot = mId.equal(int(Mat.PERIDOTITE)).select(pow(saturate(float(1).sub(vy.div(Y_MANTLE_TOP))), 1.6), float(0));
    return mix(col, col.mul(vec3(1.2, 0.55, 0.3)), hot.mul(0.7));
  })().toVar('sideColor');

  const emissive = Fn(() => {
    const flicker = mx_noise_float(vec3(p.x.mul(14), vy.mul(0.6), p.z.mul(14))).mul(0.35).add(0.9);
    const hot = mId.equal(int(Mat.PERIDOTITE))
      .select(pow(saturate(float(1).sub(vy.div(Y_MANTLE_TOP))), 2.2), float(1));
    // Magma: cartoon lava, bright cells with darker cooling veins.
    const veins = mix(0.45, 1.0, smoothstep(-0.25, 0.25, mx_noise_float(vec3(p.x.mul(40), vy.mul(1.2), p.z.mul(40)))));
    const lava = mId.equal(int(Mat.MAGMA)).select(veins, float(1));
    // Faint display lighting so faces turned away from the sun still read as strata, not black.
    return pal.emissive(mId).mul(flicker).mul(hot).mul(lava).add(color.mul(0.1));
  })();

  const rockMat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  rockMat.positionNode = positionNode;
  rockMat.colorNode = color;
  rockMat.roughnessNode = pal.rough(mId);
  rockMat.emissiveNode = emissive;
  const rock = new THREE.Mesh(geo, rockMat);
  rock.name = 'sides';
  rock.frustumCulled = false;

  // ---- water on the faces: between surfY and the wet water level, translucent ----
  const wInfo = Fn(() => {
    const q = positionWorld;
    const wy = tVoxelY(q.y).toVar();
    const c = S.corners(tWorldToCell(q.x), tWorldToCell(q.z));
    const wl = S.level(c);
    Discard(wy.lessThan(S.height(c)).or(wy.greaterThan(wl.level)).or(wl.wet.lessThan(0.001)));
    return wl.level.sub(wy);
  }).once()().toVar('sideWaterDepth');
  const wd = wInfo;
  const waterMat = new THREE.MeshStandardNodeMaterial({ metalness: 0, roughness: 0.12, transparent: true, depthWrite: false });
  waterMat.positionNode = positionNode;
  const surfLine = float(1).sub(smoothstep(0.0, 0.45, wd));
  waterMat.colorNode = mix(mix(biomeColor('waterShallow'), biomeColor('waterSide'), saturate(wd.div(6))),
    biomeColor('waterDeep'), saturate(wd.div(28))).add(surfLine.mul(0.35));
  waterMat.opacityNode = mix(0.5, 0.88, saturate(wd.div(20))).add(surfLine.mul(0.2));
  const water = new THREE.Mesh(geo, waterMat);
  water.name = 'sides-water';
  water.frustumCulled = false;
  water.renderOrder = 1;

  const group = new THREE.Group();
  group.name = 'sides-group';
  group.add(rock, water);
  return {
    object: group,
    dispose() { geo.dispose(); rockMat.dispose(); waterMat.dispose(); },
  };
}
