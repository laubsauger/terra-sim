// T10 terrain render phase 1: heightfield mesh displaced on the GPU by `surfY`.
// Vertex heights and normals read the storage buffers directly (no 3D texture copy); reads wrap (V1),
// so interpolation across the cut edge uses the far side's columns and the world stays seamless.
import * as THREE from 'three/webgpu';
import {
  Fn, vec3, vec4, float, uniform, varying, positionGeometry, positionWorld, transformNormalToView,
  mix, smoothstep, saturate, normalize, exp, uint, mx_noise_float, mx_noise_vec3, mx_fractal_noise_float,
} from 'three/tsl';
import { NX, NZ, CELL, VOXEL_H, Y_SEA_NOMINAL, Mat } from '../sim/layout';
import { tMat } from '../sim/tslLayout';
import type { GpuFields } from '../core/gpu';
import { HALF, vertEx, tWorldY, tWorldToCell, columnSampler, voxReader, tTopVoxel } from './space';
import { createPaletteNodes, biomeColor } from './palette';

type F = THREE.Node<'float'>;

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

export function createTerrain(fields: GpuFields, opts: TerrainOptions = {}): { object: THREE.Object3D; dispose(): void } {
  const S = columnSampler(fields);
  const vox = voxReader(fields);
  const pal = createPaletteNodes();
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
  // One evaluation shared by colour and roughness.
  const shade = Fn(() => {
    const p = positionWorld;
    const c = S.corners(tWorldToCell(p.x), tWorldToCell(p.z));

    // Surface material: palette colour of each corner column's top voxel, bilinear blend (no blocky cells).
    // Steep faces show bedrock: loose SEDIMENT on top does not hold on cliffs, so they use the voxel 2 below.
    let top = vec3(0) as THREE.Node<'vec3'>;
    let rock = vec3(0) as THREE.Node<'vec3'>;
    let rockR = float(0) as F;
    for (const k of c) {
      const ty = tTopVoxel(k.surf);
      const m = tMat(vox(k.x, ty, k.z));
      const mb = tMat(vox(k.x, ty.sub(2), k.z));
      const bed = m.equal(uint(Mat.SEDIMENT)).select(mb, m);
      top = top.add(pal.color(m).mul(k.w));
      rock = rock.add(pal.color(bed).mul(k.w));
      rockR = rockR.add(pal.rough(bed).mul(k.w));
    }
    top = top.toVar();
    rock = rock.toVar();

    const h = hVary;
    const n = nVary.xyz.normalize();
    const cavity = nVary.w;
    const slope = float(1).sub(n.y).toVar();
    const alt = h.sub(seaLevel).toVar();
    const nLo = mx_fractal_noise_float(p.mul(5.0), 3, 2.0, 0.5, 1.0).toVar();
    const nHi = mx_noise_float(p.mul(70.0)).toVar();

    // Placeholder biome: green lowlands → dry uplands, forest patches, rock on steep, snow high, beach.
    const grass = mix(biomeColor('grassLush'), biomeColor('grassDry'), saturate(alt.div(26).add(nLo.mul(0.35))));
    const forestAmt = smoothstep(0.18, 0.3, mx_fractal_noise_float(p.mul(15.0).add(3.7), 2, 2.0, 0.5, 1.0))
      .mul(float(1).sub(saturate(alt.div(24))));
    const veg = mix(grass, biomeColor('forest'), forestAmt.mul(0.8));
    const rockAmt = smoothstep(0.22, 0.42, slope.add(nHi.mul(0.04)).add(nLo.mul(0.05))).toVar();
    let land = mix(veg, rock.mul(0.88), rockAmt);
    const sandAmt = float(1).sub(smoothstep(0.4, 2.0, alt.add(nHi.mul(0.35)))).mul(float(1).sub(rockAmt.mul(0.6)));
    land = mix(land, biomeColor('sand'), sandAmt);
    const snowAmt = smoothstep(snowLine.sub(3), snowLine.add(2), alt.add(nLo.mul(5)))
      .mul(float(1).sub(smoothstep(0.35, 0.6, slope))).toVar();
    land = mix(land, biomeColor('snow'), snowAmt);
    const wetBand = float(1).sub(smoothstep(0.0, 0.7, alt)).mul(sandAmt);
    land = mix(land, biomeColor('wetSand'), wetBand.mul(0.6));

    // Seabed: rock + sediment, darker and bluer with depth.
    const wl = S.level(c);
    const depth = wl.level.sub(h).toVar();
    const under = wl.wet.greaterThan(0.001).and(depth.greaterThan(0.0));
    const seabed = mix(top, biomeColor('sand'), 0.3).mul(mix(vec3(0.55, 0.75, 0.8), vec3(1), exp(depth.mul(-0.12))))
      .mul(mix(0.35, 1.0, exp(depth.mul(-0.06))));

    // Soft fake AO: valleys darken, crests catch a little extra light.
    const ao = float(1).sub(saturate(cavity.mul(0.05)).mul(0.25)).add(saturate(cavity.mul(-0.05)).mul(0.08));
    const col = under.select(seabed, land).mul(nHi.mul(0.07).add(1)).mul(nLo.mul(0.06).add(1)).mul(ao);
    const landR = mix(mix(0.88, rockR, rockAmt), 0.45, snowAmt).mul(float(1).sub(wetBand.mul(0.5)));
    return vec4(col, under.select(float(0.8), landR));
  }).once()().toVar('terrShade');

  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0 });
  mat.positionNode = positionNode;
  // Grainy detail: small 3D-noise normal perturbation (3D noise needs no triplanar blend).
  const bump = mx_noise_vec3(positionWorld.mul(90)).mul(vec3(0.12, 0, 0.12));
  mat.normalNode = normalize(transformNormalToView(nVary.xyz.normalize().add(bump)));
  mat.colorNode = shade.rgb;
  mat.roughnessNode = shade.a;

  const mesh = new THREE.Mesh(createColumnGrid(), mat);
  mesh.name = 'terrain';
  mesh.frustumCulled = false; // displaced on GPU; CPU bounds are meaningless
  return {
    object: mesh,
    dispose() { mesh.geometry.dispose(); mat.dispose(); },
  };
}
