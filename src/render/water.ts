// §T.37 water (pulled forward): flat sheet at the wet water level with animated normal-map waves on
// the ambience clock (V17), Beer-Lambert absorption over the refracted seabed (viewport texture +
// viewport depth), sky Fresnel reflection, lit in-scatter (so mountain shadows fall on the sea),
// shoreline foam from column depth, and the sun glint from the standard GGX lobe (blooms).
// Also exports the shared optics used by the glass water column on the cut faces (sides.ts).
import * as THREE from 'three/webgpu';
import {
  Fn, vec2, vec3, float, varying, positionGeometry, positionWorld, positionView,
  mix, smoothstep, saturate, pow, dot, normalize, exp, texture, reflect, transformNormalToView, abs, sin, cos, sign,
  viewportSharedTexture, viewportDepthTexture, perspectiveDepthToViewZ, cameraNear, cameraFar, screenUV, color,
} from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { tWorldY, tWorldToCell, columnSampler, ambTime, heatSampler, seamGlow, volcanoSampler } from './space';
import { viewDirWorld } from './space';
import { createColumnGrid, MAGMA_ORANGE } from './terrain';
import { lookTextures } from './textures';
import { skyColor, skyU } from './sky';
import { cloudShadowAt } from '../atmo/atmosphere';

type F = THREE.Node<'float'>;
type V2 = THREE.Node<'vec2'>;
type V3 = THREE.Node<'vec3'>;

/** Seafloor magma glow let through the sea (spreading ridges, submarine lava and vents; 0 = plain absorption). */
export const RIDGE_HALO = 1;

/** Absorption per world unit (red goes first): shallow turquoise → deep sapphire. */
export const WATER_SIGMA = new THREE.Vector3(13, 1.9, 1.05);
/** Lit in-scatter albedo of a thick water column (deep sapphire). */
export const WATER_SCATTER = new THREE.Color(0x0a4d7a);
/** In-scatter of thin water over bright sand (luminous lagoon turquoise). */
export const WATER_SHALLOW = new THREE.Color(0x2fd6c4);

/** View-space z of the opaque scene at a screen uv (the water itself does not write it yet). */
export const sceneViewZ = (uv: V2): F => perspectiveDepthToViewZ(viewportDepthTexture(uv).x, cameraNear, cameraFar) as F;

/** Transmittance through `thick` world units of water. */
export const waterTransmit = (thick: F): V3 => exp(vec3(WATER_SIGMA.x, WATER_SIGMA.y, WATER_SIGMA.z).mul(thick).negate()) as V3;

/**
 * Wind-aligned swell (world units, added to the water sheet): a few sum-of-sines trains running
 * along the zonal wind bands (direction flips per band like trades / westerlies), damped in the
 * shallows so the shore stays calm. Shared by the sheet and the glass meniscus on the cut faces.
 * depth = water depth in voxel layers. Returns (height, dh/dx, dh/dz).
 */
export function swell(xz: V2, depth: F): V3 {
  return Fn(() => {
    const band = sin(xz.y.mul(Math.PI * 2 / 4 * 3)).toVar();               // 3 wind bands across the block
    const wind = normalize(vec2(sign(band).add(float(0.001)), band.mul(0.35)));
    const damp = smoothstep(0.5, 5.0, depth);
    const h = float(0).toVar(), dx = float(0).toVar(), dz = float(0).toVar();
    // (wavelength world, amplitude world, speed rad/s, angle offset)
    for (const [lam, amp, spd, ang] of [[0.55, 0.0032, 1.1, 0], [0.31, 0.0018, 1.6, 0.5], [0.19, 0.001, 2.1, -0.6]] as const) {
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const dir = vec2(wind.x.mul(ca).sub(wind.y.mul(sa)), wind.x.mul(sa).add(wind.y.mul(ca)));
      const k = (Math.PI * 2) / lam;
      const ph = dot(xz, dir).mul(k).sub(ambTime.mul(spd));
      h.addAssign(sin(ph).mul(amp));
      const g = cos(ph).mul(amp * k);
      dx.addAssign(g.mul(dir.x)); dz.addAssign(g.mul(dir.y));
    }
    return vec3(h, dx, dz).mul(damp);
  })() as V3;
}

/**
 * How much the seafloor behind `bed` (world point where a view ray through water meets the bed) glows
 * with magma: seams, submarine lava, vents (0..1). Where it does, water passes the light neutrally
 * (see createWater); the glass water on the cut faces does the same.
 */
export function seabedGlowK(fields: GpuFields, bed: V3): F {
  const heat = heatSampler(fields)(tWorldToCell(bed.x), tWorldToCell(bed.z));
  const volc = volcanoSampler(fields)(tWorldToCell(bed.x), tWorldToCell(bed.z));
  const rg = seamGlow(volc.zw, float(1)); // the seafloor seam it lets through (at full pulse: covers it)
  const lavaK = smoothstep(0.005, 0.2, heat.y).mul(smoothstep(600, 950, heat.z));
  return saturate(rg.x.mul(1.2).add(rg.y.mul(2)).add(lavaK).add(smoothstep(0.02, 0.4, volc.x))).mul(RIDGE_HALO) as F;
}

export function createWater(fields: GpuFields): { object: THREE.Mesh; dispose(): void } {
  const S = columnSampler(fields);
  const tex = lookTextures();
  const gu = tWorldToCell(positionGeometry.x), gv = tWorldToCell(positionGeometry.z);
  const levelAt = () => S.level(S.corners(gu, gv)).level;
  // Depth of the sheet above the (interpolated) seabed; negative where the flat level runs into land.
  const depthVary = varying(Fn(() => {
    const c = S.corners(gu, gv);
    return S.level(c).level.sub(S.height(c));
  })(), 'vWaterDepth');
  // Fraction of wet corner columns: < 1 only within a cell of dry land → the real shoreline.
  // Share of dry-land corner columns: > 0 only within a cell of the true coastline.
  const dryVary = varying(Fn(() => S.level(S.corners(gu, gv)).dry)(), 'vWaterDry');

  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0, transparent: true, depthWrite: true });
  const swellV = varying(swell(positionGeometry.xz, depthVary), 'vSwell'); // (h, dh/dx, dh/dz)
  mat.positionNode = Fn(() => vec3(positionGeometry.x, tWorldY(levelAt()).add(swellV.x), positionGeometry.z))();

  const p = positionWorld;
  const t = ambTime;
  // Two scrolled wave layers (speeds are k/3600 texture periods per second → seamless ambTime wrap).
  // Small, calm ripples: at diorama scale big normal-map waves read as noise.
  const wA = texture(tex.waves, p.xz.mul(1.3).add(vec2(t.mul(0.01), t.mul(0.005)))).toVar('wA');
  const wB = texture(tex.waves, p.xz.mul(3.4).add(vec2(t.mul(-0.0075), t.mul(0.0125)))).toVar('wB');
  // Calm the normals with distance and at grazing angles (where they would only alias into noise).
  const calm = float(1).div(positionView.length().mul(0.12).add(1))
    .mul(smoothstep(0.02, 0.35, abs(dot(viewDirWorld, vec3(0, 1, 0)))));
  // Lagoons are calm: ripple normals fade out over shallow flats (no sparkle carpet).
  const shallowCalm = mix(float(0.3), float(1), smoothstep(0.3, 4.0, depthVary));
  const slope = wA.rg.sub(0.5).mul(0.13).add(wB.rg.sub(0.5).mul(0.07)).mul(calm).mul(shallowCalm).toVar('wSlope');
  const nW = normalize(vec3(slope.x.add(swellV.y).negate(), 1, slope.y.add(swellV.z).negate())).toVar('wN');

  const d = depthVary.max(0);
  const V = viewDirWorld; // camera → fragment
  const ndv = saturate(dot(nW, V.negate()));
  // Capped: without SSR the grazing mirror would show only the hazy horizon band.
  // Thin sheets over flats read as wet ground, not a mirror: Fresnel fades in with depth.
  const fres = float(0.02).add(pow(float(1).sub(ndv), 5).mul(0.98)).min(0.5).mul(smoothstep(0.0, 0.9, depthVary)).toVar('wFres');

  // Thickness along the view ray from the viewport depth; refraction offset scaled by it, and
  // rejected where the offset sample lands in front of the water (land in the foreground).
  const fragZ = positionView.z;
  const rayScale = positionView.length().div(fragZ.negate().max(1e-4)).toVar('wRay');
  const thick0 = fragZ.sub(sceneViewZ(screenUV)).max(0).mul(rayScale).toVar('wThick0');
  const uvR = screenUV.add(slope.mul(0.06).mul(saturate(thick0.mul(6)))).toVar('wUvR');
  const zR = sceneViewZ(uvR).toVar('wZR');
  const ok = zR.lessThan(fragZ);
  const okF = float(ok); // branch-free (see BRANCH-FREE NOTE in space.ts)
  const uvF = mix(screenUV, uvR, okF);
  const thick = mix(thick0, fragZ.sub(zR).mul(rayScale), okF).max(0).toVar('wThick');
  const refr = viewportSharedTexture(uvF).rgb;
  const T = waterTransmit(thick).toVar('wT');

  // Foam: shoreline band from the column depth, broken up by noise and the wave height.
  const shoreN = texture(tex.detail, p.xz.mul(3.0).add(vec2(t.mul(0.005), 0))).b;
  // Swash: the foam band surges up the beach and retreats (two out-of-phase sets, noise-staggered).
  const surge = sin(t.mul(0.9).add(shoreN.mul(5))).mul(0.5).add(0.5).mul(sin(t.mul(0.37).add(1.3)).mul(0.25).add(0.75));
  // Foam hugs the shoreline (next to dry columns), not every thin sheet over a flat.
  const nearLand = smoothstep(0.02, 0.3, dryVary);
  const shore = float(1).sub(smoothstep(0.0, mix(0.35, 1.1, surge), d.add(shoreN.mul(0.8)).sub(wB.b.mul(0.3)))).mul(nearLand);
  const foam = saturate(shore.mul(0.95)).toVar('wFoam');

  // ripples tilt reflections upward: sample the sky a little above the mirror direction
  const refl = skyColor(normalize(reflect(V, nW).add(vec3(0, 0.12, 0))) as V3, false);
  const lum = float(1).sub(T.g); // how much of the column scatters instead of transmitting
  const body = mix(color(WATER_SHALLOW), color(WATER_SCATTER), smoothstep(0.05, 0.45, thick));
  const scatter = body.mul(lum).mul(float(1).sub(fres));
  // A little self-lit body colour keeps the sea luminous on the shadow side and at dusk.
  const glow = body.mul(lum).mul(skyU.sunIntensity.mul(0.03).add(0.02));
  // Magma glow on the seafloor (terrain.ts: spreading ridges, submarine lava and vents) seen through
  // the sea. Real absorption takes red first, so orange would arrive green: where the bed glows, its
  // light is attenuated neutrally with the path length instead (darker and a little greyer with depth,
  // never green), and the cyan body in front of it thins out so the vent does not read blue. The glow
  // is looked up where the view ray meets the bed (parallax-correct), so nothing floats on the surface.
  const glowK = seabedGlowK(fields, p.add(V.mul(thick))).toVar('wGlowK');
  const tN = exp(thick.mul(-3.0)); // neutral transmittance of the glow's light
  const Tg = mix(T, vec3(tN), glowK);
  const refrG = mix(refr, vec3(dot(refr, vec3(0.3, 0.59, 0.11))), float(1).sub(tN).mul(0.15).mul(glowK));
  const thin = float(1).sub(glowK.mul(0.9)); // less cyan body in front of the glow
  // faint warm scatter in the water column, from the bed below (dies out with depth)
  const warmScatter = vec3(...MAGMA_ORANGE).mul(glowK.mul(tN).mul(0.12));
  const waterEm = refrG.mul(Tg).mul(float(1).sub(fres)).add(glow.mul(thin)).add(warmScatter).add(refl.mul(fres).mul(0.8));
  mat.emissiveNode = waterEm.mul(float(1).sub(foam));
  mat.colorNode = mix(scatter.mul(thin), vec3(0.92, 0.95, 0.97), foam);
  mat.roughnessNode = mix(float(0.13), float(0.85), foam); // not glassier: the sun glint would blow out into glare
  mat.normalNode = transformNormalToView(nW);
  mat.receivedShadowNode = Fn(([s]: [F]) => s.mul(cloudShadowAt(positionWorld))) as unknown as () => THREE.Node;
  // Where the flat level is below the land, the depth test already hides it; drop it outright.
  mat.maskNode = depthVary.greaterThan(-0.05);

  const mesh = new THREE.Mesh(createColumnGrid(), mat);
  mesh.name = 'water';
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;
  mesh.receiveShadow = true;
  return { object: mesh, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}

