// Render test page: synthetic world + terrain/sides/water/lighting, driven from render.spec.ts via window.rt.
import * as THREE from 'three/webgpu';
import { GpuFields } from '../../../src/core/gpu';
import { registerSimFields } from '../../../src/sim/fields';
import { NX, NZ, colIdx, Mat } from '../../../src/sim/layout';
import { createTerrain } from '../../../src/render/terrain';
import { createSides, createFaceShading, createFaceQuad, setFaceTectonics } from '../../../src/render/sides';
import { createWater } from '../../../src/render/water';
import { createLighting } from '../../../src/render/lighting';
import { HALF, cellToWorld, voxelToWorldY, updateRenderColumns, Y_RENDER_BOTTOM, setRenderMotion } from '../../../src/render/space';
import { fillSynthWorld, LAND_X, OCEAN_X, SEA } from './synthWorld';

export const BG = 0x151924; // same as stage.ts

interface Pt { px: number; py: number; y?: number }

async function main() {
  const adapter = (await navigator.gpu.requestAdapter())!;
  const l = adapter.limits;
  const renderer = new THREE.WebGPURenderer({
    antialias: false,
    requiredLimits: {
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxBufferSize: l.maxBufferSize,
      maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    },
  } as THREE.WebGPURendererParameters);
  renderer.setPixelRatio(1);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();
  document.body.appendChild(renderer.domElement);
  const device = (renderer.backend as unknown as { device: GPUDevice }).device;
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU uncaptured error: ' + (e as GPUUncapturedErrorEvent).error.message));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 100);

  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  const info = fillSynthWorld(fields);

  createLighting(scene);
  const sides = createSides(fields).object;
  scene.add(createTerrain(fields).object, sides, createWater(fields).object);
  let cap: ReturnType<typeof createFaceQuad> | null = null;

  const project = (x: number, y: number, z: number): Pt => {
    const v = new THREE.Vector3(x, y, z).project(camera);
    return { px: Math.round((v.x * 0.5 + 0.5) * window.innerWidth), py: Math.round((-v.y * 0.5 + 0.5) * window.innerHeight) };
  };

  const frame = async () => {
    await new Promise((r) => requestAnimationFrame(r));
    updateRenderColumns(renderer, fields, 1e9); // display columns, fully settled (the stage loop eases them in the app)
    renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
    await new Promise((r) => requestAnimationFrame(r));
    await new Promise((r) => requestAnimationFrame(r));
  };

  const views: Record<string, () => void> = {
    beauty: () => { camera.fov = 35; camera.position.set(3.2, 2.4, 3.2); camera.lookAt(0, -0.15, 0); },
    corner: () => { camera.fov = 30; camera.position.set(-3.6, 1.5, 3.9); camera.lookAt(0, -0.2, 0); },
    close: () => { camera.fov = 30; camera.position.set(0.2, 0.9, 3.4); camera.lookAt(-0.6, 0.0, 1.6); },
    top: () => { camera.fov = 35; camera.position.set(0.01, 7.5, 0.01); camera.lookAt(0, 0, 0); },
  };

  // Horizontal camera facing the +Z cut face (z = HALF, column z = NZ-1).
  const landSurf = info.surf[colIdx(LAND_X, NZ - 1)]!;
  const oceanSurf = info.surf[colIdx(OCEAN_X, NZ - 1)]!;
  const ySil = SEA + 0.5 * (landSurf - SEA); // above the sea everywhere at OCEAN_X, below the land crest
  const front = () => {
    const yw = voxelToWorldY(ySil);
    camera.fov = 28; camera.position.set(0, yw, 9); camera.lookAt(0, yw, 0);
  };

  const w = window as unknown as { rt: unknown };
  w.rt = {
    ready: true,
    meta: { landSurf, oceanSurf, ySil, sea: SEA },
    async view(name: string) {
      if (name === 'front') front(); else views[name]!();
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
      await frame();
      if (name !== 'front') return {};
      const lx = cellToWorld(LAND_X), ox = cellToWorld(OCEAN_X);
      const strata: Pt[] = [];
      // the slice is drawn from Y_RENDER_BOTTOM up (render-only crop of the lower asthenosphere)
      for (let y = Y_RENDER_BOTTOM + 4; y < landSurf - 4; y += 5) strata.push({ ...project(lx, voxelToWorldY(y + 0.5), HALF), y });
      return {
        strata,
        skyOverOcean: project(ox, voxelToWorldY(ySil), HALF),
        rockUnderLand: project(lx, voxelToWorldY(ySil), HALF),
      };
    },
    setVoxParity(p: number) { fields.setParity('vox', p); },
    /** Copy the real world into the decoy buffer too, so both ping-pong halves hold the same voxels. */
    mirrorVox() {
      (fields.cpuArray('vox', 0) as Uint32Array).set(fields.cpuArray('vox', 1) as Uint32Array);
      fields.markDirty('vox');
    },
    /** Fill the rows next to the +Z face row (z = NZ-2 and z = 0, across the seam) with MAGMA below the surface. */
    markNeighbourRows() {
      const vox = fields.cpuArray('vox', 1) as Uint32Array;
      for (const z of [NZ - 2, 0]) for (let x = 0; x < 256; x++) for (let y = 0; y < 128; y++) {
        const i = x + z * 256 + y * 65536;
        if ((vox[i]! & 0xff) !== 0) vox[i] = (vox[i]! & ~0xff) | Mat.MAGMA;
      }
      fields.markDirty('vox');
    },
    /** Fake plate motion: every plate at sub-cell offset (ox, oz) cells (the sim's RenderMotion contract). */
    setPlateOffset(ox: number, oz: number) {
      setRenderMotion(fields, { runId: 0, plateOffsets(out: Float32Array) { for (let i = 0; i < out.length; i += 2) { out[i] = ox; out[i + 1] = oz; } } });
    },
    /**
     * Slice-cap stand-in: the outer faces hidden, a movable cut quad (slice.ts's caps) on the +Z face
     * plane instead, i.e. the same column row the outer face shows.
     */
    showCap(on: boolean) {
      if (!cap) {
        cap = createFaceQuad(createFaceShading(fields), 'test-cap');
        const z = HALF - 1e-4;
        cap.set(-HALF, z, HALF, z, 0, 1);
        scene.add(cap.object);
      }
      cap.object.visible = on; sides.visible = !on;
    },
    /** Two plates split at column `split` (x); plate velocities (cells/My) and the Tectonics face layer. */
    setPlates(split: number, v0: [number, number], v1: [number, number], opacity: number) {
      for (const par of [0, 1]) {
        const pid = fields.cpuArray('plateId', par as 0 | 1) as Uint32Array;
        for (let c = 0; c < pid.length; c++) pid[c] = (c % NX) < split ? 0 : 1;
      }
      fields.markDirty('plateId');
      setFaceTectonics(opacity, [{ x: v0[0], y: v0[1], z: 1, w: 0 }, { x: v1[0], y: v1[1], z: 1, w: 0 }] as THREE.Vector4[]);
    },
    /** Screen point of a face-plane position (x world, voxel y, on the +Z face). */
    project(x: number, vy: number) { return project(x, voxelToWorldY(vy), HALF); },
    /** Render one frame (after a parity change) without moving the camera. */
    async frame() { await frame(); },
    /** Decode a PNG screenshot and sample it (3×3 mean around each point) + global stats. */
    async analyze(b64: string, pts: Pt[]) {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
      const cv = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = cv.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0);
      const img = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const px = (x: number, y: number) => { const o = (y * bmp.width + x) * 4; return [img[o]!, img[o + 1]!, img[o + 2]!]; };
      const samples = pts.map((p) => {
        const acc = [0, 0, 0];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const c = px(Math.min(bmp.width - 1, Math.max(0, p.px + dx)), Math.min(bmp.height - 1, Math.max(0, p.py + dy)));
          for (let i = 0; i < 3; i++) acc[i]! += c[i]! / 9;
        }
        return acc.map(Math.round);
      });
      let sum = 0, sum2 = 0, nonBg = 0, n = 0;
      const b0 = px(2, 2); // top-left corner is always background in every view
      for (let y = 0; y < bmp.height; y += 2) for (let x = 0; x < bmp.width; x += 2) {
        const c = px(x, y);
        const lum = 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
        sum += lum; sum2 += lum * lum; n++;
        if (Math.abs(c[0]! - b0[0]!) + Math.abs(c[1]! - b0[1]!) + Math.abs(c[2]! - b0[2]!) > 24) nonBg++;
      }
      const mean = sum / n;
      return { samples, bgPixel: b0, lumStd: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), nonBgFrac: nonBg / n, size: [bmp.width, bmp.height] };
    },
  };
}

main().catch((e) => { console.error('render test page failed: ' + (e as Error).stack); });
