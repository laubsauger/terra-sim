// Full-look test page: synthetic world through createLook (plinth, studio backdrop, sky, shadows,
// water, post pipeline) on a real Stage, driven from render.spec.ts via window.rl.
// ?quality=low|high picks the post tier. ambTime is frozen so frames are comparable.
import * as THREE from 'three/webgpu';
import { GpuFields } from '../../../src/core/gpu';
import { registerSimFields } from '../../../src/sim/fields';
import { NCOL } from '../../../src/sim/layout';
import { createStage } from '../../../src/render/stage';
import { createLook } from '../../../src/render/look';
import { BASE_W, BASE_H } from '../../../src/render/backdrop';
import { fillSynthWorld, SEA } from './synthWorld';

interface Pt { px: number; py: number }

async function main() {
  const q = new URLSearchParams(location.search);
  const adapter = (await navigator.gpu.requestAdapter())!;
  const stage = await createStage(document.getElementById('app')!, adapter);
  stage.renderer.setPixelRatio(1);
  const device = (stage.renderer.backend as unknown as { device: GPUDevice }).device;
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU uncaptured error: ' + (e as GPUUncapturedErrorEvent).error.message));

  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  const info = fillSynthWorld(fields);
  // Land gets a grassland biome so the ground uses the biome palette path.
  if (fields.names().includes('biome') && fields.names().includes('veg')) {
    const b = fields.cpuArray('biome') as Uint32Array, v = fields.cpuArray('veg') as Float32Array;
    for (let c = 0; c < NCOL; c++) { const land = info.surf[c]! > SEA; b[c] = land ? 5 : 0; v[c] = land ? 0.7 : 0; }
    fields.markDirty('biome'); fields.markDirty('veg');
  }

  const look = createLook(stage, fields, { highQuality: q.get('quality') !== 'low' });
  stage.onFrame(() => look.frame(12));
  stage.heroView();
  stage.start();

  const project = (x: number, y: number, z: number): Pt => {
    const v = new THREE.Vector3(x, y, z).project(stage.camera);
    return { px: Math.round((v.x * 0.5 + 0.5) * window.innerWidth), py: Math.round((-v.y * 0.5 + 0.5) * window.innerHeight) };
  };

  (window as unknown as { rl: unknown }).rl = {
    ready: true,
    async frames(n: number) { for (let i = 0; i < n; i++) await new Promise((r) => requestAnimationFrame(r)); await device.queue.onSubmittedWorkDone(); },
    /** Middle of the slate plinth's front (+z) face, and a backdrop pixel above the block. */
    points() {
      const floorY = look.backdrop.floorY;
      return { plinth: project(0.6, floorY + BASE_H * 0.5, BASE_W / 2), backdrop: { px: 24, py: 24 } };
    },
    /** Decode a PNG screenshot: samples (3×3 mean), global luminance stats, black / speckle counts. */
    async analyze(b64: string, pts: Pt[]) {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
      const cv = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = cv.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0);
      const W = bmp.width, H = bmp.height;
      const img = ctx.getImageData(0, 0, W, H).data;
      const lum = (x: number, y: number) => { const o = (y * W + x) * 4; return 0.2126 * img[o]! + 0.7152 * img[o + 1]! + 0.0722 * img[o + 2]!; };
      const samples = pts.map((p) => {
        const acc = [0, 0, 0];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const o = ((Math.min(H - 1, Math.max(0, p.py + dy))) * W + Math.min(W - 1, Math.max(0, p.px + dx))) * 4;
          for (let i = 0; i < 3; i++) acc[i]! += img[o + i]! / 9;
        }
        return acc.map(Math.round);
      });
      let sum = 0, sum2 = 0, n = 0, black = 0, speckles = 0;
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const o = (y * W + x) * 4;
        const l = lum(x, y);
        sum += l; sum2 += l * l; n++;
        if (Math.max(img[o]!, img[o + 1]!, img[o + 2]!) <= 2) black++;
        // NaN/Inf in the HDR chain shows up as isolated dead pixels inside lit areas
        if (l < 6 && lum(x - 1, y) > 45 && lum(x + 1, y) > 45 && lum(x, y - 1) > 45 && lum(x, y + 1) > 45) speckles++;
      }
      const mean = sum / n;
      return { samples, mean, lumStd: Math.sqrt(Math.max(0, sum2 / n - mean * mean)), blackFrac: black / n, speckles, size: [W, H] };
    },
  };
}

main().catch((e) => { console.error('render look page failed: ' + (e as Error).stack); });
