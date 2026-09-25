// Display continuity test page: two plates, the "moving" one carries a Gaussian mountain at constant
// speed. The page plays the sim's side of the contract by hand (RenderMotion): sub-cell plate offset
// grows each frame; when it passes a whole cell the data shifts one cell, tecAct maps every
// destination to its source, runId increments and the offset drops by one cell. A compute probe
// evaluates the same columnSampler the terrain mesh uses along the mountain's row, so the rendered
// peak position is measured exactly (window.rm).
import * as THREE from 'three/webgpu';
import { Fn, float, instanceIndex, instancedArray } from 'three/tsl';
import { GpuFields } from '../../../src/core/gpu';
import { registerSimFields } from '../../../src/sim/fields';
import { NX, NZ, NCOL } from '../../../src/sim/layout';
import { columnSampler, updateRenderColumns, setRenderMotion, snapRenderColumns, type RenderMotion } from '../../../src/render/space';

const ROW = 128, SAMPLES = NX * 4; // probe resolution: quarter cells
const H0 = 80, SIGMA = 6;
let AMP = 20;

async function main() {
  const adapter = (await navigator.gpu.requestAdapter())!;
  const l = adapter.limits;
  const renderer = new THREE.WebGPURenderer({ requiredLimits: {
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage } } as THREE.WebGPURendererParameters);
  await renderer.init();
  const device = (renderer.backend as unknown as { device: GPUDevice }).device;
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU uncaptured error: ' + (e as GPUUncapturedErrorEvent).error.message));

  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  let cx = 100; // mountain centre column (integer, in the sim grid)
  const surf = fields.cpuArray('surfY') as Float32Array;
  const water = fields.cpuArray('water') as Float32Array;
  const pid = [fields.cpuArray('plateId', 0), fields.cpuArray('plateId', 1)] as Uint32Array[];
  const act = fields.cpuArray('tecAct') as Uint32Array;
  const writeWorld = () => {
    for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
      const c = x + z * NX;
      const dx = x - cx, dz = z - ROW;
      surf[c] = H0 + AMP * Math.exp(-(dx * dx + dz * dz) / (2 * SIGMA * SIGMA));
      water[c] = 0;
      // plate 0 carries the mountain (a band of rows), plate 1 elsewhere
      pid[0]![c] = pid[1]![c] = Math.abs(dz) < 40 ? 0 : 1;
    }
    for (const n of ['surfY', 'water', 'plateId']) fields.markDirty(n);
  };
  writeWorld();

  let offset = 0, runId = 0;
  const motion: RenderMotion = {
    get runId() { return runId; },
    plateOffsets(out: Float32Array) { out.fill(0); out[0] = offset; },
  };
  setRenderMotion(fields, motion);

  const S = columnSampler(fields);
  const probe = instancedArray(SAMPLES, 'float');
  const kProbe = Fn(() => {
    const u = float(instanceIndex).div(4);
    probe.element(instanceIndex).assign(S.height(S.corners(u, float(ROW))));
  })().compute(SAMPLES);

  const readPeak = async () => {
    renderer.compute(kProbe);
    const a = new Float32Array(await renderer.getArrayBufferAsync(probe.value as THREE.StorageBufferAttribute));
    let bi = 0; for (let i = 1; i < SAMPLES; i++) if (a[i]! > a[bi]!) bi = i;
    // parabolic refinement around the max sample
    const y0 = a[(bi + SAMPLES - 1) % SAMPLES]!, y1 = a[bi]!, y2 = a[(bi + 1) % SAMPLES]!;
    const d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) || 1);
    return { u: (bi + d) / 4, h: y1 };
  };

  (window as unknown as { rm: unknown }).rm = {
    ready: true,
    /** Run `frames` frames at dt with the plate moving `cellsPerSec`; returns rendered peak u per frame. */
    async run(frames: number, dt: number, cellsPerSec: number, foreignCalls = false) {
      snapRenderColumns(fields); updateRenderColumns(renderer, fields, 1 / 60);
      const peaks: number[] = [], heights: number[] = [], shifts: number[] = [];
      for (let f = 0; f < frames; f++) {
        // in-place uplift (isostasy / orogeny style) every 25 frames: the display must ease into it
        if (f > 0 && f % 25 === 0) { AMP += 12; writeWorld(); }
        offset += cellsPerSec * dt;
        if (offset >= 1) {
          // sim step: shift plate 0's rows one cell +x (destination d takes source d−1), offset −1
          cx += 1; offset -= 1; runId++;
          for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
            const c = x + z * NX;
            act[c] = Math.abs(z - ROW) < 40 ? ((x + NX - 1) % NX) + z * NX : c;
          }
          fields.markDirty('tecAct');
          writeWorld();
          shifts.push(f);
        }
        // another module (life/flora) asking for the display mid-frame must not snap or re-ease it
        if (foreignCalls && f % 3 === 0) updateRenderColumns(renderer, fields);
        updateRenderColumns(renderer, fields, dt);
        if (foreignCalls && f % 3 === 1) updateRenderColumns(renderer, fields);
        const p = await readPeak();
        peaks.push(p.u); heights.push(p.h);
      }
      void NCOL;
      return { peaks, heights, shifts };
    },
    /** Rendered peak on each 'plateId' parity (the sim swaps it every tectonics run). */
    async parityPeaks() {
      const out: number[] = [];
      for (const par of [0, 1, 0, 1]) {
        fields.setParity('plateId', par);
        updateRenderColumns(renderer, fields, 1 / 60);
        out.push((await readPeak()).u);
      }
      return out;
    },
  };
}

main().catch((e) => { console.error('render motion page failed: ' + (e as Error).stack); });
