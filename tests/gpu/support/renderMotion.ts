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

  let offset = 0, runId = 0, travel = 0;
  let offset1 = 0; // plate 1: stationary at this sub-cell offset (reassignment test)
  const motion: RenderMotion = {
    get runId() { return runId; },
    plateOffsets(out: Float32Array) { out.fill(0); out[0] = offset; out[2] = offset1; },
    plateTravel(out: Float64Array) { out.fill(0); out[0] = travel; out[2] = offset1; },
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
    /**
     * Run `frames` frames at dt with the plate moving `cellsPerSec`; returns rendered peak u per frame.
     * stepEvery > 1: the plate advances only every stepEvery frames (the sim runs tectonics every
     * TEC_EVERY ticks, so at normal speed plates step every few frames).
     */
    async run(frames: number, dt: number, cellsPerSec: number, foreignCalls = false, stepEvery = 1, extraRun = false) {
      snapRenderColumns(fields); updateRenderColumns(renderer, fields, 1 / 60);
      const peaks: number[] = [], heights: number[] = [], shifts: number[] = [];
      for (let f = 0; f < frames; f++) {
        // in-place uplift (isostasy / orogeny style) every 25 frames: the display must ease into it
        if (f > 0 && f % 25 === 0) { AMP += 12; writeWorld(); }
        if (f % stepEvery === 0) { offset += cellsPerSec * dt * stepEvery; travel += cellsPerSec * dt * stepEvery; }
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
          if (extraRun) {
            // a second tectonics run in the same frame that shifts nothing: 'tecAct' now holds only this
            // run's identity map (the sim at high speed runs several per frame)
            runId++;
            for (let c = 0; c < NCOL; c++) act[c] = c;
            fields.markDirty('tecAct');
          }
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
    /**
     * Water sheet along ROW for an island (Gaussian, peak 100) in a sea at level 76: `film` layers of
     * rain on every land column, and optionally a real 0.6-layer stream down the island's +x flank.
     * Returns the sampled bilinear ground height and water level (quarter cells) and the island centre.
     */
    async waterProfile(film: number, stream = false) {
      const SEA = 76, C = 100;
      offset = 0; travel = 0;
      for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
        const c = x + z * NX, dx = x - C, dz = z - ROW;
        surf[c] = 60 + 40 * Math.exp(-(dx * dx + dz * dz) / (2 * 8 * 8));
        water[c] = surf[c]! < SEA ? SEA - surf[c]! : film;
        if (stream && z === ROW && dx > 0 && surf[c]! >= SEA) water[c] = 0.6;
        pid[0]![c] = pid[1]![c] = 0;
      }
      for (const n of ['surfY', 'water', 'plateId']) fields.markDirty(n);
      snapRenderColumns(fields); updateRenderColumns(renderer, fields, 1 / 60);
      const probeW = instancedArray(SAMPLES * 2, 'float');
      const k = Fn(() => {
        const u = float(instanceIndex).div(4);
        const c = S.corners(u, float(ROW));
        probeW.element(instanceIndex.mul(2)).assign(S.height(c));
        probeW.element(instanceIndex.mul(2).add(1)).assign(S.level(c).level);
      })().compute(SAMPLES);
      renderer.compute(k);
      const a = new Float32Array(await renderer.getArrayBufferAsync(probeW.value as THREE.StorageBufferAttribute));
      const h: number[] = [], lv: number[] = [];
      for (let i = 0; i < SAMPLES; i++) { h.push(a[i * 2]!); lv.push(a[i * 2 + 1]!); }
      return { h, lv, centre: C * 4, sea: SEA };
    },
    /**
     * Plate reassignment (split / merge / majority clean): the mountain sits still on plate 0 at sub-cell
     * offset `o0`; at frame `at` its rows move to plate 1 (offset 0). Returns the rendered peak per frame.
     */
    async reassign(frames: number, dt: number, o0: number, at: number) {
      offset = o0; travel = o0; offset1 = 0;
      AMP = 20; cx = 100;
      writeWorld();
      snapRenderColumns(fields); updateRenderColumns(renderer, fields, dt);
      const peaks: number[] = [];
      for (let f = 0; f < frames; f++) {
        if (f === at) {
          for (const par of [0, 1]) for (let c = 0; c < NCOL; c++) if (pid[par]![c] === 0) pid[par]![c] = 1;
          fields.markDirty('plateId');
        }
        updateRenderColumns(renderer, fields, dt);
        peaks.push((await readPeak()).u);
      }
      return peaks;
    },
    /** Rendered peak on each 'plateId' parity (the sim swaps it every tectonics run). */
    async parityPeaks() {
      const out: number[] = [];
      for (const par of [0, 1, 0, 1]) {
        fields.setParity('plateId', par);
        snapRenderColumns(fields); // settled display: only the parity may differ between samples (not a gliding plate)
        updateRenderColumns(renderer, fields, 1 / 60);
        out.push((await readPeak()).u);
      }
      return out;
    },
  };
}

main().catch((e) => { console.error('render motion page failed: ' + (e as Error).stack); });
