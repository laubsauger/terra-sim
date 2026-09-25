// Look-dev page (lookdev.html): worldgen + one derive pass, then render only — no sim ticking, so
// sim work in progress never blocks look development.
// URL: ?seed=<u32>&quality=low|high&tod=<0..1>&day=<s per day>&t=<frozen ambTime s>&view=<name>&vx=<vert. exag.>
//      &tm=aces|agx|neutral&feat=-ssgi,+dof…&hud=0&warm=<climate+biome warm-up ticks, default 400>
// window.look: { view(name), setTimeOfDay(t), setHighQuality(b), look, stage } for screenshots.
import type * as THREE from 'three/webgpu';
import { checkWebGPU, showErrorScreen } from './ui/errorScreen';
import { createStage } from './render/stage';
import { createPerfHud } from './ui/perfHud';
import { GpuFields } from './core/gpu';
import { registerSimFields, uploadWorld } from './sim/fields';
import { generateWorld } from './sim/worldgen';
import { createDerivePass } from './sim/derive';
import { createClimatePass } from './sim/climate';
import { createBiomePass } from './sim/biome';
import { MantlePass } from './sim/mantle';
import { Params } from './core/params';
import { createLook } from './render/look';
import { setVertEx, updateAllRenderColumns } from './render/space';

/** Camera presets: position, orbit target, vertical fov. */
const VIEWS: Record<string, [number[], number[], number]> = {
  hero: [[5.5, 2.7, 5.8], [0, -0.2, 0], 30],
  grazing: [[4.3, 0.42, 2.0], [0, 0.12, -0.6], 30],
  macro: [[3.0, 1.25, 3.2], [1.3, 0.12, 1.5], 30],
  side: [[1.6, -0.25, 4.9], [0.5, -0.35, 2], 32],
  top: [[0.01, 10.5, 0.01], [0, 0, 0], 30],
  // looking into the golden-hour sun (sun disc + halo over the block, glint trail on the sea)
  sun: [[4.2, 0.55, -2.0], [-1.2, 0.75, 0.6], 34],
};

async function main() {
  const q = new URLSearchParams(location.search);
  const gpu = await checkWebGPU();
  if (!gpu.ok) { showErrorScreen('WebGPU unavailable', gpu.reason); return; }
  const stage = await createStage(document.getElementById('app')!, gpu.adapter);
  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  const world = generateWorld(Number(q.get('seed') ?? 1));
  uploadWorld(fields, world);
  createDerivePass(fields).run(stage.renderer);
  // Settle climate + biome/veg (read-only use of the sim passes) so the ground shows real biomes.
  // No tectonics / hydrology: sim bugs do not block look-dev. ?warm=0 skips it.
  const warm = Number(q.get('warm') ?? 400);
  if (warm > 0) {
    // mantle + crust temperature initial state (plume conduits, geotherm, chamber halos) for the cut faces
    new MantlePass(fields, new Params(), world).init(stage.renderer);
    const climate = createClimatePass(fields, new Params());
    const biome = createBiomePass(fields);
    for (let i = 0; i < warm; i++) { climate.step(stage.renderer); biome.step(stage.renderer); }
  }
  setVertEx(Number(q.get('vx') ?? 1.0)); // main.ts follows the 'verticalExaggeration' param (default 1.0)

  const look = createLook(stage, fields, {
    highQuality: q.get('quality') !== 'low',
    timeOfDay: q.has('tod') ? Number(q.get('tod')) : undefined,
    dayLength: Number(q.get('day') ?? 0),
    // ?feat=-ssgi,-dof,+traa … toggles single post passes for perf triage
    toneMapping: ({ aces: 4, agx: 6, neutral: 7 } as Record<string, number>)[q.get('tm') ?? ''] as THREE.ToneMapping | undefined,
    postFeatures: Object.fromEntries((q.get('feat') ?? '').split(',').filter(Boolean).map((f) => [f.slice(1), f[0] === '+'])),
  });
  const frozenT = q.has('t') ? Number(q.get('t')) : null;
  const t0 = performance.now();
  const hud = q.get('hud') === '0' ? null : createPerfHud(stage.renderer, () => fields.bytes());
  const ambNow = () => frozenT ?? (performance.now() - t0) / 1000;
  stage.onFrame((dt) => {
    look.frame(ambNow());
    hud?.frame(dt, stage.cpuMs);
  });

  /**
   * GPU-bound frame time without vsync: render n frames back to back and wait for the queue.
   * (Summed per-pass timestamps overcount on tiled GPUs, so this is the honest number.)
   */
  const device = (stage.renderer.backend as unknown as { device: GPUDevice }).device;
  const bench = async (n = 120): Promise<number> => {
    stage.renderer.setAnimationLoop(null);
    // Outside the animation loop the node frame does not advance, so FRAME-updated passes
    // (scene pass, SSGI, TRAA, bloom) would be skipped; step it like Animation does.
    const r = stage.renderer as unknown as { _nodes: { nodeFrame: { update(): void; frameId: number } }; info: { frame: number } };
    const once = () => {
      r._nodes.nodeFrame.update(); r.info.frame = r._nodes.nodeFrame.frameId;
      look.frame(ambNow()); updateAllRenderColumns(stage.renderer); look.post.render();
    };
    for (let i = 0; i < 10; i++) once();
    await device.queue.onSubmittedWorkDone();
    const t = performance.now();
    for (let i = 0; i < n; i++) once();
    await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t) / n;
    stage.start();
    return ms;
  };

  const view = (name: string) => {
    const v = VIEWS[name];
    if (!v) throw new Error(`unknown view '${name}'`);
    stage.camera.position.fromArray(v[0]);
    stage.controls.target.fromArray(v[1]);
    stage.camera.fov = v[2];
    stage.camera.updateProjectionMatrix();
    stage.controls.update();
  };
  view(q.get('view') ?? 'hero');
  stage.start();

  const w = window as unknown as Record<string, unknown>;
  w.look = {
    look, stage, fields, view, bench,
    setTimeOfDay: (t: number) => look.setTimeOfDay(t),
    setHighQuality: (b: boolean) => look.setHighQuality(b),
  };
  w.lookReady = true;
}

main().catch((e) => { console.error('lookdev failed: ' + (e as Error).stack); });
