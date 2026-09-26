// Atmosphere test page: worldgen (+ optional sim spin-up for a natural climate), the look pipeline
// (or bare lighting with ?look=0) and createAtmosphere. Driven from atmo.spec.ts via window.at;
// frames render on demand.
// URL: ?seed=<u32>&world=sim|paint&ticks=<n>&look=0|1&quality=low|high&shadow=0|1
import * as THREE from 'three/webgpu';
import { Fn, positionWorld, instanceIndex, instancedArray, vec3, uniformArray } from 'three/tsl';
import { createStage } from '../../../src/render/stage';
import { GpuFields } from '../../../src/core/gpu';
import { Params } from '../../../src/core/params';
import { registerSimFields, uploadWorld } from '../../../src/sim/fields';
import { generateWorld } from '../../../src/sim/worldgen';
import { createDerivePass } from '../../../src/sim/derive';
import { Sim } from '../../../src/sim/sim';
import { NCOL, NX } from '../../../src/sim/layout';
import { createLook, type Look } from '../../../src/render/look';
import { createLighting } from '../../../src/render/lighting';
import { createTerrain } from '../../../src/render/terrain';
import { createSides } from '../../../src/render/sides';
import { createWater } from '../../../src/render/water';
import { updateRenderColumns, cellToWorld } from '../../../src/render/space';
import { createAtmosphere, cloudShadowAt, cloudShadowU } from '../../../src/atmo/atmosphere';
import { ATMO } from '../../../src/atmo/atmoModel';
import { paintWeather, type Paint } from './atmoWorld';

async function main() {
  const q = new URLSearchParams(location.search);
  const adapter = (await navigator.gpu.requestAdapter())!;
  const stage = await createStage(document.getElementById('app')!, adapter);
  const { renderer, scene, camera } = stage;
  const device = (renderer.backend as unknown as { device: GPUDevice }).device;
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU uncaptured error: ' + (e as GPUUncapturedErrorEvent).error.message));

  const fields = new GpuFields();
  registerSimFields(fields);
  fields.freeze();
  const world = generateWorld(Number(q.get('seed') ?? 1));
  uploadWorld(fields, world);
  const mode = q.get('world') ?? 'sim';
  let sim: Sim | null = null;
  if (mode === 'sim') {
    sim = new Sim(renderer, fields, world, new Params(), 0.05);
    const ticks = Number(q.get('ticks') ?? 400);
    for (let n = 0; n < ticks;) { n += sim.runTicks(Math.min(20, ticks - n)); await new Promise((r) => setTimeout(r, 0)); }
  } else {
    createDerivePass(fields).run(renderer);
  }

  let look: Look | null = null;
  if (q.get('look') !== '0') {
    try { look = createLook(stage, fields, { highQuality: q.get('quality') !== 'low' }); } catch (e) { console.warn('atmo page: createLook failed, bare lighting: ' + (e as Error).message); }
  }
  if (!look) {
    renderer.shadowMap.enabled = true;
    createLighting(scene, { shadows: true });
    scene.add(createTerrain(fields).object, createSides(fields).object, createWater(fields).object);
  }
  const quality = q.get('quality') !== 'low';
  const atmo = createAtmosphere(fields, renderer, scene, camera, { highQuality: quality });
  for (const name of (q.get('hide') ?? '').split(',').filter(Boolean)) { const o = atmo.object.getObjectByName(name); if (o) o.visible = false; }

  // Integration demo (what render/ would do): terrain + water receive the cloud shadow on the sun term.
  if (q.get('shadow') !== '0') {
    const recv = Fn(([s]: [THREE.Node<'float'>]) => s.mul(cloudShadowAt(positionWorld)));
    scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.NodeMaterial | undefined;
      if (!m || !(o as THREE.Mesh).receiveShadow || o.parent?.name === 'atmosphere') return;
      (m as unknown as { receivedShadowNode: unknown }).receivedShadowNode = recv;
      m.needsUpdate = true;
    });
  }

  let t = Number(q.get('t0') ?? 30);
  let atmoOn = true;
  const frame = async (dt = 1 / 60) => {
    t += dt;
    look?.frame(t);
    if (atmoOn) atmo.update(dt, t, sim ? { events: sim.events.log, seaLevel: sim.stats?.seaLevel } : undefined);
    updateRenderColumns(renderer, fields); // the stage loop does this in the app
    if (look) look.post.render(); else renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
  };
  const frames = async (n: number, dt = 1 / 60) => { for (let i = 0; i < n; i++) await frame(dt); await new Promise((r) => requestAnimationFrame(r)); };
  const f32 = async (n: THREE.StorageBufferNode<'vec4'> | { value: THREE.StorageBufferAttribute }) =>
    new Float32Array(await renderer.getArrayBufferAsync((n as { value: THREE.StorageBufferAttribute }).value));

  /** Weather probe grid (refreshed every 15 frames): cell centre, coverage, storm, cloud base, ground. */
  async function clouds() {
    const C = await f32(atmo.clouds.cells);
    const n = ATMO.CELLS, out = [];
    for (let c = 0; c < n * n; c++) {
      out.push({ x: ((c % n) + 0.5) / n * 4 - 2, z: (Math.floor(c / n) + 0.5) / n * 4 - 2, cov: C[c * 4]!, storm: C[c * 4 + 1]!, base: C[c * 4 + 2]!, ground: C[c * 4 + 3]! });
    }
    return out;
  }
  async function summary() { return Array.from(await f32(atmo.clouds.summary)); }
  /** Live particles by kind with positions. */
  async function particles() {
    const P0 = await f32(atmo.plumes.P0), P1 = await f32(atmo.plumes.P1), P2 = await f32(atmo.plumes.P2);
    const live: { x: number; y: number; z: number; age: number; kind: number; vx: number; vy: number; vz: number; seed: number; ceil: number }[] = [];
    for (let i = 0; i < P0.length / 4; i++) if (P0[i * 4 + 3]! < P1[i * 4 + 3]!) live.push({ x: P0[i * 4]!, y: P0[i * 4 + 1]!, z: P0[i * 4 + 2]!, age: P0[i * 4 + 3]!, kind: P2[i * 4]!, vx: P1[i * 4]!, vy: P1[i * 4 + 1]!, vz: P1[i * 4 + 2]!, seed: P2[i * 4 + 1]!, ceil: P2[i * 4 + 2]! });
    const vents = await f32(atmo.plumes.vents);
    const ctr = new Int32Array(await renderer.getArrayBufferAsync(atmo.plumes.ctr.value as unknown as THREE.StorageBufferAttribute));
    return { live, ventCount: ctr[0]!, hydroCount: ctr[2]!, vents: Array.from(vents.slice(0, Math.min(ctr[0]!, 16) * 4)),
      hydro: Array.from(vents.slice(ATMO.VENTS_MAX * 4, (ATMO.VENTS_MAX + Math.min(ctr[2]!, 64)) * 4)) };
  }
  /** Live rain/snow drops: position, state, spawn column. */
  async function rain() {
    const R0 = await f32(atmo.rain.R0), R1 = await f32(atmo.rain.R1);
    const live: { x: number; y: number; z: number; snow: boolean; col: number }[] = [];
    for (let i = 0; i < R0.length / 4; i++) if (R0[i * 4 + 3]! > 0.5) live.push({ x: R0[i * 4]!, y: R0[i * 4 + 1]!, z: R0[i * 4 + 2]!, snow: R0[i * 4 + 3]! > 1.5, col: R1[i * 4 + 3]! });
    return live;
  }
  /** Evaluate the exported cloudShadowAt() at world points (the same TSL the terrain would run). */
  const MAXP = 64;
  const shadowOut = instancedArray(MAXP, 'float');
  const shadowPts = uniformArray(Array.from({ length: MAXP }, () => new THREE.Vector4()), 'vec4');
  const shadowK = Fn(() => {
    const p = (shadowPts.element(instanceIndex) as unknown as THREE.Node<'vec4'>).xyz;
    shadowOut.element(instanceIndex).assign(cloudShadowAt(vec3(p)));
  })().compute(MAXP);
  async function shadowAt(pts: [number, number, number][]) {
    pts.forEach((p, i) => (shadowPts.array[i] as THREE.Vector4).set(p[0], p[1], p[2], 0));
    renderer.compute(shadowK);
    return Array.from(new Float32Array(await renderer.getArrayBufferAsync(shadowOut.value as unknown as THREE.StorageBufferAttribute))).slice(0, pts.length);
  }

  /** GPU ms per frame: atmosphere compute alone, whole frame with and without the atmosphere. */
  async function perf(n = 60) {
    const hasTs = renderer.hasFeature('timestamp-query');
    if (!hasTs) return { hasTs };
    const run = async (on: boolean) => {
      atmoOn = on; atmo.object.visible = on;
      await frames(10);
      await renderer.resolveTimestampsAsync('compute'); await renderer.resolveTimestampsAsync('render');
      let comp = 0, rend = 0, atmoComp = 0, wall = 0;
      for (let i = 0; i < n; i++) {
        const w0 = performance.now();
        t += 1 / 60;
        look?.frame(t);
        updateRenderColumns(renderer, fields);
        comp += (await renderer.resolveTimestampsAsync('compute')) ?? 0;
        if (on) {
          atmo.update(1 / 60, t, sim ? { events: sim.events.log } : undefined);
          atmoComp += (await renderer.resolveTimestampsAsync('compute')) ?? 0;
        }
        if (look) look.post.render(); else renderer.render(scene, camera);
        await device.queue.onSubmittedWorkDone();
        wall += performance.now() - w0;
        rend += (await renderer.resolveTimestampsAsync('render')) ?? 0;
        comp += (await renderer.resolveTimestampsAsync('compute')) ?? 0;
      }
      return { compute: comp / n, atmoCompute: atmoComp / n, render: rend / n, wall: wall / n };
    };
    const on = await run(true);
    const off = await run(false);
    atmoOn = true; atmo.object.visible = true;
    return { hasTs, width: renderer.domElement.width, height: renderer.domElement.height, on, off,
      atmoTotal: on.atmoCompute + on.render - off.render };
  }

  /**
   * GPU-bound cost of the atmosphere per frame: N back-to-back updates (all atmo compute + the cloud
   * march pass) with one wait at the end, vs the same number of bare frames with atmo hidden.
   * Reported per update (ms); the full-res composite is measured as whole-frame on/off difference.
   */
  async function perfParts(n = 30) {
    await device.queue.onSubmittedWorkDone();
    let t0 = performance.now();
    for (let i = 0; i < n; i++) { t += 1 / 60; atmo.update(1 / 60, t); }
    await device.queue.onSubmittedWorkDone();
    const update = (performance.now() - t0) / n;
    const frameLoop = async (on: boolean) => {
      atmoOn = on; atmo.object.visible = on;
      for (let i = 0; i < 5; i++) await frame();
      await device.queue.onSubmittedWorkDone();
      const s0 = performance.now();
      for (let i = 0; i < n; i++) {
        t += 1 / 60; look?.frame(t);
        if (on) atmo.update(1 / 60, t);
        updateRenderColumns(renderer, fields);
        if (look) look.post.render(); else renderer.render(scene, camera);
      }
      await device.queue.onSubmittedWorkDone();
      return (performance.now() - s0) / n;
    };
    const frameOn = await frameLoop(true), frameOff = await frameLoop(false);
    // timestamp split: atmo compute, the cloud march pass (a render pass inside update), whole post frame
    await renderer.resolveTimestampsAsync('compute'); await renderer.resolveTimestampsAsync('render');
    let tsCompute = 0, tsMarch = 0, tsFrame = 0, tsFrameOff = 0;
    for (let i = 0; i < n; i++) {
      t += 1 / 60; look?.frame(t); updateRenderColumns(renderer, fields);
      await renderer.resolveTimestampsAsync('compute'); await renderer.resolveTimestampsAsync('render');
      atmo.update(1 / 60, t);
      tsCompute += (await renderer.resolveTimestampsAsync('compute')) ?? 0;
      tsMarch += (await renderer.resolveTimestampsAsync('render')) ?? 0;
      if (look) look.post.render(); else renderer.render(scene, camera);
      tsFrame += (await renderer.resolveTimestampsAsync('render')) ?? 0;
    }
    atmo.object.visible = false;
    for (let i = 0; i < n; i++) {
      t += 1 / 60; look?.frame(t); updateRenderColumns(renderer, fields);
      await renderer.resolveTimestampsAsync('render');
      if (look) look.post.render(); else renderer.render(scene, camera);
      tsFrameOff += (await renderer.resolveTimestampsAsync('render')) ?? 0;
    }
    atmo.object.visible = true;
    atmoOn = true; atmo.object.visible = true;
    t0 = performance.now();
    return { update, frameOn, frameOff, atmoFrame: frameOn - frameOff, ts: { compute: tsCompute / n, march: tsMarch / n, frame: tsFrame / n, frameOff: tsFrameOff / n }, width: renderer.domElement.width, height: renderer.domElement.height };
  }

  const w = window as unknown as Record<string, unknown>;
  w.at = {
    frames, clouds, summary, particles, rain, shadowAt, perf, perfParts,
    paint: (p: Paint) => paintWeather(fields, p),
    setCam: (pos: number[], target: number[]) => { camera.position.set(pos[0]!, pos[1]!, pos[2]!); stage.controls.target.set(target[0]!, target[1]!, target[2]!); stage.controls.update(); camera.lookAt(stage.controls.target); },
    hero: () => stage.heroView(),
    trigger: (e: { kind: 'volcano' | 'meteor' | 'floodBasalt'; x: number; z: number; magnitude: number }) => atmo.trigger(e),
    godEvent: (e: { kind: 'volcano' | 'meteor' | 'floodBasalt'; x: number; z: number; magnitude: number }) => { sim?.events.enqueue({ ...e, my: sim.geoMy }); },
    tick: async (n: number) => { if (!sim) return 0; let k = 0; while (k < n) { k += sim.runTicks(Math.min(10, n - k)); await new Promise((r) => setTimeout(r, 0)); } return k; },
    setQuality: (v: boolean) => { atmo.setHighQuality(v); look?.setHighQuality(v); },
    setTime: (tod: number) => look?.setTimeOfDay(tod),
    stats: () => ({ ...atmo.stats }),
    setCut: (x: number, z: number) => atmo.setCut(x, z),
    shadowParams: () => ({ key: cloudShadowU.keyDir.value.toArray(), planeY: cloudShadowU.planeY.value, on: cloudShadowU.on.value }),
    colWorld: (x: number, z: number) => [cellToWorld(x), cellToWorld(z)],
    /** Water depth per column (voxel layers). */
    water: async () => Array.from(new Float32Array(await fields.read(renderer, 'water'))),
    NCOL, NX,
  };
  w.atReady = true;
}

main().catch((e) => { console.error('atmo page failed: ' + (e as Error).stack); });
