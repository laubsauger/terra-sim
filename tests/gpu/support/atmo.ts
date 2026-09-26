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
import { tSpritePuff, tSpriteStreak } from '../../../src/atmo/plumes';
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
  const atmo = createAtmosphere(fields, renderer, scene, camera, { highQuality: quality, lightning: q.get('lightning') !== '0' });
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
    /** Show / hide one atmosphere draw by name (e.g. 'plumesUnder') for on/off image diffs. */
    setVisible: (name: string, v: boolean) => { const o = atmo.object.getObjectByName(name); if (o) o.visible = v; return !!o; },
    /** Screen pixel (CSS px) of a world point with the current camera. */
    project: (x: number, y: number, z: number) => {
      const v = new THREE.Vector3(x, y, z).project(camera), r = renderer.domElement.getBoundingClientRect();
      return [(v.x * 0.5 + 0.5) * r.width, (0.5 - v.y * 0.5) * r.height];
    },
    /**
     * Sprite footprints (the shader's own TSL) sampled over the quad: max value on the outer band
     * (max(|x|, |y|) ≥ 0.92) and at the centre region, for the puff (worst-case noise 1, strongest erosion
     * setting 0.75) and the ejecta streak.
     */
    spriteEdges: async () => {
      const N = 64;
      const out = instancedArray(N * N, 'vec2');
      const k = Fn(() => {
        const i = THREE.TSL.float(instanceIndex);
        const q = THREE.TSL.vec2(i.mod(N).add(0.5).div(N).mul(2).sub(1), i.div(N).floor().add(0.5).div(N).mul(2).sub(1));
        out.element(instanceIndex).assign(THREE.TSL.vec2(tSpritePuff(q, THREE.TSL.float(1), THREE.TSL.float(0.75)), tSpriteStreak(q)));
      })().compute(N * N);
      renderer.compute(k);
      const a = new Float32Array(await renderer.getArrayBufferAsync(out.value as unknown as THREE.StorageBufferAttribute));
      const r = { puffEdge: 0, streakEdge: 0, puffMax: 0, streakMax: 0 };
      for (let j = 0; j < N * N; j++) {
        const x = ((j % N) + 0.5) / N * 2 - 1, y = (Math.floor(j / N) + 0.5) / N * 2 - 1;
        const edge = Math.max(Math.abs(x), Math.abs(y)) >= 0.92;
        r.puffMax = Math.max(r.puffMax, a[j * 2]!); r.streakMax = Math.max(r.streakMax, a[j * 2 + 1]!);
        if (edge) { r.puffEdge = Math.max(r.puffEdge, a[j * 2]!); r.streakEdge = Math.max(r.streakEdge, a[j * 2 + 1]!); }
      }
      return r;
    },
    /** Surface height per column (voxel y). */
    surf: async () => fields.read(renderer, 'surfY'),
    /** Water depth per column (voxel layers). */
    water: async () => Array.from(new Float32Array(await fields.read(renderer, 'water'))),
    NCOL, NX,
  };
  // ---- weather / cloud diagnostics (clouds.ts): weather-map and density probes, relief edits, uniform air ----
  const { texture3D, texture, vec2: tv2, vec4: tv4, float: tf, int: ti } = THREE.TSL;
  const am = await import('../../../src/atmo/atmoModel');
  const { GodTools } = await import('../../../src/sim/godTools');
  const god = new GodTools(fields), derive = createDerivePass(fields);
  const WXN = 128;
  const wxOut = instancedArray(WXN * WXN, 'vec4');
  const wxArg = uniformArray([new THREE.Vector4(0, 0, 4, 64), new THREE.Vector4()], 'vec4'); // (x0, z0, size, n), (y)
  const arg = (i: number) => wxArg.element(i) as unknown as THREE.Node<'vec4'>;
  const gridXZ = () => {
    const a = arg(0), n = a.w;
    const ix = tf(instanceIndex).mod(n), iz = tf(instanceIndex).div(n).floor();
    return tv2(a.x.add(ix.add(0.5).div(n).mul(a.z)), a.y.add(iz.add(0.5).div(n).mul(a.z)));
  };
  const weatherK = Fn(() => { const p = gridXZ(); wxOut.element(instanceIndex).assign(atmo.clouds.weatherAt(p.x, p.y)); })().compute(WXN * WXN);
  const cirK = Fn(() => { const p = gridXZ(); wxOut.element(instanceIndex).assign(texture(atmo.clouds.cirrusMap, p.add(2).div(4)).level(ti(0))); })().compute(WXN * WXN);
  const cu = atmo.clouds.uniforms;
  const densK = Fn(() => {
    const p = gridXZ(), y = arg(1).x;
    const d = texture3D(atmo.clouds.density, vec3(p.x.add(2).div(4), y.sub(cu.yLo).div(cu.yHi.sub(cu.yLo)), p.y.add(2).div(4))).level(ti(0));
    wxOut.element(instanceIndex).assign(tv4(d.r, d.g, d.b, d.a));
  })().compute(WXN * WXN);
  const runGrid = async (k: THREE.ComputeNode, x0: number, z0: number, size: number, n: number, y = 0) => {
    (wxArg.array[0] as THREE.Vector4).set(x0, z0, size, n); (wxArg.array[1] as THREE.Vector4).set(y, 0, 0, 0);
    renderer.compute(k);
    return Array.from(new Float32Array(await renderer.getArrayBufferAsync(wxOut.value as unknown as THREE.StorageBufferAttribute))).slice(0, n * n * 4);
  };
  Object.assign(w.at as Record<string, unknown>, {
    /** Weather map (coverage, storm, stratus − congestus, base world y) on an n×n grid (n ≤ 128) over a world square. */
    weatherGrid: (n = 64, x0 = -2, z0 = -2, size = 4) => runGrid(weatherK, x0, z0, size, Math.min(n, WXN)),
    /** Cloud density volume (density, storm, ash share, cirrus share) on an n×n grid at world height y. */
    densityGrid: (y: number, n = 64, x0 = -2, z0 = -2, size = 4) => runGrid(densK, x0, z0, size, Math.min(n, WXN), y),
    /** Cirrus map (r = where high veils may form) on an n×n grid. */
    cirrusGrid: (n = 64, x0 = -2, z0 = -2, size = 4) => runGrid(cirK, x0, z0, size, Math.min(n, WXN)),
    /** GPU ms per cloud compute kernel (each run alone n×, median), for perf work. */
    kernelTimes: async (n = 40) => {
      const ks = (atmo.clouds as unknown as { kernels?: Record<string, THREE.ComputeNode> }).kernels ?? {};
      const out: Record<string, number> = {};
      await renderer.resolveTimestampsAsync('compute');
      for (const [name, k] of Object.entries(ks)) {
        const v: number[] = [];
        for (let i = 0; i < n; i++) { renderer.compute(k); v.push((await renderer.resolveTimestampsAsync('compute')) ?? 0); }
        v.sort((a, b) => a - b); out[name] = +v[v.length >> 1]!.toFixed(4);
      }
      return JSON.stringify(out);
    },
    /** Slab bounds (world y). */
    slab: () => [cu.yLo.value, cu.yHi.value],
    /** God-tool uplift (voxel columns, immediate) + derive: a real new mountain under the clouds. */
    uplift: (x: number, z: number, radius: number, layers: number) => { god.uplift(renderer, x, z, radius, layers); derive.run(renderer); updateRenderColumns(renderer, fields, 10); },
    /** Uniform air everywhere: relative humidity, surface temperature (°C), precip per tick. */
    paintAir: (a: { rh: number; t: number; precip?: number }) => {
      const cap = am.capacity(a.t);
      (fields.cpuArray('vapor') as Float32Array).fill(a.rh * cap);
      (fields.cpuArray('precip') as Float32Array).fill(a.precip ?? 0);
      (fields.cpuArray('surfTemp') as Float32Array).fill(a.t);
      for (const n of ['vapor', 'precip', 'surfTemp']) fields.markDirty(n);
    },
    /** Set the page's ambience clock (s): replay the same cloud noise over a changed terrain / weather. */
    setClock: (v: number) => { t = v; },
    /** Drift velocities (world/s) per wind band: the low cloud layer and the cirrus layer. */
    bands: () => ({ low: am.bandSpeeds(), high: am.cirrusBands() }),
  });
  w.atReady = true;
}

main().catch((e) => { console.error('atmo page failed: ' + (e as Error).stack); });
