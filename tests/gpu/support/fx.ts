// Earthquake / tsunami FX test page: worldgen + a warmed-up Sim (natural coasts, shelves, mountains), the
// full look pipeline and createFx. Driven from fx.spec.ts via window.fxp; frames render on demand.
// URL: ?seed=<u32>&ticks=<n>&quality=low|high&shake=0|1&look=0|1
import * as THREE from 'three/webgpu';
import { createStage } from '../../../src/render/stage';
import { GpuFields } from '../../../src/core/gpu';
import { Params } from '../../../src/core/params';
import { registerSimFields, uploadWorld } from '../../../src/sim/fields';
import { generateWorld } from '../../../src/sim/worldgen';
import { Sim } from '../../../src/sim/sim';
import { NX, NZ, NCOL, colIdx } from '../../../src/sim/layout';
import { createLook, type Look } from '../../../src/render/look';
import { createLighting } from '../../../src/render/lighting';
import { createTerrain } from '../../../src/render/terrain';
import { createSides } from '../../../src/render/sides';
import { createWater } from '../../../src/render/water';
import { updateRenderColumns, cellToWorld } from '../../../src/render/space';
import { createFx } from '../../../src/fx/fx';
import { TSUNAMI } from '../../../src/fx/fxModel';

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
  const sim = new Sim(renderer, fields, world, new Params(), 0.05);
  const ticks = Number(q.get('ticks') ?? 400);
  for (let n = 0; n < ticks;) { n += sim.runTicks(Math.min(20, ticks - n)); await new Promise((r) => setTimeout(r, 0)); }

  let look: Look | null = null;
  if (q.get('look') !== '0') look = createLook(stage, fields, { highQuality: q.get('quality') !== 'low' });
  else {
    renderer.shadowMap.enabled = true;
    createLighting(scene, { shadows: true });
    scene.add(createTerrain(fields).object, createSides(fields).object, createWater(fields).object);
  }
  const fx = createFx(fields, renderer, scene, camera, sim, {
    highQuality: q.get('quality') !== 'low', shake: q.get('shake') !== '0', controls: stage.controls, quakes: false,
  });
  const quakes: unknown[] = [];
  fx.onQuake((e) => quakes.push({ ...e, position: e.position.toArray() }));

  let t = Number(q.get('t0') ?? 30);
  let fxOn = true;
  const frame = async (dt = 1 / 60) => {
    t += dt;
    look?.frame(t);
    if (fxOn) fx.update(dt, t);
    stage.controls.update();
    camera.lookAt(stage.controls.target);
    updateRenderColumns(renderer, fields); // the stage loop does this in the app
    if (look) look.post.render(); else renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
  };
  const frames = async (n: number, dt = 1 / 60) => { for (let i = 0; i < n; i++) await frame(dt); await new Promise((r) => requestAnimationFrame(r)); };
  const f32 = async (n: { value: unknown }) => new Float32Array(await renderer.getArrayBufferAsync(n.value as THREE.StorageBufferAttribute));
  const torus = (a: number, b: number, n: number) => { const d = Math.abs(a - b) % n; return Math.min(d, n - d); };

  /** Column fields + distance (cells, 4-neighbour BFS on the torus) of every column to dry land / to water. */
  async function columns() {
    const water = new Float32Array(await fields.read(renderer, 'water'));
    const surf = new Float32Array(await fields.read(renderer, 'surfY'));
    const bfs = (seed: (c: number) => boolean) => {
      const d = new Int32Array(NCOL).fill(-1), queue = new Int32Array(NCOL);
      let h = 0, tl = 0;
      for (let c = 0; c < NCOL; c++) if (seed(c)) { d[c] = 0; queue[tl++] = c; }
      while (h < tl) {
        const c = queue[h++]!, x = c % NX, z = (c / NX) | 0;
        for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const n = colIdx(x + dx!, z + dz!);
          if (d[n]! < 0) { d[n] = d[c]! + 1; queue[tl++] = n; }
        }
      }
      return d;
    };
    const dry = (c: number) => water[c]! < 0.05;
    return { water, surf, toLand: bfs(dry), toWater: bfs((c) => !dry(c)) };
  }

  /** An open-ocean site (deep, 18–60 cells off the nearest coast) and a steep mountain site. */
  async function sites() {
    const { water, surf, toLand, toWater } = await columns();
    const sea = sim.stats?.seaLevel ?? 76;
    let ocean = { x: -1, z: -1, score: -1e9, depth: 0, toLand: 0 }, land = { x: -1, z: -1, score: -1e9, alt: 0, slope: 0 };
    for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
      const c = colIdx(x, z);
      const dl = toLand[c]!;
      if (water[c]! > 6 && dl >= 18 && dl <= 60) {
        const s = water[c]! - Math.abs(dl - 32) * 0.3;
        if (s > ocean.score) ocean = { x, z, score: s, depth: water[c]!, toLand: dl };
      }
      if (water[c]! < 0.05 && toWater[c]! >= 12) {
        const slope = Math.max(Math.abs(surf[colIdx(x + 1, z)]! - surf[colIdx(x - 1, z)]!), Math.abs(surf[colIdx(x, z + 1)]! - surf[colIdx(x, z - 1)]!)) / 2;
        const s = surf[c]! - sea + 4 * slope;
        if (s > land.score) land = { x, z, score: s, alt: surf[c]! - sea, slope };
      }
    }
    let landFrac = 0;
    for (let c = 0; c < NCOL; c++) if (water[c]! < 0.05) landFrac++;
    return { ocean, land, sea, landFrac: landFrac / NCOL };
  }

  /** Tsunami front summary from the arrival-time field. */
  async function tsu(ex: number, ez: number) {
    const T = await f32(fx.tsunami.T);
    const { water, surf } = await columns();
    const age = fx.tsunami.age, sea = sim.stats?.seaLevel ?? 76;
    let front = 0, reachedWater = 0, reachedLand = 0, coastReached = 0, maxLandAlt = 0, finite = 0, maxT = 0;
    for (let c = 0; c < NCOL; c++) {
      const tc = T[c]!;
      if (tc >= TSUNAMI.INF * 0.5) continue;
      finite++;
      maxT = Math.max(maxT, tc);
      if (tc > age) continue;
      const x = c % NX, z = (c / NX) | 0;
      const wet = water[c]! >= 0.05;
      if (wet) {
        reachedWater++;
        front = Math.max(front, Math.hypot(torus(x, ex, NX), torus(z, ez, NZ)));
        const coastal = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dz]) => water[colIdx(x + dx!, z + dz!)]! < 0.05);
        if (coastal) coastReached++;
      } else { reachedLand++; maxLandAlt = Math.max(maxLandAlt, surf[c]! - sea); }
    }
    // local slowness |∇T| (s per cell) over reached open water, deep vs shallow: shallow-water waves go at √depth
    const slow = { deep: 0, nDeep: 0, shallow: 0, nShallow: 0, relDeep: 0, relShallow: 0 };
    const expSlow = (d: number) => 1 / (TSUNAMI.C0 * Math.sqrt(Math.max(d, TSUNAMI.DMIN) / TSUNAMI.DREF));
    for (let c = 0; c < NCOL; c++) {
      const x = c % NX, z = (c / NX) | 0, d = water[c]!;
      const n = [T[colIdx(x - 1, z)]!, T[colIdx(x + 1, z)]!, T[colIdx(x, z - 1)]!, T[colIdx(x, z + 1)]!];
      if (T[c]! > age || n.some((v) => v >= TSUNAMI.INF * 0.5)) continue;
      // classify by the 3×3 neighbourhood: a single shallow column is crossed by the faster path around it
      const nb: number[] = [];
      for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) nb.push(water[colIdx(x + dx, z + dz)]!);
      const deep = nb.every((v) => v > 8), shallow = nb.every((v) => v > 0.3 && v < 5);
      if (!deep && !shallow) continue;
      const g = Math.hypot((n[1]! - n[0]!) / 2, (n[3]! - n[2]!) / 2);
      if (deep) { slow.deep += g; slow.nDeep++; slow.relDeep += g / expSlow(d); } else { slow.shallow += g; slow.nShallow++; slow.relShallow += g / expSlow(d); }
    }
    return { age, active: fx.tsunami.active, front, reachedWater, reachedLand, coastReached, maxLandAlt, finite, maxT,
      slowDeep: slow.deep / Math.max(1, slow.nDeep), slowShallow: slow.shallow / Math.max(1, slow.nShallow), nDeep: slow.nDeep, nShallow: slow.nShallow,
      relDeep: slow.relDeep / Math.max(1, slow.nDeep), relShallow: slow.relShallow / Math.max(1, slow.nShallow) };
  }

  /** Dust particle summary. */
  async function dust() {
    const P0 = await f32(fx.quake.P0), P1 = await f32(fx.quake.P1), P2 = await f32(fx.quake.P2);
    let alive = 0, lifted = 0, maxRise = 0;
    const pts: number[][] = [];
    for (let i = 0; i < P0.length / 4; i++) {
      const age = P0[i * 4 + 3]!, end = P1[i * 4 + 3]!;
      if (age >= end) continue;
      alive++;
      if (age > P2[i * 4]!) { lifted++; if (pts.length < 400) pts.push([P0[i * 4]!, P0[i * 4 + 1]!, P0[i * 4 + 2]!]); }
    }
    for (const p of pts) maxRise = Math.max(maxRise, p[1]!);
    return { alive, lifted, pts, maxRise };
  }

  /** GPU ms: fx compute alone (timestamps) and the whole frame with the fx drawn vs hidden. */
  async function perf(n = 40) {
    const hasTs = renderer.hasFeature('timestamp-query');
    await renderer.resolveTimestampsAsync('compute'); await renderer.resolveTimestampsAsync('render');
    let fxComp = 0, rendOn = 0, rendOff = 0;
    for (let i = 0; i < n; i++) {
      t += 1 / 60; look?.frame(t); updateRenderColumns(renderer, fields);
      await renderer.resolveTimestampsAsync('compute');
      fx.update(1 / 60, t);
      fxComp += (await renderer.resolveTimestampsAsync('compute')) ?? 0;
      await renderer.resolveTimestampsAsync('render');
      if (look) look.post.render(); else renderer.render(scene, camera);
      await device.queue.onSubmittedWorkDone();
      rendOn += (await renderer.resolveTimestampsAsync('render')) ?? 0;
      fx.object.visible = false;
      if (look) look.post.render(); else renderer.render(scene, camera);
      await device.queue.onSubmittedWorkDone();
      rendOff += (await renderer.resolveTimestampsAsync('render')) ?? 0;
      fx.object.visible = true;
    }
    return { hasTs, width: renderer.domElement.width, height: renderer.domElement.height, fxCompute: fxComp / n, renderOn: rendOn / n, renderOff: rendOff / n,
      fxTotal: (fxComp + rendOn - rendOff) / n, stats: fx.stats };
  }

  /**
   * GPU-bound ms per frame, back to back (one wait at the end), with only the named fx children drawn
   * (fx.update still runs). Compare against show=[] for the cost of what is shown.
   */
  async function bench(n: number, show: string[]) {
    const drawn: THREE.Object3D[] = [];
    fx.object.traverse((o) => { if ((o as THREE.Mesh).isMesh || (o as THREE.Sprite).isSprite) drawn.push(o); });
    const vis = drawn.map((o) => o.visible);
    // outside the animation loop the node frame does not advance (FRAME-updated passes would be skipped): step it
    const rr = renderer as unknown as { _nodes: { nodeFrame: { update(): void; frameId: number } }; info: { frame: number } };
    const run = async () => {
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      for (let i = 0; i < n; i++) {
        rr._nodes.nodeFrame.update(); rr.info.frame = rr._nodes.nodeFrame.frameId;
        t += 1 / 60; look?.frame(t); // fx state frozen: the cost of what is on screen now
        drawn.forEach((o, k) => { o.visible = vis[k]! && show.includes(o.name); });
        updateRenderColumns(renderer, fields);
        if (look) look.post.render(); else renderer.render(scene, camera);
      }
      await device.queue.onSubmittedWorkDone();
      return (performance.now() - t0) / n;
    };
    await run();
    const ms = await run();
    drawn.forEach((o, k) => { o.visible = vis[k]!; });
    return ms;
  }
  /** Interleaved A/B (fx drawn vs hidden) in blocks: median per-frame difference, robust to other GPU load. */
  async function benchAB(n = 20, reps = 8, show = ['tsunami', 'quakeDust', 'quakeGround']) {
    const on: number[] = [], off: number[] = [];
    for (let r = 0; r < reps; r++) { on.push(await bench(n, show)); off.push(await bench(n, [])); }
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1]!;
    const diffs = on.map((v, i) => v - off[i]!);
    return { on: med(on), off: med(off), diff: med(diffs) };
  }
  /** Add k−1 clones (shared geometry + material) of the named fx child: amplifies its cost k× above the noise. */
  function amplify(name: string, k: number) {
    const o = fx.object.getObjectByName(name)!;
    const clones = Array.from({ length: k - 1 }, () => o.clone());
    fx.object.add(...clones);
    return () => fx.object.remove(...clones);
  }
  async function benchAmp(name: string, k = 8, n = 15, reps = 7) {
    const undo = amplify(name, k);
    const r = await benchAB(n, reps, [name]);
    undo();
    return { ...r, perCopy: r.diff / k };
  }
  /** GPU ms of one frame's tsunami relaxation (SWEEPS in-place sweeps), from n back-to-back dispatch batches. */
  async function benchSweeps(n = 200) {
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 0; i < n; i++) fx.tsunami.update(renderer, 0, t);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / n;
  }
  const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;

  const w = window as unknown as Record<string, unknown>;
  w.fxp = {
    bench, benchAB, benchAmp, benchSweeps,
    /** Compute dispatches issued by one fx.update (0 when idle). */
    updateCalls: () => { const c0 = renderer.info.compute.calls; fx.update(1 / 60, t); return renderer.info.compute.calls - c0; },
    gpu: info ? `${info.vendor} ${info.architecture} ${info.description}` : '?',
    frames, sites, tsu, dust, perf, quakes,
    trigger: async (x: number, z: number, mag: number) => { const e = await fx.triggerQuake(x, z, mag); return { ...e, position: e.position.toArray() }; },
    stats: () => ({ ...fx.stats, last: null }),
    visible: () => fx.object.children.filter((o) => !(o as THREE.Group).isGroup).map((o) => [o.name, o.visible]),
    meteorVisible: () => fx.meteor.object.children.filter((o) => !(o as THREE.Light).isLight).map((o) => [o.name, o.visible]),
    tsunamiActive: () => fx.tsunami.active,
    setShake: (v: boolean) => fx.setShake(v),
    setEnabled: (v: boolean) => { fx.setEnabled(v); fxOn = v; },
    setCam: (pos: number[], target: number[]) => { camera.position.set(pos[0]!, pos[1]!, pos[2]!); stage.controls.target.set(target[0]!, target[1]!, target[2]!); stage.controls.update(); },
    camPos: () => [...camera.position.toArray(), ...stage.controls.target.toArray()],
    hero: () => stage.heroView(),
    setTime: (tod: number) => look?.setTimeOfDay(tod),
    /** God-tool flow: the cinematic strike, the sim meteor event enqueued at the impact frame (crater at the flash). */
    strike: async (x: number, z: number, dir: number, magnitude: number, radius?: number) => {
      let impactAt = -1;
      const q = await fx.meteorStrike(x, z, dir, magnitude, () => {
        impactAt = t;
        sim.events.enqueue({ kind: 'meteor', x, z, magnitude, radius, dir, my: sim.geoMy });
      }, radius);
      sim.flushEvents(); // main.ts does this at the start of every frame
      return { ...q, position: q.position.toArray(), impactAt };
    },
    /** Starts a strike without waiting for its impact (frames advance the sequence). */
    strikeAsync: (x: number, z: number, dir: number, magnitude: number, radius?: number) => {
      (window as any).__strike = fx.meteorStrike(x, z, dir, magnitude, () => {
        sim.events.enqueue({ kind: 'meteor', x, z, magnitude, radius, dir, my: sim.geoMy });
        sim.flushEvents();
      }, radius).then((q) => ({ ...q, position: q.position.toArray() }));
    },
    strikeResult: () => (window as any).__strike,
    /**
     * Live meteor particles by kind (smoke, curtain, ember, column, secondary, fireball); curtain centroid offset
     * from (px, pz); column top above py (world).
     */
    meteorParticles: async (px = 0, pz = 0, py = 0) => {
      const P0 = await f32(fx.meteor.P0), P1 = await f32(fx.meteor.P1), P2 = await f32(fx.meteor.P2);
      const byKind = [0, 0, 0, 0, 0, 0];
      let maxY = -1e9, cx = 0, cz = 0, nc = 0, colTop = -1e9;
      for (let i = 0; i < P0.length / 4; i++) {
        if (P0[i * 4 + 3]! >= P1[i * 4 + 3]!) continue;
        const k = Math.round(P2[i * 4]!);
        byKind[k]!++; maxY = Math.max(maxY, P0[i * 4 + 1]!);
        if (k === 1) { cx += P0[i * 4]! - px; cz += P0[i * 4 + 2]! - pz; nc++; }
        if (k === 3) colTop = Math.max(colTop, P0[i * 4 + 1]! - py);
      }
      return { byKind, maxY, colTop, light: fx.meteor.light.intensity, active: fx.meteor.active, curtain: [cx / Math.max(1, nc), cz / Math.max(1, nc)] };
    },
    events: () => sim.events.log.length,
    godMeteor: async (x: number, z: number, magnitude: number) => {
      sim.events.enqueue({ kind: 'meteor', x, z, magnitude, my: sim.geoMy });
      sim.flushEvents();
    },
    colWorld: (x: number, z: number) => [cellToWorld(x), cellToWorld(z)],
    seaLevel: () => sim.stats?.seaLevel ?? 76,
  };
  w.fxReady = true;
}

main().catch((e) => { console.error('fx page failed: ' + (e as Error).stack); });
