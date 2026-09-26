// §T.41 + §T.42 atmosphere: clouds, cloud shadows, volcanic plumes, impact dust, flood-basalt haze,
// rain/snow and lightning. One entry point for main.ts; reads sim state only (V15), moves on the
// ambience clock / real dt so it keeps running while the geo clock is paused (V17).
//
// Per frame (update): clouds (summary → cloudlets → shadow texture) → plume vents + particles → rain.
// CPU side: new event-log entries become emitters (eruption ash, meteor flash + dust ring, flood-basalt
// haze); a small cloudlet readback (~15 KB, once per second) feeds lightning in storm cells.
import * as THREE from 'three/webgpu';
import { vec3, uniform } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import type { GeoEvent } from '../sim/events';
import { CELL, BLOCK_SIZE, Y_SEA_NOMINAL } from '../sim/layout';
import { cellToWorld, voxelToWorldY, vertEx } from '../render/space';
import { skyU } from '../render/sky';
import { ATMO, AMB_PERIOD, BURST, HALF, VENT_PHASE, VENT_SHALLOW, windProfile, decodeVent } from './atmoModel';
import { createClouds, cloudShadowU, type Clouds } from './clouds';
import { atmoSeaLevel, atmoCut } from './atmoTsl';
import { createPlumes, type Plumes } from './plumes';
import { createRain, type Rain } from './rain';

export { cloudShadowAt, cloudShadowTexture, cloudShadowU } from './clouds';

export interface AtmoOptions {
  /** High tier: 48 raymarch steps, volume every frame, 6k plume particles, 4k drops. Low: 28 steps, volume every other frame, 2k / 1.5k. */
  highQuality?: boolean;
  /** Lightning bolts + flash light in storm cells (default on). */
  lightning?: boolean;
}

/** What the atmosphere reads from the sim each frame (all optional). */
export interface AtmoSimState {
  /** Sim.events.log: entries not seen before fire their FX (the log present at the first update is skipped). */
  events?: readonly GeoEvent[];
  /** Emergent sea level, voxel y (Sim.stats.seaLevel). */
  seaLevel?: number;
}

export interface Atmosphere {
  object: THREE.Group;
  clouds: Clouds;
  plumes: Plumes;
  rain: Rain;
  /** dt: real seconds since the last frame; ambTime: ambience clock (s). */
  update(dt: number, ambTime: number, simState?: AtmoSimState): void;
  setHighQuality(v: boolean): void;
  /** Slice inspection: hide everything beyond x > cutX or z > cutZ (world units; HALF = no cut). Call per frame. */
  setCut(x: number, z: number): void;
  /** Cloud opacity multiplier: 0 hidden … 1 normal (overlays dim clouds while active). Cloud shadows follow. */
  setCloudFade(v: number): void;
  /** Fire an event's FX directly (tests, god tools preview). */
  trigger(e: Pick<GeoEvent, 'kind' | 'x' | 'z' | 'magnitude'> & { radius?: number }): void;
  /** Debug counters. */
  readonly stats: { flashes: number; bursts: number; lightning: number; volcanicLightning: number; pdc: number; blasts: number };
  dispose(): void;
}

const BOLT_PTS = 18;

export function createAtmosphere(fields: GpuFields, renderer: THREE.WebGPURenderer, scene: THREE.Scene,
  camera: THREE.Camera, opts: AtmoOptions = {}): Atmosphere {
  let hq = opts.highQuality ?? true;
  const lightningOn = opts.lightning ?? true;
  const clouds = createClouds(fields, { highQuality: hq, renderer });
  const plumes = createPlumes(fields, clouds, { highQuality: hq, renderer });
  const rain = createRain(fields, clouds, { highQuality: hq, renderer });

  const object = new THREE.Group();
  object.name = 'atmosphere';
  object.add(clouds.object, plumes.object, plumes.under, plumes.flash, rain.drops, rain.curtain);

  // ---- lightning bolt (camera-facing ribbon, rebuilt per strike) + one shared flash light ----
  const boltPos = new Float32Array(BOLT_PTS * 2 * 3);
  const boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute('position', new THREE.BufferAttribute(boltPos, 3));
  const idx: number[] = [];
  for (let i = 0; i < BOLT_PTS - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  boltGeo.setIndex(idx);
  const boltAmt = uniform(0);
  const boltMat = new THREE.MeshBasicNodeMaterial();
  boltMat.transparent = true;
  boltMat.depthWrite = false;
  boltMat.blending = THREE.AdditiveBlending;
  boltMat.side = THREE.DoubleSide;
  boltMat.fog = false;
  boltMat.colorNode = vec3(0.75, 0.82, 1.0).mul(boltAmt.mul(40));
  const bolt = new THREE.Mesh(boltGeo, boltMat);
  bolt.name = 'lightning';
  bolt.frustumCulled = false;
  bolt.visible = false;
  bolt.renderOrder = 8; // additive, after the cloud composite: glows through the cloud it strikes from
  const flashLight = new THREE.PointLight(0xcfd8ff, 0, 3.5, 2);
  flashLight.name = 'atmoFlash';
  object.add(bolt, flashLight);
  scene.add(object);

  // ---- state ----
  const stats = { flashes: 0, bursts: 0, lightning: 0, volcanicLightning: 0, pdc: 0, blasts: 0 };
  let fxTime = 0;
  let first = true;
  let seaLevel = Y_SEA_NOMINAL;
  const seen = new WeakSet<GeoEvent>();
  let impact: { t0: number; pos: THREE.Vector3; mag: number } | null = null;
  let strike: { t0: number; dur: number; idx: number; mid: THREE.Vector3; volcanic: boolean } | null = null;
  // lava vents (small readback, throttled): drive pyroclastic-current episodes and volcanic lightning
  let vents: { x: number; y: number; z: number; heat: number; wet: boolean; phase: number; depth: number; coast: number }[] = [];
  // explosive openings waiting to play (a few at most; one blast per BLAST_GAP s so none drowns another)
  const openings: { x: number; y: number; z: number; heat: number; t: number; wet: boolean }[] = [];
  let readOnce = false, lastBlast = -1e9;
  // cloudlet readback for lightning (small, throttled)
  let cells: Float32Array | null = null;
  let sinceRead = 0, reading = false;
  const key = new THREE.Vector3();

  const groundY = () => voxelToWorldY(seaLevel);
  /**
   * Explosive opening of an eruption (the sim's BUILD → ACTIVE: the vent blasts its summit crater open): a
   * crater flash, a burst of incandescent bombs and spray, a fast dark ash jet and a ring of ash rushing out
   * over the flanks (base surge). y = vent surface (bursts never start below the local ground).
   */
  function blast(x: number, y: number, z: number, mag: number, wet = false): void {
    const add = (kind: number, dur: number, radius: number, m: number, rate?: number) => { plumes.addBurst({ x, y, z, kind, t0: fxTime, dur, mag: m, radius, rate }); stats.bursts++; };
    if (wet) {
      // shallow submarine (Surtseyan) opening: black tephra jets and a steam burst, nothing incandescent
      add(BURST.TEPHRA, 1.5, 0.02, 0.8 + 0.2 * mag, 90);
      add(BURST.STEAM, 3, 0.03, 1, 220);
      stats.blasts++;
      return;
    }
    add(BURST.ASH, 2.5, 0.02, 1.2 + 0.4 * mag, 300);
    add(BURST.BOMB, 1.4, 0.02, 0.9 + 0.3 * mag, 160);
    add(BURST.FOUNTAIN, 1.6, 0.02, 0.9 + 0.3 * mag, 250);
    add(BURST.DUST, 1.2, 0.03 + 0.02 * mag, 0.3 + 0.1 * mag);
    impact = { t0: fxTime, pos: new THREE.Vector3(x, y + 0.03, z), mag: 0.6 + 0.4 * mag };
    stats.blasts++;
  }
  function trigger(e: Pick<GeoEvent, 'kind' | 'x' | 'z' | 'magnitude'> & { radius?: number }): void {
    const x = cellToWorld(e.x), z = cellToWorld(e.z), mag = e.magnitude;
    const y = groundY();
    const add = (kind: number, dur: number, radius: number, m = mag) => { plumes.addBurst({ x, y, z, kind, t0: fxTime, dur, mag: m, radius }); stats.bursts++; };
    switch (e.kind) {
      case 'volcano': {
        // eruption: ash column, lava fountain, ballistic bombs, a crater flash, then a pyroclastic current
        blast(x, y, z, mag);
        impact!.pos.y = y + 0.08; // y is sea level here (the GPU lifts the bursts onto the ground)
        add(BURST.ASH, 14 + 8 * mag, 0.05);
        add(BURST.FOUNTAIN, 6 + 4 * mag, 0.02);
        add(BURST.BOMB, 8 + 4 * mag, 0.02);
        plumes.addBurst({ x, y, z, kind: BURST.PDC, t0: fxTime + 2.5, dur: 3, mag: 0.7 + 0.4 * mag, radius: 0.03 }); stats.bursts++;
        break;
      }
      case 'meteor': {
        // the cinematic strike (flash, curtain, dust column) lives in src/fx; atmosphere only adds the
        // lingering regional haze so the two don't double up
        // (short and faint: a long, dense haze read as fog sitting over the view)
        const r = (e.radius ?? 4 + 8 * mag) * CELL;
        add(BURST.HAZE, ATMO.IMPACT_HAZE_S, r * 3, ATMO.IMPACT_HAZE_MAG);
        break;
      }
      case 'floodBasalt':
        add(BURST.HAZE, 40, 0.5);
        add(BURST.ASH, 20, 0.12, 0.6);
        break;
      default:
        break; // hotspot (mantle only), iceAge, storm (precip does it), uplift, split: no atmosphere FX
    }
  }

  function readClouds(): void {
    reading = true;
    Promise.all([
      renderer.getArrayBufferAsync(clouds.cells.value as unknown as THREE.StorageBufferAttribute),
      renderer.getArrayBufferAsync(plumes.vents.value as unknown as THREE.StorageBufferAttribute, null, 0, ATMO.VENTS_MAX * 16),
      renderer.getArrayBufferAsync(plumes.ctr.value as unknown as THREE.StorageBufferAttribute, null, 0, 4),
    ]).then(([a, v, c]) => {
      cells = new Float32Array(a);
      const vf = new Float32Array(v), n = Math.min(new Int32Array(c)[0]!, ATMO.VENTS_MAX);
      const prev = vents;
      vents = [];
      for (let i = 0; i < n; i++) {
        const d = decodeVent(vf[i * 4 + 3]!);
        // 'wet' = no subaerial ash column: submarine deeper than the Surtseyan range
        vents.push({ x: vf[i * 4]!, y: vf[i * 4 + 1]!, z: vf[i * 4 + 2]!, heat: d.heat, wet: d.depth > VENT_SHALLOW, phase: d.phase, depth: d.depth, coast: d.coast });
      }
      // explosive openings: a dry vent erupting now that was not erupting at the last readback (~1 s ago)
      if (readOnce) {
        const erupting = (v: { phase: number }) => v.phase === VENT_PHASE.ACTIVE || v.phase === VENT_PHASE.WANING;
        for (const v of vents) {
          if (v.wet || v.phase !== VENT_PHASE.ACTIVE || v.heat < 0.5) continue;
          if (prev.some((q) => erupting(q) && Math.hypot(q.x - v.x, q.z - v.z) < ATMO.BLAST_NEAR)) continue;
          if (!openings.some((q) => Math.hypot(q.x - v.x, q.z - v.z) < ATMO.BLAST_NEAR)) openings.push({ x: v.x, y: v.y, z: v.z, heat: v.heat, t: fxTime, wet: v.depth > ATMO.STEAM_WET || v.coast >= 2 }); // mostly-sea cell: no bombs over the water
        }
        // openings are common (the budget, not the sim, limits them): play fresh ones, strongest first, and
        // prefer vents inside the block (a blast at the cut face is half hidden)
        const score = (o: { x: number; z: number; heat: number }) => o.heat - 2 * Math.max(0, 0.25 - (HALF - Math.max(Math.abs(o.x), Math.abs(o.z))));
        for (let i = openings.length - 1; i >= 0; i--) if (fxTime - openings[i]!.t > 2.5) openings.splice(i, 1);
        openings.sort((a, b) => score(b) - score(a)).splice(4);
      }
      readOnce = true;
    })
      .catch((err) => console.error('atmosphere: cloud readback failed: ' + (err as Error).message))
      .finally(() => { reading = false; });
  }

  // ---- big plumes in the cloud volume: strongest dry vents + eruption bursts, eased so they never pop ----
  const slots: { x: number; y: number; z: number; s: number; target: number; fed: boolean }[] = [];
  function updatePlumes(dt: number): void {
    const cands: { x: number; y: number; z: number; s: number }[] = [];
    for (const b of plumes.bursts) if (b.kind === BURST.ASH && fxTime >= b.t0 && fxTime - b.t0 < b.dur) cands.push({ x: b.x, y: -1, z: b.z, s: Math.min(1.5, b.mag) });
    // only erupting (active / waning) vents on land or in the Surtseyan range get a big ash plume
    // (sticky: vents that already carry a plume come first, so the ≤ PLUMES picks don't churn between the many
    // similar vents and leave orphaned plumes hanging where no vent erupts any more)
    const held = (v: { x: number; z: number }) => (slots.some((q) => q.s > 0.05 && Math.hypot(q.x - v.x, q.z - v.z) < 0.2) ? 1 : 0);
    for (const v of [...vents].sort((a, b) => held(b) - held(a) || b.heat - a.heat)) if (!v.wet && v.heat > 0.3 && v.phase >= VENT_PHASE.ACTIVE) cands.push({ x: v.x, y: v.y, z: v.z, s: v.heat * (v.phase === VENT_PHASE.WANING ? 0.6 : 1) });
    const picked: typeof cands = [];
    for (const c of cands) {
      if (picked.length >= ATMO.PLUMES) break;
      const near = picked.find((q) => Math.hypot(q.x - c.x, q.z - c.z) < 0.2);
      if (near) { near.s = Math.max(near.s, c.s); if (near.y < 0) near.y = c.y; } else picked.push({ ...c });
    }
    for (const sl of slots) { sl.target = 0; sl.fed = false; }
    for (const c of picked) {
      let sl = slots.find((q) => Math.hypot(q.x - c.x, q.z - c.z) < 0.2);
      if (!sl && slots.length < ATMO.PLUMES) { sl = { x: c.x, y: c.y, z: c.z, s: 0, target: 0, fed: false }; slots.push(sl); }
      if (!sl) { const idle = slots.find((q) => q.s < 0.02); if (idle) { Object.assign(idle, { x: c.x, y: c.y, z: c.z, s: 0 }); sl = idle; } }
      if (sl) {
        sl.target = c.s; sl.fed = true;
        // follow the vent as it moves with its plate (the column stays on its source)
        const kf = 1 - Math.exp(-dt * 2);
        sl.x += (c.x - sl.x) * kf; sl.z += (c.z - sl.z) * kf;
        if (c.y >= 0) sl.y = c.y;
      }
    }
    // eases in, lingers on the way out while its vent still wanes (an ending eruption's umbrella thins over
    // ~8 s); a plume whose vent is gone fades in ~2 s: with no column under it, it would float detached
    for (const sl of slots) {
      const down = sl.fed ? ATMO.PLUME_DECAY : ATMO.PLUME_ORPHAN_DECAY;
      sl.s += (sl.target - sl.s) * (1 - Math.exp(-dt * (sl.target > sl.s ? ATMO.PLUME_RISE : down)));
    }
    const yHi = clouds.uniforms.yHi.value;
    clouds.setPlumes(slots.filter((q) => q.s > 0.01).map((q) => {
      const vy = q.y >= 0 ? q.y : groundY() + 0.05;
      const top = Math.min(vy + 0.2 + 0.32 * Math.min(q.s, 1.2), yHi - 0.12);
      const w = windProfile(q.z, Math.min(1, (top - groundY()) / ATMO.WIND_H));
      const sp = Math.hypot(w.u, w.v) || 1e-6;
      return { x: q.x, y: vy, z: q.z, s: Math.min(q.s, 1.2), dx: w.u / sp, dz: w.v / sp, len: Math.min(ATMO.PLUME_LEN_MAX, (0.35 + Math.min(0.8, sp * 8) * Math.min(1, q.s)) * ATMO.PLUME_DRIFT), top };
    }));
  }

  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), side = new THREE.Vector3(), view = new THREE.Vector3();
  function buildBolt(top: THREE.Vector3, bottom: THREE.Vector3): void {
    const pts: THREE.Vector3[] = [];
    let off = 0, offz = 0;
    for (let i = 0; i < BOLT_PTS; i++) {
      const t = i / (BOLT_PTS - 1);
      if (i > 0 && i < BOLT_PTS - 1) { off += (Math.random() - 0.5) * 0.05; offz += (Math.random() - 0.5) * 0.05; }
      pts.push(new THREE.Vector3().lerpVectors(top, bottom, t).add(new THREE.Vector3(off, 0, offz)));
    }
    for (let i = 0; i < BOLT_PTS; i++) {
      const p = pts[i]!;
      tmpA.copy(pts[Math.min(i + 1, BOLT_PTS - 1)]!).sub(pts[Math.max(i - 1, 0)]!).normalize();
      view.copy(camera.position).sub(p).normalize();
      side.crossVectors(tmpA, view).normalize().multiplyScalar(0.0045 * (1.2 - 0.6 * (i / BOLT_PTS)));
      tmpB.copy(p).add(side); boltPos.set([tmpB.x, tmpB.y, tmpB.z], i * 6);
      tmpB.copy(p).sub(side); boltPos.set([tmpB.x, tmpB.y, tmpB.z], i * 6 + 3);
    }
    (boltGeo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
  }

  function updateLightning(dt: number): void {
    let light = 0;
    if (strike) {
      const t = (fxTime - strike.t0) / strike.dur;
      if (t > 1) { strike = null; bolt.visible = false; boltAmt.value = 0; clouds.uniforms.flash.value = 0; }
      else {
        // two or three flickers inside the envelope
        const flick = Math.max(0, Math.sin(t * Math.PI * 5.5)) * (1 - t) + (t < 0.12 ? 1 : 0);
        boltAmt.value = flick;
        if (!strike.volcanic) clouds.uniforms.flash.value = flick; else clouds.uniforms.flash.value = 0;
        light = flick * (strike.volcanic ? 1.2 : 2.5);
        flashLight.position.copy(strike.mid);
        flashLight.color.setRGB(0.8, 0.85, 1);
      }
    }
    if (!strike && lightningOn && cells) {
      const n = ATMO.CELLS, cw = BLOCK_SIZE / n;
      for (let c = 0; c < n * n; c++) {
        const cov = cells[c * 4]!, storm = cells[c * 4 + 1]!;
        if (storm < ATMO.STORM_LIGHTNING || cov < 0.3) continue;
        if (Math.random() > ATMO.FLASH_RATE * storm * dt) continue;
        const top = new THREE.Vector3(((c % n) + Math.random()) * cw - BLOCK_SIZE / 2, cells[c * 4 + 2]! + 0.01, (Math.floor(c / n) + Math.random()) * cw - BLOCK_SIZE / 2);
        const bottom = new THREE.Vector3(top.x + (Math.random() - 0.5) * 0.08, cells[c * 4 + 3]!, top.z + (Math.random() - 0.5) * 0.08);
        if (top.x > atmoCut.value.x || top.z > atmoCut.value.y) break; // beyond the slice cut
        buildBolt(top, bottom);
        bolt.visible = true;
        strike = { t0: fxTime, dur: 0.32 + Math.random() * 0.2, idx: c, mid: top.clone().lerp(bottom, 0.3), volcanic: false };
        clouds.uniforms.flashPos.value.copy(top).lerp(bottom, -0.4);
        stats.lightning++;
        break;
      }
    }
    // volcanic lightning: short bolts flickering inside big ash columns (vivid at night)
    if (!strike && lightningOn) {
      const night = skyU.night.value;
      for (const v of vents) {
        if (v.wet || v.heat < 0.5 || v.phase !== VENT_PHASE.ACTIVE || v.x > atmoCut.value.x || v.z > atmoCut.value.y) continue;
        if (Math.random() > ATMO.VOLC_FLASH_RATE * v.heat * (0.15 + 0.85 * night) * dt) continue;
        const j = () => (Math.random() - 0.5) * 0.06;
        const top = new THREE.Vector3(v.x + j(), v.y + 0.22 + Math.random() * 0.22, v.z + j());
        const bottom = new THREE.Vector3(top.x + j() * 2, top.y - 0.08 - Math.random() * 0.1, top.z + j() * 2);
        buildBolt(top, bottom);
        bolt.visible = true;
        strike = { t0: fxTime, dur: 0.18 + Math.random() * 0.15, idx: -1, mid: top.clone().lerp(bottom, 0.5), volcanic: true };
        stats.volcanicLightning++;
        break;
      }
    }
    // meteor flash: bright white-orange, decays over ~0.8 s
    if (impact) {
      const t = fxTime - impact.t0;
      if (t > 1.2) { impact = null; plumes.uniforms.flashAmt.value = 0; }
      else {
        const env = Math.exp(-t * 5) * Math.min(1, t * 30);
        plumes.uniforms.flashAmt.value = env * (0.6 + 0.4 * impact.mag);
        plumes.uniforms.flashPos.value.copy(impact.pos);
        plumes.uniforms.flashSize.value = (0.12 + t * 0.6) * (0.6 + 0.4 * impact.mag);
        if (env * 6 > light) { light = env * 6 * impact.mag; flashLight.position.copy(impact.pos); flashLight.color.setRGB(1, 0.62, 0.35); }
      }
    }
    flashLight.intensity = light;
  }

  return {
    object, clouds, plumes, rain, stats,
    update(dtIn, ambTime, sim) {
      const dt = Math.min(Math.max(dtIn, 0), 0.1);
      fxTime += dt;
      if (sim?.seaLevel !== undefined && Number.isFinite(sim.seaLevel)) seaLevel = sim.seaLevel;
      atmoSeaLevel.value = seaLevel;
      const yLo = voxelToWorldY(seaLevel + ATMO.SLAB_BASE);
      clouds.uniforms.yLo.value = yLo;
      clouds.uniforms.yHi.value = yLo + ATMO.SLAB_H * vertEx.value;
      // key light direction exactly as lighting.ts places the shadow caster
      key.copy(skyU.night.value > 0.02 ? skyU.moonDir.value : skyU.sunDir.value);
      key.y = Math.max(key.y, 0.06);
      cloudShadowU.keyDir.value.copy(key.normalize());
      cloudShadowU.planeY.value = yLo;

      clouds.uniforms.time.value = ((ambTime % AMB_PERIOD) + AMB_PERIOD) % AMB_PERIOD;
      rain.uniforms.time.value = fxTime % 1000;
      clouds.compute(renderer, camera);
      plumes.compute(renderer, dt, fxTime);
      rain.compute(renderer, dt);

      if (sim?.events) {
        for (const e of sim.events) {
          if (seen.has(e)) continue;
          seen.add(e);
          if (!first) trigger(e);
        }
      }
      first = false;
      sinceRead += dt;
      if (!reading && sinceRead >= 1) { sinceRead = 0; readClouds(); }
      updateLightning(dt);
      updatePlumes(dt);
      if (openings.length && fxTime - lastBlast >= ATMO.BLAST_GAP) {
        const o = openings.shift()!;
        if (o.x <= atmoCut.value.x && o.z <= atmoCut.value.y) { blast(o.x, o.y, o.z, o.heat, o.wet); lastBlast = fxTime; }
      }
      // pyroclastic density currents: occasional episodes at strong, dry vents (≤ 2 at a time)
      let livePdc = plumes.bursts.filter((b) => b.kind === BURST.PDC && fxTime - b.t0 < b.dur).length;
      for (const v of vents) {
        if (livePdc >= 2 || v.wet || v.depth > 0.05 || v.heat < 0.6 || v.phase !== VENT_PHASE.ACTIVE) continue;
        if (Math.random() > ATMO.PDC_RATE * v.heat * dt) continue;
        plumes.addBurst({ x: v.x, y: v.y, z: v.z, kind: BURST.PDC, t0: fxTime, dur: 2.5, mag: 0.6 + 0.6 * v.heat, radius: 0.03 });
        stats.pdc++; livePdc++;
      }
    },
    setHighQuality(v) {
      hq = v;
      clouds.setHighQuality(v); plumes.setHighQuality(v); rain.setHighQuality(v);
    },
    setCut(x, z) { atmoCut.value.set(x, z); },
    setCloudFade(v) {
      const f = Math.min(1, Math.max(0, v));
      clouds.uniforms.fade.value = f;
      cloudShadowU.on.value = f; // shadows fade with the clouds
    },
    trigger,
    dispose() {
      scene.remove(object);
      clouds.dispose(); plumes.dispose(); rain.dispose();
      boltGeo.dispose(); boltMat.dispose(); flashLight.dispose();
    },
  };
}
