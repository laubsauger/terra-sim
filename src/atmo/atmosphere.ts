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
import { ATMO, AMB_PERIOD, BURST } from './atmoModel';
import { createClouds, cloudShadowU, type Clouds } from './clouds';
import { atmoSeaLevel } from './atmoTsl';
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
  /** Fire an event's FX directly (tests, god tools preview). */
  trigger(e: Pick<GeoEvent, 'kind' | 'x' | 'z' | 'magnitude'> & { radius?: number }): void;
  /** Debug counters. */
  readonly stats: { flashes: number; bursts: number; lightning: number };
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
  bolt.renderOrder = 4;
  const flashLight = new THREE.PointLight(0xcfd8ff, 0, 3.5, 2);
  flashLight.name = 'atmoFlash';
  object.add(bolt, flashLight);
  scene.add(object);

  // ---- state ----
  const stats = { flashes: 0, bursts: 0, lightning: 0 };
  let fxTime = 0;
  let first = true;
  let seaLevel = Y_SEA_NOMINAL;
  const seen = new WeakSet<GeoEvent>();
  let impact: { t0: number; pos: THREE.Vector3; mag: number } | null = null;
  let strike: { t0: number; dur: number; idx: number; mid: THREE.Vector3 } | null = null;
  // cloudlet readback for lightning (small, throttled)
  let cells: Float32Array | null = null;
  let sinceRead = 0, reading = false;
  const key = new THREE.Vector3();

  const groundY = () => voxelToWorldY(seaLevel);
  function trigger(e: Pick<GeoEvent, 'kind' | 'x' | 'z' | 'magnitude'> & { radius?: number }): void {
    const x = cellToWorld(e.x), z = cellToWorld(e.z), mag = e.magnitude;
    const y = groundY();
    const add = (kind: number, dur: number, radius: number, m = mag) => { plumes.addBurst({ x, y, z, kind, t0: fxTime, dur, mag: m, radius }); stats.bursts++; };
    switch (e.kind) {
      case 'volcano':
        add(BURST.ASH, 14 + 8 * mag, 0.05);
        break;
      case 'meteor': {
        const r = (e.radius ?? 4 + 8 * mag) * CELL;
        add(BURST.DUST, 2.5, r * 2.5, 1);
        add(BURST.HAZE, 18, r * 4, 0.5);
        impact = { t0: fxTime, pos: new THREE.Vector3(x, y + 0.05, z), mag };
        stats.flashes++;
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
    renderer.getArrayBufferAsync(clouds.cells.value as unknown as THREE.StorageBufferAttribute)
      .then((a) => { cells = new Float32Array(a); })
      .catch((err) => console.error('atmosphere: cloud readback failed: ' + (err as Error).message))
      .finally(() => { reading = false; });
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
        clouds.uniforms.flash.value = flick;
        light = flick * 2.5;
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
        buildBolt(top, bottom);
        bolt.visible = true;
        strike = { t0: fxTime, dur: 0.32 + Math.random() * 0.2, idx: c, mid: top.clone().lerp(bottom, 0.3) };
        clouds.uniforms.flashPos.value.copy(top).lerp(bottom, -0.4);
        stats.lightning++;
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
        plumes.uniforms.flashSize.value = 0.25 + t * 0.9;
        if (env * 6 > light) { light = env * 6; flashLight.position.copy(impact.pos); flashLight.color.setRGB(1, 0.75, 0.5); }
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
      if (lightningOn && !reading && sinceRead >= 1) { sinceRead = 0; readClouds(); }
      updateLightning(dt);
    },
    setHighQuality(v) {
      hq = v;
      clouds.setHighQuality(v); plumes.setHighQuality(v); rain.setHighQuality(v);
    },
    trigger,
    dispose() {
      scene.remove(object);
      clouds.dispose(); plumes.dispose(); rain.dispose();
      boltGeo.dispose(); boltMat.dispose(); flashLight.dispose();
    },
  };
}
