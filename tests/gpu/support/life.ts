// Life test page: worldgen + derive, hand-painted biome/veg, the full look (or bare lighting with
// ?look=0) and createLife. Driven from life.spec.ts via window.lt; frames render on demand.
// URL: ?seed=<u32>&paint=stripes|natural&look=0|1&quality=low|high
import * as THREE from 'three/webgpu';
import { createStage } from '../../../src/render/stage';
import { GpuFields } from '../../../src/core/gpu';
import { registerSimFields, uploadWorld } from '../../../src/sim/fields';
import { generateWorld } from '../../../src/sim/worldgen';
import { createDerivePass } from '../../../src/sim/derive';
import { NCOL } from '../../../src/sim/layout';
import { createLook, type Look } from '../../../src/render/look';
import { createLighting } from '../../../src/render/lighting';
import { createTerrain } from '../../../src/render/terrain';
import { createSides } from '../../../src/render/sides';
import { createWater } from '../../../src/render/water';
import { HALF, voxelToWorldY, updateRenderColumns } from '../../../src/render/space';
import { createLife } from '../../../src/life/life';
import { birdFloor, fishValid } from '../../../src/life/creatures';
import {
  FLORA, FLORA_RULES, KIND_COUNT, KIND_BASE, KIND_NAMES, NSLOT, SPECIES_KIND, CREATURES, GRAZING, slotInfo, slotXZ,
  floraDecide, cornerWater, slopeAt, heightAt, columnAt, waterNear, smoothSurf, type ColumnMap,
} from '../../../src/life/lifeModel';
import { paintStripes, paintNatural, repaint } from './lifeWorld';

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
  createDerivePass(fields).run(renderer);
  const surfY = new Float32Array(await fields.read(renderer, 'surfY'));
  const water = world.water;
  let wetN = 0, wetSum = 0;
  for (let c = 0; c < NCOL; c++) if (water[c]! >= 1) { wetN++; wetSum += surfY[c]! + water[c]!; }
  const sea = wetN ? wetSum / wetN : 76;
  const paint = q.get('paint') ?? 'stripes';
  const biome = paint === 'natural' ? paintNatural(fields, surfY, water, sea) : paintStripes(fields, surfY, water, sea);

  let look: Look | null = null;
  if (q.get('look') !== '0') {
    try { look = createLook(stage, fields, { highQuality: q.get('quality') !== 'low' }); } catch (e) { console.warn('life page: createLook failed, bare lighting: ' + (e as Error).message); }
  }
  if (!look) {
    renderer.shadowMap.enabled = true;
    createLighting(scene, { shadows: true });
    scene.add(createTerrain(fields).object, createSides(fields).object, createWater(fields).object);
  }
  const quality = q.get('quality') !== 'low';
  const life = createLife(fields, renderer, scene, { highQuality: quality });

  let t = 0;
  const frame = async (dt = 1 / 60) => {
    t += dt;
    look?.frame(t);
    life.update(dt, t);
    if (look) look.post.render(); else { updateRenderColumns(renderer, fields); renderer.render(scene, camera); }
    await device.queue.onSubmittedWorkDone();
    await new Promise((r) => requestAnimationFrame(r));
  };

  const readMap = async (): Promise<ColumnMap & { veg: Float32Array }> => {
    const s = new Float32Array(await fields.read(renderer, 'surfY'));
    return {
      surfY: s, surfR: smoothSurf(s),
      water: new Float32Array(await fields.read(renderer, 'water')),
      biome: new Uint32Array(await fields.read(renderer, 'biome')),
      veg: new Float32Array(await fields.read(renderer, 'veg')),
    };
  };
  const readAttr = async (a: THREE.BufferAttribute) => renderer.getArrayBufferAsync(a as unknown as THREE.StorageBufferAttribute);

  async function flora(opts: { snapped: boolean; quality?: number }) {
    const m = await readMap();
    const st = new Float32Array(await readAttr(life.flora.buffers.state));
    const args = new Uint32Array(await readAttr(life.flora.buffers.args));
    const lists = new Uint32Array(await readAttr(life.flora.buffers.lists));
    const kinds = Array.from({ length: KIND_COUNT }, () => ({ live: 0, byBiome: {} as Record<number, number>, drawn: 0, listOk: true }));
    const bad = { wrongBiome: 0, wet: 0, steep: 0, cpuMismatch: 0, yMismatch: 0, examples: [] as string[] };
    const liveSlots: Set<number>[] = kinds.map(() => new Set());
    let live = 0, growing = 0, shrinking = 0;
    const perSpecies: Record<number, number> = {};
    for (let s = 0; s < NSLOT; s++) {
      const y = st[s * 4]!, prev = st[s * 4 + 1]!, tgt = st[s * 4 + 2]!, sp = st[s * 4 + 3]!;
      if (opts.snapped) {
        const ref = floraDecide(m, s, opts.quality ?? (quality ? 1 : FLORA.LOW_QUALITY));
        if (ref.species !== sp || Math.abs(ref.scale - tgt) > 1e-4 || Math.abs(prev - tgt) > 1e-6) {
          bad.cpuMismatch++;
          if (bad.examples.length < 5) bad.examples.push(`slot ${s}: gpu sp ${sp} tgt ${tgt} prev ${prev}, cpu ${ref.species} ${ref.scale}`);
        }
      }
      if (sp === 0 || Math.max(prev, tgt) <= 0) continue;
      live++;
      if (tgt > prev) growing++; else if (tgt < prev) shrinking++;
      perSpecies[sp] = (perSpecies[sp] ?? 0) + 1;
      const k = SPECIES_KIND[sp]!;
      const [x, z] = slotXZ(s);
      const b = m.biome[columnAt(x, z)]!;
      kinds[k]!.live++;
      kinds[k]!.byBiome[b] = (kinds[k]!.byBiome[b] ?? 0) + 1;
      liveSlots[k]!.add(s);
      const rule = FLORA_RULES[b]![slotInfo(s).set]!;
      if (tgt > 0 && sp !== rule.a && sp !== rule.b) bad.wrongBiome++;
      if (cornerWater(m, x, z) > FLORA.WET_MAX) bad.wet++;
      if (slopeAt(m, x, z) > FLORA.SLOPE_MAX) bad.steep++;
      if (Math.abs(y - heightAt(m, x, z)) > 1e-3) bad.yMismatch++;
    }
    for (let k = 0; k < KIND_COUNT; k++) {
      const n = args[k * 5 + 1]!;
      kinds[k]!.drawn = n;
      const seen = new Set<number>();
      for (let i = 0; i < n; i++) seen.add(lists[KIND_BASE[k]! + i]!);
      kinds[k]!.listOk = n === liveSlots[k]!.size && seen.size === n && [...seen].every((s) => liveSlots[k]!.has(s));
    }
    const named = Object.fromEntries(kinds.map((k, i) => [KIND_NAMES[i], k]));
    const biomeCols: Record<number, number> = {};
    for (let c = 0; c < NCOL; c++) biomeCols[m.biome[c]!] = (biomeCols[m.biome[c]!] ?? 0) + 1;
    return { live, growing, shrinking, kinds: named, perSpecies, bad, biomeCols };
  }

  /** Step creatures for `seconds` of fixed dt on the current map; check invariants every step. */
  async function creatures(seconds: number, dt = 1 / 60) {
    const m = await readMap();
    life.creatures.setMap(m);
    const v = { birdOut: 0, birdLow: 0, birdHigh: 0, critterWet: 0, critterBiome: 0, fishOut: 0, fishAbove: 0, examples: [] as string[] };
    let minClear = Infinity, maxAbs = 0;
    const start = life.creatures.snapshot();
    const path = { bird: 0, critter: 0, fish: 0 };
    let prev = start;
    const steps = Math.round(seconds / dt);
    for (let i = 0; i < steps; i++) {
      t += dt;
      life.creatures.step(dt, t);
      const s = life.creatures.snapshot();
      for (const b of s.birds) {
        maxAbs = Math.max(maxAbs, Math.abs(b.x), Math.abs(b.z));
        if (Math.abs(b.x) > HALF || Math.abs(b.z) > HALF) v.birdOut++;
        const fl = birdFloor(m, b.x, b.z) - CREATURES.BIRD_CLEARANCE;
        minClear = Math.min(minClear, b.y - fl);
        if (b.y < fl + CREATURES.BIRD_CLEARANCE - 1e-6) { v.birdLow++; if (v.examples.length < 5) v.examples.push(`bird low ${b.x.toFixed(3)},${b.y.toFixed(3)},${b.z.toFixed(3)}`); }
        if (b.y > CREATURES.BIRD_CEIL + 1e-6) v.birdHigh++;
      }
      for (const c of s.critters) {
        if (m.water[columnAt(c.x, c.z)]! > 0 || waterNear(m, c.x, c.z) > CREATURES.CRITTER_WET) { v.critterWet++; if (v.examples.length < 5) v.examples.push(`critter wet ${c.x},${c.z}`); }
        if (!GRAZING.has(m.biome[columnAt(c.x, c.z)]!)) v.critterBiome++;
      }
      for (const f of s.fish) {
        if (!fishValid(m, f.x, f.z)) v.fishOut++;
        const col = columnAt(f.x, f.z);
        if (f.y >= voxelToWorldY(m.surfY[col]! + m.water[col]!) || f.y <= voxelToWorldY(m.surfY[col]!)) v.fishAbove++;
      }
      const d = (a: { x: number; z: number }[], b: { x: number; z: number }[]) => a.reduce((acc, p, j) => acc + Math.hypot(p.x - b[j]!.x, p.z - b[j]!.z), 0);
      if (prev.birds.length === s.birds.length) path.bird += d(s.birds, prev.birds);
      if (prev.critters.length === s.critters.length) path.critter += d(s.critters, prev.critters);
      if (prev.fish.length === s.fish.length) path.fish += d(s.fish, prev.fish);
      prev = s;
      if (i % 600 === 599) await new Promise((r) => setTimeout(r, 0));
    }
    return { counts: life.creatures.counts, violations: v, minClear, maxAbs, path };
  }

  const setCam = (pos: number[], target: number[], fov = 30) => {
    camera.position.fromArray(pos);
    stage.controls.target.fromArray(target);
    camera.fov = fov;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    camera.lookAt(stage.controls.target);
    camera.updateMatrixWorld();
  };

  /** A world point showing `what`, for close-up framing. */
  async function spot(what: string): Promise<number[] | null> {
    const snap = life.creatures.snapshot();
    const m = await readMap();
    const worldAt = (x: number, z: number) => [x, voxelToWorldY(heightAt(m, x, z)), z];
    if (what === 'critter') {
      // the critter with most herd mates around (open pasture, not a lone one at a forest edge)
      let best: { x: number; z: number } | null = null, bestN = -1;
      for (const c of snap.critters) {
        const n = snap.critters.filter((o) => Math.hypot(o.x - c.x, o.z - c.z) < 0.12).length;
        if (n > bestN && Math.abs(c.x) < HALF - 0.4 && Math.abs(c.z) < HALF - 0.4) { bestN = n; best = c; }
      }
      return best ? worldAt(best.x, best.z) : null;
    }
    if (what === 'fish') return snap.fish[0] ? [snap.fish[0].x, snap.fish[0].y, snap.fish[0].z] : null;
    if (what === 'bird') return snap.birds[0] ? [snap.birds[0].x, snap.birds[0].y, snap.birds[0].z] : null;
    const kind = KIND_NAMES.indexOf(what as (typeof KIND_NAMES)[number]);
    if (kind < 0) return null;
    const st = new Float32Array(await readAttr(life.flora.buffers.state));
    // densest spot: the live slot of this kind with most same-kind neighbours, sampled
    let best: number[] | null = null, bestN = -1;
    const pts: [number, number][] = [];
    for (let s = 0; s < NSLOT; s++) if (st[s * 4 + 3]! > 0 && SPECIES_KIND[st[s * 4 + 3]!] === kind && st[s * 4 + 2]! > 0) pts.push(slotXZ(s));
    for (let i = 0; i < pts.length; i += Math.max(1, Math.floor(pts.length / 300))) {
      const [x, z] = pts[i]!;
      if (Math.abs(x) > HALF - 0.4 || Math.abs(z) > HALF - 0.4) continue;
      let n = 0;
      for (let j = 0; j < pts.length; j += Math.max(1, Math.floor(pts.length / 2000))) if (Math.hypot(pts[j]![0] - x, pts[j]![1] - z) < 0.25) n++;
      if (n > bestN) { bestN = n; best = worldAt(x, z); }
    }
    return best;
  }

  const w = window as unknown as Record<string, unknown>;
  w.lt = {
    meta: { sea, paint, quality, hasLook: !!look },
    life, fields, renderer, stage,
    frame, flora, creatures, spot,
    async frames(n: number, dt = 1 / 60) { for (let i = 0; i < n; i++) await frame(dt); },
    refresh(snap: boolean) { life.flora.refresh(renderer, snap); },
    advance(s: number) { life.flora.frame(s, t); },
    repaint(from: number, to: number) { repaint(fields, biome, from, to); },
    reset() { biome.set(paint === 'natural' ? paintNatural(fields, surfY, water, sea) : paintStripes(fields, surfY, water, sea)); },
    setCam,
    async view(name: string, dist = 1) {
      if (name === 'hero') setCam([5.6, 3.4, 5.9], [0, -0.12, 0]);
      else if (name === 'top') setCam([0.01, 9.5, 0.01], [0, 0, 0]);
      else {
        const p = await spot(name);
        if (!p) return false;
        const off = name === 'fish' ? [0.22, 0.3, 0.22] : name === 'bird' ? [0.35, 0.12, 0.4] : name === 'critter' ? [0.26, 0.15, 0.3] : [0.55, 0.35, 0.6];
        setCam([p[0]! + off[0]! * dist, p[1]! + off[1]! * dist, p[2]! + off[2]! * dist], p);
      }
      return true;
    },
    drawInfo() { const i = renderer.info.render; return { calls: i.drawCalls, triangles: i.triangles }; },
  };
  w.ltReady = true;
}

main().catch((e) => { console.error('life test page failed: ' + (e as Error).stack); });
