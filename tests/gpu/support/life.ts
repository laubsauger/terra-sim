// Life test page: worldgen + derive, hand-painted biome/veg, the full look (or bare lighting with
// ?look=0) and createLife. Driven from life.spec.ts via window.lt; frames render on demand.
// URL: ?seed=<u32>&paint=stripes|natural&look=0|1&bare=0|1&quality=low|high&tod=<0..1>
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
  TREELINE, TREELINE_BIOME, LIFE_BIOMES, Sp, NSLOT_LS, grazingPatches,
  floraDecide, cornerWater, slopeAt, heightAt, columnAt, waterNear, baseSink, type ColumnMap,
} from '../../../src/life/lifeModel';
import { paintStripes, paintNatural, repaint, paintAll } from './lifeWorld';

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
  const world = generateWorld(Number(q.get('seed') ?? 5));
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
    try { look = createLook(stage, fields, { highQuality: q.get('quality') !== 'low', timeOfDay: q.has('tod') ? Number(q.get('tod')) : undefined }); } catch (e) { console.warn('life page: createLook failed, bare lighting: ' + (e as Error).message); }
  }
  if (!look) {
    renderer.shadowMap.enabled = true;
    createLighting(scene, { shadows: true });
    // bare=1: life only (data tests stay independent of in-progress terrain/sides/water shaders)
    if (q.get('bare') !== '1') scene.add(createTerrain(fields).object, createSides(fields).object, createWater(fields).object);
  }
  const quality = q.get('quality') !== 'low';
  const life = createLife(fields, renderer, scene, { highQuality: quality, camera });
  life.setSeaLevel(sea);

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
      surfY: s, surfR: new Float32Array(await readAttr(life.flora.buffers.renderH)),
      water: new Float32Array(await fields.read(renderer, 'water')),
      biome: new Uint32Array(await fields.read(renderer, 'biome')),
      veg: new Float32Array(await fields.read(renderer, 'veg')),
    };
  };
  const readAttr = async (a: THREE.BufferAttribute) => renderer.getArrayBufferAsync(a as unknown as THREE.StorageBufferAttribute);

  async function flora(opts: { snapped: boolean; quality?: number }) {
    const m = await readMap();
    // fine ground cover: compact every live slot (no camera culling) so lists can be checked exactly
    life.flora.setCamera(null);
    life.flora.cullFine(renderer);
    const st = new Float32Array(await readAttr(life.flora.buffers.state));
    const args = new Uint32Array(await readAttr(life.flora.buffers.args));
    const lists = new Uint32Array(await readAttr(life.flora.buffers.lists));
    const kinds = Array.from({ length: KIND_COUNT }, () => ({ live: 0, byBiome: {} as Record<number, number>, drawn: 0, listOk: true }));
    const bad = { wrongBiome: 0, wet: 0, steep: 0, cpuMismatch: 0, yMismatch: 0, aboveBare: 0, examples: [] as string[] };
    const zone = [0, 1, 2, 3].map(() => ({ cols: 0, trees: 0, scaleSum: 0, pines: 0, tempTrees: 0, line: 0 }));
    const liveSlots: Set<number>[] = kinds.map(() => new Set());
    let live = 0, growing = 0, shrinking = 0;
    const perSpecies: Record<number, number> = {};
    for (let s = 0; s < NSLOT; s++) {
      const y = st[s * 4]!, prev = st[s * 4 + 1]!, tgt = st[s * 4 + 2]!, sp = Math.floor(st[s * 4 + 3]!);
      if (opts.snapped) {
        const ref = floraDecide(m, s, opts.quality ?? (quality ? 1 : FLORA.LOW_QUALITY), sea);
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
      const set = slotInfo(s).set;
      const rule = FLORA_RULES[b < LIFE_BIOMES ? b : 0]![set]!;
      // allowed: the rule's species, plus the altitude substitutes (pine for broadleaf, treeline shrubs/tufts)
      const allowed = new Set<number>([...rule.sp, rule.flower]);
      if (set === 2 && rule.density > 0) allowed.add(Sp.REED);
      if (set === 0 && (allowed.has(Sp.BROADLEAF) || allowed.has(Sp.BROADLEAF_RAIN))) allowed.add(Sp.PINE);
      if (set === 1 && TREELINE_BIOME[b]) { allowed.add(Sp.SHRUB_TUNDRA); allowed.add(Sp.GRASS_ALPINE); }
      allowed.delete(Sp.NONE);
      if (tgt > 0 && !allowed.has(sp)) { bad.wrongBiome++; if (bad.examples.length < 5) bad.examples.push(`species ${sp} on biome ${b}`); }
      const alt = m.surfY[columnAt(x, z)]! - sea;
      if (alt > TREELINE.BARE_A1) bad.aboveBare++;
      // treeline stats (forest biomes): per altitude bin, large-set plants, their scale, pines, line shrubs/tufts
      if (TREELINE_BIOME[b]) {
        const bin = alt < TREELINE.TREE_A0 ? 0 : alt < (TREELINE.TREE_A0 + TREELINE.TREE_A1) / 2 ? 1 : alt < TREELINE.TREE_A1 ? 2 : 3;
        const z0 = zone[bin]!;
        if (set === 0) { z0.trees++; z0.scaleSum += tgt; if (sp === Sp.PINE && b === 4) z0.pines++; if (b === 4) z0.tempTrees++; }
        else if (sp === Sp.SHRUB_TUNDRA || sp === Sp.GRASS_ALPINE) z0.line++;
      }
      const cw = cornerWater(m, x, z);
      if (sp === Sp.REED ? cw > Math.fround(FLORA.REED_MAX) : cw > Math.fround(FLORA.WET_MAX)) bad.wet++;
      if (slopeAt(m, x, z) > FLORA.SLOPE_MAX_SET[set]!) bad.steep++;
      // base = terrain as drawn, sunk on slopes by the footprint (lifeModel.baseSink)
      const yRef = heightAt(m, x, z) - baseSink(slopeAt(m, x, z), sp, tgt);
      if (Math.abs(y - yRef) > 2e-3) { bad.yMismatch++; if (bad.examples.length < 5) bad.examples.push(`y slot ${s} sp ${sp}: gpu ${y} cpu ${yRef} (h ${heightAt(m, x, z)})`); }
    }
    for (let k = 0; k < KIND_COUNT; k++) {
      const n = args[k * 5 + 1]!;
      kinds[k]!.drawn = n;
      const seen = new Set<number>();
      for (let i = 0; i < n; i++) seen.add(lists[KIND_BASE[k]! + i]!);
      kinds[k]!.listOk = n === liveSlots[k]!.size && seen.size === n && [...seen].every((s) => liveSlots[k]!.has(s));
    }
    life.flora.setCamera(camera);
    const named = Object.fromEntries(kinds.map((k, i) => [KIND_NAMES[i], k]));
    const biomeCols: Record<number, number> = {};
    for (let c = 0; c < NCOL; c++) {
      const b = m.biome[c]!;
      biomeCols[b] = (biomeCols[b] ?? 0) + 1;
      if (!TREELINE_BIOME[b]) continue;
      const alt = m.surfY[c]! - sea;
      zone[alt < TREELINE.TREE_A0 ? 0 : alt < (TREELINE.TREE_A0 + TREELINE.TREE_A1) / 2 ? 1 : alt < TREELINE.TREE_A1 ? 2 : 3]!.cols++;
    }
    return { live, growing, shrinking, kinds: named, perSpecies, bad, biomeCols, zone };
  }

  /**
   * Clumping of the fine ground cover (painted world, snapped): per 0.2-unit cell, live fine plants
   * over eligible fine slots (inside, dry, gentle). Returns mean, p95, bare-cell share and flower stats.
   */
  async function clumps() {
    const m = await readMap();
    const st = new Float32Array(await readAttr(life.flora.buffers.state));
    const CELLW = 0.2, NC = Math.round((2 * HALF) / CELLW);
    const live = new Float64Array(NC * NC), elig = new Float64Array(NC * NC), flw = new Float64Array(NC * NC);
    const species: Record<number, number> = {};
    let fineLive = 0;
    for (let s = NSLOT_LS; s < NSLOT; s++) {
      const [x, z] = slotXZ(s);
      if (Math.abs(x) > HALF - 0.01 || Math.abs(z) > HALF - 0.01) continue;
      if (cornerWater(m, x, z) > FLORA.WET_MAX || slopeAt(m, x, z) > FLORA.SLOPE_MAX_SET[2]!) continue;
      const ci = Math.min(NC - 1, Math.floor((x + HALF) / CELLW)) + Math.min(NC - 1, Math.floor((z + HALF) / CELLW)) * NC;
      elig[ci] = elig[ci]! + 1;
      const sp = Math.floor(st[s * 4 + 3]!);
      if (sp === 0 || st[s * 4 + 2]! <= 0) continue;
      live[ci] = live[ci]! + 1; fineLive++;
      species[sp] = (species[sp] ?? 0) + 1;
      if (sp === Sp.F_FLOWER) flw[ci] = flw[ci]! + 1;
    }
    const dens: number[] = [], fsh: number[] = [];
    for (let i = 0; i < NC * NC; i++) if (elig[i]! >= 40) { dens.push(live[i]! / elig[i]!); fsh.push(live[i]! ? flw[i]! / live[i]! : 0); }
    dens.sort((a, b) => a - b); fsh.sort((a, b) => a - b);
    const mean = dens.reduce((a, b) => a + b, 0) / Math.max(1, dens.length);
    const q = (arr: number[], f: number) => arr[Math.min(arr.length - 1, Math.floor(arr.length * f))] ?? 0;
    const fmean = fsh.reduce((a, b) => a + b, 0) / Math.max(1, fsh.length);
    return { cells: dens.length, mean, p50: q(dens, 0.5), p95: q(dens, 0.95), max: q(dens, 1), bareShare: dens.filter((d) => d < mean * 0.25).length / Math.max(1, dens.length),
      flowerMean: fmean, flowerP95: q(fsh, 0.95), flowerCellsNone: fsh.filter((f) => f === 0).length / Math.max(1, fsh.length), fineLive, species };
  }

  /** Step creatures for `seconds` of fixed dt on the current map; check invariants every step. */
  async function creatures(seconds: number, dt = 1 / 60) {
    const m = await readMap();
    life.creatures.setMap(m);
    const v = { birdOut: 0, birdLow: 0, birdHigh: 0, critterWet: 0, critterBiome: 0, critterSmallPatch: 0, fishOut: 0, fishAbove: 0, examples: [] as string[] };
    const patches = grazingPatches(m);
    let minClear = Infinity, maxAbs = 0, maxFishDepth = 0;
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
        const l = patches.label[columnAt(c.x, c.z)]!;
        if (l < 0 || patches.size[l]! < CREATURES.HERD_MIN_AREA) v.critterSmallPatch++;
      }
      for (const f of s.fish) {
        if (!fishValid(m, f.x, f.z)) v.fishOut++;
        const col = columnAt(f.x, f.z);
        // at least FISH_SURFACE_MIN below the water level, above the bed
        if (f.y > voxelToWorldY(m.surfY[col]! + m.water[col]! - CREATURES.FISH_SURFACE_MIN) + 1e-6 || f.y <= voxelToWorldY(m.surfY[col]!)) v.fishAbove++;
        if (f.depth < CREATURES.FISH_SURFACE_MIN - 1e-6) v.fishAbove++;
        maxFishDepth = Math.max(maxFishDepth, f.depth);
      }
      const d = (a: { x: number; z: number }[], b: { x: number; z: number }[]) => a.reduce((acc, p, j) => acc + Math.hypot(p.x - b[j]!.x, p.z - b[j]!.z), 0);
      if (prev.birds.length === s.birds.length) path.bird += d(s.birds, prev.birds);
      if (prev.critters.length === s.critters.length) path.critter += d(s.critters, prev.critters);
      if (prev.fish.length === s.fish.length) path.fish += d(s.fish, prev.fish);
      prev = s;
      if (i % 600 === 599) await new Promise((r) => setTimeout(r, 0));
    }
    return { counts: life.creatures.counts, violations: v, minClear, maxAbs, path, maxFishDepth };
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
    if (what === 'fish' || what === 'fishlow') return snap.fish[0] ? [snap.fish[0].x, snap.fish[0].y, snap.fish[0].z] : null;
    if (what === 'bird') return snap.birds[0] ? [snap.birds[0].x, snap.birds[0].y, snap.birds[0].z] : null;
    if (what === 'treeline') {
      // a forest column just below the treeline, with high ground and lowland forest around
      let best: number[] | null = null, bestScore = -1;
      for (let c = 0; c < NCOL; c += 7) {
        const alt = m.surfY[c]! - sea;
        if (!TREELINE_BIOME[m.biome[c]!] || alt < TREELINE.TREE_A1 - 6 || alt > TREELINE.TREE_A1) continue;
        const x = ((c % 256) + 0.5) / 64 - 2, z = (Math.floor(c / 256) + 0.5) / 64 - 2;
        if (Math.abs(x) > HALF - 0.5 || Math.abs(z) > HALF - 0.5) continue;
        const score = alt;
        if (score > bestScore) { bestScore = score; best = worldAt(x, z); }
      }
      return best;
    }
    const groups: Record<string, number[]> = {
      meadow: [Sp.F_GRASS, Sp.F_GRASS_TALL, Sp.F_SEED, Sp.F_CLOVER, Sp.F_FLOWER], meadowmid: [Sp.F_GRASS, Sp.F_GRASS_TALL, Sp.F_SEED, Sp.F_CLOVER, Sp.F_FLOWER],
      steppe: [Sp.FEATHER, Sp.F_STRAW, Sp.TUSSOCK], steppemid: [Sp.FEATHER, Sp.F_STRAW, Sp.TUSSOCK],
      ground: [Sp.BOULDER, Sp.STONE, Sp.BUSH_GREEN, Sp.SAGEBRUSH, Sp.TUSSOCK],
    };
    const kind = KIND_NAMES.indexOf(what as (typeof KIND_NAMES)[number]);
    if (kind < 0 && !groups[what]) return null;
    const want = (sp: number) => (groups[what] ? groups[what]!.includes(sp) : SPECIES_KIND[sp] === kind);
    const st = new Float32Array(await readAttr(life.flora.buffers.state));
    // densest spot: the live slot of this kind with most same-kind neighbours, sampled
    let best: number[] | null = null, bestN = -1;
    const pts: [number, number][] = [];
    for (let s = 0; s < NSLOT; s++) if (st[s * 4 + 3]! > 0 && want(Math.floor(st[s * 4 + 3]!)) && st[s * 4 + 2]! > 0) pts.push(slotXZ(s));
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
    life, fields, renderer, stage, look,
    frame, flora, creatures, spot, clumps,
    async frames(n: number, dt = 1 / 60) { for (let i = 0; i < n; i++) await frame(dt); },
    refresh(snap: boolean) { life.flora.refresh(renderer, snap); },
    advance(s: number) { life.flora.frame(renderer, s, t); },
    repaint(from: number, to: number) { repaint(fields, biome, from, to); },
    paintAll(b: number) { paintAll(fields, biome, water, b); },
    reset() { biome.set(paint === 'natural' ? paintNatural(fields, surfY, water, sea) : paintStripes(fields, surfY, water, sea)); },
    setCam,
    async view(name: string, dist = 1) {
      if (name === 'hero') setCam([5.6, 3.4, 5.9], [0, -0.12, 0]);
      else if (name === 'top') setCam([0.01, 9.5, 0.01], [0, 0, 0]);
      else {
        const p = await spot(name);
        if (!p) return false;
        const off = name === 'fish' ? [0.22, 0.3, 0.22] : name === 'fishlow' ? [0.3, 0.07, 0.3] : name === 'bird' ? [0.35, 0.12, 0.4] : name === 'critter' ? [0.26, 0.15, 0.3] : name === 'treeline' ? [0.9, 0.35, 1.0] : name === 'meadow' || name === 'steppe' ? [0.28, 0.1, 0.32] : name === 'ground' ? [0.13, 0.075, 0.16] : name === 'meadowmid' || name === 'steppemid' ? [0.85, 0.42, 0.95] : [0.55, 0.35, 0.6];
        let o = off;
        if (name === 'fishlow') {
          // grazing view across open water: turn the offset until the camera hovers over the sea
          const m = await readMap();
          for (let k = 0; k < 16; k++) {
            const a = (k / 16) * Math.PI * 2, r = Math.hypot(off[0]!, off[2]!);
            const c = [Math.cos(a) * r, off[1]!, Math.sin(a) * r];
            const cx = p[0]! + c[0]!, cz = p[2]! + c[2]!;
            if (Math.abs(cx) < HALF && Math.abs(cz) < HALF && m.water[columnAt(cx, cz)]! > 1 && m.water[columnAt((cx + p[0]!) / 2, (cz + p[2]!) / 2)]! > 1) { o = c; break; }
          }
        }
        setCam([p[0]! + o[0]! * dist, p[1]! + o[1]! * dist, p[2]! + o[2]! * dist], p);
      }
      return true;
    },
    drawInfo() { const i = renderer.info.render; return { calls: i.drawCalls, triangles: i.triangles }; },
  };
  w.ltReady = true;
}

main().catch((e) => { console.error('life test page failed: ' + (e as Error).stack); });
