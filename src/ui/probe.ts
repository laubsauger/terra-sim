// Inspect probe UI (§T.12, §I.probe): click the terrain or a cut face, see what the column is made of.
// Picking is a CPU raymarch against a fresh surfY readback, so it matches exactly what the sim holds.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import { NX, NZ, NY, Mat, colIdx } from '../sim/layout';
import { Probe, type ProbeResult } from '../sim/probe';
import { Biome } from '../sim/biomeModel';
import { MAT_LOOK } from '../render/palette';
import { HALF, voxelToWorldY, worldToCell, worldToVoxelY } from '../render/space';

const MAT_NAME = Object.fromEntries(Object.entries(Mat).map(([k, v]) => [v, k.toLowerCase()])) as Record<number, string>;
const BIOME_NAME = Object.fromEntries(Object.entries(Biome).map(([k, v]) => [v, k.toLowerCase().replace('_', ' ')])) as Record<number, string>;

export interface PickHit { x: number; z: number; y?: number; face: 'top' | 'side' }

/** Raymarch a world ray against the heightfield and the block's cut faces. */
export function pickColumn(ray: THREE.Ray, surfY: Float32Array): PickHit | null {
  const yTop = voxelToWorldY(NY), yBot = voxelToWorldY(0);
  const box = new THREE.Box3(new THREE.Vector3(-HALF, yBot, -HALF), new THREE.Vector3(HALF, yTop, HALF));
  const enter = ray.intersectBox(box, new THREE.Vector3());
  if (!enter) return null;
  const surfAt = (cx: number, cz: number) => surfY[colIdx(Math.min(NX - 1, Math.max(0, Math.round(cx))), Math.min(NZ - 1, Math.max(0, Math.round(cz))))]!;
  // entry on a side face below the surface → side hit
  const onSide = Math.abs(Math.abs(enter.x) - HALF) < 1e-4 || Math.abs(Math.abs(enter.z) - HALF) < 1e-4;
  const cx0 = worldToCell(enter.x), cz0 = worldToCell(enter.z);
  if (onSide && worldToVoxelY(enter.y) < surfAt(cx0, cz0)) {
    return { x: Math.round(Math.min(NX - 1, Math.max(0, cx0))), z: Math.round(Math.min(NZ - 1, Math.max(0, cz0))), y: Math.floor(worldToVoxelY(enter.y)), face: 'side' };
  }
  const p = enter.clone(), step = ray.direction.clone().multiplyScalar(0.004);
  for (let i = 0; i < 4000; i++) {
    p.add(step);
    if (Math.abs(p.x) > HALF || Math.abs(p.z) > HALF || p.y < yBot) return null;
    const cx = worldToCell(p.x), cz = worldToCell(p.z);
    if (worldToVoxelY(p.y) <= surfAt(cx, cz)) return { x: Math.round(cx), z: Math.round(cz), face: 'top' };
  }
  return null;
}

export function createProbeUI(renderer: THREE.WebGPURenderer, camera: THREE.Camera, fields: GpuFields) {
  const probe = new Probe(fields);
  const el = document.createElement('div');
  el.className = 'terra-probe';
  el.innerHTML = `<canvas class="strip" width="18" height="128"></canvas><div class="body"><div class="title"></div><div class="rows"></div><canvas class="spark" width="160" height="28"></canvas></div><button class="close">×</button>`;
  el.style.display = 'none';
  document.body.appendChild(el);
  const strip = el.querySelector<HTMLCanvasElement>('.strip')!;
  const spark = el.querySelector<HTMLCanvasElement>('.spark')!;
  let pinned: PickHit | null = null;
  let history: number[] = [];
  let timer = 0;
  el.querySelector('.close')!.addEventListener('click', () => { pinned = null; el.style.display = 'none'; });

  const hex = (c: number) => `#${c.toString(16).padStart(6, '0')}`;
  function draw(r: ProbeResult, hit: PickHit) {
    const g = strip.getContext('2d')!;
    g.clearRect(0, 0, strip.width, strip.height);
    for (let y = 0; y < NY; y++) {
      const v = r.voxels[y]!;
      if (v.mat === Mat.AIR) continue;
      g.fillStyle = hex(MAT_LOOK[v.mat]!.color);
      g.fillRect(0, NY - 1 - y, strip.width, 1);
    }
    const waterTop = Math.min(NY, r.surfY + r.water);
    if (r.water > 0.05) { g.fillStyle = 'rgba(60,150,220,0.55)'; g.fillRect(0, NY - waterTop, strip.width, r.water); }
    if (hit.y !== undefined) { g.fillStyle = '#fff'; g.fillRect(0, NY - 1 - hit.y, strip.width, 1); }
    const top = r.voxels[Math.max(0, Math.ceil(r.surfY) - 1)]!;
    const at = hit.y !== undefined ? r.voxels[hit.y] : top;
    el.querySelector('.title')!.textContent = `${MAT_NAME[at?.mat ?? 0]} · col ${hit.x},${hit.z}${hit.y !== undefined ? ` · y ${hit.y}` : ''}`;
    const rows: [string, string][] = [
      ['elevation', `${r.surfY.toFixed(1)}`],
      ['water', r.water > 0.05 ? `${r.water.toFixed(1)}` : '—'],
      ['plate', `#${r.plateId} · ${r.continental ? 'continental' : 'oceanic'}`],
      ['crust', `${r.crustLayers.toFixed(0)} layers · ${r.crustAgeMy.toFixed(0)} My`],
      ['rock age', at ? `${at.ageMy.toFixed(0)} My` : '—'],
      ['temp', `${r.surfTemp.toFixed(1)}°`],
      ['biome', BIOME_NAME[r.biome] ?? `${r.biome}`],
    ];
    el.querySelector('.rows')!.innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
    const s = spark.getContext('2d')!;
    s.clearRect(0, 0, spark.width, spark.height);
    if (history.length > 1) {
      const lo = Math.min(...history), hi = Math.max(...history, lo + 1);
      s.strokeStyle = '#e8c27a'; s.lineWidth = 1.5; s.beginPath();
      history.forEach((h, i) => { const px = (i / (history.length - 1)) * spark.width, py = spark.height - 2 - ((h - lo) / (hi - lo)) * (spark.height - 4); if (i) s.lineTo(px, py); else s.moveTo(px, py); });
      s.stroke();
    }
  }

  async function sample(hit: PickHit) {
    const r = await probe.read(renderer, hit.x, hit.z, { veg: 0, ice: 0, vapor: 0, sedSusp: 0 });
    if (pinned !== hit) return;
    history.push(r.surfY);
    if (history.length > 120) history.shift();
    draw(r, hit);
  }

  async function pickAt(clientX: number, clientY: number) {
    const rect = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    const surf = new Float32Array(await fields.read(renderer, 'surfY'));
    const hit = pickColumn(ray.ray, surf);
    if (!hit) { pinned = null; el.style.display = 'none'; return; }
    pinned = hit; history = [];
    el.style.display = '';
    el.style.left = `${Math.min(window.innerWidth - 260, clientX + 16)}px`;
    el.style.top = `${Math.min(window.innerHeight - 180, clientY + 16)}px`;
    await sample(hit);
  }

  // click without drag (orbit controls own drags)
  let down: [number, number] | null = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && e.button === 0) void pickAt(e.clientX, e.clientY);
    down = null;
  });

  return {
    el,
    pickAt,
    /** Call once per frame; refreshes the pinned column about once a second. */
    update(realDt: number) {
      if (!pinned) return;
      timer += realDt;
      if (timer >= 1) { timer = 0; void sample(pinned); }
    },
    setVisible(v: boolean) { if (!v) el.style.display = 'none'; else if (pinned) el.style.display = ''; },
  };
}
