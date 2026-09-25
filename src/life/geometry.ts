// §T.45 low-poly life meshes, built once at init. Every mesh carries a `lifeP` vec4 attribute the
// vertex shaders animate from:
//   flora:     (h01 height 0..1, part, flex 0..1, 0)  part: 0 trunk, 1 foliage, 2 petal/accent, 3 blade
//   bird:      (part, |x| span for the wing flap, 0, 0)
//   critter:   (part, leg sign ±1, leg hip z, 0)    part: 0 body, 1 head, 2 leg
//   fish:      (along-body 0 nose..1 tail, part, 0, 0)
import * as THREE from 'three/webgpu';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

interface Part {
  g: THREE.BufferGeometry;
  /** Per-vertex lifeP from the (already transformed) vertex position. */
  attr: (p: THREE.Vector3) => [number, number, number, number];
  /** Bend normals toward radial from this point (soft, puffy crowns). */
  puff?: { c: THREE.Vector3; k: number };
  /** Blend normals toward +Y (grass reads with the ground). */
  up?: number;
}

const v = new THREE.Vector3();
const n = new THREE.Vector3();

function prep(p: Part): THREE.BufferGeometry {
  let g = p.g;
  g.deleteAttribute('uv');
  if (!g.index) g = mergeVertices(g);
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const nrm = g.getAttribute('normal') as THREE.BufferAttribute;
  const lp = new Float32Array(pos.count * 4);
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    lp.set(p.attr(v), i * 4);
    if (p.puff || p.up) {
      n.fromBufferAttribute(nrm, i);
      if (p.puff) n.lerp(v.clone().sub(p.puff.c).normalize(), p.puff.k);
      if (p.up) n.lerp(new THREE.Vector3(0, 1, 0), p.up);
      n.normalize();
      nrm.setXYZ(i, n.x, n.y, n.z);
    }
  }
  g.setAttribute('lifeP', new THREE.BufferAttribute(lp, 4));
  return g;
}

function merge(parts: Part[], normalizeHeight = true): THREE.BufferGeometry {
  const g = mergeGeometries(parts.map(prep), false)!;
  if (normalizeHeight) {
    // unit height: y ∈ [0, 1], and h01 = y
    g.computeBoundingBox();
    const top = g.boundingBox!.max.y;
    g.scale(1 / top, 1 / top, 1 / top);
    const pos = g.getAttribute('position') as THREE.BufferAttribute;
    const lp = g.getAttribute('lifeP') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) lp.setX(i, Math.max(0, pos.getY(i)));
  }
  g.computeBoundingSphere();
  return g;
}

const at = (g: THREE.BufferGeometry, x: number, y: number, z: number) => g.translate(x, y, z);
const blob = (r: number, detail: number, x: number, y: number, z: number, sx = 1, sy = 1, sz = 1) =>
  at(new THREE.IcosahedronGeometry(r, detail).scale(sx, sy, sz), x, y, z);
const cyl = (r0: number, r1: number, h: number, seg: number, open = true) => new THREE.CylinderGeometry(r1, r0, h, seg, 1, open);

const flora = (part: number, flexK = 1) => (p: THREE.Vector3): [number, number, number, number] => [p.y, part, flexK, 0];

/** Grass-like blade: tapered two-segment strip from the origin, leaning toward (dx, dz). */
function blade(w: number, h: number, dx: number, dz: number, yaw: number): THREE.BufferGeometry {
  const pts = [[-w, 0], [w, 0], [-w * 0.55, h * 0.55], [w * 0.55, h * 0.55], [0, h]] as const;
  const pos: number[] = [];
  for (const [x, y] of pts) {
    const t = y / h, lean = t * t;
    pos.push(x * Math.cos(yaw) + dx * lean, y, x * Math.sin(yaw) + dz * lean);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex([0, 1, 2, 1, 3, 2, 2, 3, 4]);
  g.computeVertexNormals();
  return g;
}

// ---- flora (unit height, base at y = 0) ----

export function treeGeometry(): THREE.BufferGeometry {
  const c = new THREE.Vector3(0, 0.62, 0);
  const crown = (g: THREE.BufferGeometry): Part => ({ g, attr: flora(1), puff: { c, k: 0.55 } });
  return merge([
    { g: at(cyl(0.075, 0.05, 0.5, 6), 0, 0.25, 0), attr: flora(0, 0.3) },
    crown(blob(0.3, 1, 0, 0.64, 0)),
    crown(blob(0.23, 0, 0.21, 0.5, 0.07)),
    crown(blob(0.22, 0, -0.15, 0.55, -0.14)),
    crown(blob(0.2, 0, 0.02, 0.86, 0.03)),
    crown(blob(0.18, 0, -0.05, 0.5, 0.2)),
  ]);
}

export function pineGeometry(): THREE.BufferGeometry {
  const tier = (r: number, h: number, base: number) => ({ g: at(new THREE.ConeGeometry(r, h, 6, 1, false), 0, base + h / 2, 0), attr: flora(1), puff: { c: new THREE.Vector3(0, base, 0), k: 0.25 } });
  return merge([
    { g: at(cyl(0.06, 0.04, 0.3, 5), 0, 0.15, 0), attr: flora(0, 0.2) },
    tier(0.34, 0.42, 0.14),
    tier(0.27, 0.38, 0.37),
    tier(0.18, 0.34, 0.61),
  ]).scale(0.72, 1, 0.72); // slim conifers
}

export function cactusGeometry(): THREE.BufferGeometry {
  const cap = (r: number, x: number, y: number) => at(new THREE.SphereGeometry(r, 7, 3, 0, Math.PI * 2, 0, Math.PI / 2), x, y, 0);
  const body = (g: THREE.BufferGeometry): Part => ({ g, attr: flora(1, 0.25) });
  return merge([
    body(at(cyl(0.12, 0.11, 0.8, 7), 0, 0.4, 0)),
    body(cap(0.11, 0, 0.8)),
    body(at(cyl(0.065, 0.065, 0.2, 6).rotateZ(Math.PI / 2), 0.15, 0.38, 0)),
    body(at(cyl(0.065, 0.065, 0.3, 6), 0.24, 0.53, 0)),
    body(cap(0.065, 0.24, 0.68)),
    body(at(cyl(0.06, 0.06, 0.18, 6).rotateZ(Math.PI / 2), -0.14, 0.28, 0)),
    body(at(cyl(0.06, 0.06, 0.22, 6), -0.22, 0.39, 0)),
    body(cap(0.06, -0.22, 0.5)),
    { g: blob(0.055, 0, 0.02, 0.9, 0.02), attr: flora(2, 0.25) }, // a pink bloom on top
  ]);
}

export function acaciaGeometry(): THREE.BufferGeometry {
  const c = new THREE.Vector3(0, 0.78, 0);
  return merge([
    { g: at(cyl(0.05, 0.035, 0.52, 5).rotateZ(0.12), -0.03, 0.26, 0), attr: flora(0, 0.2) },
    { g: at(cyl(0.035, 0.022, 0.32, 4).rotateZ(-0.7), 0.07, 0.62, 0), attr: flora(0, 0.5) },
    { g: at(cyl(0.035, 0.022, 0.3, 4).rotateZ(0.65), -0.14, 0.6, 0.02), attr: flora(0, 0.5) },
    { g: blob(0.46, 1, 0.02, 0.8, 0, 1, 0.2, 0.85), attr: flora(1), puff: { c, k: 0.3 }, up: 0.35 },
    { g: blob(0.26, 1, 0.24, 0.72, 0.12, 1, 0.26, 1), attr: flora(1), puff: { c, k: 0.3 }, up: 0.35 },
  ]);
}

/** Columnar cypress: short trunk, tall narrow flame-shaped crown. */
export function cypressGeometry(): THREE.BufferGeometry {
  const c = new THREE.Vector3(0, 0.55, 0);
  return merge([
    { g: at(cyl(0.05, 0.035, 0.16, 5), 0, 0.08, 0), attr: flora(0, 0.2) },
    { g: blob(0.5, 1, 0, 0.5, 0, 0.3, 0.84, 0.3), attr: flora(1, 0.8), puff: { c, k: 0.35 } },
    { g: blob(0.2, 1, 0, 0.9, 0, 0.55, 0.9, 0.55), attr: flora(1, 1), puff: { c, k: 0.35 } },
  ]);
}

export function shrubGeometry(): THREE.BufferGeometry {
  const c = new THREE.Vector3(0, 0.15, 0);
  const b = (g: THREE.BufferGeometry): Part => ({ g, attr: flora(1, 0.6), puff: { c, k: 0.5 } });
  return merge([
    b(blob(0.38, 1, 0, 0.28, 0, 1, 0.78, 1)),
    b(blob(0.28, 0, 0.3, 0.2, 0.1)),
    b(blob(0.26, 0, -0.26, 0.18, -0.12)),
  ]);
}

export function grassGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  const blades = [[0.0, 1.0, 0.1, 0.05], [0.9, 0.8, 0.35, 0.1], [2.1, 0.75, -0.25, 0.3], [3.3, 0.85, -0.2, -0.3], [4.5, 0.7, 0.3, -0.25], [5.5, 0.9, 0.05, -0.1]] as const;
  for (const [yaw, h, dx, dz] of blades) parts.push({ g: blade(0.07, h, dx, dz, yaw), attr: flora(3), up: 0.6 });
  return merge(parts);
}

export function flowerGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  const heads = [[0.0, 0.0, 1.0], [0.28, 0.12, 0.78], [-0.2, 0.22, 0.66]] as const;
  for (const [x, z, h] of heads) {
    parts.push({ g: at(blade(0.03, h, 0, 0, x * 7), x, 0, z), attr: flora(3), up: 0.5 });
    parts.push({ g: blob(0.13, 0, x, h, z, 1, 0.55, 1), attr: flora(2), up: 0.3 });
  }
  parts.push({ g: blade(0.09, 0.4, 0.25, -0.1, 1.2), attr: flora(3), up: 0.6 });
  parts.push({ g: blade(0.09, 0.35, -0.2, -0.15, 2.4), attr: flora(3), up: 0.6 });
  return merge(parts);
}

// ---- ground cover, rocks, tussocks, hedges ----

/** Simple tapered blade (one triangle strip of 3 triangles) leaning toward (dx, dz). */
const thin = (w: number, h: number, dx: number, dz: number, yaw: number) => blade(w, h, dx, dz, yaw);

/** Short clumped tuft: 4 blades. */
export function fineShortGeometry(): THREE.BufferGeometry {
  const b = [[0.2, 1.0, 0.12, 0.05], [1.2, 0.75, -0.35, 0.18], [2.3, 0.9, 0.05, -0.33], [3.4, 0.7, 0.32, 0.28], [4.5, 0.85, -0.2, -0.25], [5.5, 0.65, 0.3, -0.12]] as const;
  return merge(b.map(([yaw, h, dx, dz]) => ({ g: thin(0.15, h, dx, dz, yaw), attr: flora(3), up: 0.65 })));
}
/** Tall wispy grass / feather grass / reeds: 3 long thin arching blades. */
export function fineTallGeometry(): THREE.BufferGeometry {
  const b = [[0.4, 1.0, 0.35, 0.1], [2.5, 0.85, -0.3, 0.25], [4.4, 0.92, 0.05, -0.38]] as const;
  return merge(b.map(([yaw, h, dx, dz]) => ({ g: thin(0.07, h, dx, dz, yaw), attr: flora(3), up: 0.55 })));
}
/** Seed-head grass: two stems with elongated seed heads, plus a low leaf. */
export function seedHeadGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  for (const [x, z, h, lean] of [[0, 0, 1.0, 0.12], [0.12, 0.08, 0.8, -0.15]] as const) {
    parts.push({ g: at(thin(0.025, h, lean, 0.05, x * 9), x, 0, z), attr: flora(3), up: 0.4 });
    parts.push({ g: at(new THREE.OctahedronGeometry(0.06, 0).scale(0.7, 2.4, 0.7), x + lean, h - 0.02, z + 0.05), attr: flora(1), up: 0.3 });
  }
  parts.push({ g: thin(0.08, 0.35, 0.2, -0.1, 1.1), attr: flora(3), up: 0.6 });
  return merge(parts);
}
/** Clover / low leafy cover: three flat round leaves at slightly different heights. */
export function cloverGeometry(): THREE.BufferGeometry {
  const leaf = (x: number, y: number, z: number, r: number, tilt: number) =>
    at(new THREE.CircleGeometry(r, 6).rotateX(-Math.PI / 2 + tilt), x, y, z);
  return merge([
    { g: leaf(0, 0.8, 0.12, 0.3, 0.15), attr: flora(1, 0.4), up: 0.5 },
    { g: leaf(0.22, 0.6, -0.12, 0.28, -0.1), attr: flora(1, 0.4), up: 0.5 },
    { g: leaf(-0.22, 0.5, -0.08, 0.26, 0.12), attr: flora(1, 0.4), up: 0.5 },
    { g: leaf(0.05, 0.35, 0.3, 0.2, 0.2), attr: flora(1, 0.4), up: 0.5 },
  ]);
}
/** Wildflower cluster: three stems with flat five-petal heads facing up. */
export function fineFlowerGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  for (const [x, z, h] of [[0, 0, 1.0], [0.22, 0.14, 0.78], [-0.18, 0.2, 0.62]] as const) {
    parts.push({ g: at(thin(0.03, h, 0, 0, x * 7), x, 0, z), attr: flora(3), up: 0.5 });
    parts.push({ g: at(new THREE.CircleGeometry(0.15, 5).rotateX(-Math.PI / 2), x, h, z), attr: flora(2), up: 0.4 });
  }
  return merge(parts);
}
/** Boulder: lumpy faceted rock, half sunk, with a smaller companion; lichen spots on part 2 vertices. */
export function rockGeometry(): THREE.BufferGeometry {
  const lump = (r: number, x: number, z: number, seed: number) => {
    const g = new THREE.IcosahedronGeometry(r, 1);
    const pos = g.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const vx = pos.getX(i), vy = pos.getY(i), vz = pos.getZ(i);
      const k = 1 + 0.18 * Math.sin(vx * 17.1 + seed) * Math.cos(vz * 13.3 - seed) + 0.1 * Math.sin(vy * 21.7 + seed * 2);
      pos.setXYZ(i, vx * k, vy * k * 0.62, vz * k * 0.85);
    }
    const flat = g.index ? g.toNonIndexed() : g;
    flat.computeVertexNormals();
    return at(flat, x, r * 0.3, z);
  };
  const lichen = (p: THREE.Vector3): [number, number, number, number] => {
    const h = Math.sin(p.x * 91.7 + p.z * 47.3 + p.y * 13.1) * 43758.5453;
    return [p.y, h - Math.floor(h) > 0.72 && p.y > 0.15 ? 2 : 1, 0, 0];
  };
  return merge([{ g: lump(0.5, 0, 0, 1.3), attr: lichen }, { g: lump(0.24, 0.5, 0.25, 4.1), attr: lichen }]);
}
/** Tussock: a dense bunch of blades splaying outward. */
export function tussockGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2 + (i % 3) * 0.2;
    const h = 0.7 + 0.3 * (((i * 7) % 5) / 4);
    const r0 = 0.06 * (i % 2);
    parts.push({ g: at(thin(0.1, h, Math.cos(a) * 0.42, Math.sin(a) * 0.42, a + Math.PI / 2), Math.cos(a) * r0, 0, Math.sin(a) * r0), attr: flora(3), up: 0.6 });
  }
  return merge(parts);
}
/** Hedge row: a line of puffy blobs, about three times longer than tall. */
export function hedgeGeometry(): THREE.BufferGeometry {
  const parts: Part[] = [];
  for (let i = 0; i < 5; i++) {
    const x = -1.0 + i * 0.5;
    parts.push({ g: blob(0.36 + 0.05 * (i % 2), 1, x, 0.34 + 0.04 * ((i * 3) % 2), 0.03 * (i % 3), 1, 0.95, 0.8), attr: flora(1, 0.3), puff: { c: new THREE.Vector3(x, 0.2, 0), k: 0.4 } });
  }
  return merge(parts);
}

/** Contact-shadow disc: 5×5 vertex grid over [-1, 1]² (conforms to the terrain per vertex), uv = local coords remapped to 0..1. */
export function blobGeometry(): THREE.BufferGeometry {
  const N = 5, pos: number[] = [], uv: number[] = [], idx: number[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const u = (i / (N - 1)) * 2 - 1, v = (j / (N - 1)) * 2 - 1;
    pos.push(u, 0, v); uv.push(u * 0.5 + 0.5, v * 0.5 + 0.5);
  }
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
    const a = i + j * N, b = a + 1, c = a + N, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}

// ---- creatures (forward = +Z, up = +Y) ----

function tris(pos: number[], idx: number[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  // flat-ish look: split to per-face normals, then re-index
  const flat = g.toNonIndexed();
  flat.computeVertexNormals();
  return flat;
}

/** Gull-like bird, span ~1.5, body length ~1. */
export function birdGeometry(): THREE.BufferGeometry {
  const body = tris(
    [0, 0, 0.5, 0.09, 0, 0.05, -0.09, 0, 0.05, 0, 0.08, 0.05, 0, -0.06, 0.05, 0, 0.01, -0.42],
    [0, 3, 1, 0, 2, 3, 0, 1, 4, 0, 4, 2, 5, 1, 3, 5, 3, 2, 5, 4, 1, 5, 2, 4],
  );
  const wing = (s: number) => tris(
    [s * 0.07, 0.01, 0.16, s * 0.07, 0.01, -0.12, s * 0.42, 0.02, 0.12, s * 0.78, 0.0, -0.16],
    s > 0 ? [0, 1, 2, 2, 1, 3] : [0, 2, 1, 2, 3, 1],
  );
  const tail = tris([0, 0.01, -0.36, 0.16, 0.01, -0.6, -0.16, 0.01, -0.6], [0, 2, 1]);
  const a = (part: number) => (p: THREE.Vector3): [number, number, number, number] => [part, Math.max(0, Math.abs(p.x) - 0.07), 0, 0];
  return merge([{ g: body, attr: a(0) }, { g: wing(1), attr: a(1) }, { g: wing(-1), attr: a(1) }, { g: tail, attr: a(0) }], false);
}

/**
 * Sheep/goat/deer: long body on four legs, neck + head forward and up (grazing tilts part 1 down
 * about the neck base). Height ~1.1, length ~1.1. Sheep puff the body up in the shader.
 */
export function critterGeometry(): THREE.BufferGeometry {
  const a = (part: number, leg = 0, hipZ = 0) => (): [number, number, number, number] => [part, leg, hipZ, 0];
  const leg = (x: number, z: number, s: number): Part => ({ g: at(cyl(0.05, 0.04, 0.48, 5, false), x, 0.24, z), attr: a(2, s, z) });
  return merge([
    { g: blob(0.5, 1, 0, 0.64, -0.02, 0.44, 0.42, 0.66), attr: a(0), puff: { c: new THREE.Vector3(0, 0.62, 0), k: 0.3 } },
    { g: blob(0.08, 0, 0, 0.7, -0.36), attr: a(0) }, // tail
    { g: at(cyl(0.08, 0.06, 0.3, 5, true).rotateX(0.75), 0, 0.82, 0.3), attr: a(1) }, // neck
    { g: blob(0.15, 1, 0, 0.98, 0.46, 0.78, 0.8, 1.3), attr: a(1) },
    { g: at(new THREE.ConeGeometry(0.045, 0.14, 4).rotateZ(1.3), 0.12, 1.03, 0.4), attr: a(1) },
    { g: at(new THREE.ConeGeometry(0.045, 0.14, 4).rotateZ(-1.3), -0.12, 1.03, 0.4), attr: a(1) },
    leg(0.13, 0.2, 1), leg(-0.13, 0.2, -1), leg(0.13, -0.22, -1), leg(-0.13, -0.22, 1),
  ], false);
}

/** Small fish, length 1 (nose z = 0.5, tail fin to z = -0.55). */
export function fishGeometry(): THREE.BufferGeometry {
  const body = tris(
    [0, 0, 0.5, 0.13, 0, 0.08, -0.13, 0, 0.08, 0, 0.22, 0.05, 0, -0.17, 0.05, 0, 0, -0.3],
    [0, 3, 1, 0, 2, 3, 0, 1, 4, 0, 4, 2, 5, 1, 3, 5, 3, 2, 5, 4, 1, 5, 2, 4],
  );
  const fin = tris([0, 0, -0.25, 0, 0.24, -0.58, 0, -0.24, -0.58], [0, 1, 2]);
  const a = (part: number) => (p: THREE.Vector3): [number, number, number, number] => [(0.5 - p.z) / 1.05, part, 0, 0];
  return merge([{ g: body, attr: a(0) }, { g: fin, attr: a(1) }], false);
}
