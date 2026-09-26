// Slice inspection (game UI): move the cut planes on X and Z with 3D brass grips on the pedestal edge.
// Everything that belongs to the block is reparented into a ClippingGroup (pedestal/backdrop stay whole);
// cap faces at the cut show the voxel cross-section there with the same shading as the outer cut faces.
import * as THREE from 'three/webgpu';
import type { GpuFields } from '../core/gpu';
import { HALF } from '../render/space';
import { createFaceShading, createFaceQuad } from '../render/sides';
import { TRAY_H } from '../render/backdrop';
import type { Stage } from '../render/stage';

/** Scene objects that belong to the block and must be clipped. */
const BLOCK_PARTS = ['terrain', 'sides-group', 'water', 'life', 'overlay-sheet', 'overlay-tectonics', 'overlay-faces', 'overlay-arrows'];
const MIN_CUT = -HALF + 0.35; // keep at least ~22 columns visible
const GRIP_OFFSET = 0.16;     // grips sit this far outside the block, on the tray

export interface Slice {
  readonly cut: { x: number; z: number };
  setCut(x: number, z: number): void;
  reset(): void;
  setVisible(v: boolean): void;
  update(dt: number): void;
}

export function createSlice(stage: Stage, fields: GpuFields): Slice {
  const { scene, camera, renderer, controls } = stage;
  const clip = new THREE.ClippingGroup();
  clip.name = 'slice-clip';
  const px = new THREE.Plane(new THREE.Vector3(-1, 0, 0), HALF + 0.01);
  const pz = new THREE.Plane(new THREE.Vector3(0, 0, -1), HALF + 0.01);
  clip.clippingPlanes = [px, pz];
  scene.add(clip);
  const adopt = () => {
    for (const name of BLOCK_PARTS) {
      const o = scene.getObjectByName(name);
      if (o && o.parent !== clip && !isInside(o, clip)) clip.attach(o);
    }
  };
  adopt();

  // ---- cap faces: cross-section at the cut, with the outer cut faces' shading (render/sides.ts) ----
  const capShading = createFaceShading(fields);
  const capX = createFaceQuad(capShading, 'slice-cap-x'); // plane x = cut.x, facing +x
  const capZ = createFaceQuad(capShading, 'slice-cap-z'); // plane z = cut.z, facing +z
  for (const cap of [capX, capZ]) { cap.object.visible = false; scene.add(cap.object); }

  // ---- grips ----
  const gripMat = new THREE.MeshStandardNodeMaterial({ color: 0xc9a25a, metalness: 0.85, roughness: 0.3 });
  const gripHot = new THREE.MeshStandardNodeMaterial({ color: 0xffd98a, metalness: 0.7, roughness: 0.25, emissive: new THREE.Color(0x553a10) });
  const makeGrip = () => {
    const g = new THREE.Group();
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.045, 24, 16), gripMat);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.018, 0.07, 12), gripMat);
    stem.position.y = -0.035; knob.position.y = 0.01;
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.006, 8, 32).rotateX(Math.PI / 2), gripMat);
    ring.position.y = -0.06;
    g.add(knob, stem, ring);
    g.userData.parts = [knob, stem, ring];
    return g;
  };
  const gripX = makeGrip(), gripZ = makeGrip();
  gripX.name = 'slice-grip-x'; gripZ.name = 'slice-grip-z';
  scene.add(gripX, gripZ);
  const gripY = -TRAY_H + 0.07;

  const cut = { x: HALF, z: HALF };
  function apply() {
    px.constant = cut.x + (cut.x >= HALF - 1e-4 ? 0.01 : 0);
    pz.constant = cut.z + (cut.z >= HALF - 1e-4 ? 0.01 : 0);
    // cap X lives on the plane x = cut.x spanning z ∈ [-HALF, cut.z], cap Z on z = cut.z for x ∈ [-HALF, cut.x]
    capX.object.visible = cut.x < HALF - 1e-4;
    capX.set(cut.x, cut.z, cut.x, -HALF, 1, 0);
    capZ.object.visible = cut.z < HALF - 1e-4;
    capZ.set(-HALF, cut.z, cut.x, cut.z, 0, 1);
    gripX.position.set(cut.x, gripY, HALF + GRIP_OFFSET);
    gripZ.position.set(HALF + GRIP_OFFSET, gripY, cut.z);
  }
  apply();

  // ---- dragging ----
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const planeY = new THREE.Plane(new THREE.Vector3(0, 1, 0), -gripY);
  let drag: 'x' | 'z' | null = null;
  let hover: THREE.Group | null = null;
  const setNdc = (e: PointerEvent) => {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
  };
  const pickGrip = (): THREE.Group | null => {
    const hits = ray.intersectObjects([gripX, gripZ], true);
    if (!hits.length) return null;
    let o: THREE.Object3D | null = hits[0]!.object;
    while (o && o !== gripX && o !== gripZ) o = o.parent;
    return o as THREE.Group | null;
  };
  const paint = (g: THREE.Group, hot: boolean) => { for (const p of g.userData.parts as THREE.Mesh[]) p.material = hot ? gripHot : gripMat; };
  const el = renderer.domElement;
  el.addEventListener('pointerdown', (e) => {
    if (!gripX.visible) return;
    setNdc(e);
    const g = pickGrip();
    if (!g) return;
    drag = g === gripX ? 'x' : 'z';
    controls.enabled = false;
    el.setPointerCapture(e.pointerId);
    e.stopPropagation();
  }, { capture: true });
  el.addEventListener('pointermove', (e) => {
    setNdc(e);
    if (drag) {
      const p = new THREE.Vector3();
      if (ray.ray.intersectPlane(planeY, p)) {
        const v = Math.min(HALF, Math.max(MIN_CUT, drag === 'x' ? p.x : p.z));
        cut[drag] = Math.abs(v - HALF) < 0.03 ? HALF : v; // snap back to the full block near the edge
        apply();
      }
      return;
    }
    const g = gripX.visible ? pickGrip() : null;
    if (g !== hover) { if (hover) paint(hover, false); if (g) paint(g, true); hover = g; el.style.cursor = g ? 'grab' : ''; }
  });
  const end = (e: PointerEvent) => {
    if (!drag) return;
    drag = null;
    controls.enabled = true;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el.addEventListener('dblclick', (e) => {
    setNdc(e as unknown as PointerEvent);
    const g = pickGrip();
    if (g) { cut[g === gripX ? 'x' : 'z'] = HALF; apply(); }
  });

  let t = 0;
  return {
    cut,
    setCut(x, z) { cut.x = Math.min(HALF, Math.max(MIN_CUT, x)); cut.z = Math.min(HALF, Math.max(MIN_CUT, z)); apply(); },
    reset() { cut.x = HALF; cut.z = HALF; apply(); },
    setVisible(v) { gripX.visible = gripZ.visible = v; },
    update(dt) {
      t += dt;
      adopt(); // parts created later (overlays, life) join the clip group
      // idle grips breathe a little so they read as interactive
      const s = 1 + 0.04 * Math.sin(t * 2.2);
      if (!drag) { gripX.scale.setScalar(hover === gripX ? 1.15 : s); gripZ.scale.setScalar(hover === gripZ ? 1.15 : s); }
    },
  };
}

function isInside(o: THREE.Object3D, root: THREE.Object3D): boolean {
  for (let p = o.parent; p; p = p.parent) if (p === root) return true;
  return false;
}
