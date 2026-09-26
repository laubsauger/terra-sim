// Slice inspection (game UI): move the cut planes on X and Z with 3D brass grips on the pedestal edge.
// Everything that belongs to the block is reparented into a ClippingGroup (pedestal/backdrop stay whole);
// cap faces at the cut show the voxel cross-section there with the same shading as the outer cut faces.
// Affordance: each grip rides an engraved brass rail on the tray with chevrons along its drag axis that
// breathe when idle and light up on hover (first hover: a small "drag to cut" tip). While a cut moves,
// the cap's top edge glows with a faint light sheet; starting or ending a drag sweeps a scanline down.
import * as THREE from 'three/webgpu';
import { uniform, vec3, float } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { HALF } from '../render/space';
import { createFaceShading, createFaceQuad } from '../render/sides';
import { TRAY_H } from '../render/backdrop';
import type { Stage } from '../render/stage';

/** Scene objects that belong to the block and must be clipped. */
const BLOCK_PARTS = ['terrain', 'sides-group', 'water', 'life', 'overlay-sheet', 'overlay-tectonics', 'overlay-faces', 'overlay-arrows'];
const MIN_CUT = -HALF + 0.35; // keep at least ~22 columns visible
const GRIP_OFFSET = 0.16;     // grips sit this far outside the block, on the tray
const SWEEP_S = 0.6;          // scanline sweep duration (s)

export interface Slice {
  readonly cut: { x: number; z: number };
  setCut(x: number, z: number): void;
  reset(): void;
  setVisible(v: boolean): void;
  update(dt: number): void;
  /** Interaction state for tools and tests: chevron light per grip, cut edge glow, sweep 0..1. */
  readonly fxState: { gripX: number; gripZ: number; glow: number; sweep: number; tip: boolean };
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
  const fx = { glow: uniform(0), sweep: uniform(0) };
  const capShading = createFaceShading(fields, { cutFx: fx });
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

  // ---- affordance: engraved rail + chevrons along each drag axis ----
  const railLen = HALF - MIN_CUT + 0.06;
  const railMat = new THREE.MeshStandardNodeMaterial({ color: 0x8a6a34, metalness: 0.9, roughness: 0.45 });
  const makeRail = (axis: 'x' | 'z') => {
    const g = new THREE.Group();
    const groove = new THREE.Mesh(new THREE.BoxGeometry(railLen, 0.003, 0.009), railMat);
    g.add(groove);
    for (const e of [-1, 1]) { // end stops
      const stop = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.005, 16), railMat);
      stop.position.x = (e * railLen) / 2;
      g.add(stop);
    }
    const mid = (HALF + MIN_CUT) / 2;
    if (axis === 'x') g.position.set(mid, 0.0015, HALF + GRIP_OFFSET);
    else { g.rotation.y = -Math.PI / 2; g.position.set(HALF + GRIP_OFFSET, 0.0015, mid); }
    g.name = `slice-rail-${axis}`;
    scene.add(g);
    return g;
  };
  const rails = [makeRail('x'), makeRail('z')];
  // chevron shape pointing +x in the tray plane
  const chevGeo = (() => {
    const sh = new THREE.Shape();
    sh.moveTo(0, 0.026); sh.lineTo(0.03, 0); sh.lineTo(0, -0.026); sh.lineTo(-0.012, -0.026); sh.lineTo(0.016, 0); sh.lineTo(-0.012, 0.026);
    return new THREE.ShapeGeometry(sh).rotateX(-Math.PI / 2);
  })();
  /** Chevrons on both sides of a grip (children of the grip, so they travel and pick with it). */
  const addChevrons = (g: THREE.Group, axis: 'x' | 'z') => {
    const lit = uniform(0); // 0 idle … 1 hovered / dragged
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    // brass engraving that lights to a warm glow (> 1: blooms a little) when hot
    m.colorNode = vec3(0.78, 0.6, 0.3).mul(float(0.5).add(lit.mul(2.0)));
    m.opacityNode = float(0.7).add(lit.mul(0.3));
    const chevs: THREE.Mesh[] = [];
    for (const side of [-1, 1]) for (const k of [0, 1]) {
      const c = new THREE.Mesh(chevGeo, m);
      c.rotation.y = side > 0 ? 0 : Math.PI;
      c.userData.base = side * (0.095 + k * 0.042);
      c.userData.side = side;
      c.renderOrder = 1;
      chevs.push(c);
      g.add(c);
    }
    const holder = new THREE.Group(); // local frame: +x = drag axis
    if (axis === 'z') holder.rotation.y = -Math.PI / 2;
    for (const c of chevs) holder.attach(c);
    g.add(holder);
    g.userData.chev = { lit, chevs, holder, m };
  };
  addChevrons(gripX, 'x');
  addChevrons(gripZ, 'z');

  // first-hover tip (DOM, follows the grip on screen)
  const tip = document.createElement('div');
  tip.className = 'terra-slice-tip';
  tip.textContent = 'drag to cut · double-click resets';
  Object.assign(tip.style, {
    position: 'fixed', pointerEvents: 'none', padding: '4px 9px', borderRadius: '8px', font: '12px system-ui, sans-serif',
    color: '#f3e6c8', background: 'rgba(24, 20, 14, 0.78)', border: '1px solid rgba(201, 162, 90, 0.55)',
    transform: 'translate(-50%, -140%)', opacity: '0', transition: 'opacity 0.25s', whiteSpace: 'nowrap', zIndex: '5',
  } as Partial<CSSStyleDeclaration>);
  document.body.appendChild(tip);
  let tipDone = false; // shown until the first drag

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
  let sweepT = -1; // seconds since the last sweep started (< 0: none)
  const startSweep = () => { if (capX.object.visible || capZ.object.visible) sweepT = 0; };
  const el = renderer.domElement;
  el.addEventListener('pointerdown', (e) => {
    if (!gripX.visible) return;
    setNdc(e);
    const g = pickGrip();
    if (!g) return;
    drag = g === gripX ? 'x' : 'z';
    tipDone = true;
    sweepT = 0;
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
    startSweep();
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
  const tmpV = new THREE.Vector3();
  return {
    cut,
    get fxState() {
      const lit = (g: THREE.Group) => (g.userData.chev as { lit: THREE.UniformNode<'float', number> }).lit.value;
      return { gripX: lit(gripX), gripZ: lit(gripZ), glow: fx.glow.value, sweep: fx.sweep.value, tip: tip.style.opacity === '1' };
    },
    setCut(x, z) { cut.x = Math.min(HALF, Math.max(MIN_CUT, x)); cut.z = Math.min(HALF, Math.max(MIN_CUT, z)); apply(); },
    reset() { cut.x = HALF; cut.z = HALF; apply(); },
    setVisible(v) { gripX.visible = gripZ.visible = v; },
    update(dt) {
      t += dt;
      adopt(); // parts created later (overlays, life) join the clip group
      // idle grips breathe and bob a little so they read as interactive
      const s = 1 + 0.04 * Math.sin(t * 2.2);
      if (!drag) { gripX.scale.setScalar(hover === gripX ? 1.15 : s); gripZ.scale.setScalar(hover === gripZ ? 1.15 : s); }
      for (const g of [gripX, gripZ]) {
        const active = drag === (g === gripX ? 'x' : 'z') || hover === g;
        const c = g.userData.chev as { lit: THREE.UniformNode<'float', number>; chevs: THREE.Mesh[]; holder: THREE.Group };
        c.lit.value += ((active ? 1 : 0) - c.lit.value) * (1 - Math.exp(-dt * 10));
        // chevrons drift outward along the axis (idle: slow breath; hot: a quicker nudge)
        const ph = active ? (t * 1.6) % 1 : 0.5 + 0.5 * Math.sin(t * 1.4);
        for (const m of c.chevs) m.position.x = (m.userData.base as number) + (m.userData.side as number) * 0.018 * ph;
        c.holder.scale.setScalar(1 / g.scale.x); // chevrons keep their size while the knob breathes
        c.holder.position.y = (-gripY + 0.002) / g.scale.x;
        g.children[0]!.position.y = 0.01 + (drag || hover === g ? 0 : 0.004 * Math.sin(t * 2.2 + (g === gripX ? 0 : 1.3)));
      }
      for (const r of rails) r.visible = gripX.visible;
      // tip over the hovered grip until the first drag
      const showTip = !tipDone && hover !== null && gripX.visible;
      tip.style.opacity = showTip ? '1' : '0';
      if (showTip) {
        tmpV.copy(hover!.position).project(camera);
        const rc = renderer.domElement.getBoundingClientRect();
        tip.style.left = `${rc.left + (tmpV.x * 0.5 + 0.5) * rc.width}px`;
        tip.style.top = `${rc.top + (-tmpV.y * 0.5 + 0.5) * rc.height}px`;
      }
      // cut FX: edge glow while a cut moves (fades out after), scanline sweep on start / end
      fx.glow.value += ((drag ? 1 : 0) - fx.glow.value) * (1 - Math.exp(-dt * (drag ? 12 : 3)));
      if (fx.glow.value < 1e-3) fx.glow.value = 0;
      if (sweepT >= 0) { sweepT += dt; fx.sweep.value = sweepT / SWEEP_S; if (sweepT > SWEEP_S) { sweepT = -1; fx.sweep.value = 0; } }
    },
  };
}

function isInside(o: THREE.Object3D, root: THREE.Object3D): boolean {
  for (let p = o.parent; p; p = p.parent) if (p === root) return true;
  return false;
}
