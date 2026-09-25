// §T.40 lighting: one shadow-casting key light that is the sun by day and the moon by night,
// a cool rim/back light without shadows, and a sky/ground hemisphere fill. Colours and angles come
// from the time-of-day model (sky.ts).
import * as THREE from 'three/webgpu';
import { skyAt, applySky, GOLDEN_HOUR, type SkyState } from './sky';

export interface Lighting {
  /** Shadow-casting key light (sun or moon). */
  sun: THREE.DirectionalLight;
  rim: THREE.DirectionalLight;
  hemi: THREE.HemisphereLight;
  /** Latest time-of-day state. */
  readonly sky: SkyState;
  setTimeOfDay(tod: number): void;
  dispose(): void;
}

export interface LightingOptions {
  shadows?: boolean;
  /** Shadow map resolution (square). */
  shadowMapSize?: number;
  /** Half extent of the shadow frustum around the origin, world units (block + pedestal). */
  shadowExtent?: number;
}

const KEY_DIST = 12;

/**
 * Layer seen only by the shadow camera. The shadow pass builds a material's colorNode (for alpha),
 * so heavy materials cast through a cheap proxy mesh on this layer instead of casting themselves.
 */
export const SHADOW_LAYER = 1;

/** Shadow-only twin of a GPU-displaced mesh: same positionNode (and optional mask), no shading. */
export function shadowProxy(geometry: THREE.BufferGeometry, positionNode: THREE.Node, maskNode?: THREE.Node): THREE.Mesh {
  const m = new THREE.MeshBasicNodeMaterial();
  m.positionNode = positionNode;
  // The shadow pass renders back faces by default (_shadowSide); an open heightfield seen from the
  // light has only front faces, so draw both.
  m.shadowSide = THREE.DoubleSide;
  if (maskNode) m.maskShadowNode = maskNode as THREE.Node<'bool'>;
  const mesh = new THREE.Mesh(geometry, m);
  mesh.name = 'shadow-proxy';
  mesh.layers.set(SHADOW_LAYER);
  mesh.castShadow = true;
  mesh.frustumCulled = false;
  return mesh;
}
const moonColor = new THREE.Color(0x9db4ff);
const tmp = new THREE.Color();

export function createLighting(scene: THREE.Scene, opts: LightingOptions = {}): Lighting {
  const sun = new THREE.DirectionalLight(0xffe1bd, 3.4);
  const rim = new THREE.DirectionalLight(0x9cc4ff, 0.6);
  const hemi = new THREE.HemisphereLight(0xc4dcff, 0x7a5a44, 0.95);
  if (opts.shadows) {
    const s = sun.shadow;
    sun.castShadow = true;
    s.mapSize.set(opts.shadowMapSize ?? 2048, opts.shadowMapSize ?? 2048);
    const e = opts.shadowExtent ?? 3.6;
    const cam = s.camera as THREE.OrthographicCamera;
    cam.left = -e; cam.right = e; cam.top = e; cam.bottom = -e;
    cam.near = 1; cam.far = KEY_DIST * 2;
    s.bias = -0.0004;
    s.normalBias = 0.012;
    s.radius = 3;
    cam.layers.enable(SHADOW_LAYER);
  }
  scene.add(sun, sun.target, rim, rim.target, hemi);

  let sky = skyAt(GOLDEN_HOUR);
  const apply = (tod: number) => {
    sky = skyAt(tod);
    applySky(sky);
    // Key: sun while it is up, moon at night; a short dip at the swap hides the jump.
    const moonUp = sky.night > 0.02;
    const dir = moonUp ? sky.moonDir : sky.sunDir;
    const d = dir.clone();
    d.y = Math.max(d.y, 0.06); // never grazing below the block: keeps shadows bounded and stable
    d.normalize();
    sun.position.copy(d).multiplyScalar(KEY_DIST);
    if (moonUp) {
      sun.color.copy(moonColor);
      sun.intensity = 0.3 * sky.night;
    } else {
      sun.color.copy(sky.sunColor);
      sun.intensity = sky.sunIntensity;
    }
    // Rim: opposite the key horizontally, a little above, cool sky tone.
    rim.position.set(-d.x, 0.5, -d.z).normalize().multiplyScalar(KEY_DIST);
    rim.color.copy(tmp.copy(sky.zenith).lerp(new THREE.Color(0xbcd4ff), 0.5));
    rim.intensity = 0.25 + 0.55 * (1 - sky.night);
    hemi.color.copy(sky.zenith).lerp(sky.horizon, 0.35);
    hemi.groundColor.copy(sky.ground).multiplyScalar(1.2);
    hemi.intensity = sky.hemiIntensity;
  };
  apply(GOLDEN_HOUR);

  return {
    sun, rim, hemi,
    get sky() { return sky; },
    setTimeOfDay: apply,
    dispose() { scene.remove(sun, sun.target, rim, rim.target, hemi); sun.dispose(); rim.dispose(); hemi.dispose(); },
  };
}
