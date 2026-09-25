// §T.43 post (pulled forward) + §T.40 screen-space AO/GI.
// high quality: MRT scene pass → SSGI (AO + one diffuse bounce) → TRAA → bloom → tilt-shift DOF
//               → AgX tone map → grade (split tone, contrast, saturation) → vignette + grain.
// low quality:  scene pass → bloom → tone map → grade → FXAA (no SSGI / TRAA / DOF, V9 low tier).
// Post reads the rendered image only; it never touches sim state (V15).
import * as THREE from 'three/webgpu';
import {
  pass, mrt, output, diffuseColor, normalView, velocity, packNormalToRGB, unpackRGBToNormal, sample,
  vec2, vec3, vec4, float, uniform, uv, mix, smoothstep, saturate, abs, sign, pow, length, hash,
  renderOutput, luminance, frameId, max, screenCoordinate,
} from 'three/tsl';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';

type V3 = THREE.Node<'vec3'>;
type V4 = THREE.Node<'vec4'>;

export interface Post {
  pipeline: THREE.RenderPipeline;
  readonly highQuality: boolean;
  setHighQuality(v: boolean): void;
  /** Focus distance (world units) for the tilt-shift DOF, e.g. camera → orbit target. */
  setFocus(distance: number): void;
  render(): void;
  /** Tunables (uniforms; change .value). */
  u: {
    exposure: THREE.UniformNode<'float', number>;
    bloomStrength: THREE.UniformNode<'float', number>;
    tilt: THREE.UniformNode<'float', number>;
    vignette: THREE.UniformNode<'float', number>;
    grain: THREE.UniformNode<'float', number>;
    saturation: THREE.UniformNode<'float', number>;
  };
  dispose(): void;
}

/** ACES over AgX: AgX desaturates the saturated diorama palette and glowing magma toward pastel. */
export const TONE_MAPPING = THREE.ACESFilmicToneMapping;

/**
 * Individual passes; each quality tier is a preset of these (perf triage can override any).
 * AO: `ssgi` (AO + one diffuse bounce, ~5.5 ms at 1440p on the M3 Max proxy) wins over `gtao`
 * (half-res AO, ~1 ms). High tier defaults to GTAO to hold the V9 render budget; SSGI is opt-in.
 */
export interface PostFeatures { ssgi: boolean; gtao: boolean; traa: boolean; dof: boolean; bloom: boolean }
export const tierFeatures = (high: boolean): PostFeatures => ({ ssgi: false, gtao: high, traa: high, dof: high, bloom: true });

export function createPost(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera,
  opts: { highQuality: boolean; features?: Partial<PostFeatures>; toneMapping?: THREE.ToneMapping }): Post {
  const pipeline = new THREE.RenderPipeline(renderer);
  pipeline.outputColorTransform = false; // tone map + colour space done in the graph, then grade in display space

  const u = {
    exposure: uniform(1.1),
    bloomStrength: uniform(0.32),
    tilt: uniform(1.0),
    vignette: uniform(0.42),
    grain: uniform(0.022),
    saturation: uniform(1.08),
  };
  const focus = uniform(7.0);
  const focalRange = uniform(4.0);

  let disposables: { dispose(): void }[] = [];
  let hq = opts.highQuality;

  /** Display-space grade: warm highlights / teal shadows, gentle S-curve, vignette, film grain. */
  const grade = (ldr: V3): V3 => {
    const l = luminance(ldr);
    const shadowsTeal = vec3(-0.012, 0.004, 0.018).mul(float(1).sub(smoothstep(0.0, 0.45, l)));
    const highsWarm = vec3(0.028, 0.008, -0.03).mul(smoothstep(0.45, 1.0, l));
    let c = ldr.add(shadowsTeal).add(highsWarm) as V3;
    // S-curve around mid grey
    c = mix(c, c.mul(c).mul(float(3).sub(c.mul(2))), 0.22) as V3;
    c = mix(vec3(luminance(c)), c, u.saturation) as V3;
    const q = uv().sub(0.5).mul(vec2(1.0, 1.15));
    const vig = float(1).sub(u.vignette.mul(pow(saturate(length(q).mul(1.35)), 2.6)));
    c = c.mul(vig) as V3;
    const g = hash(screenCoordinate.x.add(screenCoordinate.y.mul(1931.7)).add(float(frameId).mul(0.61).fract().mul(4513.1)));
    return saturate(c.add(g.sub(0.5).mul(u.grain))) as V3;
  };

  function build(high: boolean): V4 {
    for (const d of disposables) d.dispose();
    disposables = [];
    camera.clearViewOffset();
    const f: PostFeatures = { ...tierFeatures(high), ...opts.features };
    const scenePass = pass(scene, camera);
    disposables.push(scenePass);
    const useGtao = f.gtao && !f.ssgi;
    if (f.ssgi || useGtao || f.traa) {
      const outs: Record<string, THREE.Node> = { output };
      if (f.ssgi) outs.diffuseColor = diffuseColor;
      if (f.ssgi || useGtao) outs.normal = packNormalToRGB(normalView);
      if (f.traa) outs.velocity = velocity;
      scenePass.setMRT(mrt(outs));
      if (f.ssgi) scenePass.getTexture('diffuseColor').type = THREE.UnsignedByteType;
      if (f.ssgi || useGtao) scenePass.getTexture('normal').type = THREE.UnsignedByteType;
    }
    const color = scenePass.getTextureNode('output');
    const depth = scenePass.getTextureNode('depth');
    let hdr: V4 = color as unknown as V4;
    if (f.ssgi) {
      const diffuse = scenePass.getTextureNode('diffuseColor');
      const normalTex = scenePass.getTextureNode('normal');
      const sceneNormal = sample((c) => unpackRGBToNormal(normalTex.sample(c)));
      const gi = ssgi(color, depth, sceneNormal, camera);
      gi.sliceCount.value = 1;  // 'low' preset with temporal filtering (TRAA): ~half the cost of 2×8
      gi.stepCount.value = 10;
      gi.radius.value = 0.9;       // world units: valleys, grooves, the plinth step
      gi.thickness.value = 0.12;
      gi.aoIntensity.value = 1.25;
      gi.giIntensity.value = 1.0;
      gi.useScreenSpaceSampling.value = false;
      gi.useTemporalFiltering = f.traa;
      disposables.push(gi);
      // r186: the node itself is the AO texture (red); GI (one diffuse bounce) is a separate target.
      const ao = gi.getAONode().r, bounce = gi.getGINode().rgb;
      hdr = vec4(color.rgb.mul(ao).add(diffuse.rgb.mul(bounce)), color.a);
    }
    if (useGtao) {
      const normalTex = scenePass.getTextureNode('normal');
      const sceneNormal = sample((c) => unpackRGBToNormal(normalTex.sample(c)));
      const g = ao(depth, sceneNormal, camera);
      g.resolutionScale = 0.5;
      g.radius.value = 0.35;       // world units: valleys, grooves, the plinth step
      g.thickness.value = 0.15;
      g.samples.value = 12;
      g.useTemporalFiltering = f.traa;
      disposables.push(g);
      const occ = mix(float(1), g.getTextureNode().r, 0.9);
      hdr = vec4(color.rgb.mul(occ), color.a);
    }
    if (f.traa) {
      const aa = traa(hdr, depth, scenePass.getTextureNode('velocity'), camera);
      disposables.push(aa);
      hdr = aa as unknown as V4;
    }
    if (f.bloom) {
      // clamp what feeds the bloom: sun disc / glints are ~50× and would flood the frame with glare
      const b = bloom(vec4(hdr.rgb.min(vec3(5)), 1), u.bloomStrength as unknown as number, 0.55, 1.1);
      disposables.push(b);
      hdr = vec4(hdr.rgb.add(b.rgb), 1);
    }
    if (f.dof) {
      // Tilt-shift: fold a screen-space band into the depth the DOF sees. Middle band keeps the
      // real depth (sharp block), top and bottom of frame are pushed away from the focal plane.
      const viewZ = scenePass.getViewZNode();
      const dy = uv().y.sub(0.5);
      const band = smoothstep(0.16, 0.5, abs(dy)).mul(u.tilt);
      const tiltViewZ = viewZ.sub(sign(dy).mul(band).mul(focus).mul(0.6));
      const d = dof(hdr, tiltViewZ, focus, focalRange, 2.2);
      disposables.push(d);
      hdr = d as unknown as V4;
    }
    const ldr = renderOutput(vec4(max(hdr.rgb.mul(u.exposure), vec3(0)), 1), opts.toneMapping ?? TONE_MAPPING, THREE.SRGBColorSpace);
    const out = vec4(grade(ldr.rgb as V3), 1);
    if (f.traa) return out;
    const aa = fxaa(out);
    disposables.push(aa);
    return aa as unknown as V4;
  }

  pipeline.outputNode = build(hq);

  return {
    pipeline, u,
    get highQuality() { return hq; },
    setHighQuality(v) {
      if (v === hq) return;
      hq = v;
      pipeline.outputNode = build(hq);
      pipeline.needsUpdate = true;
    },
    setFocus(distance) {
      focus.value = distance;
      focalRange.value = distance * 0.9;
    },
    render() { pipeline.render(); },
    dispose() { for (const d of disposables) d.dispose(); pipeline.dispose(); camera.clearViewOffset(); },
  };
}

