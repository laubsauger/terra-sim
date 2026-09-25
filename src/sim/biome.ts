// §C pass 10 (derived): biome classification + vegetation on GPU (T31). Mirrors biomeStepCpu (biomeModel.ts).
// One kernel; writes own index only, reads 4 neighbours' water (coast test). Storage buffers bound: 8 (V23).
// Veg target blends smoothly across the treeline (trees → shrubs → grass → rock) for the life pass.
// 'veg' is the erosion coupling: erosion multiplies erodibility by vegErodibilityFactor(veg) = 1 - 0.7·veg.
import type * as THREE from 'three/webgpu';
import { Fn, If, Return, int, uint, vec2, max, select, smoothstep, instanceIndex, uniform, uniformArray } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL, Y_SEA_NOMINAL } from './layout';
import { BIOME as B, Biome, BIOME_VEG_TARGET } from './biomeModel';
import { CLIMATE } from './climateModel';
import { tColIdx, tColXZ } from './tslLayout';

type F = THREE.Node<'float'>;

export interface BiomePass {
  step(renderer: THREE.WebGPURenderer): void;
  /** Same emergent sea level as the climate pass. */
  setSeaLevel(y: number): void;
  uniforms: { seaLevel: THREE.UniformNode<'float', number> };
}

export function createBiomePass(fields: GpuFields): BiomePass {
  const surfTemp = fields.cur('surfTemp');
  const precip = fields.cur('precip');
  const surfY = fields.cur('surfY');
  const water = fields.cur('water');
  const ice = fields.cur('ice');
  const avg = fields.cur<'vec2'>('climAvg');
  const veg = fields.cur('veg');
  const biome = fields.cur<'uint'>('biome');
  const seaLevel = uniform(Y_SEA_NOMINAL);
  const vegTarget = uniformArray([...BIOME_VEG_TARGET], 'float');

  const k = Fn(() => {
    If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
    const i = instanceIndex;
    const a0 = avg.element(i).toVar();
    const tAvg = a0.x.add(surfTemp.element(i).sub(a0.x).mul(B.AVG_ALPHA)).toVar();
    const pAvg = a0.y.add(precip.element(i).sub(a0.y).mul(B.AVG_ALPHA)).toVar();
    avg.element(i).assign(vec2(tAvg, pAvg));
    const alt = surfY.element(i).sub(seaLevel).toVar();
    const m = pAvg.div(B.P_REF).toVar();
    const { x, z } = tColXZ(i);
    const deep = (c: THREE.Node<'uint'>) => water.element(c).greaterThan(B.OCEAN_MIN);
    const coastal = deep(tColIdx(x.add(int(1)), z)).or(deep(tColIdx(x.sub(int(1)), z)))
      .or(deep(tColIdx(x, z.add(int(1))))).or(deep(tColIdx(x, z.sub(int(1)))));
    const vt = (id: THREE.Node<'uint'>) => vegTarget.element(id) as unknown as F;
    // treeline (see classifyBiomeVeg)
    const tt = smoothstep(B.TREE_ALT0, B.TREE_ALT1, alt).mul(B.TREE_T_HIGH - B.TREE_T_LOW).add(B.TREE_T_LOW).toVar();
    const cold = select(alt.greaterThan(B.ALPINE_ALT).and(tAvg.add(alt.mul(CLIMATE.LAPSE)).greaterThanEqual(tt)),
      uint(Biome.ALPINE), uint(Biome.TUNDRA)).toVar();
    // Whittaker class above the treeline (warmBiome)
    const tw = max(tAvg, tt).toVar();
    const warm = uint(0).toVar();
    const set = (id: number) => () => { warm.assign(uint(id)); };
    If(tw.lessThan(B.T_BOREAL), () => {
      If(m.lessThan(B.M_DRY), set(Biome.COLD_DESERT)).ElseIf(m.lessThan(B.M_TAIGA), set(Biome.STEPPE)).Else(set(Biome.TAIGA));
    }).ElseIf(tw.lessThan(B.T_WARM), () => {
      If(m.lessThan(B.M_DRY), set(Biome.COLD_DESERT)).ElseIf(m.lessThan(B.M_SEMI), set(Biome.STEPPE))
        .ElseIf(m.lessThan(B.M_FOREST), set(Biome.GRASSLAND)).Else(set(Biome.TEMPERATE_FOREST));
    }).ElseIf(tw.lessThan(B.T_TROPIC), () => {
      If(m.lessThan(B.M_DRY), set(Biome.DESERT)).ElseIf(m.lessThan(B.M_SEMI), set(Biome.SHRUBLAND))
        .ElseIf(m.lessThan(B.M_FOREST), set(Biome.GRASSLAND)).Else(set(Biome.TEMPERATE_FOREST));
    }).Else(() => {
      If(m.lessThan(B.M_TROP_DRY), set(Biome.DESERT)).ElseIf(m.lessThan(B.M_TROP_SEMI), set(Biome.SHRUBLAND))
        .ElseIf(m.lessThan(B.M_RAINFOREST), set(Biome.SAVANNA)).Else(set(Biome.RAINFOREST));
    });
    // climateVeg(tw, m)
    const moist = smoothstep(0.03, 0.2, m).mul(0.25).add(smoothstep(0.2, 0.5, m).mul(0.3)).add(smoothstep(0.45, 0.9, m).mul(0.4)).add(0.05);
    const warmVeg = moist.mul(smoothstep(B.T_BOREAL - 3, B.T_WARM, tw).mul(0.25).add(0.75));
    const coldVeg = vt(cold).mul(smoothstep(tt.sub(B.ROCK_T), tt.sub(B.TREE_BAND), tAvg)).mul(smoothstep(0, 0.15, m)).toVar();
    const target = coldVeg.add(warmVeg.sub(coldVeg).mul(smoothstep(tt.sub(B.TREE_BAND), tt.add(B.TREE_BAND), tAvg))).toVar();
    const b = select(tAvg.lessThan(tt), cold, warm).toVar();
    // same override order as classifyBiomeVeg
    If(ice.element(i).greaterThan(B.ICE_MIN), () => { b.assign(uint(Biome.ICE)); target.assign(vt(uint(Biome.ICE))); })
      .ElseIf(water.element(i).greaterThan(B.OCEAN_MIN), () => { b.assign(uint(Biome.OCEAN)); target.assign(vt(uint(Biome.OCEAN))); })
      .ElseIf(alt.lessThan(B.BEACH_ALT).and(tAvg.greaterThan(0)).and(coastal), () => { b.assign(uint(Biome.BEACH)); target.assign(vt(uint(Biome.BEACH))); });
    biome.element(i).assign(b);
    const v0 = veg.element(i).toVar();
    veg.element(i).assign(v0.add(target.sub(v0).mul(B.VEG_RATE)));
  })().compute(NCOL);

  return {
    step(renderer) { renderer.compute(k); },
    setSeaLevel(y) { seaLevel.value = y; },
    uniforms: { seaLevel },
  };
}
