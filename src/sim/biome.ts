// §C pass 10 (derived): biome classification + vegetation on GPU (T31). Mirrors biomeStepCpu (biomeModel.ts).
// One kernel, own-index reads/writes only. Storage buffers bound: 8 (V23).
// 'veg' is the erosion coupling: erosion multiplies erodibility by vegErodibilityFactor(veg) = 1 - 0.7·veg.
import type * as THREE from 'three/webgpu';
import { Fn, If, Return, uint, vec2, instanceIndex, uniform, uniformArray } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL, Y_SEA_NOMINAL } from './layout';
import { BIOME as B, Biome, BIOME_VEG_TARGET } from './biomeModel';

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
    const b = uint(0).toVar();
    const set = (id: number) => () => { b.assign(uint(id)); };
    // same decision order as classifyBiome
    If(ice.element(i).greaterThan(B.ICE_MIN), set(Biome.ICE))
      .ElseIf(water.element(i).greaterThan(B.OCEAN_MIN), set(Biome.OCEAN))
      .ElseIf(alt.lessThan(B.BEACH_ALT).and(tAvg.greaterThan(0)), set(Biome.BEACH))
      .ElseIf(tAvg.lessThan(B.T_BOREAL).and(alt.greaterThan(B.ALPINE_ALT)), set(Biome.ALPINE))
      .ElseIf(tAvg.lessThan(B.T_TUNDRA), set(Biome.TUNDRA))
      .ElseIf(tAvg.lessThan(B.T_BOREAL), () => {
        If(m.greaterThan(B.M_TAIGA), set(Biome.TAIGA)).Else(set(Biome.TUNDRA));
      })
      .ElseIf(tAvg.lessThan(B.T_TROPIC), () => {
        If(m.lessThan(B.M_TEMP_DESERT), set(Biome.DESERT))
          .ElseIf(m.lessThan(B.M_TEMP_FOREST), set(Biome.GRASSLAND))
          .Else(set(Biome.TEMPERATE_FOREST));
      })
      .Else(() => {
        If(m.lessThan(B.M_HOT_DESERT), set(Biome.DESERT))
          .ElseIf(m.lessThan(B.M_RAINFOREST), set(Biome.SAVANNA))
          .Else(set(Biome.RAINFOREST));
      });
    biome.element(i).assign(b);
    const v0 = veg.element(i).toVar();
    veg.element(i).assign(v0.add((vegTarget.element(b) as unknown as THREE.Node<'float'>).sub(v0).mul(B.VEG_RATE)));
  })().compute(NCOL);

  return {
    step(renderer) { renderer.compute(k); },
    setSeaLevel(y) { seaLevel.value = y; },
    uniforms: { seaLevel },
  };
}
