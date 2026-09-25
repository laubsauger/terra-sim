// Glacial erosion: ice sliding over bedrock scours it ∝ ice thickness × slope, so valleys under glaciers deepen
// fastest at their thalweg (thickest ice) and widen into U-shapes. Scoured rock becomes suspended sediment
// (moraine/meltwater load) that hydro carries off, so crust + sediment mass is exact (V3).
import type * as THREE from 'three/webgpu';
import { Fn, If, float, int, uint, uniform, instanceIndex, Return, vec2, clamp, min } from 'three/tsl';
import { ReaderKernel, type GpuFields } from '../core/gpu';
import { NCOL } from './layout';
import { tColIdx, tColXZ } from './tslLayout';
import { tDither, quantize, scanTop, removeTop } from './erosion';

export const GLACIAL = {
  kScour: 8,     // fill units per tick per (voxel of ice-water-eq × voxel/cell slope)
  minIce: 0.15,   // below this, snowpack not glacier
  maxPerTick: 6,  // fill units (≈0.5 layer/My max; 90/24 over-scoured)
};

export class GlacialErosion {
  private k: ReaderKernel<'uint'>;
  readonly tick = uniform(0, 'uint');
  readonly scale = uniform(1);

  constructor(fields: GpuFields) {
    const surfY = fields.cur('surfY');
    const ice = fields.cur('ice');
    const water = fields.cur('water');
    const sed = fields.cur('sedSusp');
    const { tick, scale } = this;
    // storage buffers: vox, surfY, ice, water, sedSusp = 5 (V23)
    this.k = new ReaderKernel<'uint'>(fields, 'vox', (vox) => Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const g = ice.element(i).toVar();
      If(g.lessThan(GLACIAL.minIce).or(water.element(i).greaterThan(0.5)), () => { Return(); });
      const { x, z } = tColXZ(i);
      const sY = (dx: number, dz: number) => surfY.element(tColIdx(x.add(int(dx)), z.add(int(dz))));
      const slope = clamp(vec2(sY(1, 0).sub(sY(-1, 0)), sY(0, 1).sub(sY(0, -1))).mul(0.5).length(), 0.05, 3);
      const want = min(g, 3).mul(slope).mul(GLACIAL.kScour).mul(scale);
      const n = quantize(want, tDither(i, tick, 7), GLACIAL.maxPerTick);
      If(n.greaterThan(uint(0)), () => {
        const { topY } = scanTop(vox, x, z);
        const got = removeTop(vox, x, z, topY, n);
        sed.element(i).addAssign(float(got));
      });
    })().compute(NCOL));
  }

  step(renderer: THREE.WebGPURenderer, tick: number, erosionScale: number): void {
    this.tick.value = tick >>> 0;
    this.scale.value = erosionScale;
    this.k.run(renderer);
  }
}
