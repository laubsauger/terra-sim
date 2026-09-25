// Inspect probe (§T.12, §I.probe): gather one column's voxels and fields into a tiny buffer, read it back.
import type * as THREE from 'three/webgpu';
import { Fn, If, int, uint, uniform, instanceIndex, Return, floatBitsToUint } from 'three/tsl';
import type { GpuFields } from '../core/gpu';
import { NCOL, NY, voxMat, voxFill, voxAge, voxFlags, codeToAge } from './layout';
import { PROBE_COL, PROBE_SIZE } from './fields';

export interface ProbeResult {
  x: number; z: number;
  voxels: { mat: number; fill: number; ageMy: number; flags: number }[]; // index = y
  surfY: number; water: number; plateId: number; crustAgeMy: number;
  surfTemp: number; biome: number; veg: number; ice: number; vapor: number; sedSusp: number;
  crustLayers: number; continental: boolean;
}

export class Probe {
  private col = uniform(0, 'uint');
  private kVox: [THREE.ComputeNode, THREE.ComputeNode];
  private kCol: [THREE.ComputeNode, THREE.ComputeNode];

  constructor(private fields: GpuFields) {
    const out = fields.cur<'uint'>('probe');
    const col = this.col;
    const [v0, v1] = fields.pair<'uint'>('vox');
    const buildVox = (vox: THREE.StorageBufferNode<'uint'>) => Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(NY)), () => { Return(); });
      out.element(instanceIndex).assign(vox.element(col.add(instanceIndex.mul(uint(NCOL)))));
    })().compute(NY);
    this.kVox = [buildVox(v0), buildVox(v1)];
    const [p0, p1] = fields.pair<'uint'>('plateId');
    const [a0, a1] = fields.pair('crustAge');
    // ≤ 8 storage buffers (V23): out, plateId, crustAge, surfY, water, colInfo, surfTemp, biome
    const surfY = fields.cur('surfY'), water = fields.cur('water'), colInfo = fields.cur<'uvec2'>('colInfo');
    const surfTemp = fields.cur('surfTemp'), biome = fields.cur<'uint'>('biome');
    const buildCol = (pid: THREE.StorageBufferNode<'uint'>, age: THREE.StorageBufferNode<'float'>) => Fn(() => {
      If(instanceIndex.greaterThan(uint(0)), () => { Return(); });
      const b = uint(PROBE_COL);
      out.element(b).assign(floatBitsToUint(surfY.element(col)));
      out.element(b.add(1)).assign(floatBitsToUint(water.element(col)));
      out.element(b.add(2)).assign(pid.element(col));
      out.element(b.add(3)).assign(floatBitsToUint(age.element(col)));
      out.element(b.add(4)).assign(floatBitsToUint(surfTemp.element(col)));
      out.element(b.add(5)).assign(biome.element(col));
      out.element(b.add(6)).assign(colInfo.element(col).x);
      out.element(b.add(7)).assign(colInfo.element(col).y);
    })().compute(1);
    this.kCol = [buildCol(p0, a0), buildCol(p1, a1)];
    void int;
  }

  async read(renderer: THREE.WebGPURenderer, x: number, z: number, extra: { veg: number; ice: number; vapor: number; sedSusp: number }): Promise<ProbeResult> {
    const f = this.fields;
    this.col.value = (x & 255) + (z & 255) * 256;
    renderer.compute(this.kVox[f.parity('vox') as 0 | 1]);
    renderer.compute(this.kCol[f.parity('plateId') as 0 | 1]);
    const u = new Uint32Array(await f.read(renderer, 'probe', 0, PROBE_SIZE * 4));
    const fl = new Float32Array(u.buffer);
    const voxels = [];
    for (let y = 0; y < NY; y++) { const v = u[y]!; voxels.push({ mat: voxMat(v), fill: voxFill(v), ageMy: codeToAge(voxAge(v)), flags: voxFlags(v) }); }
    const b = PROBE_COL;
    return {
      x, z, voxels, surfY: fl[b]!, water: fl[b + 1]!, plateId: u[b + 2]!, crustAgeMy: fl[b + 3]!,
      surfTemp: fl[b + 4]!, biome: u[b + 5]!, crustLayers: u[b + 7]! / 255, continental: ((u[b + 6]! >> 16) & 1) === 1, ...extra,
    };
  }
}
