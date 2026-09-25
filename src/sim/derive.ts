// §C pass 10 (derived): surface height per column from the voxel grid.
import { Fn, If, Loop, Break, float, int, uint, instanceIndex, Return } from 'three/tsl';
import { ReaderKernel, type GpuFields } from '../core/gpu';
import { NCOL, NY, Mat } from './layout';
import { tMat, tFill, tVoxIdx, tColXZ } from './tslLayout';

/** surfY = y of top solid voxel + fill/255 (voxel-y units). Column with no solid → 0. */
export function createDerivePass(fields: GpuFields): ReaderKernel<'uint'> {
  const surfY = fields.cur('surfY');
  return new ReaderKernel<'uint'>(fields, 'vox', (vox) =>
    Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(instanceIndex);
      const h = float(0).toVar();
      Loop({ start: int(NY - 1), end: int(0), condition: '>=' }, ({ i }) => {
        const v = vox.element(tVoxIdx(x, i, z));
        If(tMat(v).notEqual(uint(Mat.AIR)), () => {
          h.assign(float(i).add(float(tFill(v)).div(255)));
          Break();
        });
      });
      surfY.element(instanceIndex).assign(h);
    })().compute(NCOL),
  );
}

/** CPU reference of the derive kernel (tests, save tooling). */
export function surfYCpu(vox: Uint32Array, out: Float32Array): void {
  for (let c = 0; c < NCOL; c++) {
    let h = 0;
    for (let y = NY - 1; y >= 0; y--) {
      const v = vox[c + y * NCOL]!;
      if ((v & 0xff) !== Mat.AIR) { h = y + ((v >>> 8) & 0xff) / 255; break; }
    }
    out[c] = h;
  }
}
