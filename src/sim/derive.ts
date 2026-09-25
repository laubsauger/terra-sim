// §C pass 10 (derived): per-column surface height and crust summary from the voxel grid.
import { Fn, If, Loop, float, int, uint, instanceIndex, Return, uvec2 } from 'three/tsl';
import { ReaderKernel, type GpuFields } from '../core/gpu';
import { NCOL, NY, Mat, FLAG_CONTINENTAL } from './layout';
import { tMat, tFill, tFlags, tVoxIdx, tColXZ, uMin } from './tslLayout';

export const COL_CONTINENTAL = 1;

/**
 * surfY = y of top solid voxel + fill/255 (voxel-y units); column with no solid → 0.
 * colInfo.x = base | top<<8 | flags<<16, base = lowest crust voxel, top = top solid voxel.
 * colInfo.y = crust mass: Σ fill over crust voxels (everything solid except PERIDOTITE and MAGMA).
 */
export function createDerivePass(fields: GpuFields): ReaderKernel<'uint'> {
  const surfY = fields.cur('surfY');
  const colInfo = fields.cur<'uvec2'>('colInfo');
  return new ReaderKernel<'uint'>(fields, 'vox', (vox) =>
    Fn(() => {
      If(instanceIndex.greaterThanEqual(uint(NCOL)), () => { Return(); });
      const { x, z } = tColXZ(instanceIndex);
      const h = float(0).toVar();
      const top = uint(0).toVar();
      const base = uint(NY).toVar();
      const mass = uint(0).toVar();
      const flags = uint(0).toVar();
      Loop({ start: int(0), end: int(NY), condition: '<' }, ({ i }) => {
        const v = vox.element(tVoxIdx(x, i, z)).toVar();
        const m = tMat(v);
        If(m.notEqual(uint(Mat.AIR)), () => {
          top.assign(uint(i));
          h.assign(float(i).add(float(tFill(v)).div(255)));
          If(m.notEqual(uint(Mat.PERIDOTITE)).and(m.notEqual(uint(Mat.MAGMA))), () => {
            base.assign(uMin(base, uint(i)));
            mass.addAssign(tFill(v));
            If(tFlags(v).bitAnd(uint(FLAG_CONTINENTAL)).notEqual(uint(0)), () => { flags.assign(flags.bitOr(uint(COL_CONTINENTAL))); });
          });
        });
      });
      surfY.element(instanceIndex).assign(h);
      colInfo.element(instanceIndex).assign(uvec2(uMin(base, uint(255)).bitOr(top.shiftLeft(uint(8))).bitOr(flags.shiftLeft(uint(16))), mass));
    })().compute(NCOL),
  );
}

/** colInfo is interleaved [info, mass] per column, as read back from the uvec2 buffer. */
export interface ColumnSummary { surfY: Float32Array; colInfo: Uint32Array }

/** CPU reference of the derive kernel (tests, save tooling). */
export function deriveCpu(vox: Uint32Array, out: ColumnSummary): void {
  for (let c = 0; c < NCOL; c++) {
    let h = 0, top = 0, base = NY, mass = 0, flags = 0;
    for (let y = 0; y < NY; y++) {
      const v = vox[c + y * NCOL]!;
      const m = v & 0xff;
      if (m === Mat.AIR) continue;
      top = y;
      h = y + ((v >>> 8) & 0xff) / 255;
      if (m !== Mat.PERIDOTITE && m !== Mat.MAGMA) {
        base = Math.min(base, y);
        mass += (v >>> 8) & 0xff;
        if (((v >>> 24) & FLAG_CONTINENTAL) !== 0) flags |= COL_CONTINENTAL;
      }
    }
    out.surfY[c] = h;
    out.colInfo[c * 2] = (Math.min(base, 255) | (top << 8) | (flags << 16)) >>> 0;
    out.colInfo[c * 2 + 1] = mass;
  }
}
