// Synthetic two-plate world for tectonics tests.
// Plate 0: oceanic, x ∈ [0,128). Plate 1: continental, x ∈ [128,256). Water fills to y=76.
import * as L from '/src/sim/layout.ts';
import { emptyWorld, type WorldData } from '/src/sim/worldData.ts';

export function twoPlateWorld(vel0: [number, number], vel1: [number, number], opts: { bothContinental?: boolean; bothOceanic?: boolean } = {}): WorldData {
  const w = emptyWorld(7);
  for (let z = 0; z < L.NZ; z++) for (let x = 0; x < L.NX; x++) {
    const c = L.colIdx(x, z);
    const cont = !opts.bothOceanic && (x >= 128 || !!opts.bothContinental);
    w.plateId[c] = x >= 128 ? 1 : 0;
    const base = cont ? 44 : 52;
    const top = cont ? 84 : 61; // exclusive
    for (let y = 0; y < top; y++) {
      const mat = y < base ? L.Mat.PERIDOTITE : cont ? L.Mat.GRANITE : L.Mat.BASALT;
      w.vox[L.voxIdx(x, y, z)] = L.packVoxel(mat, 255, 0, cont && y >= base ? L.FLAG_CONTINENTAL : 0);
    }
    w.crustAge[c] = cont ? 1000 : 50;
    w.water[c] = Math.max(0, 76 - top);
  }
  w.plates[0]!.alive = true; w.plates[0]!.vel = vel0;
  w.plates[1]!.alive = true; w.plates[1]!.vel = vel1; w.plates[1]!.continental = !opts.bothOceanic;
  w.mantleReservoir = 1_000_000;
  return w;
}
