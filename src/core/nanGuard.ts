// §T.35 NaN guard (V7). GPU check kernels over every float-typed field: any element whose exponent bits are
// all ones (NaN or ±Inf) atomically ORs its field's bit into a private 1-element flag buffer.
// Bit test instead of isnan(): WGSL has no isnan and compilers may fold x != x under fast-math.
//
// Flag location: its own atomic u32 buffer owned by the guard (not a GpuFields field, not a counters slot),
// so it is never saved, never hashed and never cleared by the per-window counter reset.
// Bit i ↔ checked field i (fieldNames()); fields past 31 share bit 31.
//
// Kernels bind ≤ 7 field buffers + the flag (V23). Ping-pong fields bind both buffers (the stale one held a
// state that was current one step earlier, so it is finite whenever the world was).
import type * as THREE from 'three/webgpu';
import { Fn, If, Return, uint, instanceIndex, instancedArray, atomicOr, atomicStore, floatBitsToUint } from 'three/tsl';
import type { FieldType, GpuFields, NodeElem, StorageNode } from './gpu';

const FLOAT_TYPES: Partial<Record<FieldType, number>> = { float: 1, vec2: 2, vec4: 4 };
const MAX_PER_KERNEL = 7; // + flag buffer = 8 storage buffers (V23)
const EXP_MASK = 0x7f800000;

export class NanGuard {
  private readonly names: string[] = [];
  private readonly kernels: THREE.ComputeNode[] = [];
  private readonly clear: THREE.ComputeNode;
  private readonly flag: StorageNode<'uint'>;

  /** exclude: scratch fields (rebuilt before use, may hold garbage between passes). */
  constructor(fields: GpuFields, exclude: ReadonlySet<string> = new Set()) {
    this.flag = (instancedArray as unknown as (n: number, t: string) => StorageNode<'uint'>)(1, 'uint');
    this.flag.setName('nanFlag');
    this.flag.toAtomic();
    const flag = this.flag;
    this.clear = Fn(() => { atomicStore(flag.element(0), uint(0)); })().compute(1);

    // group by element count so each kernel dispatches exactly one thread per element
    const groups = new Map<number, { buf: StorageNode<NodeElem>; comps: number; bit: number }[][]>();
    for (const name of fields.names()) {
      const comps = FLOAT_TYPES[fields.type(name)];
      if (!comps || exclude.has(name)) continue;
      const bit = Math.min(this.names.length, 31);
      this.names.push(name);
      const count = fields.count(name);
      const bufs = fieldBuffers(fields, name);
      let lists = groups.get(count);
      if (!lists) groups.set(count, (lists = [[]]));
      for (const buf of bufs) {
        if (lists[lists.length - 1]!.length >= MAX_PER_KERNEL) lists.push([]);
        lists[lists.length - 1]!.push({ buf, comps, bit });
      }
    }
    for (const [count, lists] of groups) for (const list of lists) this.kernels.push(buildCheck(list, flag, count));
  }

  /** Checked fields in bit order. */
  fieldNames(): readonly string[] { return this.names; }

  /** Field names flagged in a bit mask from check(). */
  decode(bits: number): string[] {
    return this.names.filter((_, i) => (bits >>> Math.min(i, 31)) & 1);
  }

  /**
   * Dispatch clear + check kernels now (queue order: sees every compute submitted before this call) and
   * return the flag bits once the readback lands. 0 = all finite.
   */
  check(renderer: THREE.WebGPURenderer): Promise<number> {
    renderer.compute(this.clear);
    for (const k of this.kernels) renderer.compute(k);
    return (renderer.getArrayBufferAsync(this.flag.value) as Promise<ArrayBuffer>).then((b) => new Uint32Array(b)[0]!);
  }
}

function fieldBuffers(fields: GpuFields, name: string): StorageNode<NodeElem>[] {
  try { return fields.pair<NodeElem>(name); } catch { return [fields.cur<NodeElem>(name)]; }
}

function buildCheck(list: { buf: StorageNode<NodeElem>; comps: number; bit: number }[], flag: StorageNode<'uint'>, count: number): THREE.ComputeNode {
  return Fn(() => {
    const i = instanceIndex;
    If(i.greaterThanEqual(uint(count)), () => { Return(); });
    const bad = uint(0).toVar();
    for (const { buf, comps, bit } of list) {
      const v = buf.element(i).toVar() as unknown as THREE.Node<'vec4'>;
      const parts = comps === 1 ? [v] : (['x', 'y', 'z', 'w'] as const).slice(0, comps).map((c) => v[c]);
      for (const p of parts) {
        If((floatBitsToUint(p as unknown as THREE.Node<'float'>) as unknown as THREE.Node<'uint'>).bitAnd(uint(EXP_MASK)).equal(uint(EXP_MASK)), () => {
          bad.assign(bad.bitOr(uint((1 << bit) >>> 0)));
        });
      }
    }
    If(bad.notEqual(uint(0)), () => { atomicOr(flag.element(0), bad); });
  })().compute(count);
}
