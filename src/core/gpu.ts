import type * as THREE from 'three/webgpu';
import { instancedArray } from 'three/tsl';

export type FieldType = 'float' | 'uint' | 'int' | 'vec2' | 'vec4' | 'uvec2' | 'uvec4' | 'ivec2';

const STRIDE: Record<FieldType, number> = { float: 4, uint: 4, int: 4, vec2: 8, vec4: 16, uvec2: 8, uvec4: 16, ivec2: 8 };

// TSL typings are per-element-type generics; the registry stores erased nodes and callers name the element type.
export type NodeElem = 'float' | 'uint' | 'int' | 'vec2' | 'vec4' | 'uvec2' | 'uvec4' | 'ivec2';
export type StorageNode<T extends NodeElem = 'float'> = THREE.StorageBufferNode<T>;
const makeArray = instancedArray as unknown as (count: number, type: FieldType) => StorageNode<NodeElem>;

interface Field {
  name: string;
  type: FieldType;
  count: number;
  bufs: StorageNode<NodeElem>[]; // 1 or 2 (ping-pong)
  parity: number;
}

// Fixed-size GPU storage registry. All sim state lives here; allocation only before freeze() (V10).
export class GpuFields {
  private fields = new Map<string, Field>();
  private frozen = false;

  add(name: string, type: FieldType, count: number, opts: { pingPong?: boolean; atomic?: boolean } = {}): void {
    if (this.frozen) throw new Error(`GpuFields: add('${name}') after freeze — GPU memory must be fixed after init (V10)`);
    if (this.fields.has(name)) throw new Error(`GpuFields: duplicate field '${name}'`);
    const n = opts.pingPong ? 2 : 1;
    const bufs: StorageNode<NodeElem>[] = [];
    for (let i = 0; i < n; i++) {
      const b = makeArray(count, type);
      b.setName(n === 1 ? name : `${name}_${i}`);
      if (opts.atomic) {
        if (type !== 'int' && type !== 'uint') throw new Error(`GpuFields: atomic field '${name}' must be int|uint (V2: no float atomics)`);
        b.toAtomic();
      }
      bufs.push(b);
    }
    this.fields.set(name, { name, type, count, bufs, parity: 0 });
  }

  freeze(): void { this.frozen = true; }
  get isFrozen(): boolean { return this.frozen; }

  private field(name: string): Field {
    const f = this.fields.get(name);
    if (!f) throw new Error(`GpuFields: unknown field '${name}'`);
    return f;
  }

  /** Current (read) buffer. */
  cur<T extends NodeElem = 'float'>(name: string): StorageNode<T> { const f = this.field(name); return f.bufs[f.parity] as unknown as StorageNode<T>; }
  /** Write target of a ping-pong field. */
  next<T extends NodeElem = 'float'>(name: string): StorageNode<T> {
    const f = this.field(name);
    if (f.bufs.length < 2) throw new Error(`GpuFields: '${name}' is not ping-pong`);
    return f.bufs[1 - f.parity] as unknown as StorageNode<T>;
  }
  /** Both buffers [0,1] of a ping-pong field, for building parity-specific kernels. */
  pair<T extends NodeElem = 'float'>(name: string): [StorageNode<T>, StorageNode<T>] {
    const f = this.field(name);
    if (f.bufs.length < 2) throw new Error(`GpuFields: '${name}' is not ping-pong`);
    return [f.bufs[0], f.bufs[1]] as unknown as [StorageNode<T>, StorageNode<T>];
  }
  parity(name: string): number { return this.field(name).parity; }
  swap(name: string): void { const f = this.field(name); f.parity = 1 - f.parity; }
  setParity(name: string, p: number): void { this.field(name).parity = p & 1; }
  count(name: string): number { return this.field(name).count; }
  type(name: string): FieldType { return this.field(name).type; }
  names(): string[] { return [...this.fields.keys()]; }

  /** CPU-side backing array of the current buffer (for init upload). Call markDirty after writing. */
  cpuArray(name: string, which: 0 | 1 = 0): Float32Array | Uint32Array | Int32Array {
    return this.field(name).bufs[which]!.value.array as Float32Array | Uint32Array | Int32Array;
  }
  markDirty(name: string): void { for (const b of this.field(name).bufs) b.value.needsUpdate = true; }

  bytes(): number {
    let t = 0;
    for (const f of this.fields.values()) t += STRIDE[f.type] * f.count * f.bufs.length;
    return t;
  }

  /** Async readback of the current buffer (debug, probe, stats, save). */
  async read(renderer: THREE.WebGPURenderer, name: string, offsetBytes = 0, countBytes = -1): Promise<ArrayBuffer> {
    const f = this.field(name);
    const attr = f.bufs[f.parity]!.value;
    // three allocates GPU buffers lazily on first binding; an unbound field's CPU array is still the truth.
    const backend = renderer.backend as unknown as { get(o: object): { buffer?: GPUBuffer } };
    if (!backend.get(attr).buffer) {
      const src = new Uint8Array((attr.array as ArrayBufferView).buffer);
      const end = countBytes < 0 ? src.byteLength : offsetBytes + countBytes;
      return src.slice(offsetBytes, end).buffer;
    }
    return renderer.getArrayBufferAsync(attr, null, offsetBytes, countBytes) as Promise<ArrayBuffer>;
  }
}

/** Kernel with two compiled variants, one per ping-pong parity of `field`. run() dispatches the right one then swaps. */
export class PingPongKernel<T extends NodeElem = 'float'> {
  private variants: [THREE.ComputeNode, THREE.ComputeNode];
  constructor(private fields: GpuFields, private field: string,
    build: (src: StorageNode<T>, dst: StorageNode<T>) => THREE.ComputeNode) {
    const [a, b] = fields.pair<T>(field);
    this.variants = [build(a, b), build(b, a)];
  }
  run(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.variants[this.fields.parity(this.field) as 0 | 1]);
    this.fields.swap(this.field);
  }
}

/** Kernel that only reads a ping-pong field: two variants, picks the one bound to the current buffer. No swap. */
export class ReaderKernel<T extends NodeElem = 'float'> {
  private variants: [THREE.ComputeNode, THREE.ComputeNode];
  constructor(private fields: GpuFields, private field: string, build: (cur: StorageNode<T>) => THREE.ComputeNode) {
    const [a, b] = fields.pair<T>(field);
    this.variants = [build(a), build(b)];
  }
  run(renderer: THREE.WebGPURenderer): void {
    renderer.compute(this.variants[this.fields.parity(this.field) as 0 | 1]);
  }
}
