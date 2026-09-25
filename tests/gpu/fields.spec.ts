import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/tests/gpu/support/blank.html'); });

// Ping-pong kernels must read the current buffer and write the other one, alternating every run.
// If parity bookkeeping breaks, sim passes silently read stale data.
test('PingPongKernel alternates buffers and results accumulate', async ({ page }) => {
  const out = await page.evaluate(async () => {
    const { makeRenderer, TSL } = await import('/tests/gpu/support/harness.ts');
    const { GpuFields, PingPongKernel } = await import('/src/core/gpu.ts');
    const { Fn, instanceIndex } = TSL;
    const r = await makeRenderer();
    const f = new GpuFields();
    const N = 300_000; // > 65535*4 threads → exercises 2D dispatch split
    f.add('v', 'float', N, { pingPong: true });
    f.freeze();
    const k = new PingPongKernel(f, 'v', (src, dst) =>
      Fn(() => { dst.element(instanceIndex).assign(src.element(instanceIndex).add(instanceIndex.toFloat())); })().compute(N));
    for (let i = 0; i < 3; i++) k.run(r);
    const buf = new Float32Array(await f.read(r, 'v'));
    return { parity: f.parity('v'), a: buf[0], b: buf[1], c: buf[N - 1], len: buf.length };
  });
  expect(out.parity).toBe(1);
  expect(out.len).toBe(300_000);
  expect(out.a).toBe(0);
  expect(out.b).toBe(3);
  expect(out.c).toBe(3 * 299_999);
});

// V10: GPU memory fixed after init.
test('adding a field after freeze throws', async ({ page }) => {
  const msg = await page.evaluate(async () => {
    const { GpuFields } = await import('/src/core/gpu.ts');
    const f = new GpuFields();
    f.add('a', 'uint', 16);
    f.freeze();
    try { f.add('b', 'uint', 16); return 'no throw'; } catch (e) { return (e as Error).message; }
  });
  expect(msg).toContain('V10');
});
