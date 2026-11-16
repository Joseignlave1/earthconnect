// @ts-nocheck
// Benchmark-only access to OpenLayers internals; never imported by the normal build.
import OLMap from 'ol/Map.js';
import ImageRenderer from 'ol/renderer/canvas/ImageLayer.js';
import ReprojImage from 'ol/reproj/Image.js';
import { fromBinary } from '@bufbuild/protobuf';
import { QueryResponseSchema } from '../src/gen/earth_index_pb.ts';

const stats = new Map();
let enabled = false;
const stack = [];
export const profile = {
  maps: new Map(),
  app: null,
  begin() {
    if (!enabled) return null;
    const token = { start: performance.now(), children: 0 };
    stack.push(token);
    return token;
  },
  end(name, token) {
    if (!token) return;
    const elapsed = performance.now() - token.start;
    stack.pop();
    if (stack.length) stack.at(-1).children += elapsed;
    const s = stats.get(name) ?? { calls: 0, total_ms: 0, self_ms: 0, max_ms: 0 };
    s.calls++;
    s.total_ms += elapsed;
    s.self_ms += elapsed - token.children;
    s.max_ms = Math.max(s.max_ms, elapsed);
    stats.set(name, s);
  },
  measure(name, fn) {
    const token = profile.begin();
    try {
      return fn();
    } finally {
      profile.end(name, token);
    }
  },
  asyncRecord(name, elapsed) {
    if (!enabled) return;
    const s = stats.get(name) ?? { calls: 0, total_ms: 0, self_ms: 0, max_ms: 0 };
    s.calls++;
    s.total_ms += elapsed;
    s.self_ms += elapsed;
    s.max_ms = Math.max(s.max_ms, elapsed);
    stats.set(name, s);
  },
};
globalThis.__earthProfile = profile;
profile.verifyGpu = async () => (await import('./verify-score-gpu.ts')).verifyScoreGpu();
profile.verifyWorldGpu = async () =>
  (await import('./verify-score-gpu.ts')).verifyWorldGpu(profile.maps.get('earth').map);

function wrap(proto, key, name, filter = () => true) {
  const original = proto[key];
  if (typeof original !== 'function') throw new Error(`Missing profiler hook: ${key}`);
  proto[key] = function (...args) {
    if (!enabled || !filter(this)) return original.apply(this, args);
    return profile.measure(name, () => original.apply(this, args));
  };
}
wrap(HTMLCanvasElement.prototype, 'toDataURL', 'canvas.PNG');
wrap(CanvasRenderingContext2D.prototype, 'putImageData', 'canvas.putImageData');
wrap(
  OLMap.prototype,
  'renderFrame_',
  'map.renderFrame',
  (map) => map === profile.maps.get('earth')?.map,
);
wrap(ImageRenderer.prototype, 'renderFrame', 'image.renderFrame');
wrap(ReprojImage.prototype, 'reproject_', 'image.reproject');
const fetchOriginal = globalThis.fetch;
globalThis.fetch = async (...args) => {
  const start = performance.now();
  const response = await fetchOriginal(...args);
  if (enabled && String(args[0] instanceof Request ? args[0].url : args[0]).endsWith('/Query')) {
    profile.rpc = { headers_ms: performance.now() - start };
  }
  return response;
};

const percentile = (values, p) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
};
const frame = () => new Promise((resolve) => requestAnimationFrame(resolve));
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const samples = [];
profile.samples = samples;
profile.environment = () => ({
  userAgent: navigator.userAgent,
  cores: navigator.hardwareConcurrency,
  visibility: document.visibilityState,
  viewport: [innerWidth, innerHeight],
  devicePixelRatio,
  coverage: profile.coverage,
  renderer: (() => {
    const gl = document.createElement('canvas').getContext('webgl');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
  })(),
});

profile.capture = async (name, action, settle = 500) => {
  // Let the previous scenario's images and frames finish outside this capture.
  await delay(300);
  stats.clear();
  profile.rpc = null;
  const longs = [];
  const observer = new PerformanceObserver((list) => {
    for (const e of list.getEntries())
      longs.push({ start_ms: e.startTime, duration_ms: e.duration });
  });
  observer.observe({ type: 'longtask' });
  const frameGaps = [];
  let last = 0,
    running = true;
  const observeFrame = (timestamp) => {
    if (last) frameGaps.push(timestamp - last);
    last = timestamp;
    if (running) requestAnimationFrame(observeFrame);
  };
  const initialHeap = performance.memory?.usedJSHeapSize;
  enabled = true;
  requestAnimationFrame(observeFrame);
  await frame();
  const start = performance.now();
  const renderEvents = [];
  const map = profile.maps.get('earth').map;
  const onRender = () => renderEvents.push(performance.now() - start);
  map.on('postrender', onRender);
  const result = await action();
  const actionEnd = performance.now();
  await delay(settle);
  await frame();
  running = false;
  enabled = false;
  map.un('postrender', onRender);
  const end = performance.now();
  await delay(0);
  observer.disconnect();
  const sample = {
    name,
    action_ms: actionEnd - start,
    window_ms: end - start,
    result,
    stages: Object.fromEntries(stats),
    rpc: profile.rpc,
    long_tasks: longs.filter((e) => e.start_ms >= start && e.start_ms < end),
    frame_gaps: {
      count: frameGaps.length,
      median_ms: percentile(frameGaps, 0.5),
      p95_ms: percentile(frameGaps, 0.95),
      max_ms: Math.max(0, ...frameGaps),
      over_50ms: frameGaps.filter((x) => x > 50).length,
    },
    render_events_ms: renderEvents,
    heap_start_bytes: initialHeap,
    heap_end_bytes: performance.memory?.usedJSHeapSize,
    earth: {
      resolution: profile.maps.get('earth').map.getView().getResolution(),
      layers: profile.maps.get('earth').map.getLayers().getLength(),
      canvas_count: profile.maps.get('earth').map.getViewport().querySelectorAll('canvas').length,
    },
  };
  samples.push(sample);
  return sample;
};
profile.query = async (name) => {
  const request = await (await fetch('/demo/profile-request.json')).json();
  return profile.capture(name, () => profile.app.run(request), 1200);
};
profile.interact = async (name, kind, frames = 12) => {
  const controller = profile.maps.get('earth');
  const { map } = controller;
  const view = map.getView();
  if (name.startsWith('empty-') || name.startsWith('world-')) {
    controller.reset();
    view.setResolution(view.getResolution() / 4);
  }
  const center = [...view.getCenter()],
    resolution = view.getResolution();
  return profile.capture(
    name,
    async () => {
      const started = performance.now();
      for (let i = 0; i < frames; i++) {
        // Fixed input sequence, one update per animation opportunity. Report
        // delivered frames rather than assuming a fixed animation duration.
        if (kind === 'pan')
          view.setCenter([
            center[0] + Math.sin((i / frames) * Math.PI * 2) * resolution * 40,
            center[1],
          ]);
        else view.setResolution(resolution * 2 ** (Math.sin((i / frames) * Math.PI * 2) * 2));
        await frame();
      }
      view.setCenter(center);
      view.setResolution(resolution);
      return {
        updates: frames,
        delivered_update_hz: (frames * 1000) / (performance.now() - started),
      };
    },
    500,
  );
};
profile.threshold = async (name) =>
  profile.capture(
    name,
    async () => {
      const input = document.querySelector('#score-threshold');
      input.value = '20';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await frame();
      await frame();
    },
    700,
  );

// Browser-only framing/protobuf baseline, using the exact request from the
// earlier whole-world fixture. No rendering or application state mutation.
profile.rawRpc = async () => {
  const { create, toBinary } = await import('@bufbuild/protobuf');
  const { QueryRequestSchema } = await import('../src/gen/earth_index_pb.ts');
  const { encodeParams } = await import('../src/values.ts');
  const request = await (await fetch('/demo/profile-request.json')).json();
  const message = toBinary(
    QueryRequestSchema,
    create(QueryRequestSchema, {
      datasourceId: 'earth-ed5bc9904a5f',
      query: request.query,
      params: encodeParams(request.params),
    }),
  );
  const payload = new Uint8Array(message.length + 5);
  payload.set(message, 5);
  new DataView(payload.buffer).setUint32(1, message.length, false);
  const start = performance.now();
  const response = await fetch('/earth_index.v1.EarthIndex/Query', {
    method: 'POST',
    headers: { 'content-type': 'application/grpc-web+proto', 'x-grpc-web': '1' },
    body: payload,
  });
  const headers = performance.now();
  const bytes = new Uint8Array(await response.arrayBuffer());
  const received = performance.now();
  const n = new DataView(bytes.buffer).getUint32(1, false);
  if (bytes[0] !== 0 || !response.ok) throw new Error('Unexpected gRPC-Web response');
  const result = fromBinary(QueryResponseSchema, bytes.subarray(5, 5 + n));
  const decoded = performance.now();
  if (result.rows.length !== 460) throw new Error(`Unexpected regions: ${result.rows.length}`);
  return {
    headers_ms: headers - start,
    body_ms: received - headers,
    protobuf_ms: decoded - received,
    total_ms: decoded - start,
    bytes: bytes.length,
    rows: result.rows.length,
  };
};
