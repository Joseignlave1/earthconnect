import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte';
import { create } from '@bufbuild/protobuf';
import {
  DatasourceSchema,
  QueryResponseSchema,
  SamplePointResponseSchema,
  MeasurementStatus,
} from '../../src/gen/earth_index_pb.ts';
import { scoreSummary } from '../../src/scores.ts';
import { encodeValue } from '../../src/values.ts';
import type { MapController } from '../../src/maps.ts';
import type { Target } from '../../src/types.ts';

const mocks = vi.hoisted(() => ({
  metadata: vi.fn(),
  samplePoint: vi.fn(),
  query: vi.fn(),
  choose: new Map<string, (target: Target) => void>(),
  controllers: new Map<string, MapController>(),
}));
vi.mock('../../src/api.ts', () => ({ api: mocks }));
vi.mock('../../src/maps.ts', () => ({
  createMap: (_element: HTMLElement, config: { id: string }, choose: (target: Target) => void) => {
    mocks.choose.set(config.id, choose);
    const controller = {
      reset: vi.fn(),
      setTarget: vi.fn(),
      destroy: vi.fn(),
      scores: {
        clear: vi.fn(),
        show: vi.fn(scoreSummary),
        threshold: vi.fn(),
        focusArea: vi.fn(),
        focus: vi.fn(),
        inspect: vi.fn(),
      },
    };
    mocks.controllers.set(config.id, controller);
    return controller;
  },
}));
import App from '../../src/App.svelte';

const datasource = create(DatasourceSchema, {
  id: 'earth-test',
  name: 'Earth',
  body: 'earth',
  relations: [
    {
      name: 'cells',
      columns: [
        { name: 'slope_mean_deg', unit: 'degree' },
        { name: 'bare_fraction', unit: '1' },
        { name: 'surface_low_c', unit: 'degC' },
      ],
    },
  ],
  grids: [{ siteKey: 'test', width: 2, height: 2, transform: [1000, 0, 0, 0, -1000, 0] }],
});
function sample(slope = 3) {
  return create(SamplePointResponseSchema, {
    measurements: [
      {
        variableId: 'mars.slope_1km',
        value: encodeValue(slope),
        status: MeasurementStatus.AVAILABLE,
        unit: 'degree',
      },
      {
        variableId: 'mars.modeled_surface_temperature_min',
        value: encodeValue(200),
        status: MeasurementStatus.AVAILABLE,
        unit: 'K',
      },
    ],
  });
}
const packed = new Uint8Array([254, 255, 0, 0, 255, 255, 200, 100]);
const response = create(QueryResponseSchema, {
  columns: [{ name: 'site_key' }, { name: 'scores' }],
  rows: [{ values: [encodeValue('test'), { kind: { case: 'bytesValue', value: packed } }] }],
});
afterEach(cleanup);
beforeEach(() => {
  vi.clearAllMocks();
  mocks.choose.clear();
  mocks.controllers.clear();
  mocks.metadata.mockResolvedValue({ datasources: [datasource] });
  mocks.samplePoint.mockResolvedValue(sample());
  mocks.query.mockResolvedValue(response);
});
async function ready() {
  const ui = render(App);
  await waitFor(() =>
    expect((ui.getByLabelText('Earth datasource') as HTMLSelectElement).disabled).toBe(false),
  );
  return ui;
}

it('starts with global coverage and allows choosing another datasource', async () => {
  const global = create(DatasourceSchema, {
    ...datasource,
    id: 'earth-global',
    name: 'Global Earth',
    grids: Array.from({ length: 460 }, (_, i) => ({
      siteKey: `region-${i}`,
      width: 278,
      height: 278,
    })),
    coverage: Array.from({ length: 460 }, (_, i) => ({ siteKey: `region-${i}`, srid: 4326 })),
  });
  mocks.metadata.mockResolvedValue({ datasources: [global, datasource] });
  const ui = await ready();
  const select = ui.getByLabelText('Earth datasource') as HTMLSelectElement;
  expect(select.value).toBe(global.id);
  await fireEvent.change(select, { target: { value: datasource.id } });
  expect(select.value).toBe(datasource.id);
});
// Avoid matcher plugins: tests use ordinary DOM properties throughout.
it('keeps three maps, autofills measurements, executes the builder, and filters without another RPC', async () => {
  const ui = await ready();
  expect(ui.getAllByRole('region')).toHaveLength(3);
  mocks.choose.get('mars')!({ body: 'mars', longitude: 23.47, latitude: 0.67 });
  await waitFor(() =>
    expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).value).toBe('3'),
  );
  expect(
    (ui.getByLabelText('Coldest observed nighttime composite target') as HTMLInputElement)
      .valueAsNumber,
  ).toBeCloseTo(-73.15);
  expect((ui.getByLabelText('Mean slope score') as HTMLInputElement).checked).toBe(true);
  await fireEvent.click(ui.getByText('Run', { selector: 'button' }));
  await waitFor(() => expect(ui.getByRole('slider', { name: /^Minimum score/ })).toBeTruthy());
  expect(mocks.query).toHaveBeenCalledOnce();
  expect(mocks.query.mock.lastCall?.[1]).not.toHaveProperty('timeoutMs');
  expect(mocks.query.mock.lastCall?.[0].params).toHaveProperty('target_0');
  await fireEvent.input(ui.getByRole('slider', { name: /^Minimum score/ }), {
    target: { value: '90' },
  });
  await waitFor(() =>
    expect(mocks.controllers.get('earth')?.scores?.threshold).toHaveBeenLastCalledWith(58981),
  );
  expect(mocks.query).toHaveBeenCalledOnce();
  await fireEvent.click(ui.getByRole('tab', { name: 'Console' }));
  expect((ui.getByLabelText('SQL') as HTMLTextAreaElement).value).toContain('pack_u16');
  await fireEvent.input(ui.getByLabelText('SQL'), { target: { value: 'SELECT 42 AS answer' } });
  expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).disabled).toBe(true);
  mocks.query.mockResolvedValue(
    create(QueryResponseSchema, {
      columns: [{ name: 'answer' }],
      rows: [{ values: [encodeValue(42)] }],
    }),
  );
  await fireEvent.click(ui.getByText('Run', { selector: 'button' }));
  await waitFor(() => expect(ui.getByText('42', { selector: 'td' })).toBeTruthy());
  expect(mocks.query.mock.lastCall?.[0].query).toBe('SELECT 42 AS answer');
  await fireEvent.click(ui.getByText('Use form query'));
  expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).disabled).toBe(false);
  expect((ui.getByLabelText('SQL') as HTMLTextAreaElement).value).toContain('pack_u16');
});

it('ignores stale targets and query responses, clears an invalid coordinate draft, and cleans up maps', async () => {
  let completeSample: (value: ReturnType<typeof sample>) => void = () => {};
  mocks.samplePoint.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        completeSample = resolve;
      }),
  );
  const ui = await ready();
  mocks.choose.get('mars')!({ body: 'mars', longitude: 1, latitude: 2 });
  await waitFor(() => expect(mocks.samplePoint).toHaveBeenCalledOnce());
  mocks.choose.get('mars')!({ body: 'mars', longitude: 3, latitude: 4 });
  await waitFor(() =>
    expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).value).toBe('3'),
  );
  completeSample(sample(99));
  await Promise.resolve();
  expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).value).toBe('3');
  let completeQuery: (value: typeof response) => void = () => {};
  mocks.query.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        completeQuery = resolve;
      }),
  );
  await fireEvent.click(ui.getByText('Run', { selector: 'button' }));
  const signal = mocks.query.mock.lastCall?.[1].signal as AbortSignal;
  await fireEvent.click(ui.getByText('Cancel', { selector: 'button' }));
  expect(signal.aborted).toBe(true);
  completeQuery(response);
  await Promise.resolve();
  expect(ui.queryByRole('slider', { name: /^Minimum score/ })).toBeNull();
  await fireEvent.input(ui.getByLabelText('Latitude'), { target: { value: '' } });
  await fireEvent.change(ui.getByLabelText('Latitude'));
  expect((ui.getByLabelText('Longitude') as HTMLInputElement).value).toBe('3');
  expect((ui.getByLabelText('Mean slope target') as HTMLInputElement).value).toBe('');
  expect((ui.getByText('Run', { selector: 'button' }) as HTMLButtonElement).disabled).toBe(true);
  const controllers = [...mocks.controllers.values()];
  ui.unmount();
  for (const controller of controllers) expect(controller.destroy).toHaveBeenCalledOnce();
});
