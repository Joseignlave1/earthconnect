import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/svelte';
import ResultControls from '../../src/components/ResultControls.svelte';
import type { AreaPatch } from '../../src/types.ts';

afterEach(cleanup);
const patches: AreaPatch[] = [
  {
    grid: { siteKey: 'test', width: 3, height: 2 },
    scores: new Uint16Array([60000, 0, 50000, 65535, 0, 50000]),
  },
];

describe('Svelte result navigation', () => {
  it('filters locally without moving the map, navigates areas, and reveals Best', async () => {
    const onthreshold = vi.fn(),
      onpick = vi.fn();
    const ui = render(ResultControls, { patches, onthreshold, onpick });
    await waitFor(() => expect(onpick).toHaveBeenCalledOnce());
    expect(ui.getByLabelText('Top areas').children).toHaveLength(2);
    expect(ui.container.querySelector('#pick-summary')?.textContent).toContain(
      '3 cells · 2 areas · 1 unknown',
    );
    await fireEvent.click(ui.getByLabelText('Next top area'));
    expect(onpick.mock.lastCall?.[0]).toMatchObject({ peak: 50000, count: 2 });
    onpick.mockClear();
    await fireEvent.input(ui.getByRole('slider', { name: /^Minimum score/ }), {
      target: { value: '100' },
    });
    await waitFor(() => expect(ui.getByLabelText('Top areas').children).toHaveLength(0));
    expect(onthreshold.mock.lastCall?.[0]).toBe(65534);
    expect(onpick).not.toHaveBeenCalled();
    ui.component.best();
    await waitFor(() => expect(onpick).toHaveBeenCalledOnce());
    expect(onpick.mock.lastCall?.[0].peak).toBe(60000);
    expect(
      Number((ui.getByRole('slider', { name: /^Minimum score/ }) as HTMLInputElement).value),
    ).toBeLessThan(100);
    await fireEvent.keyDown(ui.getByLabelText('Minimum score 0%'), { key: 'Enter' });
    expect(onthreshold.mock.lastCall?.[0]).toBe(0);
    expect(onpick).toHaveBeenCalledOnce();
  });

  it('cancels scheduled threshold work when removed', async () => {
    const onthreshold = vi.fn();
    const ui = render(ResultControls, { patches, onthreshold, onpick: vi.fn() });
    await waitFor(() => expect(onthreshold).toHaveBeenCalledOnce());
    await fireEvent.input(ui.getByRole('slider', { name: /^Minimum score/ }), {
      target: { value: '100' },
    });
    ui.unmount();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    expect(onthreshold).toHaveBeenCalledOnce();
  });
});
