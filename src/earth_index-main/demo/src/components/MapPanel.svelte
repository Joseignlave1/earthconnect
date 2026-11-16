<script lang="ts">
  import { onMount } from 'svelte';
  import type { Snippet } from 'svelte';
  import type { MapConfig, Target } from '../types.ts';
  import { createMap } from '../maps.ts';
  import type { MapController, InspectedCell } from '../maps.ts';
  let {
    config,
    target,
    onchoose,
    onready,
    oninspect = () => {},
    header,
    children,
  }: {
    config: MapConfig;
    target: Target | null;
    onchoose: (target: Target) => void;
    onready: (id: string, controller: MapController | null) => void;
    oninspect?: (cell: InspectedCell) => void;
    header?: Snippet;
    children?: Snippet;
  } = $props();
  let element: HTMLDivElement;
  let controller = $state.raw<MapController | null>(null);
  onMount(() => {
    const mounted = createMap(element, config, onchoose, oninspect);
    controller = mounted;
    onready(config.id, mounted);
    return () => {
      onready(config.id, null);
      mounted.destroy();
    };
  });
  $effect(() => controller?.setTarget(target));
</script>

<section class="map-panel" class:selected={target?.body === config.id} data-body={config.id}>
  <header>
    <h2>{config.label}</h2>
    {@render header?.()}
    <button
      type="button"
      data-reset={config.id}
      aria-label={`Reset ${config.label} map`}
      onclick={() => controller?.reset()}>Reset</button
    >
  </header>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex (OpenLayers provides keyboard navigation on the map.) -->
  <div
    bind:this={element}
    id={`map-${config.id}`}
    class="map"
    class:selectable={config.id !== 'earth'}
    role="region"
    aria-label={`${config.label} map`}
    tabindex="0"
  ></div>
  {@render children?.()}
</section>
