<script lang="ts">
  import ResultControls from './ResultControls.svelte';
  import type { AreaPatch, Area } from '../types.ts';
  let { patches }: { patches: AreaPatch[] } = $props();
  let controls: ResultControls;
  let selected = $state('');
  function pick(area: Area) {
    const width = patches.find((p) => p.grid.siteKey === area.siteKey)!.grid.width;
    selected = `${area.siteKey} · peak cell [${(area.peakIndex % width) + 1},${Math.floor(area.peakIndex / width) + 1}]`;
  }
</script>

<button type="button" onclick={() => controls.best()}>Best</button>
<ResultControls bind:this={controls} {patches} onthreshold={() => {}} onpick={pick} />
<output id="selected-area">{selected}</output>
