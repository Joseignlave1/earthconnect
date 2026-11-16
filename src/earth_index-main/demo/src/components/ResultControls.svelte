<script lang="ts">
  import { untrack } from 'svelte';
  import { scoreDistribution, findAreas, thresholdCode, SCORE_MAX } from '../picks.ts';
  import { purpleCss, scoreIntensity } from '../score-visuals.ts';
  import type { AreaPatch, Area } from '../types.ts';
  let {
    patches,
    onthreshold,
    onpick,
  }: {
    patches: AreaPatch[];
    onthreshold: (cutoff: number) => void;
    onpick: (area: Area) => void;
  } = $props();
  let threshold = $state(75);
  let result = $state<ReturnType<typeof findAreas>>({ count: 0, areaCount: 0, top: [] });
  let index = $state(-1);
  let focused = $state(false);
  let frame = 0;
  const distribution = $derived(scoreDistribution(patches));
  const largestBin = $derived(Math.max(1, ...distribution.bins));
  const pct = (code: number) => `${((100 * code) / SCORE_MAX).toFixed(1)}%`;
  const area = $derived(result.top[index]);
  function pick(next = index) {
    index = next;
    const selected = result.top[index];
    if (!selected) return;
    focused = true;
    onpick(selected);
  }
  function apply() {
    frame = 0;
    const cutoff = thresholdCode(threshold);
    onthreshold(cutoff);
    result = findAreas(patches, cutoff);
    index = result.top.length ? 0 : -1;
    focused = false;
  }
  function schedule() {
    if (!frame) frame = requestAnimationFrame(apply);
  }
  function chooseBand(band: number) {
    threshold = band * 2;
    apply();
  }
  export function best() {
    if (!distribution.best) return;
    if (thresholdCode(threshold) > distribution.best) {
      threshold = Math.floor((distribution.best / SCORE_MAX) * 1000) / 10;
      apply();
    }
    pick(0);
  }
  $effect(() => {
    // A new matrix resets the controls; editing the threshold never refetches it.
    const next = patches;
    untrack(() => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      threshold = Math.min(75, Math.floor((scoreDistribution(next).best / SCORE_MAX) * 900) / 10);
      apply();
      if (result.top.length) pick(0);
    });
    return () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };
  });
</script>

<div class="pick-controls">
  <svg
    id="score-distribution"
    viewBox="0 0 500 70"
    role="img"
    aria-label="Score distribution; logarithmic cell counts"
  >
    {#each [...distribution.bins] as count, i}
      {@const height = count ? Math.max(1, (Math.log1p(count) / Math.log1p(largestBin)) * 60) : 0}
      <rect
        x={i * 10}
        y={65 - height}
        width="9"
        {height}
        fill={purpleCss(
          0.2 + 0.8 * scoreIntensity(((i + 0.5) / 50) * SCORE_MAX, distribution.best),
        )}
      >
        <title>{i * 2}–{(i + 1) * 2}%: {count.toLocaleString()} cells</title>
      </rect>
      <g
        role="button"
        tabindex="0"
        aria-label={`Minimum score ${i * 2}%`}
        onclick={() => chooseBand(i)}
        onkeydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            chooseBand(i);
          }
        }}
      >
        <rect x={i * 10} y="0" width="10" height="70" fill="transparent" />
      </g>
    {/each}
    <line
      x1={threshold * 5}
      x2={threshold * 5}
      y1="0"
      y2="70"
      stroke="currentColor"
      stroke-width="2"
      pointer-events="none"
    />
  </svg>
  <div class="distribution-labels">
    <span>0%</span><span>Distribution · log count</span><span>100%</span>
  </div>
  <label class="threshold-label" for="score-threshold"
    >Minimum score <output id="threshold-value">{threshold.toFixed(1)}%</output></label
  >
  <input
    id="score-threshold"
    type="range"
    min="0"
    max="100"
    step="0.1"
    bind:value={threshold}
    aria-valuetext={`${threshold.toFixed(1)}%`}
    oninput={schedule}
  />
  <output id="pick-summary"
    >{result.count.toLocaleString()} cells · {result.areaCount.toLocaleString()} areas{distribution.missing
      ? ` · ${distribution.missing.toLocaleString()} unknown`
      : ''}{!result.count ? ' · no positive scores at this threshold' : ''}</output
  >
  <div class="pick-navigation">
    <button
      id="previous-pick"
      type="button"
      aria-label="Previous top area"
      disabled={index <= 0}
      onclick={() => pick(index - 1)}>←</button
    >
    <select
      id="top-pick"
      aria-label="Top areas"
      disabled={!result.top.length}
      value={index}
      onchange={(e) => pick(Number(e.currentTarget.value))}
    >
      {#each result.top as a, i}<option value={i}
          >{i + 1}. {a.siteKey} · {pct(a.peak)} · {a.count.toLocaleString()} cells</option
        >{/each}
    </select>
    <button id="go-pick" type="button" disabled={!result.top.length} onclick={() => pick()}
      >Go</button
    >
    <button
      id="next-pick"
      type="button"
      aria-label="Next top area"
      disabled={index < 0 || index >= result.top.length - 1}
      onclick={() => pick(index + 1)}>→</button
    >
  </div>
  <output id="pick-detail"
    >{focused && area
      ? `Peak ${pct(area.peak)} · mean ${pct(area.mean)} · top ${index + 1}/${result.top.length}`
      : result.top.length
        ? `${result.top.length} of ${result.areaCount.toLocaleString()} areas listed`
        : 'No picks'}</output
  >
</div>
