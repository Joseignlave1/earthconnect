<script lang="ts">
  import { onMount } from 'svelte';
  import configs from './maps.json';
  import MapPanel from './components/MapPanel.svelte';
  import QueryBuilder from './components/QueryBuilder.svelte';
  import Measurements from './components/Measurements.svelte';
  import ResultControls from './components/ResultControls.svelte';
  import ResultTable from './components/ResultTable.svelte';
  import { api } from './api.ts';
  import { targetAt } from './target.ts';
  import { targetLoader } from './characteristics.ts';
  import { applyMeasurements } from './matching.ts';
  import { fieldsFor, buildQuery } from './query.ts';
  import { encodeParams, resultRows } from './values.ts';
  import { prepareScores, MAX_SCORE } from './scores.ts';
  import { percent, errorMessage } from './types.ts';
  import type { Target, Sample, Field, Criterion, ScorePatch, Area } from './types.ts';
  import type { MapController, InspectedCell } from './maps.ts';
  import type { Datasource, QueryResponse } from './gen/earth_index_pb.ts';

  const maps = new Map<string, MapController>();
  let currentTarget = $state<Target | null>(null);
  let currentSample = $state.raw<Sample | null>(null);
  let datasources = $state.raw<Datasource[]>([]);
  let datasourceId = $state('');
  let currentDatasource = $state.raw<Datasource | null>(null);
  let fields = $state.raw<Field[]>([]);
  let criteria = $state<Criterion[]>([]);
  let queryMode = $state<'form' | 'custom'>('form');
  let tab = $state<'form' | 'console'>('form');
  let sql = $state('');
  let params = $state('{}');
  let formValid = $state(false);
  let sampling = $state(false);
  let running = $state(false);
  let queryStatus = $state('');
  let queryError = $state(false);
  let targetStatus = $state('');
  let targetError = $state('');
  let datasourceStatus = $state('Loading datasources…');
  let datasourceError = $state('');
  let body = $state('');
  let latitude = $state<number | undefined>();
  let longitude = $state<number | undefined>();
  let patches = $state.raw<ScorePatch[]>([]);
  let table = $state.raw<QueryResponse | null>(null);
  let cellScore = $state('');
  let scorePatch = $state('');
  let bestScore = $state(0);
  let picks: ResultControls | undefined = $state();
  let formTab: HTMLButtonElement;
  let consoleTab: HTMLButtonElement;
  let queryController: AbortController | undefined;
  let queryRevision = 0;
  let destroyed = false;
  const canRun = $derived(
    !!currentDatasource && !running && (queryMode === 'custom' || (formValid && !sampling)),
  );
  const evidence = $derived.by(() => {
    const selected = criteria.filter((c) => c.enabled);
    const measured = selected.filter((c) => c.kind === 'measured').length;
    return selected.length
      ? `${selected.length} scored variables · ${measured} measured · ${selected.filter((c) => c.kind === 'approximate').length} approximate thermal · ${selected.filter((c) => c.kind === 'preference').length} preference${measured < 3 ? ' · limited measured evidence' : ''}`
      : '';
  });
  function invalidateResults() {
    queryController?.abort();
    queryRevision++;
    maps.get('earth')?.scores?.clear();
    patches = [];
    table = null;
    bestScore = 0;
    cellScore = '';
    scorePatch = '';
    queryStatus = '';
    queryError = false;
    running = false;
  }
  function syncQuery() {
    if (queryMode !== 'form') return;
    try {
      const request = buildQuery(fields, criteria);
      sql = request.query;
      params = JSON.stringify(request.params, null, 2);
      formValid = true;
      queryStatus = '';
    } catch (error) {
      formValid = false;
      queryStatus = errorMessage(error);
    }
  }
  function editCriteria() {
    invalidateResults();
    syncQuery();
  }
  function applyTarget() {
    applyMeasurements(criteria, currentTarget, currentSample);
    syncQuery();
  }
  const loadTarget = targetLoader((result, state) => {
    if (destroyed) return;
    sampling = state === 'loading';
    if (queryMode === 'form') invalidateResults();
    currentSample = result && 'measurements' in result ? result : null;
    targetStatus = {
      empty: '',
      loading: 'Loading measurements…',
      ready: '',
      partial: 'Some sources unavailable',
      error: 'Measurements unavailable',
    }[state];
    targetError = result && 'error' in result ? errorMessage(result.error) : '';
    applyTarget();
  });
  function setTarget(target: Target | null, updateDraft = true) {
    invalidateResults();
    currentTarget = target;
    currentSample = null;
    if (updateDraft) {
      body = target?.body ?? '';
      latitude = target?.latitude;
      longitude = target?.longitude;
    }
    void loadTarget(target);
  }
  function editCoordinates() {
    if (!body) {
      setTarget(null);
      return;
    }
    if (
      latitude == null ||
      longitude == null ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude) ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      setTarget(null, false);
      return;
    }
    const target = targetAt(body, [longitude, latitude]);
    if (target) setTarget(target);
  }
  function mapReady(id: string, controller: MapController | null) {
    if (controller) {
      maps.set(id, controller);
    } else maps.delete(id);
  }
  function inspect(cell: InspectedCell) {
    cellScore = cell
      ? `${cell.siteKey} [${cell.x},${cell.y}] · ${cell.score == null ? 'No data' : percent(cell.score)}`
      : '';
  }
  function pick(area: Area) {
    maps.get('earth')?.scores?.focusArea(area);
    scorePatch = area.siteKey;
    cellScore = `${area.siteKey} · peak ${percent(area.peak / MAX_SCORE)}`;
  }
  function selectDatasource() {
    invalidateResults();
    currentDatasource = datasources.find((d) => d.id === datasourceId) ?? null;
    fields = fieldsFor(currentDatasource);
    criteria = fields.map((field) => ({
      key: field.key,
      target: null,
      enabled: false,
      tolerance: field.tolerance,
      weight: 1,
      min: null,
      max: null,
      ...criteria.find((c) => c.key === field.key),
    }));
    datasourceStatus = '';
    applyTarget();
  }
  const hasThermal = (d: Datasource) =>
    d.relations.find((r) => r.name === 'cells')?.columns.some((c) => c.name === 'surface_low_c');
  async function loadDatasources(signal: AbortSignal) {
    try {
      const metadata = await api.metadata({}, { signal, timeoutMs: 15000 });
      if (signal.aborted) return;
      datasources = metadata.datasources
        .filter((d) => d.body === 'earth')
        .sort(
          (a, b) =>
            b.coverage.length - a.coverage.length ||
            (b.relations.find((r) => r.name === 'cells')?.columns.length ?? 0) -
              (a.relations.find((r) => r.name === 'cells')?.columns.length ?? 0) ||
            a.id.localeCompare(b.id),
        );
      if (!datasources.length) throw new Error('No Earth datasources');
      datasourceId = datasources[0].id;
      selectDatasource();
    } catch (error) {
      if (signal.aborted) return;
      datasourceStatus = 'Datasources unavailable';
      datasourceError = errorMessage(error);
    }
  }
  async function runQuery() {
    const datasource = currentDatasource;
    if (!datasource || !canRun) return;
    invalidateResults();
    const revision = queryRevision;
    const controller = new AbortController();
    queryController = controller;
    running = true;
    queryStatus = 'Running…';
    const started = performance.now();
    try {
      const response = await api.query(
        { datasourceId: datasource.id, query: sql, params: encodeParams(JSON.parse(params)) },
        { signal: controller.signal },
      );
      if (revision !== queryRevision) return;
      if (
        response.columns.some((c) => c.name === 'site_key') &&
        response.columns.some((c) => c.name === 'scores')
      ) {
        const next = prepareScores(resultRows(response), datasource);
        const summary = maps.get('earth')?.scores?.show(next);
        if (!summary) throw new Error('Earth map is unavailable');
        patches = next;
        bestScore = summary.best?.score ?? 0;
        queryStatus = `${summary.valid.toLocaleString()} cells · ${patches.length} patches${summary.best ? ` · best ${percent(summary.best.score / MAX_SCORE)}` : ''}${summary.missing ? ` · ${summary.missing.toLocaleString()} missing` : ''} · ${((performance.now() - started) / 1000).toFixed(2)}s`;
      } else {
        table = response;
        queryStatus = `${response.rows.length.toLocaleString()} ${response.rows.length === 1 ? 'row' : 'rows'}${response.rows.length > 50 ? ' · showing first 50' : ''}`;
      }
    } catch (error) {
      if (revision === queryRevision) {
        queryStatus = errorMessage(error);
        queryError = true;
      }
    } finally {
      if (revision === queryRevision) running = false;
    }
  }
  function customQuery() {
    queryMode = 'custom';
    invalidateResults();
  }
  function useForm() {
    queryMode = 'form';
    invalidateResults();
    syncQuery();
  }
  function tabKey(event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    tab =
      event.key === 'Home'
        ? 'form'
        : event.key === 'End'
          ? 'console'
          : tab === 'form'
            ? 'console'
            : 'form';
    (tab === 'form' ? formTab : consoleTab).focus();
  }
  onMount(() => {
    const controller = new AbortController();
    void loadDatasources(controller.signal);
    return () => {
      destroyed = true;
      controller.abort();
      queryController?.abort();
      queryRevision++;
      void loadTarget(null);
    };
  });
</script>

<main>
  <div class="maps">
    {#each configs as config (config.id)}
      <MapPanel
        {config}
        target={currentTarget}
        onchoose={setTarget}
        onready={mapReady}
        oninspect={inspect}
      >
        {#snippet header()}
          {#if config.id === 'earth'}
            <select
              id="earth-datasource"
              aria-label="Earth datasource"
              disabled={!datasources.length}
              bind:value={datasourceId}
              onchange={selectDatasource}
            >
              {#each datasources as d}<option value={d.id} title={`Revision ${d.revision}`}
                  >{d.name}{hasThermal(d) ? ' · ground temperature' : ''}</option
                >{/each}
            </select>
          {/if}
        {/snippet}
        {#if config.id === 'earth'}
          <output id="datasource-status" title={datasourceError}>{datasourceStatus}</output>
          {#if patches.length}
            <div id="score-controls">
              <select
                id="score-patch"
                aria-label="Show result patch"
                bind:value={scorePatch}
                onchange={() =>
                  scorePatch
                    ? maps.get('earth')?.scores?.focus(scorePatch)
                    : maps.get('earth')?.reset()}
              >
                <option value="">World</option>{#each patches as patch}<option
                    value={patch.grid.siteKey}>{patch.grid.siteKey}</option
                  >{/each}
              </select>
              <button
                id="best-result"
                type="button"
                disabled={!bestScore}
                onclick={() => picks?.best()}>Best</button
              >
              <span
                class="legend"
                title="Brightness is relative to the best available score; percentages are absolute."
                >0% <span></span> {percent(bestScore / MAX_SCORE)}</span
              >
              <output id="cell-score">{cellScore}</output>
              <ResultControls
                bind:this={picks}
                {patches}
                onthreshold={(cutoff) => maps.get('earth')?.scores?.threshold(cutoff)}
                onpick={pick}
              />
            </div>
          {/if}
        {/if}
      </MapPanel>
    {/each}
  </div>
  <section class="query-panel">
    <div role="tablist" aria-label="Query">
      <button
        bind:this={formTab}
        id="form-tab"
        role="tab"
        aria-selected={tab === 'form'}
        aria-controls="form-panel"
        tabindex={tab === 'form' ? 0 : -1}
        type="button"
        onclick={() => (tab = 'form')}
        onkeydown={tabKey}>Form</button
      >
      <button
        bind:this={consoleTab}
        id="console-tab"
        role="tab"
        aria-selected={tab === 'console'}
        aria-controls="console-panel"
        tabindex={tab === 'console' ? 0 : -1}
        type="button"
        onclick={() => (tab = 'console')}
        onkeydown={tabKey}>Console</button
      >
    </div>
    <div id="form-panel" role="tabpanel" aria-labelledby="form-tab" hidden={tab !== 'form'}>
      <form id="target-form" onsubmit={(e) => e.preventDefault()}>
        <label
          >Target<select id="target-body" bind:value={body} onchange={editCoordinates}
            ><option value="">—</option><option value="moon">Moon</option><option value="mars"
              >Mars</option
            ></select
          ></label
        >
        <label
          >Latitude<input
            id="target-latitude"
            type="number"
            step="any"
            min="-90"
            max="90"
            bind:value={latitude}
            onchange={editCoordinates}
          /></label
        >
        <label
          >Longitude<input
            id="target-longitude"
            type="number"
            step="any"
            min="-180"
            max="180"
            bind:value={longitude}
            onchange={editCoordinates}
          /></label
        >
        <button id="clear-target" type="button" onclick={() => setTarget(null)}>Clear</button>
        <output id="target-status" title={targetError}>{targetStatus}</output>
      </form>
      <output id="match-evidence">{evidence}</output>
      <QueryBuilder
        {fields}
        bind:criteria
        disabled={queryMode !== 'form'}
        onchange={editCriteria}
      />
      <Measurements sample={currentSample} loading={sampling} />
    </div>
    <div
      id="console-panel"
      role="tabpanel"
      aria-labelledby="console-tab"
      hidden={tab !== 'console'}
    >
      <label
        >SQL<textarea
          id="query-text"
          spellcheck="false"
          aria-label="SQL"
          bind:value={sql}
          oninput={customQuery}></textarea></label
      >
      <label
        >Params<textarea
          id="query-params"
          spellcheck="false"
          aria-label="Params"
          bind:value={params}
          oninput={customQuery}></textarea></label
      >
    </div>
    <div class="query-actions">
      <button id="run-query" type="button" disabled={!canRun} onclick={runQuery}>Run</button>
      <button
        id="cancel-query"
        type="button"
        disabled={!running}
        onclick={() => {
          invalidateResults();
          queryStatus = 'Cancelled';
        }}>Cancel</button
      >
      <button
        id="clear-results"
        type="button"
        onclick={() => {
          invalidateResults();
          syncQuery();
        }}>Clear results</button
      >
      <output id="query-status" class:error={queryError}>{queryStatus}</output>
      <button id="use-form" type="button" hidden={queryMode === 'form'} onclick={useForm}
        >Use form query</button
      >
    </div>
    {#if table}<ResultTable response={table} />{/if}
  </section>
</main>
