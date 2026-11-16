<script lang="ts">
  import { MeasurementStatus } from '../gen/earth_index_pb.ts';
  import type { Sample } from '../types.ts';
  let { sample, loading }: { sample: Sample | null; loading: boolean } = $props();
  const names: Partial<Record<MeasurementStatus, string>> = {
    [MeasurementStatus.AVAILABLE]: 'Available',
    [MeasurementStatus.NO_DATA]: 'No data',
    [MeasurementStatus.OUTSIDE_COVERAGE]: 'Outside coverage',
    [MeasurementStatus.SOURCE_ERROR]: 'Source error',
    [MeasurementStatus.SOURCE_TIMEOUT]: 'Timed out',
  };
</script>

<details id="characteristics" hidden={!sample} aria-busy={loading}>
  <summary>Target measurements</summary>
  <div class="table-scroll">
    <table>
      <thead
        ><tr><th>Variable</th><th>Value</th><th>Unit</th><th>Status</th><th>Source</th></tr></thead
      >
      <tbody id="measurement-rows">
        {#each sample?.measurements ?? [] as m}
          <tr
            title={[
              m.notes,
              m.reason,
              m.pixelSpacing ? `${m.pixelSpacing} ${m.pixelSpacingUnit} grid` : '',
              ...m.qualityFlags,
            ]
              .filter(Boolean)
              .join('\n')}
          >
            <td>{m.label}</td>
            <td
              >{m.status === MeasurementStatus.AVAILABLE
                ? typeof m.decoded === 'object'
                  ? JSON.stringify(m.decoded)
                  : String(m.decoded)
                : '—'}</td
            >
            <td>{m.unit}</td><td>{names[m.status] ?? 'Unknown'}</td>
            <td
              >{#if m.sourceUrl}<a href={m.sourceUrl} target="_blank" rel="noopener noreferrer"
                  >{m.sourceId}</a
                >{:else}{m.sourceId}{/if}</td
            >
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</details>
