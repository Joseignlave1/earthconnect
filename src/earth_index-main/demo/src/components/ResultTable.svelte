<script lang="ts">
  import type { QueryResponse } from '../gen/earth_index_pb.ts';
  import { decodeValue } from '../values.ts';
  let { response }: { response: QueryResponse } = $props();
  function display(item: QueryResponse['rows'][number]['values'][number]) {
    const value = decodeValue(item);
    return value instanceof Uint8Array
      ? `${value.length} bytes`
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);
  }
</script>

<div id="result-table" class="table-scroll">
  <table>
    <thead
      ><tr
        >{#each response.columns as column}<th>{column.name}</th>{/each}</tr
      ></thead
    >
    <tbody
      >{#each response.rows.slice(0, 50) as row}<tr
          >{#each row.values as value}<td>{display(value)}</td>{/each}</tr
        >{/each}</tbody
    >
  </table>
</div>
