<script lang="ts">
  import type { Field, Criterion } from '../types.ts';
  let {
    fields,
    criteria = $bindable(),
    disabled,
    onchange,
  }: {
    fields: Field[];
    criteria: Criterion[];
    disabled: boolean;
    onchange: () => void;
  } = $props();
  type NumericKey = 'target' | 'tolerance' | 'weight' | 'min' | 'max';
  const inputs: { key: NumericKey; label: string }[] = [
    { key: 'target', label: 'target' },
    { key: 'tolerance', label: 'tolerance' },
    { key: 'weight', label: 'weight' },
    { key: 'min', label: 'minimum' },
    { key: 'max', label: 'maximum' },
  ];
  function select(c: Criterion, event: Event) {
    c.enabled = (event.currentTarget as HTMLInputElement).checked;
    c.selectionTouched = true;
    c.selectedByUser = c.enabled;
    onchange();
  }
  function edit(c: Criterion, key: NumericKey, event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const value = input.value === '' ? null : input.valueAsNumber;
    if (key === 'weight' || key === 'tolerance') c[key] = value ?? NaN;
    else c[key] = value;
    if (key === 'tolerance') c.toleranceTouched = true;
    if (key === 'weight') c.weightTouched = true;
    if (key === 'target') {
      c.targetTouched = true;
      if (c.kind !== 'preference') c.kind = 'manual';
    }
    onchange();
  }
</script>

<div class="table-scroll">
  <table id="builder">
    <thead
      ><tr
        ><th>Score</th><th>Earth variable</th><th>Target</th><th>Tolerance</th><th>Weight</th><th
          >Min</th
        ><th>Max</th></tr
      ></thead
    >
    <tbody>
      {#each fields as field, i (field.key)}
        {@const c = criteria[i]}
        <tr data-field={field.key}>
          <td
            ><input
              type="checkbox"
              aria-label={`${field.label} score`}
              checked={c.enabled}
              {disabled}
              oninput={(e) => select(c, e)}
            /></td
          >
          <td title={[field.description, c.origin].filter(Boolean).join(' ')}
            >{field.label} ({field.unit}){field.key === 'bare_fraction'
              ? ' · preference'
              : ''}{c.kind === 'approximate' ? ' · approximate' : ''}</td
          >
          {#each inputs as { key, label }}
            <td
              ><input
                type="number"
                step="any"
                data-key={key}
                aria-label={`${field.label} ${label}`}
                value={key === 'tolerance' && field.categorical ? '' : (c[key] ?? '')}
                min={key === 'weight' || key === 'tolerance' ? 0.000001 : undefined}
                disabled={disabled || (key === 'tolerance' && field.categorical)}
                oninput={(e) => edit(c, key, e)}
              /></td
            >
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
