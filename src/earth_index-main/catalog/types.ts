/** Planning/research contract. No live HTTP endpoint is implemented yet. */
export type PilotStatus =
  | "planned"
  | "pilot_processed"
  | "pilot_components_only"
  | "deferred";

export interface PilotOutput {
  id: string;
  unit: string;
  statistic: "min" | "mean" | "max" | "stddev";
  artifactColumn: string | null;
}

export interface VariableDefinition {
  id: string;
  definitionVersion: number;
  label: string;
  groupId: string;
  valueType: "number" | "category";
  proposedUnit: string;
  unitStatus: "proposed_normalization_target";
  sourceId: string;
  candidateSourceField: string | null;
  kind: "native" | "derived" | "summary";
  role: string;
  priority: "evaluate" | "optional" | "deferred";
  bodyIds: string[];
  dimensions: string[];
  temporalKind: "static" | "interval" | "annual" | "map_version" | "reference_period";
  notes: string;
  sourceMappingStatus: string;
  /** Source/version-specific dictionaries have not yet been validated/exported. */
  categoryDictionary: null;
  productionAvailable: false;
  pilot: {
    status: PilotStatus;
    regionIds: string[];
    outputs: PilotOutput[];
    manifest: string | null;
  };
}

export interface VariableCatalog {
  schemaVersion: 1;
  catalogVersion: string;
  generatedAt: string;
  sourceRegistrySha256: string;
  status: "planning_catalog_with_research_pilot_availability_no_live_API";
  groups: Array<{ id: string; label: string }>;
  counts: {
    variables: number;
    groups: number;
    byPilotStatus: Partial<Record<PilotStatus, number>>;
  };
  notes: string[];
  variables: VariableDefinition[];
}

interface ObservationBase {
  variableId: string;
  outputId: string;
  bodyId: string;
  regionId: string;
  cellId: string;
  h3Resolution: number;
  statistic: string;
  status: "valid" | "missing" | "not_processed" | "not_applicable" | "quality_filtered";
  window:
    | { kind: "static" }
    | { kind: "interval"; start: string; endExclusive: string }
    | { kind: "reference_period"; start: string; endExclusive: string };
  dimensions: Record<string, string | number | [number, number]>;
  validAreaFraction: number | null;
  spatialSupport: string;
  provenanceRef: string;
}

/** A zero is a value. Missingness uses null plus an explicit status. */
export type VariableObservation = ObservationBase &
  (
    | { valueType: "number"; value: number | null; unit: string }
    | {
        valueType: "category";
        value: string | null;
        unit: null;
        vocabularyVersion: string;
      }
  );
