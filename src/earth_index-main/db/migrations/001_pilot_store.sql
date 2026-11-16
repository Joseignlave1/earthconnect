-- First implementable subset of DESIGN.md: imported research pilots, not releases.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA catalog;
CREATE SCHEMA geo;
CREATE SCHEMA ingest;
CREATE SCHEMA data;
CREATE SCHEMA research;

CREATE TABLE catalog.variables (
    variable_key text PRIMARY KEY,
    definition_version integer NOT NULL CHECK (definition_version > 0),
    label text NOT NULL,
    value_kind text NOT NULL CHECK (value_kind IN ('number', 'category')),
    definition jsonb NOT NULL
);

CREATE TABLE catalog.features (
    feature_key text PRIMARY KEY,
    variable_key text NOT NULL REFERENCES catalog.variables
);
CREATE TABLE catalog.feature_versions (
    feature_version_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    feature_key text NOT NULL REFERENCES catalog.features,
    version integer NOT NULL CHECK (version > 0),
    canonical_unit text NOT NULL,
    statistic text NOT NULL CHECK (statistic IN ('min', 'mean', 'max', 'stddev')),
    value_kind text NOT NULL DEFAULT 'numeric' CHECK (value_kind = 'numeric'),
    definition jsonb NOT NULL,
    definition_sha256 text NOT NULL CHECK (definition_sha256 ~ '^[0-9a-f]{64}$'),
    UNIQUE (feature_key, version)
);

CREATE TABLE geo.grids (
    grid_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    grid_key text NOT NULL UNIQUE,
    body text NOT NULL CHECK (body = 'earth'),
    scheme text NOT NULL CHECK (scheme = 'h3'),
    resolution smallint NOT NULL CHECK (resolution BETWEEN 0 AND 15),
    library_version text NOT NULL,
    area_method text NOT NULL
);
CREATE TABLE geo.cells (
    grid_id bigint NOT NULL REFERENCES geo.grids,
    cell_id text NOT NULL CHECK (cell_id ~ '^[0-9a-f]{15}$'),
    geom geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom) AND NOT ST_IsEmpty(geom)),
    centroid geography(Point, 4326) NOT NULL,
    area_m2 double precision NOT NULL CHECK (area_m2 > 0 AND area_m2 < 'Infinity'::float8),
    PRIMARY KEY (grid_id, cell_id)
);
CREATE INDEX cells_geom_gist ON geo.cells USING gist (geom);
CREATE INDEX cells_centroid_gist ON geo.cells USING gist (centroid);

CREATE TABLE geo.regions (
    region_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region_key text NOT NULL,
    version text NOT NULL,
    name text NOT NULL,
    geom geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom) AND NOT ST_IsEmpty(geom)),
    source_metadata jsonb NOT NULL,
    UNIQUE (region_key, version)
);
CREATE TABLE geo.cell_supports (
    grid_id bigint NOT NULL,
    region_id bigint NOT NULL REFERENCES geo.regions,
    cell_id text NOT NULL,
    geom geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom) AND NOT ST_IsEmpty(geom)),
    target_area_m2 double precision NOT NULL CHECK (target_area_m2 > 0 AND target_area_m2 < 'Infinity'::float8),
    fraction_of_full_cell double precision NOT NULL CHECK (fraction_of_full_cell > 0 AND fraction_of_full_cell <= 1),
    area_method text NOT NULL,
    PRIMARY KEY (grid_id, region_id, cell_id),
    FOREIGN KEY (grid_id, cell_id) REFERENCES geo.cells
);
CREATE INDEX cell_supports_geom_gist ON geo.cell_supports USING gist (geom);

CREATE TABLE ingest.imports (
    import_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bundle_sha256 text NOT NULL UNIQUE CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    input_manifest jsonb NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now(),
    expected_observations bigint NOT NULL CHECK (expected_observations > 0)
);
CREATE TABLE ingest.runs (
    run_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    import_id bigint NOT NULL REFERENCES ingest.imports,
    region_id bigint NOT NULL REFERENCES geo.regions,
    source_kind text NOT NULL CHECK (source_kind IN ('terrain', 'temperature')),
    source_manifest_sha256 text NOT NULL,
    manifest jsonb NOT NULL,
    status text NOT NULL CHECK (status = 'validated_research_artifact'),
    UNIQUE (import_id, region_id, source_kind),
    UNIQUE (run_id, region_id)
);

CREATE TABLE data.time_windows (
    window_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    window_sha256 text NOT NULL UNIQUE,
    kind text NOT NULL CHECK (kind IN ('static', 'interval')),
    time_start timestamptz,
    time_end timestamptz,
    calendar text NOT NULL DEFAULT 'gregorian' CHECK (calendar = 'gregorian'),
    CHECK ((kind = 'static' AND time_start IS NULL AND time_end IS NULL)
        OR (kind = 'interval' AND time_start IS NOT NULL AND time_end IS NOT NULL AND time_end > time_start))
);
CREATE TABLE data.series (
    series_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    feature_version_id bigint NOT NULL REFERENCES catalog.feature_versions,
    grid_id bigint NOT NULL REFERENCES geo.grids,
    region_id bigint NOT NULL REFERENCES geo.regions,
    run_id bigint NOT NULL,
    window_kind text NOT NULL CHECK (window_kind IN ('static', 'interval')),
    status text NOT NULL DEFAULT 'staging' CHECK (status IN ('staging', 'sealed')),
    sealed_at timestamptz,
    support_metadata jsonb NOT NULL,
    CHECK ((status = 'sealed') = (sealed_at IS NOT NULL)),
    FOREIGN KEY (run_id, region_id) REFERENCES ingest.runs (run_id, region_id),
    UNIQUE (run_id, feature_version_id),
    UNIQUE (series_id, grid_id, region_id)
);
CREATE TABLE data.observations (
    series_id bigint NOT NULL,
    grid_id bigint NOT NULL,
    region_id bigint NOT NULL,
    cell_id text NOT NULL,
    window_id bigint NOT NULL REFERENCES data.time_windows,
    value_numeric double precision,
    missing_reason text,
    valid_fraction double precision NOT NULL CHECK (valid_fraction BETWEEN 0 AND 1),
    PRIMARY KEY (series_id, cell_id, window_id),
    FOREIGN KEY (series_id, grid_id, region_id) REFERENCES data.series (series_id, grid_id, region_id),
    FOREIGN KEY (grid_id, region_id, cell_id) REFERENCES geo.cell_supports,
    CHECK (value_numeric IS NULL OR (value_numeric > '-Infinity'::float8 AND value_numeric < 'Infinity'::float8)),
    CHECK ((value_numeric IS NULL AND missing_reason IS NOT NULL AND valid_fraction = 0)
        OR (value_numeric IS NOT NULL AND missing_reason IS NULL AND valid_fraction > 0))
);
CREATE INDEX observations_window_cell ON data.observations (window_id, cell_id);

CREATE FUNCTION data.guard_observation() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s data.series%ROWTYPE;
BEGIN
    IF TG_OP <> 'INSERT' THEN
        SELECT * INTO STRICT s FROM data.series WHERE series_id = OLD.series_id FOR SHARE;
        IF s.status = 'sealed' THEN RAISE EXCEPTION 'Sealed series observations are immutable'; END IF;
    END IF;
    IF TG_OP <> 'DELETE' THEN
        SELECT * INTO STRICT s FROM data.series WHERE series_id = NEW.series_id FOR SHARE;
        IF s.status = 'sealed' THEN RAISE EXCEPTION 'Sealed series observations are immutable'; END IF;
        IF (SELECT kind FROM data.time_windows WHERE window_id = NEW.window_id) <> s.window_kind THEN
            RAISE EXCEPTION 'Observation window kind differs from series';
        END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER guard_observation BEFORE INSERT OR UPDATE OR DELETE ON data.observations
    FOR EACH ROW EXECUTE FUNCTION data.guard_observation();

CREATE FUNCTION data.guard_sealed_series() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status = 'sealed' THEN RAISE EXCEPTION 'Sealed series are immutable'; END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER guard_sealed_series BEFORE UPDATE OR DELETE ON data.series
    FOR EACH ROW EXECUTE FUNCTION data.guard_sealed_series();

CREATE FUNCTION ingest.reject_pilot_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Versioned pilot metadata is append-only; create a new identity/version';
END $$;
DO $$ DECLARE relation_name text;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY['catalog.features','catalog.feature_versions',
        'geo.grids','geo.cells','geo.regions','geo.cell_supports','ingest.imports',
        'ingest.runs','data.time_windows'] LOOP
        EXECUTE format('CREATE TRIGGER reject_mutation BEFORE UPDATE OR DELETE ON %s FOR EACH ROW EXECUTE FUNCTION ingest.reject_pilot_mutation()', relation_name);
    END LOOP;
END $$;

-- These views intentionally expose research imports, not published releases.
CREATE VIEW research.pilot_observations AS
SELECT i.import_id, r.run_id, s.series_id, c.feature_key AS output_id,
    c.variable_key AS variable_id, f.version AS definition_version,
    f.canonical_unit AS unit, f.statistic, g.body AS body_id, g.resolution AS h3_resolution,
    o.grid_id, o.region_id, region.region_key AS region_id_external, o.cell_id,
    w.kind AS time_kind, w.time_start, w.time_end AS time_end_exclusive,
    o.value_numeric, o.missing_reason, o.valid_fraction AS valid_area_fraction,
    s.support_metadata, r.source_kind, r.source_manifest_sha256,
    i.bundle_sha256, 'research_pilot'::text AS availability
FROM data.observations o
JOIN data.series s USING (series_id, grid_id, region_id)
JOIN catalog.feature_versions f USING (feature_version_id)
JOIN catalog.features c USING (feature_key)
JOIN geo.grids g USING (grid_id)
JOIN geo.regions region USING (region_id)
JOIN data.time_windows w USING (window_id)
JOIN ingest.runs r ON r.run_id = s.run_id
JOIN ingest.imports i USING (import_id)
WHERE s.status = 'sealed';
CREATE VIEW research.pilot_features AS
SELECT f.feature_key AS output_id, f.variable_key AS variable_id,
    v.canonical_unit AS unit, v.statistic, v.version AS definition_version,
    v.definition, s.series_id, s.grid_id, s.region_id, r.region_key, s.run_id
FROM data.series s
JOIN catalog.feature_versions v USING (feature_version_id)
JOIN catalog.features f USING (feature_key)
JOIN geo.regions r USING (region_id)
WHERE s.status = 'sealed';
CREATE VIEW research.variables AS SELECT * FROM catalog.variables;
CREATE VIEW research.cells AS SELECT * FROM geo.cells;
CREATE VIEW research.cell_supports AS SELECT * FROM geo.cell_supports;

DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'earth_index_reader') THEN
        CREATE ROLE earth_index_reader NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'earth_index_api') THEN
        CREATE ROLE earth_index_api LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION IN ROLE earth_index_reader;
    END IF;
END $$;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA research TO earth_index_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA research TO earth_index_reader;
ALTER ROLE earth_index_api SET default_transaction_read_only = on;
ALTER ROLE earth_index_api SET statement_timeout = '10s';
ALTER ROLE earth_index_api SET search_path = pg_catalog, research, public;
