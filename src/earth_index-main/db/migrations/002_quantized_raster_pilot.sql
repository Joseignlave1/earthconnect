-- Independent research raster pilot. Preserve the original H3 observations.
CREATE EXTENSION IF NOT EXISTS postgis_raster;

CREATE TABLE catalog.raster_layers (
    layer_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bundle_sha256 text NOT NULL CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    site_key text NOT NULL,
    feature_key text NOT NULL,
    variable_key text NOT NULL REFERENCES catalog.variables,
    unit text NOT NULL,
    scale double precision NOT NULL CHECK (scale > 0 AND scale < 'Infinity'::float8),
    offset_value double precision NOT NULL CHECK (offset_value > '-Infinity'::float8 AND offset_value < 'Infinity'::float8),
    srid integer NOT NULL REFERENCES spatial_ref_sys,
    grid_definition jsonb NOT NULL CHECK (jsonb_typeof(grid_definition) = 'object'),
    provenance jsonb NOT NULL CHECK (jsonb_typeof(provenance) = 'object'),
    valid_cells integer NOT NULL CHECK (valid_cells BETWEEN 0 AND 10000),
    max_rounding_error double precision NOT NULL CHECK (max_rounding_error >= 0 AND max_rounding_error < 'Infinity'::float8),
    file_sha256 text NOT NULL CHECK (file_sha256 ~ '^[0-9a-f]{64}$'),
    status text NOT NULL DEFAULT 'research_pilot' CHECK (status = 'research_pilot'),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (bundle_sha256, site_key, feature_key)
);

CREATE TABLE data.raster_tiles (
    layer_id bigint NOT NULL REFERENCES catalog.raster_layers,
    tile_row integer NOT NULL,
    tile_col integer NOT NULL,
    rast raster NOT NULL,
    PRIMARY KEY (layer_id, tile_row, tile_col),
    CHECK (ST_NumBands(rast) = 1),
    CHECK (ST_BandPixelType(rast, 1) = '16BUI'),
    CHECK (ST_BandNoDataValue(rast, 1) = 0),
    CHECK (ST_Width(rast) = 100 AND ST_Height(rast) = 100),
    CHECK (ST_ScaleX(rast) = 1000 AND ST_ScaleY(rast) = -1000),
    CHECK (ST_SkewX(rast) = 0 AND ST_SkewY(rast) = 0)
);
CREATE INDEX raster_tiles_extent_gist ON data.raster_tiles USING gist (ST_ConvexHull(rast));

CREATE FUNCTION data.validate_raster_tile_grid() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE expected_srid integer;
BEGIN
    SELECT srid INTO STRICT expected_srid FROM catalog.raster_layers WHERE layer_id = NEW.layer_id;
    IF ST_SRID(NEW.rast) <> expected_srid OR NEW.tile_row <> 0 OR NEW.tile_col <> 0
       OR ST_UpperLeftX(NEW.rast) <> -50000 OR ST_UpperLeftY(NEW.rast) <> 50000 THEN
        RAISE EXCEPTION 'Raster tile does not match its pilot layer grid';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER raster_tile_grid BEFORE INSERT OR UPDATE ON data.raster_tiles
FOR EACH ROW EXECUTE FUNCTION data.validate_raster_tile_grid();

CREATE VIEW research.raster_layers AS SELECT * FROM catalog.raster_layers;
CREATE VIEW research.raster_tiles AS
SELECT l.bundle_sha256, l.site_key, l.feature_key, l.scale, l.offset_value, l.unit,
       t.layer_id, t.tile_row, t.tile_col, t.rast
FROM data.raster_tiles t JOIN catalog.raster_layers l USING (layer_id);

-- Convenience interface for bounded queries, not a materialized global cell table.
CREATE VIEW research.raster_values AS
SELECT t.bundle_sha256, t.site_key, t.feature_key, t.layer_id, p.x, p.y,
       p.val AS code, p.val * t.scale + t.offset_value AS value, t.unit
FROM research.raster_tiles t
CROSS JOIN LATERAL ST_PixelAsPoints(t.rast, 1, true) AS p;

CREATE VIEW research.priority_pilot_cells AS
SELECT bundle_sha256, site_key, x, y,
    max(value) FILTER (WHERE feature_key = 'elevation_median_m') AS elevation_median_m,
    max(value) FILTER (WHERE feature_key = 'slope_mean_deg') AS slope_mean_deg,
    max(value) FILTER (WHERE feature_key = 'terrain_valid_fraction') AS terrain_valid_fraction,
    max(value) FILTER (WHERE feature_key = 'relief_5km_m') AS relief_5km_m,
    max(value) FILTER (WHERE feature_key = 'tpi_5km_m') AS tpi_5km_m,
    max(value) FILTER (WHERE feature_key = 'relief_21km_m') AS relief_21km_m,
    max(value) FILTER (WHERE feature_key = 'tpi_21km_m') AS tpi_21km_m,
    max(value) FILTER (WHERE feature_key = 'coldest_month_air_c') AS coldest_month_air_c,
    max(value) FILTER (WHERE feature_key = 'warmest_month_air_c') AS warmest_month_air_c,
    max(value) FILTER (WHERE feature_key = 'annual_precip_mm') AS annual_precip_mm,
    max(value) FILTER (WHERE feature_key = 'bare_fraction') AS bare_fraction,
    max(value) FILTER (WHERE feature_key = 'permanent_snow_ice_fraction') AS permanent_snow_ice_fraction,
    max(value) FILTER (WHERE feature_key = 'landcover_valid_fraction') AS landcover_valid_fraction,
    max(value) FILTER (WHERE feature_key = 'permafrost_fraction') AS permafrost_fraction,
    max(value) FILTER (WHERE feature_key = 'permafrost_valid_fraction') AS permafrost_valid_fraction
FROM research.raster_values
GROUP BY bundle_sha256, site_key, x, y;

GRANT SELECT ON research.raster_layers, research.raster_tiles, research.raster_values,
    research.priority_pilot_cells TO earth_index_reader;
