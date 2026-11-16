-- Keep immutable 1 km bundles while admitting explicitly declared 4 km grids.
DO $$
DECLARE c record;
BEGIN
    FOR c IN SELECT conname FROM pg_constraint
        WHERE conrelid='data.raster_tiles'::regclass AND contype='c'
          AND (pg_get_constraintdef(oid) ILIKE '%st_width%' OR pg_get_constraintdef(oid) ILIKE '%st_scalex%')
    LOOP EXECUTE format('ALTER TABLE data.raster_tiles DROP CONSTRAINT %I',c.conname); END LOOP;
    FOR c IN SELECT conname FROM pg_constraint
        WHERE conrelid='catalog.raster_layers'::regclass AND contype='c'
          AND pg_get_constraintdef(oid) LIKE '%10000%'
    LOOP EXECUTE format('ALTER TABLE catalog.raster_layers DROP CONSTRAINT %I',c.conname); END LOOP;
END $$;
ALTER TABLE catalog.raster_layers ADD CONSTRAINT valid_cells_within_grid CHECK (
    valid_cells >= 0 AND valid_cells <=
    (grid_definition->'shape'->>0)::integer*(grid_definition->'shape'->>1)::integer
);
ALTER TABLE data.raster_tiles ADD CONSTRAINT north_up_positive_grid CHECK (
    ST_Width(rast)>0 AND ST_Height(rast)>0 AND ST_ScaleX(rast)>0 AND ST_ScaleY(rast)<0
);
CREATE OR REPLACE FUNCTION data.validate_raster_tile_grid() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE expected_srid integer; g jsonb;
BEGIN
    SELECT srid,grid_definition INTO STRICT expected_srid,g FROM catalog.raster_layers WHERE layer_id=NEW.layer_id;
    IF ST_SRID(NEW.rast) <> expected_srid OR NEW.tile_row<>0 OR NEW.tile_col<>0
       OR ST_Width(NEW.rast)<>(g->'shape'->>1)::integer
       OR ST_Height(NEW.rast)<>(g->'shape'->>0)::integer
       OR ST_ScaleX(NEW.rast)<>(g->'transform'->>0)::float8
       OR ST_ScaleY(NEW.rast)<>(g->'transform'->>4)::float8
       OR ST_UpperLeftX(NEW.rast)<>(g->'transform'->>2)::float8
       OR ST_UpperLeftY(NEW.rast)<>(g->'transform'->>5)::float8
       OR (g->'transform'->>1)::float8<>0 OR (g->'transform'->>3)::float8<>0
       OR jsonb_array_length(g->'shape')<>2 OR jsonb_array_length(g->'transform')<>6 THEN
        RAISE EXCEPTION 'Raster tile does not match its declared layer grid';
    END IF;
    RETURN NEW;
END $$;
