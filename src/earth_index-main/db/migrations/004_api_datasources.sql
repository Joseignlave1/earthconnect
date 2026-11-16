-- API namespaces contain views only. Existing scientific data remains untouched.
CREATE TABLE catalog.api_datasources (
    datasource_id text PRIMARY KEY,
    schema_name name NOT NULL UNIQUE,
    bundle_sha256 text NOT NULL UNIQUE,
    name text NOT NULL,
    description text NOT NULL
);
CREATE VIEW research.api_datasources AS SELECT * FROM catalog.api_datasources;
GRANT SELECT ON research.api_datasources TO earth_index_reader;

CREATE FUNCTION public.pack_u16(values_to_pack integer[]) RETURNS bytea
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE result bytea;
BEGIN
    IF EXISTS (SELECT FROM unnest(values_to_pack) v WHERE v < 0 OR v > 65535) THEN
        RAISE EXCEPTION 'pack_u16 values must be between 0 and 65535' USING ERRCODE = '22003';
    END IF;
    SELECT coalesce(string_agg(
        set_byte(set_byte('\x0000'::bytea, 0, coalesce(v,65535) & 255),
                 1, (coalesce(v,65535) >> 8) & 255), ''::bytea ORDER BY ord), ''::bytea)
    INTO result FROM unnest(values_to_pack) WITH ORDINALITY AS t(v,ord);
    RETURN result;
END $$;
COMMENT ON FUNCTION public.pack_u16(integer[]) IS
    'Pack integers as little-endian uint16; NULL becomes 65535. Array order is preserved.';

-- Run after imports as owner. One immutable bundle per SQL namespace; no temp views
-- or hidden query parameters are needed to select a datasource.
CREATE FUNCTION catalog.refresh_api_datasources() RETURNS void LANGUAGE plpgsql AS $$
DECLARE b record; f record; ns text; query text; joins text; columns_sql text;
        unnest_sql text; aliases text; n integer; description_text text;
BEGIN
    FOR b IN SELECT bundle_sha256, count(DISTINCT site_key) sites
             FROM catalog.raster_layers GROUP BY bundle_sha256 LOOP
        ns := 'raster_' || left(b.bundle_sha256,32);
        EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I', ns);
        EXECUTE format('GRANT USAGE ON SCHEMA %I TO earth_index_reader', ns);
        EXECUTE format('CREATE OR REPLACE VIEW %I.layers AS SELECT * FROM research.raster_layers WHERE bundle_sha256=%L',ns,b.bundle_sha256);
        EXECUTE format('CREATE OR REPLACE VIEW %I.tiles AS SELECT * FROM research.raster_tiles WHERE bundle_sha256=%L',ns,b.bundle_sha256);
        columns_sql := ''; joins := ''; unnest_sql := ''; aliases := 'i'; n := 0;
        FOR f IN SELECT DISTINCT feature_key FROM catalog.raster_layers
                 WHERE bundle_sha256=b.bundle_sha256 ORDER BY feature_key LOOP
            columns_sql := columns_sql || format(
                ', CASE WHEN p.c%s=0 THEN NULL ELSE p.c%s*f%s.scale+f%s.offset_value END AS %I',
                n,n,n,n,f.feature_key);
            joins := joins || format(
                ' LEFT JOIN LATERAL (SELECT ST_DumpValues(t.rast,1,false) codes,t.scale,t.offset_value
                   FROM %I.tiles t WHERE t.site_key=s.site_key AND t.feature_key=%L) f%s ON true',
                ns,f.feature_key,n);
            unnest_sql := unnest_sql || format(',unnest(f%s.codes)',n);
            aliases := aliases || format(',c%s',n); n := n+1;
        END LOOP;
        query := format(
            'CREATE OR REPLACE VIEW %I.cells AS SELECT %L::text bundle_sha256,s.site_key,
             ((p.i-1) %% s.width)+1 AS x, ((p.i-1)/s.width)+1 AS y %s,
             ST_Transform(ST_MakeEnvelope(
               s.a*((p.i-1) %% s.width)+s.c,
               s.e*(((p.i-1)/s.width)+1)+s.f,
               s.a*(((p.i-1) %% s.width)+1)+s.c,
               s.e*((p.i-1)/s.width)+s.f,s.srid),4326) AS geom
             FROM (SELECT DISTINCT ON (site_key) site_key,srid,
               (grid_definition->''shape''->>1)::integer width,
               (grid_definition->''shape''->>0)::integer height,
               (grid_definition->''transform''->>0)::float8 a,
               (grid_definition->''transform''->>2)::float8 c,
               (grid_definition->''transform''->>4)::float8 e,
               (grid_definition->''transform''->>5)::float8 f
               FROM %I.layers ORDER BY site_key,layer_id) s %s
             CROSS JOIN LATERAL ROWS FROM (generate_series(1,s.width*s.height)%s) AS p(%s)',
             ns,b.bundle_sha256,columns_sql,ns,joins,unnest_sql,aliases);
        EXECUTE query;
        EXECUTE format('COMMENT ON VIEW %I.cells IS %L',ns,
            'Decoded physical values, one row per loaded cell. x/y are 1-based; geom is the WGS84 cell footprint. NULL means missing evidence, not zero.');
        FOR f IN SELECT DISTINCT feature_key,unit FROM catalog.raster_layers WHERE bundle_sha256=b.bundle_sha256 LOOP
            EXECUTE format('COMMENT ON COLUMN %I.cells.%I IS %L',ns,f.feature_key,
                           replace(f.feature_key,'_',' ') || '; unit: ' || f.unit || '. See layer provenance for source scale, aggregation, warnings and categories.');
        END LOOP;
        EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO earth_index_reader',ns);
        description_text := format('Research raster snapshot with %s loaded 100 x 100 km patches. Not global coverage; variable validity differs within footprints.',b.sites);
        INSERT INTO catalog.api_datasources VALUES
            ('earth-'||left(b.bundle_sha256,12),ns,b.bundle_sha256,
             format('Earth pilot — %s patches',b.sites),description_text)
        ON CONFLICT (bundle_sha256) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description;
    END LOOP;
END $$;
REVOKE ALL ON FUNCTION catalog.refresh_api_datasources() FROM PUBLIC;
SELECT catalog.refresh_api_datasources();
