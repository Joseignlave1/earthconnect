-- Geographic footprints are explicit for grids whose native rectangle contains
-- padding outside the planet. Preserve the existing local pilot behavior.
CREATE TABLE catalog.raster_region_coverage (
    bundle_sha256 text NOT NULL REFERENCES catalog.raster_bundle_publication,
    site_key text NOT NULL,
    geom geometry(MultiPolygon,4326) NOT NULL,
    PRIMARY KEY (bundle_sha256,site_key),
    CHECK (NOT ST_IsEmpty(geom) AND ST_IsValid(geom)),
    CHECK (ST_CoveredBy(geom,ST_MakeEnvelope(-180,-90,180,90,4326)))
);
CREATE VIEW research.raster_region_coverage AS SELECT * FROM catalog.raster_region_coverage;
GRANT SELECT ON research.raster_region_coverage TO earth_index_reader;

-- Native MODIS sinusoidal rectangle intersected with the curved Earth domain.
-- Construct directly in longitude/latitude, avoiding singular/out-of-world
-- transformations at the poles and +/-180-degree boundary.
CREATE FUNCTION public.sinusoidal_footprint(xmin float8,ymin float8,xmax float8,ymax float8)
RETURNS geometry LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE radius constant float8 := 6371007.181;
        distance float8; limit_lat float8; low_lat float8; high_lat float8;
        latitude float8; longitude float8; factor float8; steps integer; i integer;
        points geometry[] := ARRAY[]::geometry[];
BEGIN
    IF xmin>=xmax OR ymin>=ymax THEN RETURN NULL; END IF;
    distance := CASE WHEN xmin<=0 AND xmax>=0 THEN 0 ELSE least(abs(xmin),abs(xmax)) END;
    IF distance>=pi()*radius THEN RETURN NULL; END IF;
    limit_lat := degrees(acos(distance/(pi()*radius)));
    low_lat := greatest(-90.,degrees(ymin/radius),-limit_lat);
    high_lat := least(90.,degrees(ymax/radius),limit_lat);
    IF low_lat>=high_lat THEN RETURN NULL; END IF;
    steps := greatest(2,ceil((high_lat-low_lat)/.05)::integer);
    FOR i IN 0..steps LOOP
        latitude := low_lat+(high_lat-low_lat)*i/steps;
        factor := radius*greatest(cos(radians(latitude)),1e-15);
        longitude := greatest(-180.,least(180.,degrees(xmin/factor)));
        points := array_append(points,ST_MakePoint(longitude,latitude));
    END LOOP;
    FOR i IN REVERSE steps..0 LOOP
        latitude := low_lat+(high_lat-low_lat)*i/steps;
        factor := radius*greatest(cos(radians(latitude)),1e-15);
        longitude := greatest(-180.,least(180.,degrees(xmax/factor)));
        points := array_append(points,ST_MakePoint(longitude,latitude));
    END LOOP;
    points := array_append(points,points[1]);
    RETURN ST_Multi(ST_SetSRID(ST_MakePolygon(ST_MakeLine(points)),4326));
END $$;
COMMENT ON FUNCTION public.sinusoidal_footprint(float8,float8,float8,float8) IS
    'WGS84 geographic extent of a MODIS-sinusoidal rectangle clipped to Earth; edges sampled at <=.05 degree latitude. Padding outside Earth has no footprint.';

CREATE OR REPLACE FUNCTION catalog.refresh_api_datasources() RETURNS void LANGUAGE plpgsql AS $$
DECLARE b record; f record; ns text; query text; joins text; columns_sql text;
        unnest_sql text; aliases text; n integer; description_text text; name_text text;
BEGIN
    FOR b IN SELECT bundle_sha256,count(DISTINCT site_key) sites,
                    bool_and(grid_definition->>'version'='modis-sinu-278-v1') global_grid
             FROM catalog.raster_layers GROUP BY bundle_sha256 LOOP
        ns := 'raster_' || left(b.bundle_sha256,32);
        EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I',ns);
        EXECUTE format('GRANT USAGE ON SCHEMA %I TO earth_index_reader',ns);
        EXECUTE format('CREATE OR REPLACE VIEW %I.layers AS SELECT * FROM research.raster_layers WHERE bundle_sha256=%L',ns,b.bundle_sha256);
        EXECUTE format('CREATE OR REPLACE VIEW %I.tiles AS SELECT * FROM research.raster_tiles WHERE bundle_sha256=%L',ns,b.bundle_sha256);
        columns_sql := ''; joins := ''; unnest_sql := ''; aliases := 'i'; n := 0;
        FOR f IN SELECT DISTINCT feature_key FROM catalog.raster_layers
                 WHERE bundle_sha256=b.bundle_sha256 ORDER BY feature_key LOOP
            columns_sql := columns_sql || format(
                ',CASE WHEN p.c%s=0 THEN NULL ELSE p.c%s*f%s.scale+f%s.offset_value END AS %I',
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
             ((p.i-1) %% s.width)+1 AS x,((p.i-1)/s.width)+1 AS y %s,
             CASE WHEN s.grid_version=''modis-sinu-278-v1'' THEN
               CASE WHEN abs(s.a*(((p.i-1) %% s.width)+.5)+s.c) >
                   pi()*6371007.181*cos((s.e*(((p.i-1)/s.width)+.5)+s.f)/6371007.181)
                 THEN NULL ELSE public.sinusoidal_footprint(
                   s.a*((p.i-1) %% s.width)+s.c,s.e*(((p.i-1)/s.width)+1)+s.f,
                   s.a*(((p.i-1) %% s.width)+1)+s.c,s.e*((p.i-1)/s.width)+s.f) END
             ELSE ST_Transform(ST_MakeEnvelope(
               s.a*((p.i-1) %% s.width)+s.c,s.e*(((p.i-1)/s.width)+1)+s.f,
               s.a*(((p.i-1) %% s.width)+1)+s.c,s.e*((p.i-1)/s.width)+s.f,s.srid),4326)
             END AS geom
             FROM (SELECT DISTINCT ON(site_key) site_key,srid,
               (grid_definition->''shape''->>1)::integer width,
               (grid_definition->''shape''->>0)::integer height,
               (grid_definition->''transform''->>0)::float8 a,
               (grid_definition->''transform''->>2)::float8 c,
               (grid_definition->''transform''->>4)::float8 e,
               (grid_definition->''transform''->>5)::float8 f,
               grid_definition->>''version'' grid_version
               FROM %I.layers ORDER BY site_key,layer_id) s %s
             CROSS JOIN LATERAL ROWS FROM (generate_series(1,s.width*s.height)%s) AS p(%s)',
             ns,b.bundle_sha256,columns_sql,ns,joins,unnest_sql,aliases);
        EXECUTE query;
        EXECUTE format('COMMENT ON VIEW %I.cells IS %L',ns,
            'Decoded physical values, one row per stored raster cell. x/y are 1-based; geom is WGS84, NULL for padding outside Earth. NULL measurements mean missing evidence, not zero.');
        FOR f IN SELECT DISTINCT feature_key,unit FROM catalog.raster_layers WHERE bundle_sha256=b.bundle_sha256 LOOP
            EXECUTE format('COMMENT ON COLUMN %I.cells.%I IS %L',ns,f.feature_key,
                replace(f.feature_key,'_',' ') || '; unit: ' || f.unit || '. See layer provenance for source scale, aggregation, warnings and categories.');
        END LOOP;
        EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO earth_index_reader',ns);
        name_text := format('Earth pilot — %s patches',b.sites);
        description_text := format('Research raster snapshot with %s loaded projected patches. Extent and spacing are declared per grid; variable validity differs within footprints.',b.sites);
        IF b.global_grid THEN
            name_text := 'Earth — global 4 km, 25 bands';
            description_text := 'Global equal-area sinusoidal grid, 3999.822 m cells, 25 quantized bands. Coverage is the grid domain; source gaps remain NULL. POWER climate baseline 2001–2020; annual precipitation precision 1 mm.';
        END IF;
        IF NOT EXISTS (SELECT FROM catalog.raster_bundle_publication bp WHERE bp.bundle_sha256=b.bundle_sha256 AND NOT bp.published) THEN
            INSERT INTO catalog.api_datasources VALUES
                ('earth-'||left(b.bundle_sha256,12),ns,b.bundle_sha256,name_text,description_text)
            ON CONFLICT (bundle_sha256) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description;
        ELSE
            DELETE FROM catalog.api_datasources WHERE bundle_sha256=b.bundle_sha256;
        END IF;
    END LOOP;
END $$;
