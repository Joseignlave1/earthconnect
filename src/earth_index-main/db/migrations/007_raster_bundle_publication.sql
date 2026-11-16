-- A staged experiment keeps prepared SQL views without changing demo discovery.
-- Existing bundles remain published unless explicitly marked otherwise.
CREATE TABLE catalog.raster_bundle_publication (
    bundle_sha256 text PRIMARY KEY CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    published boolean NOT NULL
);
DO $$
DECLARE definition text;
BEGIN
    definition := pg_get_functiondef('catalog.refresh_api_datasources()'::regprocedure);
    definition := replace(definition,
        'Research raster snapshot with %s loaded 100 x 100 km patches. Not global coverage; variable validity differs within footprints.',
        'Research raster snapshot with %s loaded projected patches. Extent and spacing are declared per grid; variable validity differs within footprints.');
    definition := replace(definition,
        'INSERT INTO catalog.api_datasources VALUES',
        'IF NOT EXISTS (SELECT FROM catalog.raster_bundle_publication bp WHERE bp.bundle_sha256=b.bundle_sha256 AND NOT bp.published) THEN
         INSERT INTO catalog.api_datasources VALUES');
    definition := replace(definition,
        'ON CONFLICT (bundle_sha256) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description;',
        'ON CONFLICT (bundle_sha256) DO UPDATE SET name=EXCLUDED.name,description=EXCLUDED.description;
         ELSE DELETE FROM catalog.api_datasources WHERE bundle_sha256=b.bundle_sha256;
         END IF;');
    IF definition NOT LIKE '%IF NOT EXISTS (SELECT FROM catalog.raster_bundle_publication%' THEN
        RAISE EXCEPTION 'Unexpected datasource refresh definition; review publication migration';
    END IF;
    EXECUTE definition;
END $$;
