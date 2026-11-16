-- Independent clear-sky surface-temperature summaries on the existing pilot grid.
INSERT INTO catalog.variables(variable_key,definition_version,label,value_kind,definition)
VALUES ('thermal.surface_temperature',1,'MODIS land-surface temperature','number',
        '{"id":"thermal.surface_temperature","label":"MODIS clear-sky surface temperature","unit":"degree_Celsius","availability":"research_thermal_pilot","warning":"Observed composite summaries, not true daily extremes or air temperature; inspect temporal support and provenance."}'::jsonb)
ON CONFLICT (variable_key) DO NOTHING;
