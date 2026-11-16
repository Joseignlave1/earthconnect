-- Append the coarse categorical geology context to the bounded convenience view.
CREATE OR REPLACE VIEW research.priority_pilot_cells AS
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
    max(value) FILTER (WHERE feature_key = 'permafrost_valid_fraction') AS permafrost_valid_fraction,
    max(value) FILTER (WHERE feature_key = 'dominant_lithology_code') AS dominant_lithology_code
FROM research.raster_values
GROUP BY bundle_sha256, site_key, x, y;
