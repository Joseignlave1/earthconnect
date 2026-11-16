# Planetary variable catalog

Generated from `catalog.json`; see README for verification results and limitations.

`verified_transport` describes API access, not scientific validation or frontend availability.

## Moon

| Variable ID | Unit | Priority | Readiness |
| --- | --- | --- | --- |
| `moon.elevation` | m | P0 | verified_transport |
| `moon.elevation_mean` | m | P0 | requires_local_derivation |
| `moon.elevation_min` | m | P0 | requires_local_derivation |
| `moon.elevation_max` | m | P0 | requires_local_derivation |
| `moon.slope_mean` | degree | P0 | requires_local_derivation |
| `moon.slope_p90` | degree | P0 | requires_local_derivation |
| `moon.slope_max` | degree | P0 | requires_local_derivation |
| `moon.aspect_eastness` | 1 | P0 | requires_local_derivation |
| `moon.aspect_northness` | 1 | P0 | requires_local_derivation |
| `moon.relief` | m | P0 | requires_local_derivation |
| `moon.tpi` | m | P0 | requires_local_derivation |
| `moon.roughness` | method_dependent | P0 | requires_local_derivation |
| `moon.flat_fraction` | fraction | P0 | requires_local_derivation |
| `moon.surface_noon` | K | P0 | verified_transport |
| `moon.surface_midnight` | K | P0 | verified_transport |
| `moon.surface_min` | K | P0 | verified_transport |
| `moon.surface_max` | K | P0 | verified_transport |
| `moon.surface_temperature_range` | K_difference | P0 | requires_local_derivation |
| `moon.solar_illumination` | fraction | P1 | verified_transport |
| `moon.earth_visibility` | fraction | P1 | verified_transport |
| `moon.permanent_shadow` | boolean_or_area_fraction | P1 | verified_transport |
| `moon.surface_ice_detection` | distance_m_or_detection | P1 | verified_transport |
| `moon.geologic_unit` | body_specific_category | P1 | verified_transport |
| `moon.olivine` | raw_fraction | P2 | verified_transport_requires_semantic_review |
| `moon.clinopyroxene` | raw_fraction | P2 | verified_transport_requires_semantic_review |
| `moon.orthopyroxene` | raw_fraction | P2 | verified_transport_requires_semantic_review |
| `moon.plagioclase` | raw_fraction | P2 | verified_transport_requires_semantic_review |
| `moon.feo` | wt_percent_FeO | P2 | verified_transport |
| `moon.smfe` | unresolved_raw_scale | P2 | verified_transport_requires_semantic_review |
| `moon.optical_maturity` | 1 | P2 | verified_transport |
| `moon.plagioclase_grain_size` | micrometre | P2 | verified_transport |
| `moon.mineral_fit_quality` | raw_criterion | P2 | verified_transport_requires_semantic_review |
| `moon.thorium` | ppm | P2 | verified_transport |
| `moon.titanium` | wt_percent_Ti | P2 | verified_transport |
| `moon.potassium` | unresolved_raw_scale | P2 | verified_transport_requires_semantic_review |
| `moon.uranium` | unresolved_raw_scale | P2 | verified_transport_requires_semantic_review |
| `moon.thermal_neutron` | unresolved_raw_scale | P2 | verified_transport_requires_semantic_review |
| `moon.water_equivalent_hydrogen` | raw_WEH_scale_unresolved | P1 | verified_transport_requires_semantic_review |
| `moon.ice_stability_depth` | raw_depth_scale_unresolved | P1 | verified_transport_requires_semantic_review |
| `moon.radar_cpr` | ratio | P2 | verified_transport |
| `moon.radar_backscatter` | raw_linear_scale_unresolved | P2 | verified_transport_requires_semantic_review |
| `moon.radar_backscatter_malapert` | dB | P2 | verified_transport |
| `moon.christiansen_feature` | micrometre | P2 | verified_transport |
| `moon.regional_rock_count` | count_per_100m_by_100m_cell | P1 | verified_transport |
| `moon.regional_rock_abundance_unresolved` | unresolved_integer_encoding | P1 | verified_transport_requires_semantic_review |
| `moon.regional_roughness` | method_and_units_unresolved | P1 | verified_transport_requires_semantic_review |
| `moon.regional_published_slope` | degree | P1 | verified_transport |
| `moon.albedo` | product_specific | P1 | see_external_evidence_or_not_verified |
| `moon.thermal_inertia` | product_specific | P1 | see_external_evidence_or_not_verified |
| `moon.rock_area_fraction` | product_specific | P1 | see_external_evidence_or_not_verified |
| `moon.crustal_density` | kg_m3 | P3 | verified_transport_requires_semantic_review |
| `moon.crustal_thickness` | km | P3 | verified_transport_requires_semantic_review |
| `moon.gravity_anomaly` | mGal | P3 | verified_transport_requires_semantic_review |
| `moon.bouguer_anomaly` | mGal | P3 | verified_transport_requires_semantic_review |
| `moon.geoid` | m | P3 | verified_transport_requires_semantic_review |
| `moon.magnetic_field` | raw_scale_unresolved | P3 | verified_transport_requires_semantic_review |

## Mars

| Variable ID | Unit | Priority | Readiness |
| --- | --- | --- | --- |
| `mars.elevation` | m | P0 | verified_transport |
| `mars.elevation_mean` | m | P0 | requires_local_derivation |
| `mars.elevation_min` | m | P0 | requires_local_derivation |
| `mars.elevation_max` | m | P0 | requires_local_derivation |
| `mars.slope_mean` | degree | P0 | requires_local_derivation |
| `mars.slope_p90` | degree | P0 | requires_local_derivation |
| `mars.slope_max` | degree | P0 | requires_local_derivation |
| `mars.aspect_eastness` | 1 | P0 | requires_local_derivation |
| `mars.aspect_northness` | 1 | P0 | requires_local_derivation |
| `mars.relief` | m | P0 | requires_local_derivation |
| `mars.tpi` | m | P0 | requires_local_derivation |
| `mars.roughness` | method_dependent | P0 | requires_local_derivation |
| `mars.flat_fraction` | fraction | P0 | requires_local_derivation |
| `mars.modeled_surface_temperature_mean` | degree_C | P0 | verified_transport |
| `mars.modeled_surface_temperature_min` | degree_C | P0 | verified_transport |
| `mars.modeled_surface_temperature_max` | degree_C | P0 | verified_transport |
| `mars.surface_temperature_range` | K_difference | P0 | requires_local_derivation |
| `mars.albedo` | 1 | P0 | verified_transport |
| `mars.thermal_inertia_tes` | J_m-2_K-1_s-0.5 | P0 | verified_transport |
| `mars.thermal_inertia_tes_putzig_night` | J_m-2_K-1_s-0.5 | P0 | verified_transport |
| `mars.thermal_inertia_themis` | J_m-2_K-1_s-0.5 | P0 | verified_transport |
| `mars.surface_dust_index` | 1 | P0 | verified_transport |
| `mars.water_equivalent_hydrogen` | wt_percent_WEH | P1 | verified_transport |
| `mars.tes_olivine` | areal_fraction | P2 | verified_transport |
| `mars.tes_feldspar` | areal_fraction | P2 | verified_transport |
| `mars.tes_clinopyroxene` | areal_fraction | P2 | verified_transport |
| `mars.tes_plagioclase` | areal_fraction | P2 | verified_transport |
| `mars.tes_carbonate` | areal_fraction | P2 | verified_transport |
| `mars.tes_hematite` | areal_fraction | P2 | verified_transport |
| `mars.tes_amphibole` | areal_fraction | P2 | verified_transport |
| `mars.tes_quartz` | areal_fraction | P2 | verified_transport |
| `mars.tes_sulfate` | areal_fraction | P2 | verified_transport |
| `mars.tes_dust` | areal_fraction | P2 | verified_transport |
| `mars.tes_mineral_fit_error` | raw_fit_error | P2 | verified_transport |
| `mars.geologic_unit` | body_specific_category | P1 | verified_transport |
| `mars.hydrous_mineral_detection` | presence_or_distance_m | P1 | verified_transport |
| `mars.chloride_detection` | presence_or_distance_m | P1 | verified_transport |
| `mars.subsurface_ice_evidence` | product_specific_score | P1 | see_external_evidence_or_not_verified |
| `mars.dune_field` | presence_or_distance_m | P1 | verified_transport |
| `mars.valley_network` | presence_or_distance_m | P1 | verified_transport |
| `mars.fans_glaciers_gullies` | category_or_distance_m | P1 | verified_transport |
| `mars.rock_abundance` | product_specific_fraction | P1 | see_external_evidence_or_not_verified |
| `mars.air_temperature_pressure_wind` | variable_specific | P1 | see_external_evidence_or_not_verified |
