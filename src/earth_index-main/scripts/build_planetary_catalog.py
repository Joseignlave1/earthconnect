#!/usr/bin/env python3
"""Build a curated variable catalog from the saved planetary transport audits."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/planetary-sources"
trek_audit = json.loads((OUT / "trek-audit.json").read_text())
mars_audit = json.loads((OUT / "mars-cutout-audit.json").read_text())
mars_metadata = json.loads((OUT / "mars-source-metadata.json").read_text())
variables = []


def add(body, key, label, unit, priority, method, sources=(), notes="", unit_status="documented", readiness="candidate", earth_comparison="context_only"):
    evidence = []
    for name in sources:
        services = [s for s in trek_audit["services"] if s["body"] == body and s["name"] == name]
        for service in services:
            evidence.append({"provider": "NASA Trek", "service": name, "url": service["url"],
                             "status": service["status"], "audit": "trek-audit.json",
                             "pixel_spacing": service.get("reported_pixel_spacing"),
                             "extent": service.get("metadata", {}).get("data", {}).get("extent"),
                             "sr": service.get("metadata", {}).get("data", {}).get("spatialReference")})
        for layer in mars_audit["layers"] if body == "mars" else []:
            if layer["layer"] == name:
                evidence.append({"provider": "ASU/JMARS", "service": name, "url": layer["url"],
                                 "status": layer["status"], "audit": "mars-cutout-audit.json",
                                 "metadata": mars_metadata["layers"].get(name),
                                 "metadata_audit": "mars-source-metadata.json"})
    variables.append({"id": f"{body}.{key}", "body": body, "label": label, "unit": unit,
                      "unit_status": unit_status, "priority": priority, "method": method,
                      "readiness": readiness, "earth_comparison": earth_comparison,
                      "sources": evidence, "notes": notes})


MOON_DEM = "LRO_LOLA_DEM_Global_256ppd_v06"
MARS_DEM = "Mars_MOLA_blend200ppx_HRSC_DEM_clon0dd_200mpp_lzw"
for body, dem in [("moon", MOON_DEM), ("mars", MARS_DEM)]:
    add(body, "elevation", "Reference-surface elevation", "m", "P0", "numeric_cutout", [dem],
        "Moon heights reference a 1737.4 km sphere; Mars heights reference the MOLA areoid. Absolute elevations do not share a terrestrial sea-level datum.",
        readiness="verified_transport", earth_comparison="different_vertical_datum")
    for key, label, unit in [
        ("elevation_mean", "Window mean elevation", "m"),
        ("elevation_min", "Window minimum elevation", "m"),
        ("elevation_max", "Window maximum elevation", "m"),
        ("slope_mean", "Window mean slope", "degree"),
        ("slope_p90", "Window 90th percentile slope", "degree"),
        ("slope_max", "Window maximum slope", "degree"),
        ("aspect_eastness", "Mean east-facing component", "1"),
        ("aspect_northness", "Mean north-facing component", "1"),
        ("relief", "Window elevation range", "m"),
        ("tpi", "Topographic position relative to neighborhood", "m"),
        ("roughness", "Defined-scale terrain roughness", "method_dependent"),
        ("flat_fraction", "Fraction below a user-selected slope threshold", "fraction"),
    ]:
        add(body, key, label, unit, "P0", "derive_from_dem_cutout", [dem],
            "Derivable, not a separately verified API field. Match footprint, derivative baseline, area weighting and NODATA support to Earth; use planet-specific distances. Roughness requires an explicit definition.",
            readiness="requires_local_derivation", earth_comparison="comparable_after_support_alignment")

for key, suffix, label in [
    ("surface_noon", "hour12", "Local-noon bolometric surface temperature"),
    ("surface_midnight", "hour00", "Local-midnight bolometric surface temperature"),
    ("surface_min", "min", "Minimum bolometric surface temperature"),
    ("surface_max", "max", "Maximum bolometric surface temperature"),
]:
    add("moon", key, label, "K", "P0", "numeric_point_or_cutout", [f"diviner_tbol_{suffix}"],
        "Composite Diviner surface brightness temperature, not air temperature or current weather. Source compilation 2009-07-05 to 2015-04-01. Service spacing 0.5 degree (~15.2 km at equator); some prose inconsistently says 0.5 ppd. Absolute-temperature conversion to Celsius is K-273.15; mask source missing values before conversion.",
        readiness="verified_transport", earth_comparison="needs_earth_surface_temperature")
add("moon", "surface_temperature_range", "Maximum-minus-minimum surface temperature", "K_difference", "P0", "derive_from_temperature_extrema", ["diviner_tbol_max", "diviner_tbol_min"],
    "Difference requires spatially matched valid source pixels. No Celsius offset for a temperature difference.", readiness="requires_local_derivation", earth_comparison="needs_matched_thermal_definition")

for key, family, names, label, unit, notes in [
    ("solar_illumination", "illumination", ["AVGVISIB_65N_240M_201608_NP", "AVGVISIB_65S_240M_201608_SP", "AVGVISIB_85N_060M_201608_NP", "AVGVISIB_85S_060M_201608_SP"], "Long-term sunlight visibility fraction", "fraction", "Polar coverage; 240/60 m service pixels. Long-term modeled visibility, typically the 18.6-year nutation cycle; specify observer height/reference and epoch. Not instantaneous insolation."),
    ("earth_visibility", "illumination", ["AVGVISIB_65N_240M_201608_EARTH_NP", "AVGVISIB_65S_240M_201608_EARTH_SP", "AVGVISIB_85N_060M_201608_EARTH_NP", "AVGVISIB_85S_060M_201608_EARTH_SP"], "Long-term Earth visibility fraction", "fraction", "Polar coverage; 240/60 m service pixels. Any visible part of Earth can count, so this does not establish a usable DSN/radio link."),
    ("permanent_shadow", "shadow", ["PSR_NP", "PSR_SP"], "Intersection with mapped permanently shadowed region", "boolean_or_area_fraction", "Polar vector polygons; sample a point or intersect a window. Polygon quality/coverage fields matter."),
    ("surface_ice_detection", "ice", ["NP_IceLoc_280m", "SP_IceLoc_280m"], "Proximity to mapped surface-ice detection", "distance_m_or_detection", "Detection points on a nominal 280 m product. An empty result is not evidence of ice absence; radius must be explicit."),
    ("geologic_unit", "geology", ["Unified_Global_Geologic_Map_of_the_Moon_Geologic_Units"], "Geologic unit and description", "body_specific_category", "Queryable global polygons; preserve USGS unit code, description and interpretation. Codes are not interchangeable with Earth or Mars lithology codes."),
]:
    add("moon", key, label, unit, "P1", "spatial_feature_query" if family in ["shadow", "ice", "geology"] else "numeric_point_or_cutout", names, notes,
        readiness="verified_transport", earth_comparison="context_only" if family != "geology" else "requires_cross_body_geologic_mapping")

mineral = "Lunar_Kaguya_MIMap_MineralDeconv_"
for key, suffix, label, unit, semantics in [
    ("olivine", "OlivinePercent", "Olivine abundance", "raw_fraction", "Four mineral raw fractions sum to approximately one at all three probe locations, whereas FGDC calls them wt%. Multiply by 100 only under an explicit, reviewed normalization; do not label raw 0.066 as 0.066 wt%."),
    ("clinopyroxene", "ClinopyroxenePercent", "Clinopyroxene abundance", "raw_fraction", "Same fraction-versus-wt% scaling issue as olivine."),
    ("orthopyroxene", "OrthopyroxenePercent", "Orthopyroxene abundance", "raw_fraction", "Same fraction-versus-wt% scaling issue as olivine."),
    ("plagioclase", "PlagioclasePercent", "Plagioclase abundance", "raw_fraction", "Same scaling issue; mature soils can bias mineral interpretation, so retain OMAT and fit quality."),
    ("feo", "FeOWeightPercent", "Iron oxide abundance", "wt_percent_FeO", "FGDC explicitly defines FeO wt%; raw samples are approximately 6.8-20.4 in this audit."),
    ("smfe", "AbundanceSMFe", "Submicroscopic metallic iron abundance", "unresolved_raw_scale", "Raw 2.9-6.9; source labels do not resolve the raw service scaling. Do not publish a physical abundance yet."),
    ("optical_maturity", "OpticalMaturityIndex", "Optical maturity index (OMAT)", "1", "Space-weathering/composition context; not an Earth soil-maturity index."),
    ("plagioclase_grain_size", "PlagioclaseGrainSizeMicrons", "Modeled plagioclase grain size", "micrometre", "Spectral-model parameter for plagioclase; not measured bulk soil particle size."),
    ("mineral_fit_quality", "WeightedCriteriaValue", "Mineral-model weighted fit criterion", "raw_criterion", "Quality/provenance channel, not a resource or environmental score. NaN observed at one probe; no universal quality cutoff established."),
]:
    add("moon", key, label, unit, "P2", "numeric_point_or_cutout", [mineral + suffix + "_50N50S"],
        "Coverage 50S-50N. Numeric service and samples report 0.25-degree spacing (~7.6 km at equator), despite source imagery advertised at 512 ppd (~59 m). A fine pixelSize request still returned 0.25 degree. " + semantics,
        unit_status="scaling_unresolved" if key in ["olivine", "clinopyroxene", "orthopyroxene", "plagioclase", "smfe"] else "documented" if key in ["feo", "plagioclase_grain_size", "optical_maturity"] else "raw_only",
        readiness="verified_transport_requires_semantic_review" if key not in ["feo", "optical_maturity", "plagioclase_grain_size"] else "verified_transport",
        earth_comparison="earth_equivalent_not_loaded")

for key, names, label, unit, priority, notes, status in [
    ("thorium", ["LP_GRS_Th_Global_2ppd", "LP_GRS_ThoriumAbundance_0_5ppd"], "Thorium abundance", "ppm", "P2", "Very coarse geochemical context; tested service spacings 1 and 0.5 degree. Not a local radiation-dose estimate.", "documented_family"),
    ("titanium", ["LP_GRS_TitaniumAbundance_2ppd"], "Elemental titanium abundance", "wt_percent_Ti", "P2", "FGDC defines elemental Ti wt%, not TiO2. Service spacing 2 degrees (~61 km); product name is not a reliable ppd declaration.", "documented"),
    ("potassium", ["KPLO_KGRS_Potassium_2ppd"], "KPLO potassium channel", "unresolved_raw_scale", "P2", "Numeric samples verified; 2-degree service grid. Raw physical scaling/unit unresolved.", "unresolved"),
    ("uranium", ["KPLO_KGRS_Uranium_2ppd"], "KPLO uranium channel", "unresolved_raw_scale", "P2", "Numeric samples verified; 2-degree service grid. Raw physical scaling/unit unresolved.", "unresolved"),
    ("thermal_neutron", ["KPLO_KGRS_Thermal_Neutron_2ppd"], "Thermal-neutron channel", "unresolved_raw_scale", "P2", "Numeric channel, not temperature or water abundance. Detector normalization/units unresolved.", "unresolved"),
    ("water_equivalent_hydrogen", ["np_weh_5km_adapt_map", "sp_weh_5km_adapt_map"], "Polar water-equivalent-hydrogen proxy", "raw_WEH_scale_unresolved", "P1", "North samples verified at 5 km spacing; south endpoint timed out. Confirm raw abundance scaling before exposing %. Hydrogen does not uniquely establish accessible water ice.", "unresolved"),
    ("ice_stability_depth", ["np_ice_depth_new_240m_mat_today_27_Oct_2016", "sp_ice_depth_new_240m_mat_today_27_Oct_2016"], "Modeled ice-stability depth", "raw_depth_scale_unresolved", "P1", "North samples verified at 240 m spacing, some exactly 2.5; possible cap/sentinel requires source documentation. South endpoint timed out. Modeled stability, not measured depth to an ice deposit.", "unresolved"),
    ("radar_cpr", ["LRO_MiniRF_CPR_NPole70_118mp", "LRO_MiniRF_CPR_SPole70_118mp"], "Radar circular polarization ratio", "ratio", "P2", "Polar 118 m grid. Roughness and material properties affect CPR; elevated CPR alone is not ice confirmation.", "documented_family"),
    ("radar_backscatter", ["LRO_MiniRF_S1_NPole70_118mp", "LRO_MiniRF_S1_SPole70_118mp"], "Radar S1 backscatter", "raw_linear_scale_unresolved", "P2", "Polar 118 m service. Do not mix its positive linear values with the separate Malapert dB product.", "unresolved"),
    ("radar_backscatter_malapert", ["minirf_s1_49dnorm_EQ"], "Incidence-normalized Malapert radar backscatter", "dB", "P2", "Small south-polar region; FGDC defines dB. Metadata title says CPR but abstract and S1 layer describe backscatter; retain that ambiguity.", "documented_with_label_conflict"),
    ("christiansen_feature", ["dgdr_std_cf_clc_cyl_128_jp2"], "Christiansen-feature wavelength", "micrometre", "P2", "Composition proxy, 60S-60N, 1/128-degree service grid (~237 m equator). Metadata min contains invalid negative wavelengths; enforce documented masks, not a generic numeric check.", "documented"),
    ("regional_rock_count", ["LRO_NAC_RockAbundance_100mpp_SiteB"], "Regional detected-rock count", "count_per_100m_by_100m_cell", "P1", "Site B only. Exact FGDC defines detected-rock counts in each 100 m by 100 m pixel; not rock area fraction. Detection completeness and source missing mask still matter.", "documented"),
    ("regional_rock_abundance_unresolved", ["LRO_NAC_RockAbundance_100mpp_SiteC"], "Regional Site C rock channel", "unresolved_integer_encoding", "P1", "Site C is U2 with sampled 0/1; exact FGDC was empty. Do not infer a fraction or transfer Site B encoding without confirmation.", "unresolved"),
    ("regional_roughness", ["LRO_NAC_Roughness_15m_00N234E_150cmp"], "Regional NAC terrain roughness", "method_and_units_unresolved", "P1", "Regional raster with 1.5 m postings; 15 m is a processing baseline in its name, not pixel spacing. Verify roughness definition and units before comparison.", "unresolved"),
    ("regional_published_slope", ["LRO_LOLA_Slope_60mpp", "LRO_LOLA_Kagyua_TC_Slope_60mpp"], "Published regional lunar slope", "degree", "P1", "Only southern latitude bands for services checked; postings about119m for the working layer despite 60mpp name. Kagyua endpoint returned invalid-parameter error on sample probe; do not advertise it as working.", "documented_family"),
]:
    add("moon", key, label, unit, priority, "numeric_point_or_cutout", names, notes, unit_status=status,
        readiness="verified_transport" if status.startswith("documented") else "verified_transport_requires_semantic_review")

for key, label, notes in [
    ("albedo", "Photometrically defined surface albedo", "Useful for heating and surface context; no calibrated numeric lunar albedo service was verified in this audit. Do not infer it from basemap brightness."),
    ("thermal_inertia", "Regolith thermal inertia / thermal-property model", "Useful for material and thermal analogues. Published Diviner products exist, but an appropriate quantitative small-cutout endpoint and exact definition still need verification."),
    ("rock_area_fraction", "Diviner-scale surface rock area fraction", "Distinct from NAC detected-rock counts. Published thermal-model rock-abundance products exist; quantitative cutout and quality mask were not verified here."),
]:
    add("moon", key, label, "product_specific", "P1", "external_product_not_yet_verified", notes=notes,
        unit_status="product_specific", readiness="see_external_evidence_or_not_verified")

for key, name, label, unit in [
    ("crustal_density", "grain_density_310_eq", "Modeled crustal grain density", "kg_m3"),
    ("crustal_thickness", "Model2_thick_eq", "Modeled crustal thickness", "km"),
    ("gravity_anomaly", "gggrx_1200a_anom_l180_eq", "Gravity anomaly", "mGal"),
    ("bouguer_anomaly", "gggrx_1200a_boug_l180_eq", "Bouguer gravity anomaly", "mGal"),
    ("geoid", "gggrx_1200a_geoid_l660_eq", "Geoid model", "m"),
    ("magnetic_field", "Ravatetal_JGR2020_LunarL1MagneticModelFields_LPgrad_Total_Field", "Modeled total magnetic field", "raw_scale_unresolved"),
]:
    add("moon", key, label, unit, "P3", "numeric_point_or_cutout", [name],
        "Geophysical/model context, low priority for surface Earth-analogue matching. Unit/model version/normalization must be pinned before frontend exposure; grain density is not surface bulk regolith density.",
        unit_status="family_interpretation_needs_product_confirmation", readiness="verified_transport_requires_semantic_review")

for key, suffix, label in [
    ("modeled_surface_temperature_mean", "avg", "Modeled annual approximate mean surface temperature"),
    ("modeled_surface_temperature_min", "minimum", "Modeled annual minimum surface temperature"),
    ("modeled_surface_temperature_max", "maximum", "Modeled annual maximum surface temperature"),
]:
    add("mars", key, label, "degree_C", "P0", "numeric_cutout", [f"mars_yearly_{suffix}_temperature_celsius"],
        "KRC v2.3.2 model using TES albedo; 8 ppd (~7.4 km equator). Samples at 5am/1pm and 10-degree solar-longitude increments; mean is approximate [day+night]/2. Not measured current weather, air temperature, or an exhaustive diurnal extreme.",
        readiness="verified_transport", earth_comparison="needs_earth_surface_temperature_and_model_flag")
add("mars", "surface_temperature_range", "Modeled annual maximum-minus-minimum surface temperature", "K_difference", "P0", "derive_from_model_extrema", ["mars_yearly_maximum_temperature_celsius", "mars_yearly_minimum_temperature_celsius"],
    "Do not use mars_yearly_temperature_range_celsius directly: all 64 probe pixels equal (max-min)-273.15 within float rounding, producing negative ranges. Derive max-min from matching valid extrema instead.", readiness="requires_local_derivation", earth_comparison="needs_matched_thermal_definition")
for key, layer, label, unit, notes in [
    ("albedo", "TES_Lambert_Albedo_numeric", "TES Lambert albedo", "1", "8 ppd (~7.4 km equator). Reflectance under a specific photometric convention; not a generic RGB brightness or universally interchangeable Earth albedo."),
    ("thermal_inertia_tes", "TES_Thermal_Inertia_numeric", "TES apparent thermal inertia", "J_m-2_K-1_s-0.5", "8 ppd product; preserve model/version/quality and distinguish from Putzig day/night alternatives."),
    ("thermal_inertia_tes_putzig_night", "TES_TI_Putzig_night_numeric", "TES Putzig nighttime apparent thermal inertia", "J_m-2_K-1_s-0.5", "20 ppd service grid (~3 km equator); model alternative to Christensen TES map, not an additional independent physical property."),
    ("thermal_inertia_themis", "thm_ti_100m_32bit", "THEMIS apparent thermal inertia", "J_m-2_K-1_s-0.5", "Nominal100m / max593ppd. Quantitative32bit mosaic; mask-32768 and report actual valid support. Different scenes/epochs are mosaicked."),
    ("surface_dust_index", "TES_dust_index_16ppd_numeric", "TES surface dust-cover index", "1", "16ppd (~3.7km equator). Distinct from atmospheric dust opacity and mineral-model surface-dust fraction. Source NullValue must be respected."),
    ("water_equivalent_hydrogen", "mars_odyssey_grs_mons_perc_wt", "Odyssey neutron-derived water-equivalent hydrogen", "wt_percent_WEH", "2ppd (~29.6km equator), with much larger measurement/reconstruction support possible. Hydrogen-equivalent proxy, not direct surface ice fraction or recoverable resource mass."),
]:
    add("mars", key, label, unit, "P0" if key not in ["water_equivalent_hydrogen"] else "P1", "numeric_cutout", [layer], notes,
        readiness="verified_transport", earth_comparison="earth_equivalent_not_loaded")

for suffix, label in [("Olivine", "Olivine"), ("Feldspar", "K-feldspar"), ("Clinopyroxene", "Clinopyroxene"),
                      ("Plagioclase", "Plagioclase"), ("Carbonate", "Carbonates"), ("Hematite", "Hematite"),
                      ("Amphibole", "Amphibole"), ("Quartz", "Quartz"), ("Sulfate", "Sulfates"), ("Dust", "Surface dust")]:
    add("mars", f"tes_{suffix.lower()}", f"TES {label} spectral-model abundance", "areal_fraction", "P2", "numeric_cutout", [f"TES_{suffix}_numeric"],
        "4ppd (~14.8km equator); source NODATA=-1, not zero. Spectral-mixture model fraction, not elemental wt%; compare only with a compatible terrestrial measurement.", readiness="verified_transport", earth_comparison="earth_equivalent_not_loaded")
add("mars", "tes_mineral_fit_error", "TES mineral-model RMS error", "raw_fit_error", "P2", "numeric_cutout", ["TES_RMS_Error_numeric"],
    "Quality/provenance channel, not an environmental preference. Source NODATA=-1.", readiness="verified_transport", earth_comparison="quality_only")

# Feature queries are transport-verified where the saved Trek audit confirms it.
feature_sources = {
    "geologic_unit": ["SIM3292_Global_Geology"],
    "hydrous_mineral_detection": ["Hydrous_Mineral_Detections_OMEGA_CRISM_clon0", "Aquieous_Mineral_Distribution_Options_clon0"],
    "chloride_detection": ["Chloride_Salt_bearing_Material_THEMIS_clon0"],
    "dune_field": ["Dune_Field"],
    "valley_network": ["Hynek_Valley_Networks"],
    "fans_glaciers_gullies": ["Mars_Fans_Glaciers_Gullies"],
}
for key, label, unit, notes in [
    ("geologic_unit", "USGS geologic unit and description", "body_specific_category", "Trek SIM3292_Global_Geology categorical polygons; ASU numeric map codes require a legend."),
    ("hydrous_mineral_detection", "Mapped hydrous-mineral detections", "presence_or_distance_m", "Trek CRISM/OMEGA detections and aqueous-mineral classes. An empty query is no observation, not absence or zero abundance."),
    ("chloride_detection", "Mapped chloride-bearing-material detections", "presence_or_distance_m", "Trek THEMIS chloride detections; no concentration field established."),
    ("subsurface_ice_evidence", "Regional subsurface ice-consistency evidence", "product_specific_score", "SWIM regional products are evidence/model scores, not probabilities or a globally valid ice-depth/fraction measurement; cutout access needs separate verification."),
    ("dune_field", "Mapped dune-field context", "presence_or_distance_m", "Trek feature mapping; availability outside mapped features is unknown, not zero sand cover."),
    ("valley_network", "Mapped valley-network context", "presence_or_distance_m", "Trek feature mapping; morphology and paleoenvironments, not present liquid water."),
    ("fans_glaciers_gullies", "Mapped fans, glaciers and gullies", "category_or_distance_m", "Trek feature catalog; distinguish separate classes and mapping coverage."),
    ("rock_abundance", "TES/Viking modeled rock abundance", "product_specific_fraction", "Published products exist, but no calibrated numeric point/cutout was verified in this audit. Do not substitute thermal inertia or browse RGB."),
    ("air_temperature_pressure_wind", "Seasonal/time-dependent atmosphere model variables", "variable_specific", "Require explicit local time, season, altitude and atmospheric model; no static Trek lookup verified. Keep separate from surface temperatures."),
]:
    names = feature_sources.get(key, [])
    verified = any(s["name"] in names and s["body"] == "mars" and s["status"] == "spatial_feature_query_verified" for s in trek_audit["services"])
    add("mars", key, label, unit, "P1", "spatial_feature_query" if names else "feature_or_external_provider", names,
        notes + (" Only the first non-group layer was probed; inspect all sublayer semantics and mapping completeness before aggregating." if names else ""),
        unit_status="product_specific", readiness="verified_transport" if verified else "see_external_evidence_or_not_verified")

result = {"schema_version": 1, "scope": "Useful scalar/categorical concepts for small-area planetary screening; not a claim that every layer is globally valid or implemented",
          "readiness_meaning": "verified_transport means a numeric or spatial API response was observed. It does not establish full scientific validation, cross-body equivalence, or demo implementation.",
          "priorities": {"P0": "First terrain/thermal profile", "P1": "Illumination, ice/resource proxies, geology and regional hazards", "P2": "Advanced composition/physical context", "P3": "Low-priority deep geophysical/model context"},
          "required_per_value_metadata": ["source_product", "provider", "body", "raw_value", "raw_unit", "physical_unit", "normalization", "temporal_support", "source_pixel_spacing", "effective_resolution_if_known", "requested_footprint", "valid_support_fraction", "quality_flags", "missing_reason", "retrieved_at"],
          "variables": variables}
(OUT / "catalog.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
lines = ["# Planetary variable catalog", "", "Generated from `catalog.json`; see README for verification results and limitations.", "",
         "`verified_transport` describes API access, not scientific validation or frontend availability.", ""]
for body in ["moon", "mars"]:
    lines += [f"## {body.title()}", "", "| Variable ID | Unit | Priority | Readiness |", "| --- | --- | --- | --- |"]
    lines += [f"| `{v['id']}` | {v['unit']} | {v['priority']} | {v['readiness']} |" for v in variables if v["body"] == body]
    lines.append("")
(OUT / "VARIABLES.md").write_text("\n".join(lines))
print(json.dumps({"variables": len(variables), "moon": sum(v["body"] == "moon" for v in variables), "mars": sum(v["body"] == "mars" for v in variables)}))
