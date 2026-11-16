"""Export the candidate registry as a frontend catalog and complete human list.

Run: python3 scripts/export_frontend_catalog.py
Planning metadata is separate from processed pilot outputs and production data.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research/variables/candidates.json"
CATALOG = ROOT / "catalog"
DOCS = ROOT / "docs"
GROUPS = {
    "terrain": "Terrain",
    "surface_thermal": "Surface temperature and emissivity",
    "albedo": "Albedo",
    "spectral_reflectance": "Spectral reflectance",
    "weather_radiation": "Weather and radiation",
    "climate_derivatives": "Derived climate indicators",
    "soil_properties": "Soil properties",
    "soil_moisture": "Soil moisture",
    "snow_ice": "Snow, ice and freeze/thaw",
    "surface_water": "Surface water",
    "hydrography": "Rivers and catchments",
    "geology": "Geology",
    "land_cover": "Land cover",
    "dust_aerosols": "Dust and aerosols",
    "biosphere_optional": "Vegetation and productivity (optional)",
    "evaporation": "Evaporation",
    "disturbance_optional": "Fire (optional)",
}
LABELS = dict(
    line.split("|", 1)
    for line in """terrain.elevation|Surface elevation
terrain.slope|Slope
terrain.aspect|Aspect (downslope direction)
terrain.local_relief|Local relief
terrain.roughness|Terrain roughness
terrain.topographic_position|Topographic position
terrain.plan_curvature|Plan curvature
terrain.profile_curvature|Profile curvature
terrain.sky_view_factor|Sky view factor
terrain.potential_solar_exposure|Potential solar exposure
temperature.surface.day|Daytime land surface temperature
temperature.surface.night|Nighttime land surface temperature
temperature.surface.composite_day_night_difference|Composite day-minus-night surface temperature
emissivity.modis.band31|MODIS band 31 emissivity
emissivity.modis.band32|MODIS band 32 emissivity
weather.air_temperature_2m|Air temperature at 2 m
weather.dewpoint_2m|Dew point at 2 m
weather.relative_humidity_2m|Relative humidity at 2 m
weather.specific_humidity_2m|Specific humidity at 2 m
weather.surface_pressure|Surface pressure
weather.precipitation|Precipitation amount
weather.wind_speed_10m|Wind speed at 10 m
weather.wind_direction_10m|Wind direction at 10 m
radiation.shortwave_down_all_sky|Downward shortwave radiation (all sky)
radiation.shortwave_down_clear_sky|Downward shortwave radiation (clear sky)
radiation.longwave_down_all_sky|Downward longwave radiation (all sky)
weather.vapor_pressure_deficit|Vapor pressure deficit
weather.air_frost_day_fraction|Fraction of days with air frost
weather.air_freeze_thaw_transition_count|Air freeze/thaw transition count
soil.sand_mass_fraction|Soil sand mass fraction
soil.silt_mass_fraction|Soil silt mass fraction
soil.clay_mass_fraction|Soil clay mass fraction
soil.coarse_fragment_volume_fraction|Soil coarse fragment volume fraction
soil.bulk_density|Soil bulk density
soil.ph_water|Soil pH in water
soil.organic_carbon_mass_fraction|Soil organic carbon mass fraction
soil.nitrogen_mass_fraction|Soil nitrogen mass fraction
soil.cation_exchange_capacity|Soil cation exchange capacity
soil.water_retention|Soil water retention
moisture.surface_retrieval_am|Retrieved surface soil moisture (AM)
moisture.surface_retrieval_pm|Retrieved surface soil moisture (PM)
moisture.surface_assimilated|Assimilated surface soil moisture
moisture.root_zone_assimilated|Assimilated root-zone soil moisture
snow.ndsi_snow_cover_signal|NDSI snow-cover signal
snow.observed_snow_day_fraction|Fraction of observed days with snow
ground.freeze_thaw_state|Ground freeze/thaw state
snow.water_equivalent|Snow water equivalent
soil.temperature|Soil temperature
water.surface_occurrence|Surface water occurrence
water.surface_seasonality|Surface water seasonality
water.distance_to_persistent_water|Distance to persistent surface water
hydro.distance_to_mapped_river|Distance to mapped river
hydro.drainage_density|Drainage density
hydro.upstream_catchment_area|Upstream catchment area
hydro.modeled_runoff|Modeled runoff
geology.dominant_lithology|Dominant lithology
geology.lithology_area_fraction|Lithology area fraction
land_cover.dominant_igbp_class|Dominant IGBP land-cover class
land_cover.igbp_area_fraction|IGBP land-cover area fraction
aerosol.optical_depth_047|Aerosol optical depth at 0.47 µm
aerosol.optical_depth_055|Aerosol optical depth at 0.55 µm
dust.surface_mass_concentration|Surface dust mass concentration
dust.column_mass_density|Dust column mass density
vegetation.ndvi|Normalized Difference Vegetation Index (NDVI)
vegetation.evi|Enhanced Vegetation Index (EVI)
vegetation.leaf_area_index|Leaf area index
vegetation.absorbed_par_fraction|Fraction of absorbed photosynthetically active radiation
vegetation.gross_primary_production|Gross primary production
vegetation.net_primary_production|Net primary production
water.evapotranspiration|Evapotranspiration amount
water.potential_evapotranspiration|Potential evapotranspiration amount
fire.burn_date|Burn date (day of year)
fire.burned_area_fraction|Burned area fraction""".splitlines()
)
for band in range(1, 8):
    LABELS[f"reflectance.modis.band{band}"] = f"MODIS band {band} surface reflectance"
for band in range(10, 15):
    LABELS[f"emissivity.aster.band{band}"] = f"ASTER band {band} emissivity"
for sky, sky_label in [("black_sky", "Black-sky"), ("white_sky", "White-sky")]:
    for band, label in [
        ("visible", "visible"),
        ("near_infrared", "near infrared"),
        ("shortwave", "shortwave"),
    ]:
        LABELS[f"albedo.{sky}.{band}"] = f"{sky_label} {label} albedo"

TERRAIN = {
    "terrain.elevation": [
        (f"terrain.elevation.{s}", "m", s) for s in ("min", "mean", "max", "stddev")
    ],
    "terrain.slope": [
        (f"terrain.slope.{s}", "degree", s) for s in ("mean", "stddev", "max")
    ],
    "terrain.roughness": [("terrain.roughness_3x3.mean", "m", "mean")],
    "terrain.topographic_position": [
        ("terrain.topographic_position_3x3.mean", "m", "mean")
    ],
    "terrain.local_relief": [("terrain.local_relief_3x3.mean", "m", "mean")],
    "terrain.aspect": [
        ("terrain.aspect_eastness.mean", "1", "mean"),
        ("terrain.aspect_northness.mean", "1", "mean"),
    ],
}
TEMPERATURE = {
    "temperature.surface.day": "day_k",
    "temperature.surface.night": "night_k",
    "temperature.surface.composite_day_night_difference": "paired_day_minus_night_k",
}


def validated(directory):
    path = ROOT / directory / "validation.json"
    return path.exists() and json.loads(path.read_text())["status"] == "passed"


def axes(feature):
    key = feature["key"]
    source = feature["source"]
    if feature["family"] == "terrain" and key != "terrain.potential_solar_exposure":
        return ["cell"], "static"
    if source == "SoilGrids":
        return ["cell", "depth_interval", "prediction_statistic"] + (
            ["matric_potential"] if key == "soil.water_retention" else []
        ), "static"
    if source == "GLiM":
        return ["cell", "map_version"] + (
            ["lithology_class"] if key.endswith("area_fraction") else []
        ), "map_version"
    if source == "MCD12Q1":
        return ["cell", "year"] + (
            ["land_cover_class"] if key.endswith("area_fraction") else []
        ), "annual"
    if source in ("AG100", "JRC_GSW"):
        return ["cell", "reference_period"], "reference_period"
    if source == "HydroATLAS":
        return [
            "cell",
            "reference_period" if key == "hydro.modeled_runoff" else "map_version",
        ], "reference_period" if key == "hydro.modeled_runoff" else "map_version"
    if source == "MOD17A3HGF":
        return ["cell", "year"], "annual"
    if key == "soil.temperature":
        return ["cell", "observation_window", "soil_layer"], "interval"
    return ["cell", "observation_window"], "interval"


def main():
    raw = json.loads(REGISTRY.read_text())
    assert set(LABELS) == {f["key"] for f in raw["features"]}
    terrain_ready = validated(Path("research/pilots/terrain"))
    temperature_ready = validated(Path("research/pilots/temperature-native"))
    definitions = []
    for feature in raw["features"]:
        key = feature["key"]
        outputs = []
        manifest = None
        status = "deferred" if feature["priority"] == "deferred" else "planned"
        if terrain_ready and key in TERRAIN:
            status = (
                "pilot_components_only"
                if key == "terrain.aspect"
                else "pilot_processed"
            )
            manifest = "research/pilots/terrain/run.json"
            outputs = [
                {"id": oid, "unit": unit, "statistic": stat, "artifactColumn": None}
                for oid, unit, stat in TERRAIN[key]
            ]
        if temperature_ready and key in TEMPERATURE:
            status = "pilot_processed"
            manifest = "research/pilots/temperature-native/run.json"
            outputs = [
                {
                    "id": key + ".mean",
                    "unit": "K",
                    "statistic": "mean",
                    "artifactColumn": TEMPERATURE[key],
                }
            ]
        dimensions, time_kind = axes(feature)
        definitions.append(
            {
                "id": key,
                "definitionVersion": 1,
                "label": LABELS[key],
                "groupId": feature["family"],
                "valueType": "category"
                if feature["proposed_unit"] == "category"
                else "number",
                "proposedUnit": feature["proposed_unit"],
                "unitStatus": "proposed_normalization_target",
                "sourceId": feature["source"],
                "candidateSourceField": feature["candidate_source_field"],
                "kind": feature["kind"],
                "role": feature["role"],
                "priority": feature["priority"],
                "bodyIds": ["earth"],
                "dimensions": dimensions,
                "temporalKind": time_kind,
                "notes": feature["notes"],
                "sourceMappingStatus": feature["mapping_status"],
                "categoryDictionary": None,
                "productionAvailable": False,
                "pilot": {
                    "status": status,
                    "regionIds": ["atacama", "iceland", "sahara"] if outputs else [],
                    "outputs": outputs,
                    "manifest": manifest,
                },
            }
        )
    catalog = {
        "schemaVersion": 1,
        "catalogVersion": "0.1.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceRegistrySha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        "status": "planning_catalog_with_research_pilot_availability_no_live_API",
        "groups": [{"id": key, "label": label} for key, label in GROUPS.items()],
        "counts": {
            "variables": len(definitions),
            "groups": len(GROUPS),
            "byPilotStatus": dict(Counter(v["pilot"]["status"] for v in definitions)),
        },
        "notes": [
            "Stable variable IDs identify quantities; output IDs identify realized statistics/operators.",
            "91 candidates are a broad roadmap, not a guarantee all sources pass scientific validation.",
            "All current candidates use Earth sources. Missing or inapplicable Moon/Mars values must not become zero.",
            "Proposed units are normalization targets, not necessarily native provider units; output units override them.",
            "Soil depth, prediction statistic, class and time are dimensions, not separate variable IDs.",
            "Categorical dictionaries, allowed ranges and operators remain source/version-specific and are not fabricated here.",
            "Aspect pilot outputs are dimensionless components, not an available angle in degrees.",
            "Pilot-processed means artifacts exist in three small regions; no production PostgreSQL/API release is loaded.",
            "Not all cells/windows have valid data; zero and missing are distinct.",
            "Consumer contract types are in catalog/types.ts; observation example is illustrative, not a live endpoint.",
        ],
        "variables": definitions,
    }
    CATALOG.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    (CATALOG / "variables.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Historical candidate variable catalog",
        "",
        "This is the early research roadmap. Its availability statements describe the",
        "original pilot, not the deployed API. For the current 25 Earth fields, source",
        "citations and planetary sampling, see [Data sources and methodology](DATA_SOURCES.md).",
        "Use the API `Metadata` RPC for the live SQL schema.",
        "",
        "Generated by `python3 scripts/export_frontend_catalog.py` from the research candidate registry.",
        "",
        f"**{len(definitions)} candidate variables across {len(GROUPS)} groups.** Stable IDs identify candidate concepts. This is a historical roadmap with explicit pilot availability; it does not describe the current live query API.",
        "",
        "Machine-readable definitions: [`catalog/variables.json`](../catalog/variables.json). Contract types: [`catalog/types.ts`](../catalog/types.ts).",
        "",
        "## Frontend rules",
        "",
        "- Use `id` as the variable key and `label` for presentation. Select a concrete `pilot.outputs[].id` when using a realized statistic.",
        "- All current sources describe Earth. Do not substitute zero for missing, unprocessed, quality-filtered, or inapplicable values.",
        "- Units below are proposed normalized units. Use an output’s own unit when one is available. Kelvin temperatures and Kelvin differences have different display-conversion rules.",
        "- Render categorical selectors from a versioned dictionary when supplied. `categoryDictionary: null` means the dictionary is not ready.",
        "- Soil depth, class, prediction statistic, matric potential, and time are dimensions. They must remain selectable coordinates rather than being flattened into additional variable names.",
        "- `pilot_processed` means research CSVs exist only for Atacama, Iceland and the Sahara. It does not mean global coverage or a production API. All `productionAvailable` flags are currently false.",
        "- Aspect is `pilot_components_only`: the available eastness/northness components use unit `1`; a degree-valued aspect filter remains planned.",
        "- EVI is deferred. Other vegetation/productivity and fire variables are optional context.",
        "",
        "## Availability",
        "",
        "| Status | Meaning |",
        "| --- | --- |",
        "| `pilot_processed` | A supported statistic has validated research artifacts for the three pilot regions. |",
        "| `pilot_components_only` | Related components exist, but the named quantity is not yet supplied directly. |",
        "| `planned` | Cataloged candidate; extraction/validation remains. |",
        "| `deferred` | Intentionally postponed candidate. |",
        "",
        "“Validated” here means operator fixtures and artifact consistency checks passed; source limitations and scientific publication review still apply.",
        "",
        "## Dimensions and time",
        "",
        "The JSON carries `dimensions` and `temporalKind` per variable. Static terrain and soil differ from interval observations, annual products, map versions, and reference-period summaries. Dimension value vocabularies and ranges will come from the selected dataset/version. Processing scale and neighborhood belong to the output definition and provenance.",
        "",
        "## Full list",
        "",
    ]
    for group, label in GROUPS.items():
        items = [v for v in definitions if v["groupId"] == group]
        lines.extend(
            [
                f"### {label} ({len(items)})",
                "",
                "| Variable ID | Label | Type | Proposed unit | Priority | Pilot status |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| `{v['id']}` | {v['label']} | {v['valueType']} | `{v['proposedUnit']}` | {v['priority']} | `{v['pilot']['status']}` |"
            for v in items
        )
        lines.append("")
    lines.extend(
        [
            "## Currently realized outputs",
            "",
            "The JSON maps canonical variables to artifact output IDs. Terrain supports include native 3×3 roughness, relief and topographic-position operators; do not label these as arbitrary-scale global metrics. Temperature outputs are strict-QA 8-day surface composites with separate valid-area fractions. Day-minus-night is computed from paired valid pixels, not a full diurnal amplitude.",
            "",
            "| Variable | Output ID | Unit | Statistic |",
            "| --- | --- | --- | --- |",
        ]
    )
    for v in definitions:
        for output in v["pilot"]["outputs"]:
            lines.append(
                f"| `{v['id']}` | `{output['id']}` | `{output['unit']}` | {output['statistic']} |"
            )
    (DOCS / "VARIABLES.md").write_text("\n".join(lines) + "\n")
    if temperature_ready:
        row = next(
            r
            for r in csv.DictReader(
                (ROOT / "research/pilots/temperature-native/atacama.csv").open()
            )
            if r["day_k"]
        )
        example = {
            "contractStatus": "illustrative_response_using_real_pilot_value_no_live_endpoint",
            "variableId": "temperature.surface.day",
            "outputId": "temperature.surface.day.mean",
            "bodyId": "earth",
            "regionId": "atacama",
            "cellId": row["cell_id"],
            "h3Resolution": 6,
            "valueType": "number",
            "value": float(row["day_k"]),
            "unit": "K",
            "status": "valid",
            "statistic": "mean",
            "window": {
                "kind": "interval",
                "start": row["window_start"],
                "endExclusive": row["window_end_exclusive"],
            },
            "dimensions": {},
            "validAreaFraction": float(row["day_k_valid_area_fraction"]),
            "spatialSupport": "h3_cell_intersection_with_pilot_region",
            "provenanceRef": "research/pilots/temperature-native/atacama-run.json",
        }
        (CATALOG / "observation-example.json").write_text(
            json.dumps(example, indent=2) + "\n"
        )
    assert len(definitions) == 91 and len({v["id"] for v in definitions}) == 91
    output_ids = [o["id"] for v in definitions for o in v["pilot"]["outputs"]]
    assert len(output_ids) == len(set(output_ids))
    print(json.dumps(catalog["counts"], indent=2))


if __name__ == "__main__":
    main()
