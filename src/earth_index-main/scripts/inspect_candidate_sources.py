"""Build a broad, unprocessed feature registry and check NASA catalog coverage.

Run with Python 3.11+: python scripts/inspect_candidate_sources.py
Metadata responses are cached. No scientific raster arrays are downloaded.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "variables"
CACHE = OUT / "catalog-cache"
CMR = "https://cmr.earthdata.nasa.gov/search/"
PERIOD = "2024-01-01T00:00:00Z,2024-12-31T23:59:59Z"

# Collection identities checked against the live NASA catalog in this investigation.
NASA = {
    "MOD11A2": ("C2269056084-LPCLOUD", False),
    "MCD43A3": ("C2278860820-LPCLOUD", False),
    "MCD12Q1": ("C2484079608-LPCLOUD", False),
    "MOD13Q1": ("C1748066515-LPCLOUD", False),
    "MOD09A1": ("C2343111356-LPCLOUD", False),
    "MOD10A1": ("C2565093311-NSIDC_CPRD", False),
    "MOD16A2GF": ("C2565791021-LPCLOUD", False),
    "MCD15A3H": ("C2343110937-LPCLOUD", False),
    "MOD17A2HGF": ("C2565791029-LPCLOUD", False),
    "MOD17A3HGF": ("C2565791034-LPCLOUD", False),
    "MCD64A1": ("C2565786756-LPCLOUD", False),
    "MCD19A2": ("C2324689816-LPCLOUD", False),
    "SPL3SMP_E": ("C2938664763-NSIDC_CPRD", False),
    "SPL3FTP_E": ("C2938664439-NSIDC_CPRD", False),
    "SPL4SMGP": ("C3480440870-NSIDC_CPRD", False),
    "M2T1NXAER": ("C1276812830-GES_DISC", False),
    "AG100": ("C2763266348-LPCLOUD", True),
}
OTHER = {
    "cop-dem-glo-30": ("https://earth-search.aws.element84.com/v1/collections/cop-dem-glo-30", "Pilot footprints already checked; static DSM, not uniformly bare-earth terrain."),
    "SoilGrids": ("https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html", "Modeled properties with depth and uncertainty dimensions; bare ground, ice and other masks can exclude analog sites."),
    "GLiM": ("https://doi.org/10.1029/2012GC004370", "Heterogeneous mapping scales; lithology is not direct mineral abundance."),
    "JRC_GSW": ("https://global-surface-water.appspot.com/download", "Surface-water history; acquisition gaps and version transitions require checks."),
    "HydroATLAS": ("https://www.hydrosheds.org/hydroatlas", "Basin/river/lake support; assigning a basin property to a cell does not make it a cell measurement."),
    "ERA5_Land": ("https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries", "Model/reanalysis estimates; retain original grid and depth/time support."),
    "NASA_POWER": ("https://power.larc.nasa.gov/docs/services/api/temporal/daily/", "NASA service with parameter-dependent coarse source grids; cataloged here, not spatially sampled."),
}

FEATURES = []


def add(family, source, role, entries, notes="", dimensions=None):
    """Rows: key | proposed canonical unit | native/derived/summary | source field.

    A source field is a candidate mapping, not proof of local valid pixels.
    Missing mappings must be completed against product headers before extraction.
    """
    for row in entries.strip().splitlines():
        key, unit, kind, field = [x.strip() for x in row.split("|")]
        priority = "optional" if family.endswith("optional") else "evaluate"
        if key == "vegetation.evi":
            priority = "deferred"
        FEATURES.append({
            "key": key, "family": family, "source": source, "role": role,
            "proposed_unit": unit, "kind": kind,
            "candidate_source_field": field or None,
            "mapping_status": "candidate_requires_header_and_QA_validation",
            "dimensions": dimensions or ["cell", "observation_window"],
            "status": "candidate_unprocessed", "notes": notes,
            "priority": priority,
        })


add("terrain", "cop-dem-glo-30", "physical_surface", """
terrain.elevation | m | native | data
terrain.slope | degree | derived |
terrain.aspect | degree | derived |
terrain.local_relief | m | derived |
terrain.roughness | m | derived |
terrain.topographic_position | m | derived |
terrain.plan_curvature | m-1 | derived |
terrain.profile_curvature | m-1 | derived |
terrain.sky_view_factor | 1 | derived |
terrain.potential_solar_exposure | J m-2 | derived |
""", "Terrain derivatives require a versioned operator, native processing scale and neighborhood. Aspect is circular. Potential exposure additionally requires time and solar geometry.")
add("surface_thermal", "MOD11A2", "physical_surface", """
temperature.surface.day | K | native | LST_Day_1km
temperature.surface.night | K | native | LST_Night_1km
emissivity.modis.band31 | 1 | native | Emis_31
emissivity.modis.band32 | 1 | native | Emis_32
temperature.surface.composite_day_night_difference | K | derived |
""", "Day/night composites can contain different accepted dates. Their difference is not a directly measured full diurnal amplitude.")
add("surface_thermal", "AG100", "physical_surface", "\n".join(
    f"emissivity.aster.band{band} | 1 | native |" for band in range(10, 15)
), "Static ASTER emissivity climatology; preserve wavelength response, acquisition support, sample count and uncertainty.", ["cell", "reference_period"])
add("albedo", "MCD43A3", "physical_surface", """
albedo.black_sky.visible | 1 | native | Albedo_BSA_vis
albedo.black_sky.near_infrared | 1 | native | Albedo_BSA_nir
albedo.black_sky.shortwave | 1 | native | Albedo_BSA_shortwave
albedo.white_sky.visible | 1 | native | Albedo_WSA_vis
albedo.white_sky.near_infrared | 1 | native | Albedo_WSA_nir
albedo.white_sky.shortwave | 1 | native | Albedo_WSA_shortwave
""", "Keep black/white-sky illumination definitions and overlapping 16-day support. These are not directly interchangeable with planetary Lambert or geometric albedo.")
add("spectral_reflectance", "MOD09A1", "physical_surface", "\n".join(
    f"reflectance.modis.band{band} | 1 | native | sur_refl_b0{band}" for band in range(1, 8)
), "Retain spectral response, viewing geometry and quality. Broad bands alone do not establish a mineral or rock type.")
add("weather_radiation", "NASA_POWER", "earth_environment", """
weather.air_temperature_2m | K | native | T2M
weather.dewpoint_2m | K | native | T2MDEW
weather.relative_humidity_2m | percent | native | RH2M
weather.specific_humidity_2m | kg kg-1 | native | QV2M
weather.surface_pressure | Pa | native | PS
weather.precipitation | kg m-2 | native | PRECTOTCORR
weather.wind_speed_10m | m s-1 | native | WS10M
weather.wind_direction_10m | degree | native | WD10M
radiation.shortwave_down_all_sky | J m-2 | native | ALLSKY_SFC_SW_DWN
radiation.shortwave_down_clear_sky | J m-2 | native | CLRSKY_SFC_SW_DWN
radiation.longwave_down_all_sky | J m-2 | native | ALLSKY_SFC_LW_DWN
""", "Proposed units require explicit conversion from provider units and accumulation periods. Query native coarse grid support rather than implying H3-scale atmospheric detail.")
add("climate_derivatives", "NASA_POWER", "earth_environment", """
weather.vapor_pressure_deficit | Pa | derived |
weather.air_frost_day_fraction | 1 | summary |
weather.air_freeze_thaw_transition_count | 1 | summary |
""", "Require specified time series, thresholds and sampling frequency. Daily means cannot determine true subdaily freeze-thaw cycling. Freeze/thaw from air temperature is not a soil-freeze measurement.")
add("soil_properties", "SoilGrids", "earth_substrate", """
soil.sand_mass_fraction | kg kg-1 | native | sand
soil.silt_mass_fraction | kg kg-1 | native | silt
soil.clay_mass_fraction | kg kg-1 | native | clay
soil.coarse_fragment_volume_fraction | 1 | native | cfvo
soil.bulk_density | kg m-3 | native | bdod
soil.ph_water | 1 | native | phh2o
soil.organic_carbon_mass_fraction | kg kg-1 | native | soc
soil.nitrogen_mass_fraction | kg kg-1 | native | nitrogen
soil.cation_exchange_capacity | cmol(c) kg-1 | native | cec
soil.water_retention | m3 m-3 | native |
""", "Predictions with depth and uncertainty. Soil is not planetary regolith. Water retention additionally requires a specified matric-potential dimension.", ["cell", "depth_interval", "prediction_statistic", "matric_potential_if_applicable"])
add("soil_moisture", "SPL3SMP_E", "earth_environment", """
moisture.surface_retrieval_am | m3 m-3 | native |
moisture.surface_retrieval_pm | m3 m-3 | native |
""", "Keep retrieval footprint, AM/PM acquisition, retrieval quality, frozen-soil and land masks; 9 km grid spacing does not imply 9 km independent radiometric resolution.")
add("soil_moisture", "SPL4SMGP", "earth_environment", """
moisture.surface_assimilated | m3 m-3 | native |
moisture.root_zone_assimilated | m3 m-3 | native |
""", "Model-assimilated fields, not a second independent direct observation of root-zone water.")
add("snow_ice", "MOD10A1", "earth_environment", """
snow.ndsi_snow_cover_signal | 1 | native | NDSI_Snow_Cover
snow.observed_snow_day_fraction | 1 | summary |
""", "NDSI snow-cover signal is not automatically fractional snow-covered area. Persistence denominator must exclude unobserved/cloud-masked days.")
add("snow_ice", "SPL3FTP_E", "earth_environment", """
ground.freeze_thaw_state | category | native |
""", "Check latitude domain, seasonal-reference validity and retrieval quality before interpreting categorical states.")
add("snow_ice", "ERA5_Land", "earth_environment", """
snow.water_equivalent | kg m-2 | native |
soil.temperature | K | native |
""", "Model values; soil temperature has a depth-layer dimension. Preserve native grid and accumulation semantics.", ["cell", "observation_window", "depth_layer_if_applicable"])
add("surface_water", "JRC_GSW", "earth_environment", """
water.surface_occurrence | 1 | native |
water.surface_seasonality | month_count | native |
water.distance_to_persistent_water | m | derived |
""", "Occurrence uses valid-observation support. Distance depends on the chosen water definition, period and coastline policy.")
add("hydrography", "HydroATLAS", "earth_environment", """
hydro.distance_to_mapped_river | m | derived |
hydro.drainage_density | m m-2 | derived |
hydro.upstream_catchment_area | m2 | native |
hydro.modeled_runoff | kg m-2 | native |
""", "Preserve basin/river support and mapping limits. A catchment attribute must not be presented as a local cell observation.")
add("geology", "GLiM", "physical_surface", """
geology.dominant_lithology | category | derived |
geology.lithology_area_fraction | 1 | derived |
""", "Class vocabularies and source map scales vary. Class fractions have a category dimension, and uncertainty at boundaries must remain visible.", ["cell", "map_version", "lithology_class_if_applicable"])
add("land_cover", "MCD12Q1", "earth_context", """
land_cover.dominant_igbp_class | category | derived | LC_Type1
land_cover.igbp_area_fraction | 1 | derived | LC_Type1
""", "Include bare, vegetated, snow/ice, water and built classes under the versioned IGBP vocabulary.", ["cell", "year", "class_if_applicable"])
add("dust_aerosols", "MCD19A2", "earth_environment", """
aerosol.optical_depth_047 | 1 | native | Optical_Depth_047
aerosol.optical_depth_055 | 1 | native | Optical_Depth_055
""", "Atmospheric column optical depth is not deposited dust or abrasiveness. At high elevations, a background value may replace an actual retrieval; QA must distinguish it.")
add("dust_aerosols", "M2T1NXAER", "earth_environment", """
dust.surface_mass_concentration | kg m-3 | native | DUSMASS
dust.column_mass_density | kg m-2 | native | DUCMASS
""", "Coarse reanalysis fields describe airborne mineral dust; they do not measure local sediment grain size or lunar regolith hazards.")
add("biosphere_optional", "MOD13Q1", "earth_biosphere", """
vegetation.ndvi | 1 | native | NDVI
vegetation.evi | 1 | native | EVI
""", "Optional Earth-side context, not a planetary match coordinate. EVI is deferred for the initial analog workflow.")
add("biosphere_optional", "MCD15A3H", "earth_biosphere", """
vegetation.leaf_area_index | m2 m-2 | native |
vegetation.absorbed_par_fraction | 1 | native |
""", "Evaluate only where vegetation/quality masks make the retrieval applicable.")
add("biosphere_optional", "MOD17A2HGF", "earth_biosphere", """
vegetation.gross_primary_production | kg C m-2 | native |
""", "Model-derived carbon flux accumulated over the source composite; optional context.")
add("biosphere_optional", "MOD17A3HGF", "earth_biosphere", """
vegetation.net_primary_production | kg C m-2 | native |
""", "Annual modeled ecosystem productivity; optional context, not a planetary observable.")
add("evaporation", "MOD16A2GF", "earth_environment", """
water.evapotranspiration | kg m-2 | native | ET
water.potential_evapotranspiration | kg m-2 | native | PET
""", "Product is primarily for vegetated land. Catalog footprint intersection does not guarantee useful values on bare desert or ice.")
add("disturbance_optional", "MCD64A1", "earth_context", """
fire.burn_date | day_of_year | native | BurnDate
fire.burned_area_fraction | 1 | derived |
""", "Optional site-history context. Decode unburned, unobserved and fill separately; never average categorical burn dates.")


def fetch(url):
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "earth-index-variable-inventory/1"}), timeout=60) as response:
                record = {"request_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                          "hits": response.headers.get("CMR-Hits"), "body": json.load(response)}
            path.write_text(json.dumps(record))
            return record
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def inspect(source, concept_id, static, areas):
    url = CMR + "collections.umm_json?" + urlencode({"concept_id": concept_id})
    record = fetch(url)
    assert len(record["body"]["items"]) == 1
    item = record["body"]["items"][0]
    result = {"source": source, "concept_id": concept_id, "revision_id": item["meta"]["revision-id"],
              "metadata": item["umm"], "metadata_url": url, "retrieved_at": record["retrieved_at"], "pilot_catalog_hits": {}}
    for area in areas:
        params = {"collection_concept_id": concept_id, "page_size": 1,
                  "polygon": ",".join(f"{n:.6f}" for p in area["geometry"]["coordinates"][0] for n in p)}
        if not static:
            params["temporal"] = PERIOD
        query = CMR + "granules.json?" + urlencode(params)
        response = fetch(query)
        result["pilot_catalog_hits"][area["properties"]["key"]] = {
            "granule_count": int(response["hits"]), "query_url": query,
            "retrieved_at": response["retrieved_at"],
            "sample_ids": [e["id"] for e in response["body"]["feed"]["entry"]],
        }
    return result


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    assert len({f["key"] for f in FEATURES}) == len(FEATURES)
    assert all(f["source"] in NASA or f["source"] in OTHER for f in FEATURES)
    areas = json.loads((ROOT / "research" / "pilots" / "areas.geojson").read_text())["features"]
    sources = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(inspect, k, cid, static, areas): k for k, (cid, static) in NASA.items()}
        for future in as_completed(jobs):
            key = jobs[future]
            sources[key] = future.result()
            print(key, {k: v["granule_count"] for k, v in sources[key]["pilot_catalog_hits"].items()}, flush=True)
    for key, (url, note) in OTHER.items():
        sources[key] = {"source": key, "documentation_url": url, "notes": note,
                        "pilot_check_status": "verified_in_previous_pilot" if key == "cop-dem-glo-30" else "documentation_reviewed_spatial_check_pending"}
    registry = {"status": "Discovery registry only; no scientific values processed", "analysis_period": PERIOD,
                "generated_at": datetime.now(timezone.utc).isoformat(), "features": FEATURES,
                "counts": {"candidate_feature_definitions": len(FEATURES), "sources": len(sources),
                           "by_family": dict(Counter(f["family"] for f in FEATURES)),
                           "by_kind": dict(Counter(f["kind"] for f in FEATURES))},
                "scope_notes": [
                    "Source fields and proposed units are candidate mappings requiring header, QA and unit-conversion validation.",
                    "Feature definitions include spectral bands, derivatives and summaries; they are not all independent physical variables.",
                    "Depth, class, uncertainty quantile and time dimensions are not counted as new independent variable definitions.",
                    "NASA catalog hit counts are source-file availability, not valid-pixel coverage, scientific suitability or independent sample counts.",
                    "Non-CMR source families have documentation review only unless a previous pilot check is explicitly linked.",
                    "No globally populated database, raster extraction or full-scale processing is performed by this script.",
                ]}
    (OUT / "candidates.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    (OUT / "source-checks.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    print(json.dumps(registry["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
