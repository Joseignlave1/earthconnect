#!/usr/bin/env python3
"""Read-only, bounded numeric-service audit. No raster downloads or DB changes.

Refresh: python3 scripts/inspect_planetary_sources.py
Outputs public service discovery, metadata and small sample responses only.
"""

import concurrent.futures
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/planetary-sources"
SERVERS = [("moon", "trekarcgis"), ("moon", "trekarcgis2"),
           ("moon", "trekarcgis3"), ("mars", "trekarcgis")]


def request(url):
    start = time.monotonic()
    result = {"url": url, "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EarthIndex-source-audit/1.0", "Origin": "http://localhost:8080"})
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Response exceeds audit limit")
            result.update(http_status=response.status, bytes=len(payload),
                          sha256=hashlib.sha256(payload).hexdigest(),
                          cors=response.headers.get("Access-Control-Allow-Origin"),
                          data=json.loads(payload))
    except (OSError, ValueError, urllib.error.URLError) as error:
        result["error"] = str(error)
    result["elapsed_ms"] = round(1000 * (time.monotonic() - start))
    return result


def url_query(base, values):
    return base + "?" + urllib.parse.urlencode(values)


def candidates(inventories):
    """Curated useful families; projection and regional replicas stay in discovery."""
    selected = []
    exact = {
        "LRO_LOLA_DEM_Global_256ppd_v06": ("terrain", "P0"),
        "LRO_LOLA_DEM_Global_128ppd_v04": ("terrain_alternative", "P2"),
        "LRO_LOLA_Kagyua_TC_Slope_60mpp": ("regional_slope", "P1"),
        "LRO_LOLA_Slope_60mpp": ("regional_slope", "P1"),
        "LRO_NAC_Roughness_15m_00N234E_150cmp": ("regional_roughness", "P1"),
        "LRO_NAC_RockAbundance_100mpp_SiteC": ("regional_rock_abundance", "P1"),
        "LRO_NAC_RockAbundance_100mpp_SiteB": ("regional_rock_abundance", "P1"),
        "LP_GRS_Th_Global_2ppd": ("thorium", "P2"),
        "LP_GRS_ThoriumAbundance_0_5ppd": ("thorium_alternative", "P2"),
        "LP_GRS_TitaniumAbundance_2ppd": ("titanium", "P2"),
        "KPLO_KGRS_Potassium_2ppd": ("potassium", "P2"),
        "KPLO_KGRS_Uranium_2ppd": ("uranium", "P2"),
        "KPLO_KGRS_Thermal_Neutron_2ppd": ("thermal_neutron_proxy", "P2"),
        "KGRS_Th_LG_smooth": ("thorium_alternative", "P2"),
        "np_weh_5km_adapt_map": ("hydrogen_proxy", "P1"),
        "sp_weh_5km_adapt_map": ("hydrogen_proxy", "P1"),
        "np_ice_depth_new_240m_mat_today_27_Oct_2016": ("modeled_ice_stability_depth", "P1"),
        "sp_ice_depth_new_240m_mat_today_27_Oct_2016": ("modeled_ice_stability_depth", "P1"),
        "np_ice_depth_new_240m_mat_paleo_27_Oct_2016": ("paleo_ice_model", "P3"),
        "LRO_MiniRF_CPR_NPole70_118mp": ("radar_cpr", "P2"),
        "LRO_MiniRF_CPR_SPole70_118mp": ("radar_cpr", "P2"),
        "LRO_MiniRF_S1_NPole70_118mp": ("radar_backscatter", "P2"),
        "LRO_MiniRF_S1_SPole70_118mp": ("radar_backscatter", "P2"),
        "minirf_s1_49dnorm_EQ": ("regional_radar_backscatter", "P2"),
        "dgdr_std_cf_clc_cyl_128_jp2": ("christiansen_feature", "P2"),
        "dgdr_nen_cf_clc_cyl_032_jp2": ("christiansen_feature_alternative", "P2"),
        "grain_density_310_eq": ("crustal_grain_density_model", "P3"),
        "LOLA_PA_310_eq": ("surface_relief_geophysics", "P3"),
        "Model2_thick_eq": ("crustal_thickness_model", "P3"),
        "Model2_cmi_eq": ("crust_mantle_interface_model", "P3"),
        "gggrx_1200a_anom_l180_eq": ("gravity_anomaly", "P3"),
        "gggrx_1200a_boug_l180_eq": ("bouguer_anomaly", "P3"),
        "gggrx_1200a_geoid_l660_eq": ("geoid_model", "P3"),
        "Ravatetal_JGR2020_LunarL1MagneticModelFields_LPgrad_Total_Field": ("magnetic_field_model", "P3"),
        "PSR_SP": ("permanent_shadow_polygon", "P1"),
        "PSR_NP": ("permanent_shadow_polygon", "P1"),
        "NP_IceLoc_280m": ("surface_ice_detection", "P1"),
        "SP_IceLoc_280m": ("surface_ice_detection", "P1"),
        "LRO_LOLA_MaxIlluminationSites_10m_SP": ("illumination_sites", "P1"),
        "Unified_Global_Geologic_Map_of_the_Moon_Geologic_Units": ("geologic_unit", "P1"),
    }
    mars_features = {
        "SIM3292_Global_Geology": "geologic_unit",
        "Hydrous_Mineral_Detections_OMEGA_CRISM_clon0": "hydrous_mineral_detection",
        "Aquieous_Mineral_Distribution_Options_clon0": "aqueous_mineral_class",
        "Chloride_Salt_bearing_Material_THEMIS_clon0": "chloride_detection",
        "Dune_Field": "dune_field",
        "Hynek_Valley_Networks": "valley_network",
        "Mars_Fans_Glaciers_Gullies": "fans_glaciers_gullies",
    }
    for body, server, inventory in inventories:
        for service in inventory.get("data", {}).get("services", []):
            name = service["name"]
            match = exact.get(name) if body == "moon" else None
            if body == "mars" and name in mars_features and service["type"] == "MapServer":
                match = (mars_features[name], "P1")
            if body == "moon" and name.startswith("Lunar_Kaguya_MIMap_MineralDeconv_"):
                match = ("mineralogy_fit_quality" if "WeightedCriteria" in name else "mineralogy", "P2")
            if body == "moon" and name.startswith("diviner_") and not name.endswith(("_NP", "_SP")):
                match = ("thermal_anisothermality" if "_c" in name else "thermal_anomaly" if "anom" in name else "surface_temperature", "P0" if name in ["diviner_tbol_hour00", "diviner_tbol_hour12", "diviner_tbol_max", "diviner_tbol_min"] else "P2")
            if body == "moon" and re.match(r"AVGVISIB_(65[NS]_240M|85[NS]_060M)_201608", name):
                match = ("earth_visibility" if "EARTH" in name else "solar_illumination", "P1")
            if body == "mars" and service["type"] == "ImageServer":
                is_dem = "dem" in name.lower() or name.startswith("mola") or name.startswith("DTEEC")
                match = ("terrain" if is_dem else "rendered_imagery", "P0" if is_dem else "excluded")
                if name.endswith(("_NP", "_SP")):
                    continue
            if match:
                family, priority = match
                selected.append({"body": body, "server": server, "name": name,
                                 "type": service["type"], "family": family, "priority": priority,
                                 "url": f"https://trek.nasa.gov/{body}/{server}/rest/services/{name}/{service['type']}"})
    return selected


def sample_points(metadata):
    extent = metadata.get("extent", metadata.get("fullExtent", {}))
    if not all(k in extent for k in ["xmin", "ymin", "xmax", "ymax"]):
        return [], "unknown"
    xmin, ymin, xmax, ymax = (extent[k] for k in ["xmin", "ymin", "xmax", "ymax"])
    geographic = xmin >= -181 and xmax <= 361 and ymin >= -91 and ymax <= 91
    if geographic and xmin <= 0 <= xmax and ymin <= 0 <= ymax:
        points = [[0, 0], [23.47, .67], [15, -30]]
        points = [p for p in points if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
    else:
        points = [[xmin + (xmax - xmin) * fx, ymin + (ymax - ymin) * fy]
                  for fx, fy in [(.5, .5), (.3, .4), (.7, .6), (.4, .7), (.6, .3)]]
    return points, "native_geographic_degrees" if geographic else "native_projected_coordinates"


def numeric(value):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > -1e30 and parsed not in [-32768, -99999] else None
    except ValueError:
        return None


def inspect(candidate):
    result = dict(candidate)
    metadata = request(candidate["url"] + "?f=pjson")
    result["metadata"] = metadata
    data = metadata.get("data", {})
    if metadata.get("error") or data.get("error"):
        result["status"] = "metadata_unavailable"
        return result
    if candidate["type"] == "MapServer":
        result["status"] = "feature_catalog_only"
        result["layers"] = data.get("layers", [])
        # Query the first actual feature layer, not a group or raster layer.
        layer = next((layer for layer in data.get("layers", []) if layer.get("subLayerIds") is None), None)
        if layer:
            result["feature_layer"] = request(f"{candidate['url']}/{layer['id']}?f=pjson")
            detail = result["feature_layer"].get("data", {})
            if detail.get("type") == "Feature Layer":
                result["feature_probe"] = request(url_query(f"{candidate['url']}/{layer['id']}/query", {
                    "f": "json", "objectIds": "0,1", "outFields": "*", "returnGeometry": "true"}))
                result["status"] = "feature_query_verified" if result["feature_probe"].get("data", {}).get("features") else "feature_query_unverified"
                features = result["feature_probe"].get("data", {}).get("features", [])
                if features:
                    geometry = features[0].get("geometry", {})
                    if "x" in geometry:
                        point = [geometry["x"], geometry["y"]]
                    else:
                        rings = geometry.get("rings", geometry.get("paths", []))
                        point = rings[0][0] if rings else None
                    if point:
                        probe_geometry = {"x": point[0], "y": point[1]}
                        sr = result["feature_probe"]["data"].get("spatialReference")
                        if sr:
                            probe_geometry["spatialReference"] = sr
                        result["spatial_feature_probe"] = request(url_query(f"{candidate['url']}/{layer['id']}/query", {
                            "f": "json", "geometry": json.dumps(probe_geometry),
                            "geometryType": "esriGeometryPoint", "spatialRel": "esriSpatialRelIntersects",
                            "returnCountOnly": "true"}))
                        if result["spatial_feature_probe"].get("data", {}).get("count", 0) > 0:
                            result["status"] = "spatial_feature_query_verified"
        return result
    result["reported_pixel_spacing"] = {"x": data.get("pixelSizeX"), "y": data.get("pixelSizeY")}
    if candidate["family"] == "rendered_imagery":
        result["status"] = "rendered_imagery_excluded"
        return result
    if data.get("bandCount") != 1:
        result["status"] = "multiband_not_validated_physical"
        return result
    points, coordinates = sample_points(data)
    result["probe_coordinates"] = {"system": coordinates, "points": points}
    if not points:
        result["status"] = "no_probe_geometry"
        return result
    probe = request(url_query(candidate["url"] + "/getSamples", {
        "f": "json", "geometry": json.dumps({"points": points}, separators=(",", ":")),
        "geometryType": "esriGeometryMultipoint", "returnFirstValueOnly": "true",
        "interpolation": "RSP_NearestNeighbor"}))
    result["probe"] = probe
    samples = probe.get("data", {}).get("samples", [])
    result["numeric_sample_count"] = sum(numeric(sample.get("value")) is not None for sample in samples)
    result["status"] = "numeric_response_verified" if result["numeric_sample_count"] else "no_numeric_sample_at_probe_locations"
    # All-zero rasters and unit/scaling conflicts still require interpretation;
    # a numeric HTTP 200 response alone is not scientific validation.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry", action="store_true", help="Reuse successful evidence; retry failed samples and feature queries")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventories = []
    for body, server in SERVERS:
        inventory = request(f"https://trek.nasa.gov/{body}/{server}/rest/services?f=pjson")
        inventories.append((body, server, inventory))
    selected = candidates(inventories)
    old_path = OUT / "trek-audit.json"
    previous = {s["url"]: s for s in json.loads(old_path.read_text())["services"]} if args.retry and old_path.exists() else {}
    def check(candidate):
        old = previous.get(candidate["url"])
        if old and candidate["family"] == "rendered_imagery":
            return {**old, **candidate, "status": "rendered_imagery_excluded"}
        if old and old["status"] in ["numeric_response_verified", "spatial_feature_query_verified"]:
            return {**old, **candidate}
        return inspect(candidate)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        services = list(executor.map(check, selected))
    output = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Curated useful numeric/feature families, not every regional DEM or image/projection replica",
              "authentication": "No API key, token, cookies or login supplied",
              "method": "Metadata plus up to five native-coordinate samples per scalar service; up to two feature records and a spatial count query for selected vector services. No full rasters downloaded. Individual requests carry their observation time.",
              "warning": "Numeric response does not prove physical units, original instrument resolution, valid global coverage, or pixel-aligned area statistics. See README and variable definitions.",
              "discovery": [{"body": b, "server": s, "response": r} for b, s, r in inventories],
              "services": services}
    (OUT / "trek-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    counts = {}
    for service in services:
        counts[service["status"]] = counts.get(service["status"], 0) + 1
    print(json.dumps({"services_checked": len(services), "statuses": counts, "output": str(OUT / "trek-audit.json")}))


if __name__ == "__main__":
    main()
