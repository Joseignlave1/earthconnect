#!/usr/bin/env python3
"""Validate saved planetary evidence and record important source inconsistencies."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "research/planetary-sources"
trek = json.loads((OUT / "trek-audit.json").read_text())
mars = json.loads((OUT / "mars-cutout-audit.json").read_text())
catalog = json.loads((OUT / "catalog.json").read_text())

ids = [v["id"] for v in catalog["variables"]]
assert len(ids) == len(set(ids)), "Duplicate variable IDs"
for variable in catalog["variables"]:
    if variable["readiness"].startswith("verified_transport"):
        assert variable["sources"], variable["id"]
        assert any(source["status"] in ["numeric_response_verified", "numeric_cutout_verified", "spatial_feature_query_verified"] for source in variable["sources"]), variable["id"]

moon = {s["name"]: s for s in trek["services"] if s["body"] == "moon"}
prefix = "Lunar_Kaguya_MIMap_MineralDeconv_"
mineral_samples = [moon[prefix + name + "_50N50S"]["probe"]["data"]["samples"]
                   for name in ["OlivinePercent", "ClinopyroxenePercent", "OrthopyroxenePercent", "PlagioclasePercent"]]
mineral_sums = [sum(float(samples[i]["value"]) for samples in mineral_samples) for i in range(3)]
assert all(abs(value - 1) < 1e-6 for value in mineral_sums)

layers = {s["layer"]: s for s in mars["layers"]}
minimum = layers["mars_yearly_minimum_temperature_celsius"]["values"]
maximum = layers["mars_yearly_maximum_temperature_celsius"]["values"]
published_range = layers["mars_yearly_temperature_range_celsius"]["values"]
assert len(minimum) == len(maximum) == len(published_range) == 64
range_residual = max(abs(actual - (high - low) + 273.15) for low, high, actual in zip(minimum, maximum, published_range))
assert all(low <= high for low, high in zip(minimum, maximum))
assert range_residual < .001, "Temperature-range source behavior changed; investigate before using it"

for service in trek["services"]:
    if service["family"] in ["solar_illumination", "earth_visibility"] and service["status"] == "numeric_response_verified":
        assert all(0 <= float(sample["value"]) <= 1 for sample in service["probe"]["data"]["samples"])

result = {
    "artifact_integrity": "passed",
    "transport_summary": {
        "trek_services_checked": len(trek["services"]),
        "trek_numeric_responses": sum(s["status"] == "numeric_response_verified" for s in trek["services"]),
        "trek_spatial_feature_queries": sum(s["status"] == "spatial_feature_query_verified" for s in trek["services"]),
        "mars_cutouts_checked": len(mars["layers"]),
        "mars_numeric_cutouts": sum(s["status"] == "numeric_cutout_verified" for s in mars["layers"]),
    },
    "findings": [
        {"id": "lunar_mineral_scale", "status": "normalization_review_required",
         "observed_four_mineral_sums": mineral_sums,
         "interpretation": "Raw mineral channels behave as fractions summing to one; product prose uses wt%. Do not present raw values as percentages."},
        {"id": "mars_temperature_range_offset", "status": "source_layer_rejected",
         "pixels_compared": 64, "offset": -273.15, "max_offset_residual": range_residual,
         "interpretation": "Published Celsius range equals max-minus-min minus273.15 in this cutout. Derive a valid difference from the extrema; do not use the range layer directly."},
        {"id": "lunar_mineral_grid", "status": "service_grid_differs_from_product_imagery",
         "service_spacing_degree": .25, "source_imagery_ppd": 512,
         "interpretation": "Actual samples and metadata report a much coarser numeric grid than product imagery. This is service spacing, not proof of effective instrument resolution."},
        {"id": "polar_visibility_fractions", "status": "probe_values_in_fraction_range",
         "interpretation": "All tested numeric visibility samples are within0..1. This does not validate the illumination model or extend coverage outside the polar products."},
    ],
    "not_validated": ["Full spatial coverage", "All source-specific quality/nodata masks", "Every raw-to-physical scale", "Cross-body comparability", "Production reliability and latency", "4km area aggregation implementation"],
}
(OUT / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result["transport_summary"]))
