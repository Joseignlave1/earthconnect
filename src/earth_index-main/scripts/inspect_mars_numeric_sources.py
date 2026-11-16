#!/usr/bin/env python3
"""Verify tiny anonymous ASU/JMARS numeric cutouts, never full global rasters."""

import concurrent.futures
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import urllib.parse
import urllib.request

OUT = Path(__file__).resolve().parents[1] / "research/planetary-sources"
BASE = "https://ms-mars.mars.asu.edu/alb32"
LAYERS = [
    "TES_Lambert_Albedo_numeric", "TES_Thermal_Inertia_numeric",
    "thm_ti_100m_32bit", "TES_dust_index_16ppd_numeric", "TES_Dust_numeric",
    "mars_yearly_avg_temperature_celsius", "mars_yearly_minimum_temperature_celsius",
    "mars_yearly_maximum_temperature_celsius", "mars_yearly_temperature_range_celsius",
    "mars_odyssey_grs_mons_perc_wt", "mgs_geologic_map_mars_usgs_numeric",
    "TES_Olivine_numeric", "TES_Feldspar_numeric", "TES_Clinopyroxene_numeric",
    "TES_Carbonate_numeric", "TES_Hematite_numeric", "TES_Amphibole_numeric",
    "TES_Quartz_numeric", "TES_Sulfate_numeric", "TES_Plagioclase_numeric",
    "TES_Low_Ca_Pyroxene_numeric", "TES_High_Ca_Pyroxene_numeric",
    "TES_RMS_Error_numeric", "TES_Sheet_Silicates_numeric",
    "TES_TI_Putzig_night_numeric",
]


def decode_vicar(payload):
    """Accept a narrow, explicitly verified BSQ single-band numeric VICAR layout."""
    size = re.match(rb"LBLSIZE\s*=\s*(\d+)", payload)
    if not size:
        raise ValueError("Not a VICAR numeric raster")
    label_size = int(size[1])
    label = payload[:label_size].decode("ascii")
    fields = {}
    for key, quoted, plain in re.findall(r"([A-Z0-9_]+)\s*=\s*(?:'([^']*)'|([^\s]+))", label):
        fields[key] = quoted or plain
    if fields.get("ORG") != "BSQ" or int(fields.get("NB", "0")) != 1 or int(fields.get("NBB", "0")) != 0:
        raise ValueError("Unsupported VICAR layout")
    formats = {"REAL": ("f", 4), "DOUB": ("d", 8), "HALF": ("h", 2), "FULL": ("i", 4), "BYTE": ("B", 1)}
    code, width = formats[fields["FORMAT"]]
    if code in ["f", "d"]:
        if fields.get("REALFMT") not in ["RIEEE", "IEEE"]:
            raise ValueError("Unsupported floating-point representation")
        endian = "<" if fields["REALFMT"] == "RIEEE" else ">"
    else:
        endian = "<" if fields.get("INTFMT") == "LOW" else ">"
    nx, ny = int(fields["NS"]), int(fields["NL"])
    record_size = int(fields["RECSIZE"])
    offset = label_size + int(fields.get("NLB", "0")) * record_size
    if record_size != nx * width or len(payload) < offset + ny * record_size:
        raise ValueError("Truncated or unsupported VICAR row layout")
    values = list(struct.unpack(endian + str(nx * ny) + code, payload[offset:offset + nx * ny * width]))
    return fields, values


def probe(layer):
    params = {"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": layer,
              "STYLES": "", "SRS": "JMARS:4", "BBOX": "0,0,1,1", "WIDTH": 8, "HEIGHT": 8,
              "FORMAT": "image/vicar"}
    # This endpoint rejected percent-encoded MIME slashes during the audit.
    url = BASE + "?" + urllib.parse.urlencode(params, safe=":/,")
    result = {"layer": layer, "url": url, "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "request_bbox_native": [0, 0, 1, 1], "requested_shape": [8, 8]}
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            body = response.read(65537)
            result.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                          bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            if len(body) > 65536:
                raise ValueError("Audit response exceeds bounded cutout size")
            if not body.startswith(b"LBLSIZE"):
                result["status"] = "no_numeric_raster"
                result["response_text"] = body.decode("utf-8", errors="replace")[:2000]
            else:
                fields, values = decode_vicar(body)
                result.update(status="numeric_cutout_verified", vicar_header=fields,
                              values=[value if math.isfinite(value) else None for value in values])
    except (OSError, ValueError, KeyError, struct.error) as error:
        result.update(status="request_failed", error=str(error))
    result["elapsed_ms"] = round(1000 * (time.monotonic() - start))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        layers = list(executor.map(probe, LAYERS))
    result = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "authentication": "Anonymous requests; no key, token or cookies",
              "method": "8x8 VICAR numeric cutouts only; no full global datasets",
              "warning": "Successful decoding is transport verification, not validation of units, NODATA, longitude convention, effective resolution or scientific quality; see catalog and README.",
              "layers": layers}
    (OUT / "mars-cutout-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checked": len(layers), "numeric_cutouts": sum(l["status"] == "numeric_cutout_verified" for l in layers)}))


if __name__ == "__main__":
    main()
