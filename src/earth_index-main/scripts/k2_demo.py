# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio==1.5.2", "numpy==2.5.4", "exactextract==0.3.0", "shapely==2.2.0", "pyproj==3.8.0", "h3==4.5.0", "pyhdf==0.11.7"]
# ///
"""Bounded k²-raster experiment, separate from the production/pilot database.

Prepare: uv run scripts/k2_demo.py --prepare --credentials-file /private/file
Repeat codec benchmark without network/auth: uv run scripts/k2_demo.py
Build the external adapter first: bash scripts/k2_demo_build.sh
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from pyproj import Transformer
from rasterio.transform import array_bounds, Affine
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, mapping

from process_pilot_terrain import derivatives, https_url
from process_native_temperature import CRS_WKT, decode, download, original_token
from process_nasa_temperature import kelvin, quality_screen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/k2-demo"
N = 100
PIN = "fac218aab831f2611b5ae94f09f0ef12d1cd29ea"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def grid_features(site, target_crs):
    project = Transformer.from_crs(site["construction_crs"], target_crs, always_xy=True)
    result = []
    for r in range(N):
        for c in range(N):
            x, y = -50000 + c * 1000, 50000 - r * 1000
            # Densify edges before projecting into the source raster CRS.
            corners = [(x,y),(x+1000,y),(x+1000,y-1000),(x,y-1000),(x,y)]
            coords = []
            for a,b in zip(corners,corners[1:]):
                coords.extend(project.transform(a[0]+(b[0]-a[0])*t/4,
                                                a[1]+(b[1]-a[1])*t/4) for t in range(4))
            result.append({"type":"Feature", "properties":{"cell_id":str(r*N+c)},
                           "geometry":mapping(Polygon(coords))})
    return result


def summarize(array, transform, crs, cells, geographic=False):
    bounds = array_bounds(*array.shape, transform)
    raster = NumPyRasterSource(array, *bounds, nodata=np.nan, srs_wkt=crs)
    weight = "area_spherical_m2" if geographic else "area_cartesian"
    rows = exact_extract(raster, cells, [f"mean(coverage_weight={weight})",
        f"valid=count(coverage_weight={weight})",
        f"total=count(default_value=0,coverage_weight={weight})"], include_cols=["cell_id"])
    means = np.full(N*N,np.nan,dtype=np.float64)
    fractions = np.zeros(N*N,dtype=np.float64)
    for row in rows:
        p = row["properties"]; i=int(p["cell_id"])
        if p["valid"] > 0:
            means[i]=p["mean"]
            fractions[i]=p["valid"]/p["total"]
    if np.any(fractions>1.000001):
        raise ValueError("Invalid support fraction")
    return means.reshape(N,N), fractions.reshape(N,N)


def prepare(credentials):
    inventory_path=ROOT/"research/pilots/inventory.json"
    site=next(s for s in json.loads(inventory_path.read_text())["sites"] if s["key"]=="atacama")
    cells=grid_features(site,"EPSG:4326")
    # This pilot is wholly covered by one DEM asset. Fail rather than silently
    # accepting a new site with multiple assets and unhandled seam/merge rules.
    items=site["products"]["cop-dem-glo-30"]["items"]
    if len(items)!=1: raise ValueError("Demo expects one covering DEM asset")
    bounds=Polygon(site["geometry"]["coordinates"][0]).bounds
    uri=https_url(items[0]["assets"]["data"]["href"])
    print("Reading the existing Atacama DEM source window",flush=True)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="45"):
        with rasterio.open(uri) as src:
            if src.crs.to_epsg()!=4326 or src.transform.b or src.transform.d:
                raise ValueError("Expected north-up WGS84 DEM")
            f=from_bounds(*bounds,transform=src.transform)
            left=max(0,math.floor(f.col_off)-1); top=max(0,math.floor(f.row_off)-1)
            right=min(src.width,math.ceil(f.col_off+f.width)+1)
            bottom=min(src.height,math.ceil(f.row_off+f.height)+1)
            window=Window(left,top,right-left,bottom-top)
            z=src.read(1,window=window,masked=True).astype("float32").filled(np.nan)
            transform=src.window_transform(window)
            source={"uri":uri,"window":[left,top,right-left,bottom-top],
                    "shape":list(z.shape),"float32_sha256":digest(z.tobytes()),
                    "window_transform":list(transform),"crs":src.crs.to_string()}
    slope=derivatives(z,transform)["slope"]
    flat=np.where(np.isfinite(slope),(slope<5).astype("float32"),np.nan)
    elevation,dem_coverage=summarize(z,transform,src.crs.to_wkt(),cells,True)
    slope_mean,slope_coverage=summarize(slope,transform,src.crs.to_wkt(),cells,True)
    flat_fraction,_=summarize(flat,transform,src.crs.to_wkt(),cells,True)
    if dem_coverage.min()<.999 or slope_coverage.min()<.999:
        raise ValueError("Demo expects complete terrain support")
    del z,slope,flat
    print("Reading one existing MOD11A2 period (2024-01-01 through 2024-01-08)",flush=True)
    tcells=grid_features(site,CRS_WKT)
    tb=np.array([Polygon(f["geometry"]["coordinates"][0]).bounds for f in tcells])
    bbox=(tb[:,0].min(),tb[:,1].min(),tb[:,2].max(),tb[:,3].max())
    granules=[g for g in site["products"]["MOD11A2"]["granules"] if ".A2024001." in g["title"]]
    if len(granules)!=1: raise ValueError("Expected exactly one temperature tile")
    token=original_token(credentials)
    payload,meta=download(granules[0],token)
    decoded,meta=decode(payload,meta,bbox)
    del payload,token
    day=quality_screen(kelvin(*decoded["LST_Day_1km"]),decoded["QC_Day"][0])
    tr=Affine(*decoded["LST_Day_1km"][1]["transform"][:6])
    temperature,tcoverage=summarize(day,tr,CRS_WKT,tcells)
    layers={
        "elevation_mean_m":(elevation,1.0,"m"),
        "slope_mean_deg":(slope_mean,.1,"degree"),
        "slope_lt_5deg_fraction":(flat_fraction,.001,"1"),
        "day_temperature_k":(temperature,.02,"K"),
        "day_temperature_valid_fraction":(tcoverage,.001,"1"),
    }
    manifest={"site":"atacama","shape":[N,N],"cell_width_m":1000,"cell_height_m":1000,
        "crs":site["construction_crs"],"transform":[1000,0,-50000,0,-1000,50000],
        "row_order":"north to south; columns west to east", "cells":N*N,
        "projected_cell_area_m2":1000000,
        "area_note":"Local WGS84 AEQD metre squares; geodesic ground areas differ slightly.",
        "prepared_at":datetime.now(timezone.utc).isoformat(),
        "inventory_sha256":digest(inventory_path.read_bytes()),"dem":source,"temperature":meta,
        "temperature_window":["2024-01-01","2024-01-09"],
        "temperature_native_pixel_side_m":abs(tr.a),
        "terrain_minimum_valid_fraction":float(min(dem_coverage.min(),slope_coverage.min())),
        "methods":["Native DEM Horn 3x3 slope, then area-weighted mean at 1 km; slope is not computed from the averaged DEM.",
            "Flat fraction: native terrain support with slope <5 degrees, aggregated by area.",
            "Fractional pixel/polygon intersections; spherical area weights for DEM and Cartesian equal-area sinusoidal weights for MODIS.",
            "MOD11A2 native QA byte exactly 0; missing samples excluded; valid footprint fraction retained separately.",
            "DEM is mixed-epoch surface elevation, EGM2008. Temperature is one 8-day daytime LST composite, not air temperature.",
            "Aligned 1 km summaries preserve co-location at this footprint; no guarantee of sub-km co-location or contiguity."],
        "source_raster_files_retained":0,"layers":{}}
    for name,(a,step,unit) in layers.items():
        valid=np.isfinite(a); rounded=np.rint(a[valid]/step).astype(np.int64)
        origin=int(rounded.min())-1
        encoded=np.zeros(a.shape,dtype="<u2")
        if rounded.max()-origin>65535: raise ValueError("16-bit overflow")
        encoded[valid]=rounded-origin
        decoded_values=(encoded[valid].astype(np.float64)+origin)*step
        max_error=float(np.abs(decoded_values-a[valid]).max())
        if max_error>step/2+1e-9: raise ValueError("Quantization error exceeded")
        raw=encoded.tobytes(); compressed=zlib.compress(raw,9)
        (OUT/f"{name}.u16").write_bytes(raw)
        (OUT/f"{name}.u16.zlib").write_bytes(compressed)
        manifest["layers"][name]={"unit":unit,"quantum":step,"integer_origin":origin,
            "decode":"physical_value=(code+integer_origin)*quantum; code 0 is missing",
            "dtype":"uint16 little-endian", "missing_code":0,"valid_cells":int(valid.sum()),
            "physical_min":float(a[valid].min()),"physical_max":float(a[valid].max()),
            "max_quantization_error":max_error,"raw16_bytes":len(raw),"zlib16_bytes":len(compressed),
            "raw16_sha256":digest(raw)}
    save(OUT/"grid.json",manifest)


def code_range(meta,lo,hi):
    # Range comparisons are against the quantized physical value.
    return (max(1,math.ceil(lo/meta["quantum"]-meta["integer_origin"]-1e-8)),
            min(65535,math.floor(hi/meta["quantum"]-meta["integer_origin"]+1e-8)))


def benchmark(binary):
    grid=json.loads((OUT/"grid.json").read_text())
    definitions={
        "elevation_mean_m":[("high_elevation",3500,10000)],
        "slope_mean_deg":[("flat_mean_slope",0,5)],
        "slope_lt_5deg_fraction":[("mostly_flat",.9,1)],
        "day_temperature_k":[("hot",310,500)],
        "day_temperature_valid_fraction":[("temperature_supported",.95,1)],
    }
    report={"upstream_commit":PIN,"codec":"Authors' v3.1 C++ SDSL implementation",
        "adapter_sha256":digest((ROOT/"scripts/k2_demo_query.cpp").read_bytes()),
        "benchmark_script_sha256":digest(Path(__file__).read_bytes()),
        "binary_sha256":digest(binary.read_bytes()),
        "upstream_changes":"Member-name typo fixes only, recorded in scripts/k2_demo_build.sh; query/codec algorithms unchanged.",
        "parameters":{"k1":2,"k2":2,"level_k1":0,"heuristic_plain_levels":1},
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "benchmark_note":"Median of 7 batches of 200 warm in-process queries. Same returned coordinates; index load/I/O excluded. Tiny grid, not a global throughput benchmark.",
        "grid_metadata_bytes":(OUT/"grid.json").stat().st_size,"layers":{},"queries":{}}
    masks={}; total_pixels=0
    with tempfile.TemporaryDirectory(prefix="earth-index-k2-") as scratch:
        for name,meta in grid["layers"].items():
            raw=(OUT/f"{name}.u16").read_bytes()
            if digest(raw)!=meta["raw16_sha256"]: raise ValueError("Input hash mismatch")
            if zlib.decompress((OUT/f"{name}.u16.zlib").read_bytes())!=raw:
                raise ValueError("zlib roundtrip failed")
            array=np.frombuffer(raw,dtype="<u2").reshape(N,N)
            raw32=Path(scratch)/"input.i32"; raw32.write_bytes(array.astype("<i4").tobytes())
            queries=[]; query_names=[]
            for label,lo,hi in definitions[name]:
                low,high=code_range(meta,lo,hi)
                queries.append((0,N-1,0,N-1,low,high)); query_names.append(label)
            # Spatial+value query, out-of-range query, and seeded random cases.
            queries.append((20,39,30,49,*queries[0][4:])); query_names.append("spatial_window")
            queries.append((0,N-1,0,N-1,65535,65535)); query_names.append("empty_range")
            rng=np.random.default_rng(42)
            for i in range(12):
                r0,r1=sorted(rng.integers(0,N,2));c0,c1=sorted(rng.integers(0,N,2))
                lo,hi=sorted(rng.integers(int(array.min()),int(array.max())+1,2))
                queries.append((int(r0),int(r1),int(c0),int(c1),int(lo),int(hi)))
                query_names.append(f"random_{i}")
            qpath=Path(scratch)/"queries.txt"
            qpath.write_text("\n".join(" ".join(map(str,q)) for q in queries)+"\n")
            sizes={**meta,"raw32_bytes":array.size*4}; codec_hits=None
            for typ,ext in [(1,"k2r"),(11,"k2h")]:
                result=subprocess.run([str(binary),str(raw32),str(N),str(N),str(OUT/f"{name}.{ext}"),str(qpath),str(typ)],
                    check=True,capture_output=True,text=True)
                measured=json.loads(result.stdout)
                if measured["bytes"]!=(OUT/f"{name}.{ext}").stat().st_size:
                    raise ValueError("Serialized size mismatch")
                total_pixels+=measured["pixels_verified"]
                compact=[]; all_hits=[]
                for label,q,out in zip(query_names,queries,measured["queries"]):
                    r0,r1,c0,c1,lo,hi=q
                    expected=np.zeros((N,N),dtype=bool)
                    expected[r0:r1+1,c0:c1+1]=((array[r0:r1+1,c0:c1+1]>=lo)&(array[r0:r1+1,c0:c1+1]<=hi))
                    hits=out.pop("cells"); all_hits.append(hits)
                    if hits!=np.flatnonzero(expected).tolist(): raise ValueError("Independent NumPy query check failed")
                    compact.append({"name":label,"bounds_codes":q,**out})
                if codec_hits is not None and codec_hits!=all_hits: raise ValueError("Codec mismatch")
                codec_hits=all_hits
                sizes[ext]={**{k:v for k,v in measured.items() if k!="queries"},"queries":compact,
                            "sha256":digest((OUT/f"{name}.{ext}").read_bytes())}
            masks[query_names[0]]=set(codec_hits[0])
            report["layers"][name]=sizes
            print(f"{name}: raw16={len(raw)}, k2={sizes['k2r']['bytes']}, heuristic={sizes['k2h']['bytes']}, zlib16={meta['zlib16_bytes']}",flush=True)
    hot=masks["hot"]&masks["temperature_supported"]
    hot_flat=hot&masks["flat_mean_slope"]
    hot_mostly_flat=hot&masks["mostly_flat"]
    report["queries"]={"hot_supported":len(hot),"hot_and_mean_slope_le_5":len(hot_flat),
        "hot_and_90pct_native_terrain_slope_lt_5":len(hot_mostly_flat),
        "joint_query_method":"Intersect exact coordinate sets from per-band compressed-index queries. Joint query not separately timed.",
        "selected_cells":sorted(hot_mostly_flat)}
    # Largest four-neighbour component, rather than equating total area with a large contiguous site.
    pending=set(hot_mostly_flat); largest=0; components=0
    while pending:
        todo=[pending.pop()]; size=0; components+=1
        while todo:
            p=todo.pop();size+=1;r,c=divmod(p,N)
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                n=rr*N+cc
                if 0<=rr<N and 0<=cc<N and n in pending: pending.remove(n);todo.append(n)
        largest=max(largest,size)
    report["queries"].update(components=components,largest_4_neighbour_component_cells=largest)
    report["validation"]={"status":"passed","serialized_pixel_values_checked":total_pixels,
        "range_queries_checked":len(grid["layers"])*2*15,"window_pixels_checked":len(grid["layers"])*2*260,
        "joint_masks_checked_against":"Per-band NumPy brute-force masks; both codecs agree"}
    report["totals"]={key:sum(v[key] for v in report["layers"].values()) for key in ("raw16_bytes","raw32_bytes","zlib16_bytes")}
    for key in ("k2r","k2h"): report["totals"][key+"_bytes"]=sum(v[key]["bytes"] for v in report["layers"].values())
    save(OUT/"results.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--credentials-file",type=Path)
    parser.add_argument("--binary",type=Path,default=Path("/tmp/earth-index-k2-upstream/build/bin/earth_index_demo"))
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        if args.credentials_file is None: parser.error("--prepare needs --credentials-file")
        prepare(args.credentials_file)
    report=benchmark(args.binary)
    print(json.dumps({"totals":report["totals"],"queries":report["queries"],"validation":report["validation"]},indent=2))


if __name__=="__main__": main()
