#!/usr/bin/env python3
"""Build compact spatial index mapping Earth's 256 ECV3 packs (16x16 grid: 11.25° x 22.5°)
to intersecting Copernicus DEM GLO-30 tiles and ESA Sentinel-2 Quarterly Cloudless MGRS tiles.

Output: scripts/ecv3/earth_sources_index.json.gz
"""

from __future__ import annotations

import gzip
import json
import math
import re
import urllib.request
from pathlib import Path

import duckdb
import mgrs

COP_LIST_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/tileList.txt"
S2_MANIFEST_BASE = (
    "https://data.source.coop/tge-labs/sentinel-2-quarterly-cloudless-mosaics/manifest"
)

BANDS = (
    16  # 11.25° latitude per band (band 0 = [78.75°, 90°], band 15 = [-90°, -78.75°])
)
LON_BLKS = 32  # 11.25° longitude per block (blk 0 = [-180°, -168.75°], blk 31 = [168.75°, 180°])
LAT_STEP = 180.0 / BANDS
LON_STEP = 360.0 / LON_BLKS

COP_RE = re.compile(r"^Copernicus_DSM_COG_10_([NS])(\d{2})_00_([EW])(\d{3})_00_DEM$")


def pack_bbox(pack_id: int) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) in degrees for pack_id in 0..255."""
    band = pack_id // LON_BLKS
    lon_blk = pack_id % LON_BLKS
    lat_max = 90.0 - band * LAT_STEP
    lat_min = lat_max - LAT_STEP
    lon_min = -180.0 + lon_blk * LON_STEP
    lon_max = lon_min + LON_STEP
    return lat_min, lat_max, lon_min, lon_max


def lon_intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Check if longitude intervals overlap modulo 360°."""
    for shift in (-360.0, 0.0, 360.0):
        if a0 + shift < b1 and a1 + shift > b0:
            return True
    return False


def fetch_copernicus_tiles() -> list[tuple[str, float, float]]:
    print("Fetching Copernicus DEM GLO-30 tileList.txt ...")
    with urllib.request.urlopen(COP_LIST_URL, timeout=60) as r:
        lines = r.read().decode("utf-8").splitlines()
    out: list[tuple[str, float, float]] = []
    for raw in lines:
        name = raw.strip().rstrip("/")
        m = COP_RE.match(name)
        if not m:
            continue
        ns, lat_s, ew, lon_s = m.groups()
        lat = int(lat_s) * (1 if ns == "N" else -1)
        lon = int(lon_s) * (1 if ew == "E" else -1)
        out.append((name, float(lat), float(lon)))
    print(f"  Parsed {len(out):,} Copernicus 1°x1° tiles.")
    return out


def fetch_sentinel2_quarter(quarter: str) -> list[str]:
    url = f"{S2_MANIFEST_BASE}/manifest_{quarter}.parquet"
    print(f"Querying Sentinel-2 manifest {quarter} ...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(
        f"""
        SELECT split_part(destination_key, '/', 4) AS tile_folder
        FROM read_parquet('{url}')
        WHERE band IN ('B02', 'B03', 'B04') AND size_bytes > 0
        GROUP BY tile_folder
        HAVING COUNT(DISTINCT band) = 3
        ORDER BY tile_folder
        """
    ).fetchall()
    folders = [r[0] for r in rows if r[0]]
    print(f"  Found {len(folders):,} complete RGB MGRS tiles in {quarter}.")
    return folders


def main() -> None:
    cop_tiles = fetch_copernicus_tiles()
    s2_q3 = set(fetch_sentinel2_quarter("2025_Q3"))
    s2_q1 = set(fetch_sentinel2_quarter("2025_Q1"))

    # Pre-index MGRS centers for every unique tile_folder across Q3 and Q1
    # Latitude band centers for MGRS C..X fallback (when Sentinel-2 tile slightly crosses UTM zone boundary)
    MGRS_BAND_LAT = {
        "C": -76.0,
        "D": -68.0,
        "E": -60.0,
        "F": -52.0,
        "G": -44.0,
        "H": -36.0,
        "J": -28.0,
        "K": -20.0,
        "L": -12.0,
        "M": -4.0,
        "N": 4.0,
        "P": 12.0,
        "Q": 20.0,
        "R": 28.0,
        "S": 36.0,
        "T": 44.0,
        "U": 52.0,
        "V": 60.0,
        "W": 68.0,
        "X": 78.0,
    }
    converter = mgrs.MGRS()
    all_s2 = sorted(s2_q3 | s2_q1)
    s2_geo: dict[str, tuple[float, float, float, float]] = {}
    fb_geocoded = 0
    for folder in all_s2:
        code = folder.split("_")[0]
        try:
            clat, clon = converter.toLatLon(code)
            dlat = 1.25
            dlon = min(35.0, 1.45 / max(math.cos(math.radians(clat)), 0.10))
        except Exception:
            zone = int(code[:2])
            band_ch = code[2]
            clat = MGRS_BAND_LAT.get(band_ch, 0.0)
            clon = -180.0 + (zone - 1) * 6.0 + 3.0
            dlat = 5.5
            dlon = min(35.0, 6.0 / max(math.cos(math.radians(clat)), 0.12))
            fb_geocoded += 1
        s2_geo[folder] = (clat - dlat, clat + dlat, clon - dlon, clon + dlon)
    print(
        f"  Geocoded {len(s2_geo):,} unique Sentinel-2 MGRS tiles ({fb_geocoded} via zone-band fallback, 0 dropped)."
    )

    packs: dict[str, dict] = {}
    non_empty = 0
    for pid in range(BANDS * LON_BLKS):
        lat0, lat1, lon0, lon1 = pack_bbox(pid)
        band = pid // LON_BLKS
        # Pad Copernicus search by 0.08° for 2 px gutter across pack seams
        p_cop = []
        for name, tlat, tlon in cop_tiles:
            if tlat + 1.0 > lat0 - 0.08 and tlat < lat1 + 0.08:
                if lon_intervals_overlap(tlon, tlon + 1.0, lon0 - 0.08, lon1 + 0.08):
                    p_cop.append(name)

        # Copernicus DEM GLO-30 covers 100% of Earth's landmasses (90°S..90°N); 0 cop tiles = pure ocean
        if not p_cop:
            continue

        p_s2_pri = []
        p_s2_fb = []
        # Northern hemisphere (bands 0..7, lat >= 0): Q3 (Jul-Sep summer) primary, Q1 fallback
        # Southern hemisphere (bands 8..15, lat < 0): Q1 (Jan-Mar summer) primary, Q3 fallback
        pri_set, fb_set = (s2_q3, s2_q1) if band < 8 else (s2_q1, s2_q3)
        pri_q, fb_q = ("2025/Q3", "2025/Q1") if band < 8 else ("2025/Q1", "2025/Q3")

        for folder, (slat0, slat1, slon0, slon1) in s2_geo.items():
            if slat1 > lat0 - 0.05 and slat0 < lat1 + 0.05:
                if lon_intervals_overlap(slon0, slon1, lon0 - 0.05, lon1 + 0.05):
                    if folder in pri_set:
                        p_s2_pri.append(folder)
                    if folder in fb_set:
                        p_s2_fb.append(folder)

        non_empty += 1
        packs[str(pid)] = {
            "band": band,
            "lon_blk": pid % LON_BLKS,
            "bbox": [lat0, lat1, lon0, lon1],
            "cop": p_cop,
            "pri_q": pri_q,
            "fb_q": fb_q,
            "s2_pri": p_s2_pri,
            "s2_fb": p_s2_fb,
        }

    out_path = Path(__file__).parent / "earth_sources_index.json.gz"
    payload = {
        "bands": BANDS,
        "lon_blks": LON_BLKS,
        "pack_count": BANDS * LON_BLKS,
        "land_pack_count": non_empty,
        "packs": packs,
    }
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))

    sz_kb = out_path.stat().st_size / 1024.0
    print(
        f"Wrote {out_path} ({sz_kb:.1f} KiB) — {non_empty}/{BANDS * LON_BLKS} land packs indexed."
    )
    # SF Bay / Sierra Nevada is band 4 (33.75°..45°N), lon_blk 5 (-123.75°..-112.5°W) => pid = 4*32 + 5 = 133
    p_sf = packs["133"]
    print(
        f"  Smoke-test pack p133 (SF/Yosemite): {len(p_sf['cop'])} Copernicus tiles, {len(p_sf['s2_pri'])} S2 primary ({p_sf['pri_q']}), {len(p_sf['s2_fb'])} S2 fallback."
    )
    # Check band 0 (Arctic) and band 15 (Antarctica)
    b0 = [pid for pid, p in packs.items() if p["band"] == 0]
    b15 = [pid for pid, p in packs.items() if p["band"] == 15]
    print(
        f"  Arctic band 0 packs ({len(b0)}): {b0[:8]} ... | Antarctic band 15 packs ({len(b15)}): {b15[:8]} ..."
    )


if __name__ == "__main__":
    main()
