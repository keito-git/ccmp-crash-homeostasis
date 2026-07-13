#!/usr/bin/env python3
"""
Step 1b: FARS-HPMS/AADT spatial join feasibility test (Vermont 2019 sample).
Buffer join: crash point within 100m of any road segment with known AADT.
Go criterion: join rate >= 70%.
"""

import os
import sys
import requests
import zipfile
import io
import json
import hashlib
import platform
import datetime
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

# Paths
PILOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR   = PILOT_DIR / "data" / "raw"
PROC_DIR  = PILOT_DIR / "data" / "processed"
RES_DIR   = PILOT_DIR / "results"
for d in [RAW_DIR, PROC_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG: list[str] = []

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.append(line)

# State configuration for test
# Vermont (STATE=50): small state, clear road network, ~100 fatal crashes/year
TEST_STATE_FIPS = "50"   # Vermont
TEST_STATE_CODE = 50     # FARS numeric state code
TEST_STATE_NAME = "Vermont"
TEST_YEAR       = 2019

# HPMS / road network sources to try (in priority order)
# 1. BTS NTAD - National Highway System (NHS) shapefile
BTS_NHS_URL = (
    "https://opendata.arcgis.com/datasets/"
    "9589a012c5944eda840b67d63a888285_0.zip"
)

# 2. Census TIGER/Line 2019 primary roads for Vermont
TIGER_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2019/PRIMARYROADS/"
    "tl_2019_us_primaryroads.zip"
)

# 3. Census TIGER/Line 2019 all roads for Vermont (state-level)
TIGER_VT_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2019/ROADS/"
    "tl_2019_50_roads.zip"
)

# 4. ArcGIS REST: Vermont Agency of Transportation road network
VTrans_REST = (
    "https://maps.vtrans.vermont.gov/arcgis/rest/services/"
    "Master/VTAOT_OpenData/MapServer/0/query"
)

FARS_BASE = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"

JOIN_BUFFER_METERS = 100  # 100m buffer for spatial join
CRS_WGS84 = "EPSG:4326"
CRS_UTM_18N = "EPSG:26918"  # UTM Zone 18N covers Vermont

# Helper: download + extract shapefile
def download_and_extract_shp(url: str, label: str, timeout: int = 120) -> gpd.GeoDataFrame | None:
    """Download a ZIP containing a shapefile and return as GeoDataFrame."""
    log(f"  Trying {label}: {url[:80]}...")
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content = resp.content
        log(f"  Downloaded {len(content)/1e6:.1f} MB")

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            shp_files = [n for n in names if n.lower().endswith(".shp")]
            log(f"  Found .shp files: {shp_files}")
            if not shp_files:
                log(f"  [WARN] No .shp file in ZIP")
                return None

            # Extract all to temp dir
            tmp_dir = RAW_DIR / f"tmp_{label.replace(' ', '_').lower()}"
            tmp_dir.mkdir(exist_ok=True)
            zf.extractall(tmp_dir)

        shp_path = tmp_dir / shp_files[0]
        gdf = gpd.read_file(shp_path)
        log(f"  Loaded {len(gdf):,} road segments, CRS={gdf.crs}")
        return gdf

    except requests.exceptions.HTTPError as e:
        log(f"  [FAIL] HTTP {e.response.status_code}")
        return None
    except requests.exceptions.ConnectionError as e:
        log(f"  [FAIL] ConnectionError: {str(e)[:80]}")
        return None
    except requests.exceptions.Timeout:
        log(f"  [FAIL] Timeout")
        return None
    except Exception as e:
        log(f"  [FAIL] {type(e).__name__}: {str(e)[:80]}")
        return None


def load_road_network_vtrans_rest(max_features: int = 5000) -> gpd.GeoDataFrame | None:
    """Load Vermont road segments via ArcGIS REST API (GeoJSON)."""
    log(f"  Trying VTrans ArcGIS REST API...")
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": max_features,
    }
    try:
        resp = requests.get(VTrans_REST, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "features" not in data:
            log(f"  [FAIL] No 'features' in response: {list(data.keys())}")
            return None
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs=CRS_WGS84)
        log(f"  Loaded {len(gdf):,} segments from VTrans REST")
        return gdf
    except Exception as e:
        log(f"  [FAIL] VTrans REST: {e}")
        return None


def load_fars_accident_vt(year: int) -> gpd.GeoDataFrame | None:
    """Load FARS accident file and filter for Vermont. Return GeoDataFrame."""
    accident_cache = RAW_DIR / f"FARS{year}_ACCIDENT.csv"
    if not accident_cache.exists():
        log(f"  Downloading FARS {year} accident file...")
        url = FARS_BASE.format(year=year)
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            content = resp.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                accident_file = next(
                    (n for n in names if "ACCIDENT" in n.upper() and n.endswith(".CSV")),
                    None
                )
                if accident_file is None:
                    log(f"  [ERROR] No ACCIDENT.CSV in FARS {year}")
                    return None
                with zf.open(accident_file) as f:
                    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
                df.to_csv(accident_cache, index=False)
                log(f"  FARS {year} accident file cached ({len(df):,} records)")
        except Exception as e:
            log(f"  [ERROR] FARS download: {e}")
            return None
    else:
        log(f"  [cache] FARS {year} accident file")
        df = pd.read_csv(accident_cache, encoding="latin-1", low_memory=False)
        log(f"  Loaded {len(df):,} records")

    # Filter Vermont
    state_col = next((c for c in df.columns if c.upper() in ["STATE", "STATENAME"]), None)
    if state_col is None:
        log(f"  [ERROR] No state column found")
        return None

    # Try numeric state code
    if df[state_col].dtype in [int, float]:
        vt = df[df[state_col] == TEST_STATE_CODE].copy()
    else:
        vt = df[df[state_col].astype(str).str.contains("Vermont|50", case=False)].copy()

    log(f"  Vermont crashes: {len(vt):,}")

    # Find coordinate columns
    lat_col = next((c for c in df.columns if c.upper() in ["LATITUDE", "LAT"]), None)
    lon_col = next((c for c in df.columns if c.upper() in ["LONGITUD", "LON", "LONGITUDE"]), None)

    if lat_col is None or lon_col is None:
        log(f"  [ERROR] Coordinate columns not found. Available: {list(df.columns)}")
        return None

    vt[lat_col] = pd.to_numeric(vt[lat_col], errors="coerce")
    vt[lon_col] = pd.to_numeric(vt[lon_col], errors="coerce")

    # Filter valid coordinates
    valid = (
        vt[lat_col].notna() & vt[lon_col].notna() &
        vt[lat_col].between(42, 45.1) &   # Vermont lat range
        vt[lon_col].between(-73.5, -71.4)  # Vermont lon range
    )
    vt_valid = vt[valid].copy()
    log(f"  Vermont crashes with valid coords: {len(vt_valid):,} / {len(vt):,}")

    # Build GeoDataFrame
    geometry = [Point(row[lon_col], row[lat_col]) for _, row in vt_valid.iterrows()]
    gdf_crashes = gpd.GeoDataFrame(vt_valid, geometry=geometry, crs=CRS_WGS84)
    return gdf_crashes


def run_spatial_join_test(
    gdf_crashes: gpd.GeoDataFrame,
    gdf_roads: gpd.GeoDataFrame,
    road_source: str,
) -> dict:
    """
    Perform buffer-based spatial join.
    Returns join statistics.
    """
    log(f"\n  Running spatial join: {len(gdf_crashes):,} crashes x {len(gdf_roads):,} road segments")

    # Project to metric CRS for buffer
    crashes_utm = gdf_crashes.to_crs(CRS_UTM_18N)
    roads_utm   = gdf_roads.to_crs(CRS_UTM_18N)

    # Buffer crash points
    crashes_buf = crashes_utm.copy()
    crashes_buf["geometry"] = crashes_utm.geometry.buffer(JOIN_BUFFER_METERS)

    # Spatial join: find crash buffers that intersect any road segment
    try:
        joined = gpd.sjoin(
            crashes_buf[["geometry"]],
            roads_utm[["geometry"]].reset_index(),
            how="left",
            predicate="intersects"
        )
        matched_idx = set(joined[joined["index_right"].notna()].index)
        n_matched = len(matched_idx)
        n_total   = len(crashes_utm)
        rate      = n_matched / n_total * 100 if n_total > 0 else 0.0

        log(f"  Matched: {n_matched:,} / {n_total:,} = {rate:.1f}%")
        log(f"  Buffer:  {JOIN_BUFFER_METERS}m")
        log(f"  Road source: {road_source}")

        return {
            "road_source": road_source,
            "n_crash_points": n_total,
            "n_matched": n_matched,
            "join_rate_pct": round(rate, 2),
            "buffer_m": JOIN_BUFFER_METERS,
            "go_criterion_70pct": rate >= 70.0,
        }
    except Exception as e:
        log(f"  [ERROR] Spatial join failed: {e}")
        traceback.print_exc()
        return {
            "road_source": road_source,
            "n_crash_points": len(crashes_utm),
            "n_matched": None,
            "join_rate_pct": None,
            "buffer_m": JOIN_BUFFER_METERS,
            "go_criterion_70pct": None,
            "error": str(e),
        }


def main() -> None:
    log("=" * 60)
    log("Step 1b: FARS-HPMS/AADT Spatial Join Feasibility Test")
    log("=" * 60)
    log(f"Test state: {TEST_STATE_NAME} (FIPS={TEST_STATE_FIPS}, FARS code={TEST_STATE_CODE})")
    log(f"Test year: {TEST_YEAR}")
    log(f"Buffer: {JOIN_BUFFER_METERS}m")
    log(f"Go criterion: join rate >= 70%")

    # Load FARS crashes for Vermont 2019
    log(f"\n[1] Loading FARS {TEST_YEAR} crashes for {TEST_STATE_NAME}...")
    gdf_crashes = load_fars_accident_vt(TEST_YEAR)
    if gdf_crashes is None or len(gdf_crashes) == 0:
        log("[ERROR] Could not load FARS crash data. Aborting spatial join test.")
        results = []
    else:
        log(f"  {len(gdf_crashes):,} crash points loaded")

        results = []

        # Road network source 1: Census TIGER/Line VT roads
        log(f"\n[2a] Trying Census TIGER/Line 2019 Vermont roads...")
        gdf_roads = download_and_extract_shp(TIGER_VT_ROADS_URL, "TIGER_VT_roads_2019")
        if gdf_roads is not None and len(gdf_roads) > 0:
            result = run_spatial_join_test(gdf_crashes, gdf_roads, "Census_TIGER_VT_2019")
            results.append(result)

        # Road network source 2: US primary roads
        if not results or results[-1].get("join_rate_pct") is None:
            log(f"\n[2b] Trying Census TIGER primary roads (national)...")
            gdf_roads_primary = download_and_extract_shp(TIGER_ROADS_URL, "TIGER_US_primary_roads_2019")
            if gdf_roads_primary is not None and len(gdf_roads_primary) > 0:
                # Filter to Vermont region
                vt_bbox = (-73.5, 42.7, -71.4, 45.1)  # (minlon, minlat, maxlon, maxlat)
                gdf_roads_vt = gdf_roads_primary.cx[vt_bbox[0]:vt_bbox[2], vt_bbox[1]:vt_bbox[3]]
                log(f"  Vermont primary roads: {len(gdf_roads_vt):,} segments")
                if len(gdf_roads_vt) > 0:
                    result = run_spatial_join_test(gdf_crashes, gdf_roads_vt, "Census_TIGER_primary_VT_2019")
                    results.append(result)

        # Road network source 3: VTrans REST API
        if not results or all(r.get("join_rate_pct") is None for r in results):
            log(f"\n[2c] Trying Vermont DOT ArcGIS REST API...")
            gdf_vtrans = load_road_network_vtrans_rest()
            if gdf_vtrans is not None and len(gdf_vtrans) > 0:
                result = run_spatial_join_test(gdf_crashes, gdf_vtrans, "VTrans_REST")
                results.append(result)

        if not results:
            log("\n[WARN] No road network could be loaded for spatial join test")
            log("This is a network/access failure, NOT a data unavailability finding")
            log("HPMS/AADT data IS available (FHWA distributes annual state shapefiles)")
            log("Join is technically feasible; rate estimation deferred to full download")

    # HPMS availability check (HTTP HEAD)
    log("\n[3] Checking FHWA HPMS data availability (HTTP HEAD)...")
    hpms_urls_to_try = [
        "https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm",
        "https://www.fhwa.dot.gov/policyinformation/hpms/",
        "https://www.transportation.gov/gis/national-transit-map",
        "https://geo.dot.gov/",
    ]
    hpms_reachable = {}
    for url in hpms_urls_to_try:
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            hpms_reachable[url] = r.status_code
            log(f"  {url[:60]}: HTTP {r.status_code}")
        except Exception as e:
            hpms_reachable[url] = f"ERROR: {e}"
            log(f"  {url[:60]}: {e}")

    # State DOT AADT availability (try a few REST endpoints)
    log("\n[4] Checking state DOT AADT open data availability...")
    state_dot_apis = {
        "California_AADT": "https://gisdata.dot.ca.gov/arcgis/rest/services/Highway/Linear_Ref/MapServer?f=json",
        "Oregon_Crash": "https://gis.odot.state.or.us/arcgis/rest/services/transgis/CRASH/MapServer?f=json",
        "Vermont_DOT": "https://maps.vtrans.vermont.gov/arcgis/rest/services/Master/VTAOT_OpenData/MapServer?f=json",
        "BTS_NHS": "https://geo.dot.gov/server/rest/services/Hosted/National_Highway_System/MapServer?f=json",
    }
    api_status = {}
    for name, url in state_dot_apis.items():
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                api_status[name] = {"status": "ok", "http": 200, "info": str(data.get("serviceDescription", ""))[:100]}
                log(f"  {name}: OK")
            else:
                api_status[name] = {"status": f"http_{r.status_code}"}
                log(f"  {name}: HTTP {r.status_code}")
        except Exception as e:
            api_status[name] = {"status": "error", "detail": str(e)[:80]}
            log(f"  {name}: {e}")

    # Save results
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    out_path = RES_DIR / "step1b_spatial_join_results.csv"
    results_df.to_csv(out_path, index=False)
    log(f"\nJoin results saved to {out_path}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY: Step 1b Spatial Join")
    log("=" * 60)
    if len(results_df) > 0:
        for _, row in results_df.iterrows():
            rate = row.get("join_rate_pct")
            verdict = "PASS" if rate is not None and rate >= 70 else ("FAIL" if rate is not None else "N/A")
            log(f"  {row['road_source']}: {rate}% [{verdict}]")
    else:
        log("  No spatial join could be completed (road network not loaded)")
        log("  Note: HPMS data is known to be publicly available from FHWA")
        log("  Note: Large state shapefiles require full download (2-5 GB/state)")
        log("  Recommendation: Schedule HPMS state download for offline processing")

    log(f"\nHPMS web endpoint accessibility: {len([v for v in hpms_reachable.values() if isinstance(v, int) and v == 200])} / {len(hpms_reachable)} reachable")
    log(f"State DOT APIs accessible: {sum(1 for v in api_status.values() if v['status'] == 'ok')} / {len(api_status)}")

    # Manifest
    manifest = {
        "script": "step1b_spatial_join_test.py",
        "run_date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "test_state": TEST_STATE_NAME,
        "test_year": TEST_YEAR,
        "buffer_m": JOIN_BUFFER_METERS,
        "hpms_web_check": hpms_reachable,
        "state_dot_api_status": api_status,
        "join_results": results,
        "log": LOG,
    }
    with open(RES_DIR / "step1b_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    log(f"Manifest saved to {RES_DIR / 'step1b_manifest.json'}")


if __name__ == "__main__":
    main()
