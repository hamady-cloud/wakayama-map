import geopandas as gpd
import pandas as pd

DATA_DIR = r"C:\Users\hamada\wakayama-map\app\data"

AREAS_PATH  = rf"{DATA_DIR}\areas_wakayama.geojson"
HOSP_PATH   = rf"{DATA_DIR}\hospital.geojson"
METRICS_IN  = rf"{DATA_DIR}\metrics_base.csv"
METRICS_OUT = rf"{DATA_DIR}\metrics.csv"

def main():
    areas = gpd.read_file(AREAS_PATH)
    hosps = gpd.read_file(HOSP_PATH)

    # --- areas key ---
    if "N03_007" not in areas.columns:
        raise KeyError(f"areas_wakayama.geojson に N03_007 がありません。columns={list(areas.columns)}")
    areas["code"] = areas["N03_007"].astype(str).str.zfill(5)

    # --- representative points (inside polygon) ---
    areas = areas.copy()
    areas["rep_pt"] = areas.geometry.representative_point()
    areas_pts = gpd.GeoDataFrame(areas[["code"]], geometry=areas["rep_pt"], crs=areas.crs)

    # --- ensure hospitals are points ---
    if not all(hosps.geometry.geom_type.isin(["Point"])):
        hosps = hosps.copy()
        hosps["geometry"] = hosps.geometry.representative_point()

    # --- project to meters (EPSG:3857) ---
    areas_m = areas_pts.to_crs("EPSG:3857")
    hosps_m = hosps.to_crs("EPSG:3857")

    # --- nearest join (distance in meters) ---
    joined = gpd.sjoin_nearest(
        areas_m,
        hosps_m[["geometry"]],
        how="left",
        distance_col="dist_m"
    )

    dist = joined[["code", "dist_m"]].copy()
    dist["hospital_dist_km"] = dist["dist_m"] / 1000.0
    dist = dist.drop(columns=["dist_m"])

    metrics = pd.read_csv(METRICS_IN, dtype={"code": str})
    out = metrics.merge(dist, on="code", how="left")

    out.to_csv(METRICS_OUT, index=False, encoding="utf-8-sig")
    print("saved:", METRICS_OUT)

if __name__ == "__main__":
    main()
