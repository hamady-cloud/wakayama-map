import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\hamada\wakayama-map\app\data")

AREAS_PATH = DATA_DIR / "areas_wakayama.geojson"
MESH_POP   = DATA_DIR / "mesh1km_pop.geojson"
HOSP_2ND   = DATA_DIR / "hospital_2nd.geojson"

OUT_CSV    = DATA_DIR / "city_hospital_p5090.csv"   # 市町村別のP50/P90（この工程の成果物）
OUT_METRICS = DATA_DIR / "metrics.csv"              # 既存があるなら上書き更新（あれば）

def weighted_quantile(values, weights, qs=(0.5, 0.9)):
    """values, weights: 1d array. qs: tuple of quantiles in [0,1]."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]; w = w[mask]
    if len(v) == 0:
        return {q: np.nan for q in qs}

    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w)
    if cw[-1] == 0:
        return {q: np.nan for q in qs}

    # 重み付きCDF（0～1）
    cdf = cw / cw[-1]

    out = {}
    for q in qs:
        idx = np.searchsorted(cdf, q, side="left")
        idx = min(idx, len(v) - 1)
        out[q] = float(v[idx])
    return out

def main():
    areas = gpd.read_file(AREAS_PATH)
    mesh  = gpd.read_file(MESH_POP)
    hosp  = gpd.read_file(HOSP_2ND)

    # --- key columns ---
    if "N03_007" not in areas.columns:
        raise KeyError(f"areas に N03_007 がありません。columns={list(areas.columns)}")
    areas["code"] = areas["N03_007"].astype(str).str.zfill(5)
    areas["name"] = areas.get("N03_004", "")

    if "KEY_CODE" not in mesh.columns:
        raise KeyError(f"mesh に KEY_CODE がありません。columns={list(mesh.columns)}")
    if "pop" not in mesh.columns:
        raise KeyError(f"mesh に pop がありません。columns={list(mesh.columns)}")

    # --- ensure CRS ---
    if areas.crs is None:
        raise ValueError("areas のCRSが不明です（読み込み元geojsonにcrsが無い）")
    if mesh.crs is None:
        # meshはshp由来ならcrs付くはずだが、念のためareasに合わせる
        mesh = mesh.set_crs(areas.crs, allow_override=True)
    if hosp.crs is None:
        hosp = hosp.set_crs(areas.crs, allow_override=True)

    # --- mesh points -> city polygon (spatial join) ---
    mesh = mesh.to_crs(areas.crs)
    hosp = hosp.to_crs(areas.crs)

    mesh_city = gpd.sjoin(mesh[["KEY_CODE","pop","geometry"]], areas[["code","name","geometry"]], how="left", predicate="within")
    mesh_city = mesh_city.drop(columns=["index_right"])

    # 市町村外（海上など）を落とす
    mesh_city = mesh_city[mesh_city["code"].notna()].copy()
    mesh_city["code"] = mesh_city["code"].astype(str).str.zfill(5)

    # --- distance to nearest 2nd hospital ---
    # 距離計算用にメートル系へ
    mesh_m = mesh_city.to_crs("EPSG:3857")
    hosp_m = hosp.to_crs("EPSG:3857")

    nn = gpd.sjoin_nearest(
        mesh_m,
        hosp_m[["geometry"]],
        how="left",
        distance_col="dist_m"
    )
    nn["dist_km"] = nn["dist_m"] / 1000.0

    # --- aggregate per city (weighted P50/P90) ---
    rows = []
    for code, g in nn.groupby("code"):
        q = weighted_quantile(g["dist_km"].values, g["pop"].values, qs=(0.5, 0.9))
        rows.append({
            "code": code,
            "name": g["name"].iloc[0] if "name" in g.columns else "",
            "mesh_count": int(len(g)),
            "pop_sum": float(np.sum(g["pop"].values)),
            "hospital_p50_km": q[0.5],
            "hospital_p90_km": q[0.9],
            "hospital_mean_km": float(np.average(g["dist_km"].values, weights=g["pop"].values)),
        })

    out = pd.DataFrame(rows).sort_values("hospital_p50_km").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved:", OUT_CSV, "rows=", len(out))

    # --- optional: merge into existing metrics.csv if exists ---
    if OUT_METRICS.exists():
        m = pd.read_csv(OUT_METRICS, dtype={"code": str})
        m["code"] = m["code"].astype(str).str.zfill(5)
        out2 = out[["code","hospital_p50_km","hospital_p90_km","hospital_mean_km"]].copy()
        merged = m.merge(out2, on="code", how="left")
        merged.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
        print("updated:", OUT_METRICS, "rows=", len(merged))
    else:
        print("note:", OUT_METRICS, "が無いので更新はスキップしました")

if __name__ == "__main__":
    main()
