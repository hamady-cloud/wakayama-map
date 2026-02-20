import glob
from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA_DIR = Path(r"C:\Users\hamada\wakayama-map\app\data")
POP_TXT  = DATA_DIR / "tblT001100S30.txt"
OUT_GEOJSON = DATA_DIR / "mesh1km_pop.geojson"

def read_pop_agg():
    df = pd.read_csv(
        POP_TXT,
        encoding="cp932",
        dtype={"KEY_CODE": str},
        header=0,
        skiprows=[1],
    )
    df = df[df["KEY_CODE"].notna()].copy()
    df["KEY_CODE"] = df["KEY_CODE"].astype(str).str.strip()

    # 総人口を数値化
    df["pop"] = pd.to_numeric(df["T001100001"].replace("*", pd.NA), errors="coerce").fillna(0)

    # ここが肝：HTKSYORIの違いは気にせず、KEY_CODEで最終人口に集約
    agg = df.groupby("KEY_CODE", as_index=False)["pop"].sum()

    print("pop rows raw:", len(df), "agg rows:", len(agg), "pop sum:", float(agg["pop"].sum()))
    return agg

def find_mesh_shps():
    paths = sorted(glob.glob(str(DATA_DIR / "MESH*.shp")))
    if not paths:
        paths = sorted(glob.glob(str(DATA_DIR / "MESHH*.shp")))
    if not paths:
        raise FileNotFoundError(f"MESH*.shp が {DATA_DIR} に見つかりません")
    return paths

def read_mesh(paths):
    gdfs = []
    for p in paths:
        g = gpd.read_file(p)
        if "KEY_CODE" not in g.columns:
            raise KeyError(f"{Path(p).name} に KEY_CODE がありません。columns={list(g.columns)}")
        g["KEY_CODE"] = g["KEY_CODE"].astype(str).str.strip()
        gdfs.append(g[["KEY_CODE", "geometry"]].copy())

    out = pd.concat(gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdfs[0].crs)
    print("mesh rows:", len(out))
    return out

def main():
    pop = read_pop_agg()
    shps = find_mesh_shps()
    print("mesh shp files:", len(shps))
    for p in shps:
        print(" -", p)

    mesh = read_mesh(shps)

    merged = mesh.merge(pop, on="KEY_CODE", how="left")
    merged["pop"] = merged["pop"].fillna(0)

    # 人口>0だけ残す
    merged = merged[merged["pop"] > 0].copy()

    # 軽量化：ポリゴン代表点
    merged["geometry"] = merged.geometry.representative_point()

    merged.to_file(OUT_GEOJSON, driver="GeoJSON", encoding="utf-8")
    print("saved:", OUT_GEOJSON, "rows=", len(merged))
    print("pop sum after join:", float(merged["pop"].sum()))

if __name__ == "__main__":
    main()
