import geopandas as gpd

DATA_DIR = r"C:\Users\hamada\wakayama-map\app\data"
IN_PATH  = rf"{DATA_DIR}\hospital.geojson"
OUT_PATH = rf"{DATA_DIR}\hospital_2nd.geojson"

ALIASES = {
    "日赤和歌山医療センター": "日本赤十字社和歌山医療センター",
    "紀北分院": "和歌山県立医科大学附属病院紀北分院",
    "和歌山病院": "（独）国立病院機構和歌山病院",
    "南和歌山医療センター": "（独）国立病院機構南和歌山医療センター",
}

SECONDARY = [
    "日赤和歌山医療センター",
    "和歌山労災病院",
    "和歌山県立医科大学附属病院",
    "和歌山生協病院",
    "済生会和歌山病院",
    "海南医療センター",
    "公立那賀病院",
    "橋本市民病院",
    "紀北分院",
    "有田市立病院",
    "済生会有田病院",
    "国保日高総合病院",
    "和歌山病院",
    "南和歌山医療センター",
    "紀南病院",
    "白浜はまゆう病院",
    "国保すさみ病院",
    "新宮市立医療センター",
    "くしもと町立病院",
    "那智勝浦町立温泉病院",
]

def norm(s: str) -> str:
    return str(s).replace("\u3000", " ").strip()

def main():
    gdf = gpd.read_file(IN_PATH).copy()
    name_col = "P04_002"

    gdf["__name"] = gdf[name_col].map(norm)

    targets = [ALIASES.get(n, n) for n in SECONDARY]
    targets = set(map(norm, targets))

    out = gdf[gdf["__name"].isin(targets)].copy()

    found = set(out["__name"].tolist())
    missing = sorted([t for t in targets if t not in found])

    print("rows total:", len(gdf))
    print("rows 2nd  :", len(out))
    print("missing (official):", missing)

    out = out.drop(columns=["__name"])
    out.to_file(OUT_PATH, driver="GeoJSON", encoding="utf-8")
    print("saved:", OUT_PATH)

if __name__ == "__main__":
    main()
