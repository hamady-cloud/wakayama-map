import pandas as pd

DATA_DIR = r"C:\Users\hamada\wakayama-map\app\data"
IN_PATH  = rf"{DATA_DIR}\in.csv"
OUT_PATH = rf"{DATA_DIR}\out.csv"
POP_PATH = rf"{DATA_DIR}\pop.csv"

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    """候補のうち、実際に存在する列名を1つ返す（なければKeyError）"""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"列が見つかりません。候補={candidates}\n実際の列={df.columns.tolist()}")

def fix_unnamed_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    e-Stat CSVでよくある：
      - header=1で読むと Unnamed:* が残る
      - 先頭行(df.iloc[0])に「本当の列名」が入っている
    のパターンを修正する
    """
    if not any(str(c).startswith("Unnamed") for c in df.columns):
        return df

    header_row = df.iloc[0]
    rename_map = {}
    for c in df.columns:
        if str(c).startswith("Unnamed"):
            v = header_row[c]
            if pd.notna(v) and str(v).strip() != "":
                rename_map[c] = str(v).strip()

    df = df.rename(columns=rename_map).iloc[1:].copy()
    return df

def load_flow(path: str):
    """
    住民基本台帳人口移動（転入/転出）のCSVを読む。
    - あなたのCSVの列構造に合わせて header=1 → Unnamed補正
    - 「全国・都道府県・市区町村コード」等を抽出
    """
    df = pd.read_csv(path, header=1)
    df = fix_unnamed_headers(df)

    col_item  = first_existing(df, ["表章項目"])
    col_code  = first_existing(df, ["全国・都道府県・市区町村コード", "全国都道府県市区町村コード"])
    col_name  = first_existing(df, ["全国・都道府県・市区町村", "全国都道府県市区町村"])
    col_total = first_existing(df, ["総数"])

    # 単位行（総数='人'）などを除外
    df = df[df[col_total].astype(str) != "人"].copy()

    df["code"] = df[col_code].astype(str).str.zfill(5)
    df["name"] = df[col_name].astype(str)
    df["total"] = to_num(df[col_total])

    # 和歌山の集計行（県計/市部/郡部）を落とす
    df = df[~df["code"].isin(["30000", "30001", "30002"])].copy()

    item = str(df[col_item].iloc[0])  # 転入/転出の判定に使う
    return df[["code", "name", "total"]], item

def load_pop(path: str):
    """
    国勢調査(pop.csv)を読む（あなたのファイル前提）
    """
    df = pd.read_csv(path, encoding="cp932", skiprows=17)

    col_code  = first_existing(df, ["全国，都道府県，市区町村 コード", "全国，都道府県，市区町村コード"])
    col_name  = first_existing(df, ["全国，都道府県，市区町村"])
    col_total = first_existing(df, ["総数"])
    col_65    = first_existing(df, ["65歳以上", "65歳 以上"])

    df["code"] = df[col_code].astype(str).str.zfill(5)

    # 和歌山（30xxxx）に絞り、県計(30000)は除外
    df = df[df["code"].str.startswith("30") & (df["code"] != "30000")].copy()

    df["pop_total"] = to_num(df[col_total])
    df["pop_65"] = to_num(df[col_65])
    df["aging_rate"] = df["pop_65"] / df["pop_total"]

    return df[["code", col_name, "pop_total", "pop_65", "aging_rate"]].rename(columns={col_name: "pop_name"})

def main():
    pop = load_pop(POP_PATH)

    f1, item1 = load_flow(IN_PATH)
    f2, item2 = load_flow(OUT_PATH)

    # item文字列で転入/転出を判定（ファイル名のin/outが逆でもOK）
    def is_inflow(item):  return "転入" in item
    def is_outflow(item): return "転出" in item

    if is_inflow(item1) and is_outflow(item2):
        inflow, outflow = f1, f2
    elif is_outflow(item1) and is_inflow(item2):
        inflow, outflow = f2, f1
    else:
        raise ValueError(f"転入/転出の判定に失敗しました: item1={item1}, item2={item2}")

    m = (
        pop
        .merge(inflow[["code", "total"]].rename(columns={"total": "inflow"}), on="code", how="left")
        .merge(outflow[["code", "total"]].rename(columns={"total": "outflow"}), on="code", how="left")
    )

    m["net_migration"] = m["inflow"] - m["outflow"]
    m["net_migration_rate_per1000"] = m["net_migration"] / m["pop_total"] * 1000

    out_path = rf"{DATA_DIR}\metrics_base.csv"
    m.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

if __name__ == "__main__":
    main()
