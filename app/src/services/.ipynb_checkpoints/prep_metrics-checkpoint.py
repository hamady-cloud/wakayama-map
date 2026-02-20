import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\hamada\wakayama-map\app\data"

IN_PATH  = rf"{DATA_DIR}\in.csv"
OUT_PATH = rf"{DATA_DIR}\out.csv"
POP_PATH = rf"{DATA_DIR}\pop.csv"

def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

def load_flow(path: str):
    # e-StatのCSV（上に単位行がある形式）を想定
    df = pd.read_csv(path, header=1)
    df = df[df["総数"] != "人"].copy()

    df["code"] = df["全国都道府県市区町村コード"].astype(str)
    df["name"] = df["全国都道府県市区町村"].astype(str)
    df["total"] = _to_num(df["総数"])

    # 県計/市部/郡部などの集計（30000/30001/30002）を落とす
    df = df[~df["code"].isin(["30000", "30001", "30002"])].copy()

    # 表章項目（転入/転出）を読む
    item = str(df["表章項目"].iloc[0])
    return df[["code", "name", "total"]], item

def load_pop(path: str):
    # pop.csv は e-Stat特有のメタ行があるので skiprows=17 で本体を読む
    df = pd.read_csv(path, encoding="cp932", skiprows=17)
    df["code"] = df["全国，都道府県，市区町村 コード"].astype(str).str.zfill(5)

    # 和歌山県の市町村だけ（30000=県計は除外）
    df = df[df["code"].str.startswith("30") & (df["code"] != "30000")].copy()

    df["pop_total"] = _to_num(df["総数"])
    df["pop_65"] = _to_num(df["65歳以上"])
    df["aging_rate"] = df["pop_65"] / df["pop_total"]

    return df[["code", "全国，都道府県，市区町村", "pop_total", "pop_65", "aging_rate"]]

def main():
    pop = load_pop(POP_PATH)

    f1, item1 = load_flow(IN_PATH)
    f2, item2 = load_flow(OUT_PATH)

    # item文字列で転入/転出を自動判定（ファイル名の逆転に耐える）
    def is_inflow(item):  return "転入" in item
    def is_outflow(item): return "転出" in item

    if is_inflow(item1) and is_outflow(item2):
        inflow, outflow = f1, f2
    elif is_outflow(item1) and is_inflow(item2):
        inflow, outflow = f2, f1
    else:
        raise ValueError(f"転入/転出の判定に失敗しました: item1={item1}, item2={item2}")

    m = pop.merge(inflow[["code","total"]].rename(columns={"total":"inflow"}), on="code", how="left") \
           .merge(outflow[["code","total"]].rename(columns={"total":"outflow"}), on="code", how="left")

    m["net_migration"] = m["inflow"] - m["outflow"]
    m["net_migration_rate_per1000"] = m["net_migration"] / m["pop_total"] * 1000

    out_path = rf"{DATA_DIR}\metrics_base.csv"
    m.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

if __name__ == "__main__":
    main()
