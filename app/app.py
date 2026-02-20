import json
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt

# =========================
# Config
# =========================
st.set_page_config(page_title="和歌山：重点エリア可視化", layout="wide")

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GEO_PATH = os.path.join(DATA_DIR, "areas_wakayama.geojson")
MET_PATH = os.path.join(DATA_DIR, "metrics.csv")

# =========================
# Utils
# =========================
@st.cache_data
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(5)

    # 1市町村=1行に集約（国籍/年次などが混ざって重複しがち）
    key_cols = ["code"]
    keep_first_cols = [c for c in ["pop_name"] if c in df.columns]

    # 数値列をなるべく numeric に
    for c in df.columns:
        if c in key_cols + keep_first_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    agg = {}
    for c in df.columns:
        if c in key_cols:
            continue
        if c in keep_first_cols:
            agg[c] = "first"
        else:
            agg[c] = "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"

    df = df.groupby("code", as_index=False).agg(agg)
    return df

def clip_minmax01(s: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    """外れ値に強い正規化：分位点でクリップmin-max(0-1)"""
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)

    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return pd.Series(np.zeros(len(x)), index=x.index)

    x2 = x.clip(lower=lo, upper=hi)
    return (x2 - lo) / (hi - lo)

def score_to_color(v01: float) -> list[int]:
    v01 = float(max(0.0, min(1.0, v01)))
    base = 60
    r = int(base + (195 * v01))
    g = int(base + (60 * (1 - v01)))
    b = int(base + (60 * (1 - v01)))
    return [r, g, b, 170]

def compute_scores(df: pd.DataFrame, w: dict, access_col: str, q_low=0.05, q_high=0.95) -> pd.DataFrame:
    d = df.copy()

    # 外れ値に強い正規化（0-1）
    d["n_aging"] = clip_minmax01(d["aging_rate"], q_low=q_low, q_high=q_high)
    d["n_access"] = clip_minmax01(d[access_col], q_low=q_low, q_high=q_high)

    # 人口流出は「マイナスが大きいほど重点」（値が大きいほど重点）に揃える
    d["n_net_raw"] = clip_minmax01(d["net_migration_rate_per1000"], q_low=q_low, q_high=q_high)
    d["n_net"] = 1.0 - d["n_net_raw"]

    # スコア（重み付き加算）
    d["score"] = (
        w["aging"] * d["n_aging"]
        + w["access"] * d["n_access"]
        + w["net"] * d["n_net"]
    )

    # 0-1に再正規化（色付け用）
    mn, mx = float(d["score"].min()), float(d["score"].max())
    d["score01"] = 0.0 if mn == mx else (d["score"] - mn) / (mx - mn)

    # 寄与（重み込み）と寄与率
    d["c_aging"] = w["aging"] * d["n_aging"]
    d["c_access"] = w["access"] * d["n_access"]
    d["c_net"] = w["net"] * d["n_net"]

    denom = (d["c_aging"] + d["c_access"] + d["c_net"]).replace(0, np.nan)
    d["p_aging"] = (d["c_aging"] / denom).fillna(0)
    d["p_access"] = (d["c_access"] / denom).fillna(0)
    d["p_net"] = (d["c_net"] / denom).fillna(0)

    contrib = pd.DataFrame({
        "高齢化": d["c_aging"],
        "医療アクセス": d["c_access"],
        "人口流出（純移動）": d["c_net"],
    })
    d["top_driver"] = contrib.idxmax(axis=1)

    return d

def merge_into_geojson(geo: dict, df: pd.DataFrame) -> dict:
    m = df.set_index("code").to_dict(orient="index")
    for feat in geo["features"]:
        props = feat.get("properties", {})
        code = str(props.get("N03_007", "")).zfill(5)
        props["code"] = code
        props["name"] = props.get("N03_004", "")
        if code in m:
            props.update(m[code])
        feat["properties"] = props
    return geo

def contrib_chart(selected_row: pd.Series) -> alt.Chart:
    data = pd.DataFrame({
        "項目": ["高齢化", "医療アクセス", "人口流出（純移動）"],
        "寄与率": [
            float(selected_row.get("p_aging", 0)),
            float(selected_row.get("p_access", 0)),
            float(selected_row.get("p_net", 0)),
        ],
        "寄与(重み込み)": [
            float(selected_row.get("c_aging", 0)),
            float(selected_row.get("c_access", 0)),
            float(selected_row.get("c_net", 0)),
        ],
    })

    ch = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("項目:N", sort="-x"),
            x=alt.X("寄与率:Q", axis=alt.Axis(format="%")),
            tooltip=["項目:N", alt.Tooltip("寄与率:Q", format=".2%"), alt.Tooltip("寄与(重み込み):Q", format=".4f")],
        )
        .properties(height=140)
    )
    txt = (
        alt.Chart(data)
        .mark_text(align="left", dx=5)
        .encode(
            y=alt.Y("項目:N", sort="-x"),
            x=alt.X("寄与率:Q"),
            text=alt.Text("寄与率:Q", format=".2%"),
        )
    )
    return (ch + txt)

def fmt2(x):
    return "-" if pd.isna(x) else f"{float(x):.2f}"

# =========================
# Load
# =========================
geo = load_geojson(GEO_PATH)
metrics = load_metrics(MET_PATH)

need_base = ["code", "aging_rate", "net_migration_rate_per1000"]
need_access = ["hospital_p50_km", "hospital_p90_km"]
missing = [c for c in need_base if c not in metrics.columns] + [c for c in need_access if c not in metrics.columns]
if missing:
    st.error(f"metrics.csv に必要列がありません: {missing}")
    st.stop()

# =========================
# Sidebar
# =========================
st.sidebar.header("設定")

st.sidebar.subheader("医療アクセス指標")
access_mode = st.sidebar.selectbox("二次医療機関までの距離をどちらで評価？", ["中央値（P50）", "上位10%（P90）"])
access_col = "hospital_p50_km" if access_mode.startswith("中央値") else "hospital_p90_km"

st.sidebar.subheader("外れ値に強い正規化（分位点クリップ）")
q_low = st.sidebar.slider("下側分位点", 0.00, 0.20, 0.05, 0.01)
q_high = st.sidebar.slider("上側分位点", 0.80, 1.00, 0.95, 0.01)

st.sidebar.subheader("重み（合計は自動で正規化）")
w_aging = st.sidebar.slider("高齢化", 0.0, 1.0, 0.35, 0.05)
w_access = st.sidebar.slider("医療アクセス（距離）", 0.0, 1.0, 0.35, 0.05)
w_net = st.sidebar.slider("人口流出（純移動）", 0.0, 1.0, 0.30, 0.05)

s = w_aging + w_access + w_net
w = {"aging": w_aging / s if s else 1/3, "access": w_access / s if s else 1/3, "net": w_net / s if s else 1/3}

color_mode = st.sidebar.selectbox("地図の色分け", ["総合（スコア）", "高齢化", "医療アクセス", "人口流出（純移動）"])
top_n = st.sidebar.number_input("上位N表示", 3, 50, 10, 1)

metrics_tmp = compute_scores(metrics, w, access_col=access_col, q_low=q_low, q_high=q_high)

st.sidebar.subheader("地域選択")
area_id = st.sidebar.selectbox("市町村コード（N03_007）", metrics_tmp["code"].tolist())
selected = metrics_tmp.loc[metrics_tmp["code"] == area_id].iloc[0]

# =========================
# Prepare geojson with colors
# =========================
geo2 = merge_into_geojson(json.loads(json.dumps(geo)), metrics_tmp)

def pick_v(props):
    if color_mode == "総合（スコア）":
        return float(props.get("score01", 0.0))
    if color_mode == "高齢化":
        return float(props.get("n_aging", 0.0))
    if color_mode == "医療アクセス":
        return float(props.get("n_access", 0.0))
    if color_mode == "人口流出（純移動）":
        return float(props.get("n_net", 0.0))
    return 0.0

for feat in geo2["features"]:
    p = feat.get("properties", {})
    v = pick_v(p)
    feat["properties"]["_fill_color"] = score_to_color(v)

view_state = pdk.ViewState(latitude=34.23, longitude=135.17, zoom=7.2)

layer = pdk.Layer(
    "GeoJsonLayer",
    data=geo2,
    stroked=True,
    filled=True,
    get_fill_color="properties._fill_color",
    get_line_color=[220, 220, 220, 120],
    line_width_min_pixels=0.6,
    pickable=True,
    auto_highlight=True,
)

tooltip = {
    "html": """
    <b>{name}</b><br/>
    コード: {code}<br/>
    総合スコア: {score:.2f}<br/>
    高齢化率: {aging_rate:.2%}<br/>
    医療アクセスP50(km): {hospital_p50_km:.2f}<br/>
    医療アクセスP90(km): {hospital_p90_km:.2f}<br/>
    純移動(千人当たり): {net_migration_rate_per1000:.2f}<br/>
    主要因: {top_driver}<br/>
    """,
    "style": {"backgroundColor": "rgba(20,20,20,0.85)", "color": "white"},
}

deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)

# =========================
# UI
# =========================
st.title("和歌山県：重点エリア可視化（説明可能性強化版）")
st.caption("国土数値情報（行政区域医療機関）＋ e-Stat（住基移動国勢調査）を統合し、分位点クリップ正規化重み付けで重点候補を抽出")

c1, c2 = st.columns([1.5, 1.0], gap="large")

with c1:
    st.subheader("地図（ホバーで値表示）")
    st.pydeck_chart(deck, use_container_width=True)

with c2:
    st.subheader("選択地域の指標")
    st.write(f"**市町村名:** {selected.get('pop_name', '')}")
    st.write(f"**コード:** {selected['code']}")
    st.metric("総合スコア", f"{selected['score']:.2f}")

    st.write({
        "高齢化率（%）": fmt2(float(selected["aging_rate"]) * 100.0) if pd.notna(selected["aging_rate"]) else "-",
        "医療アクセス P50（km）": fmt2(selected["hospital_p50_km"]),
        "医療アクセス P90（km）": fmt2(selected["hospital_p90_km"]),
        "純移動（千人当たり）": fmt2(selected["net_migration_rate_per1000"]),
        "主要因（寄与最大）": str(selected["top_driver"]),
    })

    st.subheader("寄与内訳（重み込みの寄与率）")
    st.altair_chart(contrib_chart(selected), use_container_width=True)

st.divider()

st.subheader(f"重点候補 上位{int(top_n)}（重み反映）")
rank = metrics_tmp.sort_values("score", ascending=False).head(int(top_n)).copy()
rank["rank"] = np.arange(1, len(rank) + 1)
rank["reason"] = rank["top_driver"].apply(lambda x: f"主要因は「{x}」")

# 表示用の整形（小数2桁＆日本語列名）
for c in ["score", "aging_rate", "hospital_p50_km", "hospital_p90_km", "net_migration_rate_per1000"]:
    if c in rank.columns:
        rank[c] = pd.to_numeric(rank[c], errors="coerce").round(2)

rename_map = {
    "rank": "順位",
    "code": "市町村コード",
    "pop_name": "市町村名",
    "score": "総合スコア",
    "reason": "主因",
    "aging_rate": "高齢化率",
    "hospital_p50_km": "医療アクセスP50(km)",
    "hospital_p90_km": "医療アクセスP90(km)",
    "net_migration_rate_per1000": "純移動(千人当たり)",
}
rank = rank.rename(columns=rename_map)

disp_cols = ["順位","市町村コード","市町村名","総合スコア","主因","高齢化率","医療アクセスP50(km)","医療アクセスP90(km)","純移動(千人当たり)"]
disp_cols = [c for c in disp_cols if c in rank.columns]
st.dataframe(rank[disp_cols], use_container_width=True, hide_index=True)
