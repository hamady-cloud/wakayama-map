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
st.divider()

# =========================
# Site Purpose & Explanation
# =========================
st.markdown("""
<style>
.explanation-container {
    background-color: #f0f2f6;
    padding: 2rem;
    border-radius: 10px;
    margin-top: 2rem;
    border-left: 5px solid #0068c9;
}
.explanation-container h2 {
    color: #0068c9;
    border-bottom: 2px solid #0068c9;
    padding-bottom: 0.5rem;
    margin-top: 1.5rem;
}
.explanation-container h3 {
    color: #1f77b4;
    margin-top: 1.2rem;
}
.explanation-container b, .explanation-container strong {
    color: #d33682;
}
.explanation-container ul {
    margin-bottom: 1rem;
}
.explanation-container li {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="explanation-container">

# 和歌山県「重点支援候補エリア」可視化MAP

## 0. 目的（何を作ったか）

和歌山県内の市町村について、複数の地域課題指標を統合し **「重点支援の候補エリア」**を可視化するWebアプリを作成した。
統合した指標は以下の3系統である。

- **高齢化率**（国勢調査）
- **人口流出**（純移動：住基移動）
- **医療アクセス**（二次医療機関までの距離：1kmメッシュ人口×距離分布 P50/P90）

さらに、総合スコアに対して各指標がどれだけ効いているかを **寄与内訳（重み込み寄与率）**として横棒グラフで表示し、スコアの説明可能性（Why）を担保した。

## 1. 意思決定（So what）

本ツールは「どこを先に支援するべきか」を決める一次選定（スクリーニング）に使う。

- **自治体・県庁**：限られた予算・人員で支援対象を絞る（優先順位付け）
- **医療・福祉・地域政策**：高齢化、人口減、医療アクセスの“複合課題”を同一画面で整理し、施策ポートフォリオの議論に入る
- **調査・コンサル**：初期診断→追加調査（地域実態・交通・医療圏連携・拠点配置等）の論点を短時間で作る

## 2. 論点と仮説（What）

- **論点A：複合課題の重なり**
  高齢化・人口流出・医療アクセスの悪さが同時に大きい市町村はどこか
- **論点B：医療アクセスの“取り残され”**
  平均ではなく **P90（上位10%が置かれている距離）**で、取り残されやすい住民が多い地域を炙り出せるか
- **論点C：施策優先度の説明可能性**
  なぜその市町村が高スコアなのか（どの指標が効いているのか）を寄与内訳で説明できるか

## 3. 現状の到達点（実装済み）

- **地図**：市町村ポリゴン（GeoJSON）を pydeck で表示
- **色分け**：総合スコア / 高齢化 / 医療アクセス / 人口流出 を切替表示
- **医療アクセス**：P50（中央値）/ P90（上位10%点） を切替表示
- **正規化**：外れ値に強い 分位点クリップ→min-max（0–1）
- **重み**：高齢化/医療アクセス/人口流出をスライダーで調整（合計自動正規化）
- **寄与内訳**：Altairで 重み込み寄与率 を横棒グラフで表示
- **表示改善**：小数点2桁、英語項目は日本語表記

## 4. 重要な設計判断（なぜこうしたか）

### 4.1 医療アクセスを「代表点距離」ではなく「距離分布」にした
市町村ポリゴンの代表点から病院までの距離は、居住の偏り（海沿い・山間部・集落分布）を反映しにくい。
そこで本アプリでは、1kmメッシュ人口×最近傍（二次医療機関）距離を計算し、市町村内の距離分布から P50/P90/mean を算出する方式を採用した。
これにより医療アクセスを「地理の中心」ではなく 住民ベースの到達困難さとして説明できる。

### 4.2 外れ値（山奥など）に引っ張られない正規化
距離は外れ値が出やすく、min-maxだけだと極端値に引っ張られて比較が歪む。
そのため分位点（例：5%〜95%）でクリップした上で0–1に正規化し、現実的な差を見分けられるようにした。

## 5. データと定義（Evidence）

- **行政界**：国土数値情報（N03）市町村ポリゴン
- **医療機関**：国土数値情報P04（医療機関）をベースに二次医療機関を抽出（名称マッチ＋手動補正あり）
- **人口**：e-Statの1kmメッシュ人口（KEY_CODEでJOIN）
- **人口移動**：e-Stat出力（転入・転出→純移動）
- **高齢化率**：国勢調査データから算出

主要指標は、高齢化率・純移動・医療アクセス（P50/P90/mean）を0–1正規化し、重みをかけて総合スコア化する。総合スコアは寄与内訳（重み込み寄与率）で説明できる形にしている。

## 6. 結果（Findings：発見トップ3）

（例：和歌山県）
総合スコア上位の市町村は、北山村（score=0.88）、紀美野町（0.87）、高野町（0.86）。内訳を見ると、上位3件でも主因が異なる。

- **北山村（score=0.88）**：主要因が医療アクセスである。寄与内訳でも医療アクセスの寄与が0.45と大きく、住民ベースで見た到達困難さが総合スコアを押し上げている。医療アクセスは距離分布で見ると、P50=32.49km、P90=38.22kmと高水準で、中心部の平均では見えにくい“遠い住民側”まで含めた課題が示唆される。加えて純移動は **-9.9** であり、人口流出も同時に観察されるため、医療アクセス改善（交通・遠隔・連携）と生活圏支援をセットで検討する余地がある。
- **紀美野町（0.87）**：主要因は 高齢化。高齢化の寄与が大きく、福祉・医療需要の増加側の課題が中心である。
- **高野町（0.86）**：主要因は 医療アクセス。人口分布を踏まえた距離分布（P50/P90）で見たときに、到達困難層が相対的に大きい可能性が示唆される。

このように、総合順位だけでなく「なぜ高いのか（寄与内訳）」まで同時に見せることで、同じ“上位”でも打ち手の種類が違うことを明確にできる。

## 7. 示唆（Implications：打ち手案トップ3）

- **優先度A：医療アクセス主因エリア（北山村・高野町型）**
  P90が大きい＝“取り残され”が出やすい可能性があるため、交通支援、救急搬送の運用、出張診療、遠隔医療、拠点・連携の再設計など「到達困難の解消」に直接効く打ち手を優先検討する。
- **優先度B：高齢化主因エリア（紀美野町型）**
  医療アクセスだけでなく、在宅・介護・予防・地域包括ケアなど「需要増」側の対策を厚くする。医療資源の配置というより、生活圏で支える設計（人材・サービス連携）が論点になる。
- **優先度C：上位市町村を“タイプ分け”して施策を当てる**
  総合スコア上位を一括で扱わず、寄与内訳で「医療アクセス型／高齢化型／人口流出型」に分類し、施策パッケージを切り替える。これにより合意形成が速くなり、説明責任も担保できる。

## 8. 再現性（How：更新・再現手順）

1. 市町村ポリゴン（GeoJSON）を用意
2. 医療機関データを整備し、二次医療機関を定義（リスト化）
3. 1kmメッシュ形状とメッシュ人口をKEY_CODEでJOIN
4. メッシュ→最近傍（二次医療機関）距離を計算し、市町村単位に集計（P50/P90/mean）
5. 高齢化率・純移動と結合し、分位点クリップ→0–1正規化
6. 重みをかけて総合スコア算出、寄与内訳を可視化
7. 地図表示・切替・可視化・CSV等の出力を提供

</div>
""", unsafe_allow_html=True)
