# 1) パス
$ROOT = "C:\Users\hamada\wakayama-map\app"
$README = Join-Path $ROOT "README.md"
$BK = Join-Path $ROOT ("README.md.bak_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

# 2) バックアップ（存在する場合）
if (Test-Path $README) {
  Copy-Item $README $BK -Force
  Write-Host "backup:" $BK
}

# 3) 差し替え
@'
# 和歌山県：重点エリア可視化（説明可能性強化版）

和歌山県内の市町村について、以下の観点を統合し「重点支援の候補エリア」を可視化する Streamlit アプリです。

- **高齢化率（国勢調査）**
- **人口流出（純移動：千人当たり、住民基本台帳人口移動）**
- **医療アクセス（“二次医療機関”までの距離：1kmメッシュ×距離分布 P50/P90）**

さらに、重み付けスコアの **寄与内訳（重み込み寄与率）** を表示し、説明可能性（Why）を強化しています。

---

## 1. できること（主な機能）

- 市町村ポリゴンを地図で表示し、スコアに応じて色分け
- 医療アクセス指標を **P50（中央値） / P90（上位10%）** で切り替え可能
- 外れ値に強い正規化：**分位点クリップ（例：下位5%〜上位95%）**
- 重み（高齢化/医療アクセス/人口流出）をスライダーで調整し、結果に反映
- 選択した市町村について、
  - 主要因（寄与最大）
  - 寄与内訳（横棒グラフ）
  - 上位Nのランキング
  を表示

---

## 2. 設計意図（なぜこの作りか）

### 2.1 「代表点距離」ではなく「距離分布（P50/P90）」を採用
市町村の代表点からの距離だけだと、山間部・沿岸部などで **実態（居住地の偏り）** を反映しづらいです。  
そこで **1kmメッシュ人口 × 二次医療機関までの距離** を作り、メッシュごとの距離分布から

- P50：住民の“中央値”のアクセス
- P90：アクセス不利な住民側（上位10%）の距離

を算出します。

### 2.2 説明可能性（Why）の強化
総合スコアだけだと「なぜ重点なのか」が伝わりにくいので、  
**重み込み寄与（c_aging / c_access / c_net）と寄与率（p_aging / p_access / p_net）** を可視化します。

---

## 3. データソース

### 3.1 行政区域（市町村ポリゴン）
- 国土数値情報：行政区域データ（N03）
- 例：`areas_wakayama.geojson`

### 3.2 医療機関（ポイント）
- 国土数値情報：医療機関データ（P04）
- Shapefile（.shp/.dbf/.shx/.prj）で配布されるため、GeoJSONに変換して利用
- 例：`hospital.geojson`（全医療機関）
- 二次医療機関のみ抽出版：`hospital_2nd.geojson`

### 3.3 人口（1kmメッシュ）
- e-Stat：1kmメッシュ人口（テキスト `tblT001100S30.txt` など）
- キー：`KEY_CODE`（メッシュコード）

### 3.4 人口移動（市町村）
- e-Stat：住民基本台帳人口移動報告（転入・転出）
- 例：`in.csv`, `out.csv`

### 3.5 国勢調査（市町村）
- e-Stat：国勢調査（高齢化率算出用）
- 例：`pop.csv`

---

## 4. ファイル構成（重要ファイル）
app/
app.py # Streamlit アプリ本体
README.md # このファイル
data/
areas_wakayama.geojson # 行政区域（和歌山）
hospital.geojson # 医療機関（全）
hospital_2nd.geojson # 二次医療機関（抽出）
metrics_base.csv # 基本指標（高齢化・純移動など）
metrics.csv # アプリ用（P50/P90など結合済み）
city_hospital_p5090.csv # 市町村別 P50/P90 集計結果
mesh1km_pop.geojson # 1kmメッシュ人口（GeoJSON）
tblT001100S30.txt # 1kmメッシュ人口（元データ）
MESH05xxxx.shp など # 1kmメッシュ形状（複数ファイル）
src/
services/
prep_metrics.py # 指標前処理（必要に応じて）
... # 追加の前処理スクリプト群
make.ps1 # 前処理→起動（任意：ワンコマンド化）

---

## 5. セットアップ

### 5.1 仮想環境（任意）
例：venv を使う場合

```powershell
cd C:\Users\hamada\wakayama-map\app
python -m venv .venv
.\.venv\Scripts\Activate.ps1

##5.2 依存関係
（requirements.txt がある場合）
最低限の主要パッケージ例：
streamlit
pandas / numpy
geopandas
pydeck
altair
pyogrio（環境により）

##6. 実行方法

##6.1 通常起動
cd C:\Users\hamada\wakayama-map\app
streamlit run app.py

##6.2 ワンコマンド化（make.ps1 がある場合）
cd C:\Users\hamada\wakayama-map\app
powershell -ExecutionPolicy Bypass -File .\make.ps1

##7. 出力指標の意味（アプリ内）

総合スコア：正規化後（0-1）の指標を重み付きで加算したもの
高齢化率（%）：aging_rate × 100
純移動（千人当たり）：転入−転出を人口で割り、千人当たりに換算した値
医療アクセス P50/P90（km）：二次医療機関までの距離分布（住民ベース）

##8. 注意点・改善余地（次の打ち手）

二次医療機関の定義は、抽出ルール（名称一致など）次第で変わるため、将来的には
医療機能データ（病床・診療科・救急指定など）
二次医療圏の定義
と合わせると説明力がさらに上がります。
距離は直線距離（投影座標系での距離）です。より実態に近づけるなら
道路ネットワーク距離
所要時間（車/公共交通）
も検討対象です（コストは上がります）。

##9. ライセンス・クレジット

国土数値情報 / e-Stat の利用規約に従って利用してください。
地図表示：pydeck / Map tiles（OpenStreetMap 等）
'@ | Set-Content -Encoding UTF8 $README
