# make.ps1
# 前処理（metrics生成）→ Streamlit起動までをワンコマンド化

$ErrorActionPreference = "Stop"

$APP_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA_DIR = Join-Path $APP_DIR "data"
$SRC_DIR  = Join-Path $APP_DIR "src"
$SVC_DIR  = Join-Path $SRC_DIR "services"

$PREP_PY  = Join-Path $SVC_DIR "prep_metrics.py"
$DIST_PY  = Join-Path $SVC_DIR "calc_hospital_distance.py"
$APP_PY   = Join-Path $APP_DIR "app.py"

$AREAS_EXPECT = Join-Path $DATA_DIR "areas_wakayama.geojson"
$AREAS_SRC    = Join-Path $DATA_DIR "N03-23_30_230101.geojson"

$HOSP_EXPECT  = Join-Path $DATA_DIR "hospital.geojson"
$HOSP_SHP     = Join-Path $DATA_DIR "P04-14_30-g_MedicalInstitution.shp"

$IN_CSV   = Join-Path $DATA_DIR "in.csv"
$OUT_CSV  = Join-Path $DATA_DIR "out.csv"
$POP_CSV  = Join-Path $DATA_DIR "pop.csv"

$MET_BASE = Join-Path $DATA_DIR "metrics_base.csv"
$MET_OUT  = Join-Path $DATA_DIR "metrics.csv"

function Assert-File($path, $msg) {
  if (-not (Test-Path $path)) { throw $msg }
  $item = Get-Item $path
  if ($item.Attributes -match "Directory") { throw "$msg （※ $path がフォルダになっています。ファイルにしてください）" }
}

function Ensure-AreasGeoJSON() {
  # areas_wakayama.geojson がフォルダなら退避
  if (Test-Path $AREAS_EXPECT) {
    $it = Get-Item $AREAS_EXPECT
    if ($it.Attributes -match "Directory") {
      Write-Host "[fix] areas_wakayama.geojson is a directory -> renaming to areas_wakayama_geojson_dir"
      Rename-Item $AREAS_EXPECT "areas_wakayama_geojson_dir"
    }
  }

  # 期待ファイルが無ければ、N03ファイルからコピーして作る
  if (-not (Test-Path $AREAS_EXPECT)) {
    Assert-File $AREAS_SRC "行政区域GeoJSONが見つかりません: $AREAS_SRC を data に置いてください"
    Write-Host "[prep] copy $AREAS_SRC -> $AREAS_EXPECT"
    Copy-Item $AREAS_SRC $AREAS_EXPECT -Force
  }

  Assert-File $AREAS_EXPECT "行政区域GeoJSONの読み込み準備に失敗しました"
}

function Ensure-HospitalGeoJSON() {
  # hospital.geojson がフォルダなら退避
  if (Test-Path $HOSP_EXPECT) {
    $it = Get-Item $HOSP_EXPECT
    if ($it.Attributes -match "Directory") {
      Write-Host "[fix] hospital.geojson is a directory -> renaming to hospital_geojson_dir"
      Rename-Item $HOSP_EXPECT "hospital_geojson_dir"
    }
  }

  # hospital.geojson が無ければ Shapefile から作る
  if (-not (Test-Path $HOSP_EXPECT)) {
    Assert-File $HOSP_SHP "医療機関Shapefileが見つかりません: $HOSP_SHP を data に置いてください（.dbf/.shx/.prjも同一ベース名で必要）"
    Write-Host "[prep] converting shapefile -> GeoJSON: $HOSP_SHP -> $HOSP_EXPECT"
    python -c "import os; os.environ['SHAPE_RESTORE_SHX']='YES'; import geopandas as gpd; src=r'$HOSP_SHP'; dst=r'$HOSP_EXPECT'; gdf=gpd.read_file(src); gdf.to_file(dst, driver='GeoJSON'); print('saved:', dst, 'rows=', len(gdf))"
  }

  Assert-File $HOSP_EXPECT "医療機関GeoJSONの読み込み準備に失敗しました"
}

Write-Host "==============================="
Write-Host " Wakayama Map: build & run"
Write-Host " APP_DIR : $APP_DIR"
Write-Host " DATA_DIR: $DATA_DIR"
Write-Host "==============================="

# 入力CSVの存在確認
Assert-File $IN_CSV  "in.csv が見つかりません: $IN_CSV"
Assert-File $OUT_CSV "out.csv が見つかりません: $OUT_CSV"
Assert-File $POP_CSV "pop.csv が見つかりません: $POP_CSV"

# GeoJSONの準備（フォルダ問題を自動回避）
Ensure-AreasGeoJSON
Ensure-HospitalGeoJSON

# 前処理スクリプト存在確認
Assert-File $PREP_PY "prep_metrics.py が見つかりません: $PREP_PY"
Assert-File $DIST_PY "calc_hospital_distance.py が見つかりません: $DIST_PY"
Assert-File $APP_PY  "app.py が見つかりません: $APP_PY"

# 1) 指標生成
Write-Host "[run] prep_metrics.py"
python $PREP_PY

if (-not (Test-Path $MET_BASE)) {
  throw "metrics_base.csv が生成されていません: $MET_BASE"
}

# 2) 最寄り病院距離の付与
Write-Host "[run] calc_hospital_distance.py"
python $DIST_PY

if (-not (Test-Path $MET_OUT)) {
  throw "metrics.csv が生成されていません: $MET_OUT"
}

Write-Host "[ok] metrics.csv generated: $MET_OUT"

# 3) 起動
Write-Host "[run] streamlit app"
Set-Location $APP_DIR
streamlit run $APP_PY
