#!/usr/bin/env bash
# run_experiment.sh  –  GPU を環境変数で指定して実験を実行

# ---------- 引数 ----------
GPU_ID=${1:-2}          # 第1引数: GPU 番号 (デフォルト 2)
TEAMS=${2:-TeamMixed}   # 第2引数: チーム名 CSV
N_CASE=${3:-50}         # 第3引数: 問題数
WORKERS=${4:-2}         # 第4引数: プロセス数

echo "== Run experiment on GPU ${GPU_ID} =="
export CUDA_VISIBLE_DEVICES=${GPU_ID}  # GPU 制限  :contentReference[oaicite:2]{index=2}

python main.py \
  --teams "${TEAMS}" \
  --n_case "${N_CASE}" \
  --workers "${WORKERS}"
