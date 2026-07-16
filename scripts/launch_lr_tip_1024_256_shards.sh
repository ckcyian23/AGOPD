#!/usr/bin/env bash
set -euo pipefail

cd /data_b/qtwei/ckcyi/AGOPD

BASE_OUT="outputs/lr_tip_1024_256_shards"
PROMPTS_FILE="data/prompts_1024.jsonl"
mkdir -p "${BASE_OUT}"

for SHARD in 0 1 2 3 4 5 6 7; do
  GPU=$((SHARD / 2))
  SESSION="lr_tip_1024_s${SHARD}"
  OUT_DIR="${BASE_OUT}/shard_${SHARD}"
  mkdir -p "${OUT_DIR}"

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session ${SESSION} already exists; skipping."
    continue
  fi

  tmux new -d -s "${SESSION}" \
    "cd /data_b/qtwei/ckcyi/AGOPD && CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python -m agopd.experiments.offline_lr_tip_eval --prompts-file ${PROMPTS_FILE} --limit 1024 --num-shards 8 --shard-index ${SHARD} --max-new-tokens 256 --execution-mode joint --device cuda --dtype float16 --budgets 0.05,0.1,0.2 --output-kl-direction forward --output-dir ${OUT_DIR} > ${OUT_DIR}/run.log 2>&1"
  echo "Started ${SESSION} on GPU ${GPU}; output ${OUT_DIR}"
done

tmux ls
