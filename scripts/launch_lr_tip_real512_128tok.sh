#!/usr/bin/env bash
set -euo pipefail

cd /data_b/qtwei/ckcyi/AGOPD

PROMPTS_FILE="data/real_prompts_512.jsonl"

launch_group() {
  local NAME="$1"
  local EXTRA_ARGS="$2"
  local BASE_OUT="outputs/lr_tip_real512_128tok_${NAME}"
  mkdir -p "${BASE_OUT}"

  for SHARD in 0 1 2 3; do
    local GPU="${SHARD}"
    local SESSION="lr_tip_real512_${NAME}_s${SHARD}"
    local OUT_DIR="${BASE_OUT}/shard_${SHARD}"
    mkdir -p "${OUT_DIR}"

    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      echo "Session ${SESSION} already exists; skipping."
      continue
    fi

    tmux new -d -s "${SESSION}" \
      "cd /data_b/qtwei/ckcyi/AGOPD && CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python -m agopd.experiments.offline_lr_tip_eval --prompts-file ${PROMPTS_FILE} --limit 512 --num-shards 4 --shard-index ${SHARD} --max-new-tokens 128 --execution-mode joint --device cuda --dtype float16 --budgets 0.05,0.1,0.2 --output-kl-direction forward --progress-every 8 ${EXTRA_ARGS} --output-dir ${OUT_DIR} > ${OUT_DIR}/run.log 2>&1"
    echo "Started ${SESSION} on GPU ${GPU}; output ${OUT_DIR}"
  done
}

launch_group "main" ""
launch_group "window8" "--window 8"

tmux ls
