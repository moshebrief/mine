#!/usr/bin/env bash
# Starts the model (vLLM, internal port 8001) and the Mine gateway (public port 8000).
# Keeps both running; restarts them if they crash. Logs: /workspace/mine-data/*.log
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf
export MINE_DATA_DIR=/workspace/mine-data
export MINE_TRAIN_PYTHON=/workspace/venv-train/bin/python
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
BASE="${MINE_BASE_MODEL:-continuedev/instinct}"
mkdir -p "$MINE_DATA_DIR"
if [ ! -s "$MINE_DATA_DIR/token.txt" ]; then
  python3 -c "import secrets; print('mine_' + secrets.token_urlsafe(24))" > "$MINE_DATA_DIR/token.txt"
  chmod 600 "$MINE_DATA_DIR/token.txt"
fi

# Give vLLM ~24 GB and leave the rest of the GPU for training.
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${TOTAL_MB:-0}" -lt 40000 ]; then
  echo "⚠️  This GPU has ${TOTAL_MB} MB. Mine needs a 48 GB+ GPU to serve AND train at the same time."
fi
UTIL=${MINE_GPU_UTIL:-$(python3 -c "print(max(0.30, min(0.60, 24000/${TOTAL_MB:-48000})))")}

pkill -f "vllm serve" 2>/dev/null; pkill -f "mine.gateway" 2>/dev/null; sleep 2

(
  while true; do
    /workspace/venv-serve/bin/vllm serve "$BASE" \
      --served-model-name base --host 127.0.0.1 --port 8001 \
      --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization "$UTIL" \
      --enable-lora --max-lora-rank 32 --max-loras 3 --enable-prefix-caching \
      >> "$MINE_DATA_DIR/vllm.log" 2>&1
    echo "vLLM exited; restarting in 5s" >> "$MINE_DATA_DIR/vllm.log"; sleep 5
  done
) &

(
  while true; do
    /workspace/venv-serve/bin/uvicorn mine.gateway:app --host 0.0.0.0 --port 8000 \
      >> "$MINE_DATA_DIR/gateway.log" 2>&1
    echo "gateway exited; restarting in 3s" >> "$MINE_DATA_DIR/gateway.log"; sleep 3
  done
) &

sleep 4
echo
echo "=============================================================="
echo " Mine is starting. The model takes 2–5 minutes to load."
echo " Your API token (copy it into VS Code and the dashboard):"
echo
echo "   $(cat $MINE_DATA_DIR/token.txt 2>/dev/null || echo 'not created yet – run: cat /workspace/mine-data/token.txt')"
echo
echo " Watch progress:  tail -f /workspace/mine-data/vllm.log"
echo "=============================================================="
