#!/usr/bin/env bash
# One-time install on the GPU pod. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/.uv-cache
mkdir -p /workspace/mine-data /workspace/hf

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Creating serving environment (vLLM)…"
[ -d /workspace/venv-serve ] || uv venv --python 3.12 /workspace/venv-serve
uv pip install --python /workspace/venv-serve/bin/python -r requirements-serve.txt

echo "==> Creating training environment…"
[ -d /workspace/venv-train ] || uv venv --python 3.12 /workspace/venv-train
uv pip install --python /workspace/venv-train/bin/python -r requirements-train.txt

echo "==> Downloading the base model (Instinct, ~15 GB)…"
/workspace/venv-serve/bin/python -c "from huggingface_hub import snapshot_download as s; s('${MINE_BASE_MODEL:-continuedev/instinct}')"

echo
echo "✅ Setup finished. Now run:  bash $(pwd)/start.sh"
