import os
import secrets
from pathlib import Path

DATA_DIR = Path(os.environ.get("MINE_DATA_DIR", "/workspace/mine-data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = DATA_DIR / "adapters"
ADAPTER_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "mine.db"

BASE_MODEL = os.environ.get("MINE_BASE_MODEL", "continuedev/instinct")
SERVED_BASE_NAME = "base"
VLLM_URL = os.environ.get("MINE_VLLM_URL", "http://127.0.0.1:8001")
TRAIN_PYTHON = os.environ.get("MINE_TRAIN_PYTHON", "/workspace/venv-train/bin/python")

# Train automatically once this many new learnable examples have arrived.
TRAIN_EVERY = int(os.environ.get("MINE_TRAIN_EVERY", "40"))
MIN_EXAMPLES = int(os.environ.get("MINE_MIN_EXAMPLES", "30"))
MAX_TOKENS = int(os.environ.get("MINE_MAX_TOKENS", "512"))
# Wait for user to be idle for this many seconds before auto-training.
IDLE_BEFORE_TRAIN = int(os.environ.get("MINE_IDLE_BEFORE_TRAIN", "120"))


def _token() -> str:
    env = os.environ.get("MINE_TOKEN")
    if env:
        return env
    f = DATA_DIR / "token.txt"
    if f.exists():
        return f.read_text().strip()
    t = "mine_" + secrets.token_urlsafe(24)
    f.write_text(t)
    os.chmod(f, 0o600)
    return t


API_TOKEN = _token()
