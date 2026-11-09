# Mine: a self-hosted next-edit model that learns from you

Mine works like Cursor Tab or Continue's Instinct, but it runs on your own GPU and keeps training on your own coding.

- **Model:** Continue's open Instinct model (7.6B parameters, fine-tuned from Qwen2.5-Coder-7B, Apache-2.0), served with vLLM.
- **Editor:** a VS Code extension.
  - Shows ghost text when the model only wants to insert at your cursor. Press **Tab** to accept.
  - Shows a highlighted edit when it wants to rewrite nearby lines. Press **Tab** to apply, **Esc** to dismiss.
  - After you accept, it looks for your next edit straight away (Tab, Tab, Tab…).
- **Learning loop:**
  - Every suggestion is logged.
    - When you **accept** one, that's a "do more of this" example.
    - When you **reject or ignore** one, the extension waits 20 seconds and then sends what you actually wrote in that spot. Your version becomes the right answer, and the rejected one becomes a "not this" example.
  - After every 40 new examples, once you've been idle for 2 minutes, the server trains a fresh LoRA adapter on all your feedback. It runs two passes: supervised fine-tuning (SFT) on your real edits, then DPO, a preference pass, on (what you did) vs. (what you rejected).
  - The new version is tested against held-out examples from your own coding. It only replaces the current one if it scores at least as well, and the switch is live with no restart.
  - The dashboard shows the accept rate for each version, so you can see whether it's learning you. One tap rolls back.

```
VS Code (your Mac/PC) ──HTTPS──▶ Mine gateway :8000 ──▶ vLLM :8001 (Instinct + your LoRA)
     ▲   accept/reject                 │
     └───────────────────────────────  ├─ SQLite: every suggestion + outcome
                                       └─ trainer (QLoRA SFT + DPO) → new adapter → eval → hot-swap
```

## What you need
- A rented GPU with **48 GB of memory or more**, such as an RTX A6000, L40S, RTX 6000 Ada or A100. It serves the model and trains at the same time. Expect roughly $0.40–$1 per hour, and stop the pod when you're not coding.
- VS Code on your MacBook or Windows laptop.

## Files
- `server/`: the gateway, the trainer and the dashboard. Setup runs on the GPU machine.
- `extension/mine-next-edit.vsix`: the VS Code extension, ready to install.

## Server commands (run in the pod's terminal)
```
bash /workspace/mine/server/setup.sh   # one time, about 10–15 min
bash /workspace/mine/server/start.sh   # each time the pod starts
cat /workspace/mine-data/token.txt     # shows your API token
tail -f /workspace/mine-data/vllm.log  # model loading progress
```

## Tuning (environment variables before start.sh)
| Variable | Default | Meaning |
|---|---|---|
| `MINE_TRAIN_EVERY` | 40 | Number of new examples that triggers auto-training |
| `MINE_IDLE_BEFORE_TRAIN` | 120 | Seconds you must be idle before training starts |
| `MINE_MIN_DPO_PAIRS` | 20 | Number of rejection pairs needed before the DPO pass runs |
| `MINE_LORA_R` | 16 | LoRA rank (max 32) |
| `MINE_GPU_UTIL` | auto | Share of GPU memory vLLM may use |
| `MINE_BASE_MODEL` | continuedev/instinct | Base model |

## API
All endpoints need the header `Authorization: Bearer <token>`.
`POST /v1/next_edit`, `POST /v1/feedback`, `GET /v1/stats`, `POST /v1/train`, `POST /v1/rollback?to=base|mine-vN`, `GET /v1/train_log`, `GET /` (dashboard).
