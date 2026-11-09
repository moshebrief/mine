"""Mine gateway: next-edit API, feedback collection, and the self-training loop.

Runs in front of a local vLLM server. Exposed port: 8000.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import db
from .config import (
    ADAPTER_DIR, API_TOKEN, DATA_DIR, IDLE_BEFORE_TRAIN, MAX_TOKENS, MIN_EXAMPLES,
    SERVED_BASE_NAME, TRAIN_EVERY, TRAIN_PYTHON, VLLM_URL,
)
from .prompt import build_prompt, messages_for, parse_completion, same

app = FastAPI(title="Mine", version="1.0")
http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
TRAIN_LOG = DATA_DIR / "train.log"
state = {"training": False, "last_activity": 0.0, "train_error": None, "phase": "idle"}


# ---------------------------------------------------------------- auth
def auth(authorization: str = Header(default="")):
    if authorization.removeprefix("Bearer ").strip() != API_TOKEN:
        raise HTTPException(401, "bad token")


# ---------------------------------------------------------------- vLLM helpers
def active_model() -> str:
    return db.kv_get("active_model", SERVED_BASE_NAME)


async def vllm_chat(model: str, user_prompt: str, max_tokens: int) -> str:
    r = await http.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages_for(user_prompt),
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


async def vllm_load_adapter(name: str, path: str) -> None:
    r = await http.post(f"{VLLM_URL}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
    if r.status_code >= 400 and "already" not in r.text.lower():
        raise RuntimeError(f"load adapter failed: {r.status_code} {r.text}")


async def vllm_unload_adapter(name: str) -> None:
    try:
        await http.post(f"{VLLM_URL}/v1/unload_lora_adapter", json={"lora_name": name})
    except Exception:
        pass


async def wait_for_vllm(timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = await http.get(f"{VLLM_URL}/v1/models")
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(3)
    raise RuntimeError("vLLM did not come up")


@app.on_event("startup")
async def startup():
    async def boot():
        await wait_for_vllm()
        name = active_model()
        if name != SERVED_BASE_NAME:
            rows = db.q("SELECT path FROM models WHERE name=?", (name,))
            if rows and Path(rows[0]["path"]).exists():
                try:
                    await vllm_load_adapter(name, rows[0]["path"])
                except Exception as e:
                    print("could not reload adapter, falling back to base:", e, file=sys.stderr)
                    db.kv_set("active_model", SERVED_BASE_NAME)
            else:
                db.kv_set("active_model", SERVED_BASE_NAME)
        print(f"[mine] ready. active model = {active_model()}", flush=True)
    asyncio.create_task(boot())
    asyncio.create_task(auto_train_loop())


# ---------------------------------------------------------------- API models
class Edit(BaseModel):
    filepath: str
    diff: str


class Ctx(BaseModel):
    filepath: str
    content: str


class NextEditReq(BaseModel):
    filepath: str
    lines: list[str]
    lines_start: int = 0
    cursor_line: int
    cursor_char: int
    recent_edits: list[Edit] = Field(default_factory=list)
    context: list[Ctx] = Field(default_factory=list)


class FeedbackReq(BaseModel):
    id: str
    outcome: str                 # accepted | rejected | ignored
    final: str | None = None     # region text ~20s after a rejection


# ---------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    try:
        r = await http.get(f"{VLLM_URL}/v1/models", timeout=3)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return {"gateway": True, "model_ready": ok, "active_model": active_model()}


@app.post("/v1/next_edit", dependencies=[Depends(auth)])
async def next_edit(req: NextEditReq):
    state["last_activity"] = time.time()
    bp = build_prompt(
        req.filepath, req.lines, req.lines_start, req.cursor_line, req.cursor_char,
        [e.model_dump() for e in req.recent_edits], [c.model_dump() for c in req.context],
    )
    model = active_model()
    max_tokens = min(MAX_TOKENS, len(bp.original_region) // 2 + 200)
    t0 = time.time()
    try:
        raw = await vllm_chat(model, bp.user_prompt, max_tokens)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"model not ready: {e}")
    new_region = parse_completion(raw)
    latency = int((time.time() - t0) * 1000)
    if same(new_region, bp.original_region):
        return {"id": None, "latency_ms": latency, "model": model}
    sid = uuid.uuid4().hex
    db.x(
        "INSERT INTO suggestions(id,created,model,filepath,prompt,original,suggestion,latency_ms) VALUES(?,?,?,?,?,?,?,?)",
        (sid, db.now(), model, req.filepath, bp.user_prompt, bp.original_region, new_region, latency),
    )
    return {
        "id": sid,
        "model": model,
        "latency_ms": latency,
        "region_start": bp.region_start,
        "region_end": bp.region_end,
        "original": bp.original_region,
        "suggestion": new_region,
    }


@app.post("/v1/feedback", dependencies=[Depends(auth)])
async def feedback(req: FeedbackReq):
    rows = db.q("SELECT original, suggestion FROM suggestions WHERE id=?", (req.id,))
    if not rows:
        raise HTTPException(404, "unknown suggestion id")
    original, suggestion = rows[0]["original"], rows[0]["suggestion"]
    outcome = req.outcome if req.outcome in ("accepted", "rejected", "ignored") else "rejected"
    label, negative = None, None
    if outcome == "accepted":
        label = suggestion
    elif req.final is not None:
        final = req.final.strip("\n")
        if same(final, suggestion):
            label, outcome = final, "accepted"   # they typed it themselves
        elif same(final, original):
            label, negative = original, suggestion   # "no edit" was right
        else:
            label, negative = final, suggestion      # what they really did
    holdout = 1 if label is not None and random.random() < 0.12 else 0
    db.x(
        "UPDATE suggestions SET outcome=?, final=?, label=?, negative=?, holdout=? WHERE id=?",
        (outcome, req.final, label, negative, holdout, req.id),
    )
    return {"ok": True}


@app.get("/v1/stats", dependencies=[Depends(auth)])
async def stats():
    per_model = [
        dict(r) for r in db.q(
            """SELECT model,
                      SUM(outcome='accepted') AS accepted,
                      SUM(outcome IN ('rejected','ignored')) AS rejected,
                      COUNT(outcome) AS shown,
                      CAST(AVG(latency_ms) AS INT) AS avg_latency_ms
               FROM suggestions WHERE outcome IS NOT NULL GROUP BY model ORDER BY MIN(created)"""
        )
    ]
    for m in per_model:
        m["accept_rate"] = round(m["accepted"] / m["shown"], 3) if m["shown"] else None
    learnable = db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL")[0]["n"]
    since = learnable - db.kv_get("trained_on_count", 0)
    models = [dict(r) for r in db.q("SELECT version,name,created,n_train,score,base_score,promoted,notes FROM models ORDER BY version")]
    return {
        "active_model": active_model(),
        "training": state["training"],
        "phase": state["phase"],
        "train_error": state["train_error"],
        "learnable_examples": learnable,
        "new_since_last_train": since,
        "train_every": TRAIN_EVERY,
        "per_model": per_model,
        "versions": models,
    }


@app.post("/v1/train", dependencies=[Depends(auth)])
async def train_now():
    if state["training"]:
        return {"started": False, "reason": "already training"}
    n = db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL AND holdout=0")[0]["n"]
    if n < 5:
        return {"started": False, "reason": f"need at least 5 examples, have {n}"}
    asyncio.create_task(run_training())
    return {"started": True}


@app.post("/v1/rollback", dependencies=[Depends(auth)])
async def rollback(to: str = SERVED_BASE_NAME):
    if to != SERVED_BASE_NAME:
        rows = db.q("SELECT path FROM models WHERE name=?", (to,))
        if not rows:
            raise HTTPException(404, "no such version")
        await vllm_load_adapter(to, rows[0]["path"])
    db.kv_set("active_model", to)
    return {"active_model": to}


@app.get("/v1/train_log", dependencies=[Depends(auth)], response_class=PlainTextResponse)
async def train_log():
    if not TRAIN_LOG.exists():
        return ""
    return TRAIN_LOG.read_text(errors="replace")[-6000:]


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()


# ---------------------------------------------------------------- training loop
def export_dataset(out_dir: Path) -> tuple[int, int]:
    """Writes sft.jsonl / dpo.jsonl from all labeled, non-holdout feedback."""
    rows = db.q(
        "SELECT prompt, original, label, negative FROM suggestions "
        "WHERE label IS NOT NULL AND holdout=0 ORDER BY created DESC LIMIT 6000"
    )
    edits, noops, prefs = [], [], []
    for r in rows:
        ex = {"prompt": messages_for(r["prompt"]), "completion": [{"role": "assistant", "content": r["label"]}]}
        (noops if same(r["label"], r["original"]) else edits).append(ex)
        if r["negative"] and not same(r["negative"], r["label"]):
            prefs.append({
                "prompt": messages_for(r["prompt"]),
                "chosen": [{"role": "assistant", "content": r["label"]}],
                "rejected": [{"role": "assistant", "content": r["negative"]}],
            })
    # keep "don't change anything" examples to <= 1/3 of the data so it stays helpful
    noops = noops[: max(1, len(edits) // 2)]
    sft = edits + noops
    random.shuffle(sft)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sft.jsonl", "w") as f:
        for ex in sft:
            f.write(json.dumps(ex) + "\n")
    with open(out_dir / "dpo.jsonl", "w") as f:
        for ex in prefs:
            f.write(json.dumps(ex) + "\n")
    return len(sft), len(prefs)


def similarity(a: str, b: str) -> float:
    if same(a, b):
        return 1.0
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio() * 0.9


async def evaluate(model: str, rows) -> float:
    sem = asyncio.Semaphore(8)

    async def one(r):
        async with sem:
            try:
                out = parse_completion(await vllm_chat(model, r["prompt"], MAX_TOKENS))
            except Exception:
                return 0.0
            return similarity(out, r["label"])

    scores = await asyncio.gather(*(one(r) for r in rows))
    return sum(scores) / max(1, len(scores))


async def run_training():
    state.update(training=True, train_error=None, phase="exporting data")
    try:
        version = (db.q("SELECT COALESCE(MAX(version),0) v FROM models")[0]["v"]) + 1
        name = f"mine-v{version}"
        work = DATA_DIR / "runs" / name
        out = ADAPTER_DIR / name
        n_sft, n_dpo = export_dataset(work)
        learnable_now = db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL")[0]["n"]

        state["phase"] = f"training {name} on {n_sft} examples ({n_dpo} preference pairs)"
        with open(TRAIN_LOG, "w") as log:
            log.write(state["phase"] + "\n")
            log.flush()
            proc = await asyncio.create_subprocess_exec(
                TRAIN_PYTHON, "-m", "mine.train", "--data", str(work), "--out", str(out),
                cwd=str(Path(__file__).resolve().parent.parent),
                stdout=log, stderr=asyncio.subprocess.STDOUT,
            )
            code = await proc.wait()
        if code != 0 or not (out / "adapter_config.json").exists():
            raise RuntimeError(f"trainer exited with code {code}; see training log")

        state["phase"] = f"evaluating {name}"
        await vllm_load_adapter(name, str(out))
        hold = db.q(
            "SELECT prompt, label FROM suggestions WHERE label IS NOT NULL AND holdout=1 "
            "ORDER BY created DESC LIMIT 80"
        )
        current = active_model()
        if len(hold) >= 5:
            new_score = await evaluate(name, hold)
            cur_score = await evaluate(current, hold)
        else:
            new_score, cur_score = None, None
        promote = new_score is None or new_score >= cur_score - 0.005
        notes = "promoted" if promote else f"kept {current} (it scored higher)"
        db.x(
            "INSERT INTO models(version,name,path,created,n_train,score,base_score,promoted,notes) VALUES(?,?,?,?,?,?,?,?,?)",
            (version, name, str(out), db.now(), n_sft, new_score, cur_score, int(promote), notes),
        )
        db.kv_set("trained_on_count", learnable_now)
        if promote:
            db.kv_set("active_model", name)
            if current != SERVED_BASE_NAME:
                await vllm_unload_adapter(current)
        else:
            await vllm_unload_adapter(name)
        shutil.rmtree(work, ignore_errors=True)
        state["phase"] = f"done: {notes}"
    except Exception as e:  # noqa: BLE001
        state["train_error"] = str(e)
        state["phase"] = "failed"
        # don't immediately retry the same failure
        db.kv_set("trained_on_count", db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL")[0]["n"])
    finally:
        state["training"] = False


async def auto_train_loop():
    while True:
        await asyncio.sleep(60)
        try:
            if state["training"]:
                continue
            learnable = db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL AND holdout=0")[0]["n"]
            total = db.q("SELECT COUNT(*) n FROM suggestions WHERE label IS NOT NULL")[0]["n"]
            new = total - db.kv_get("trained_on_count", 0)
            idle = time.time() - state["last_activity"] > IDLE_BEFORE_TRAIN
            if learnable >= MIN_EXAMPLES and new >= TRAIN_EVERY and idle:
                await run_training()
        except Exception as e:  # noqa: BLE001
            print("auto-train error:", e, file=sys.stderr)
