"""SQLite storage for suggestions, feedback, and model versions."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from .config import DB_PATH

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL")
_conn.executescript(
    """
CREATE TABLE IF NOT EXISTS suggestions (
    id TEXT PRIMARY KEY,
    created REAL,
    model TEXT,
    filepath TEXT,
    prompt TEXT,
    original TEXT,
    suggestion TEXT,
    latency_ms INTEGER,
    outcome TEXT,          -- accepted | rejected | ignored | NULL
    final TEXT,            -- what the region looked like ~20s after rejection
    label TEXT,            -- training target derived from outcome
    negative TEXT,         -- rejected output, for preference training
    holdout INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_outcome ON suggestions(outcome);
CREATE TABLE IF NOT EXISTS models (
    version INTEGER PRIMARY KEY,
    name TEXT,
    path TEXT,
    created REAL,
    n_train INTEGER,
    score REAL,
    base_score REAL,
    promoted INTEGER,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""
)
_conn.commit()


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        return _conn.execute(sql, args).fetchall()


def x(sql: str, args: tuple = ()) -> None:
    with _lock:
        _conn.execute(sql, args)
        _conn.commit()


def kv_get(k: str, default=None):
    r = q("SELECT v FROM kv WHERE k=?", (k,))
    return json.loads(r[0]["v"]) if r else default


def kv_set(k: str, v) -> None:
    x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))


def now() -> float:
    return time.time()
