"""Durable, atomic storage for all three GA3 agents.

Every question forbids keeping durable state in process memory and requires
store-before-respond, exact replay, changed-content conflicts and one winner
for the cancel/receipt race. SQLite in WAL mode with ``BEGIN IMMEDIATE`` write
transactions gives all of that in a single file with no extra service.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

DB_PATH = os.environ.get("GA3_DB_PATH") or os.path.join(
    os.environ.get("GA3_DATA_DIR", "/tmp"), "ga3-state.sqlite3"
)

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  ns         TEXT NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT,
  state      TEXT NOT NULL DEFAULT 'pending',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (ns, key)
);
CREATE INDEX IF NOT EXISTS records_ns_created ON records (ns, created_at);
"""


def _connect() -> sqlite3.Connection:
    global _initialised
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=20000")
    with _init_lock:
        if not _initialised:
            conn.executescript(SCHEMA)
            _initialised = True
    _local.conn = conn
    return conn


@contextmanager
def write_tx() -> Iterator[sqlite3.Connection]:
    """Serialised write transaction: the atomicity primitive for races."""
    conn = _connect()
    for attempt in range(60):
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError:
            if attempt == 59:
                raise
            time.sleep(0.05)
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def get(ns: str, key: str) -> Optional[Any]:
    row = _connect().execute(
        "SELECT value, state FROM records WHERE ns=? AND key=?", (ns, key)
    ).fetchone()
    if row is None or row["state"] != "ready" or row["value"] is None:
        return None
    return json.loads(row["value"])


def raw(ns: str, key: str) -> Optional[sqlite3.Row]:
    return _connect().execute(
        "SELECT * FROM records WHERE ns=? AND key=?", (ns, key)
    ).fetchone()


def put(ns: str, key: str, value: Any) -> None:
    now = time.time()
    payload = json.dumps(value, ensure_ascii=False)
    with write_tx() as conn:
        conn.execute(
            "INSERT INTO records (ns,key,value,state,created_at,updated_at)"
            " VALUES (?,?,?,'ready',?,?)"
            " ON CONFLICT(ns,key) DO UPDATE SET value=excluded.value,"
            " state='ready', updated_at=excluded.updated_at",
            (ns, key, payload, now, now),
        )


def claim(ns: str, key: str) -> Tuple[bool, Optional[Any]]:
    """Atomically reserve ``key``.

    Returns ``(True, None)`` for the single caller that won the reservation and
    must do the work, or ``(False, value_or_None)`` for everyone else. A loser
    that sees ``None`` is racing the winner and should wait via :func:`await_ready`.
    """
    now = time.time()
    with write_tx() as conn:
        row = conn.execute(
            "SELECT value, state FROM records WHERE ns=? AND key=?", (ns, key)
        ).fetchone()
        if row is not None:
            value = json.loads(row["value"]) if row["value"] is not None else None
            return False, (value if row["state"] == "ready" else None)
        conn.execute(
            "INSERT INTO records (ns,key,value,state,created_at,updated_at)"
            " VALUES (?,?,NULL,'pending',?,?)",
            (ns, key, now, now),
        )
    return True, None


def release(ns: str, key: str) -> None:
    """Drop a reservation so a failed attempt does not wedge the key."""
    with write_tx() as conn:
        conn.execute(
            "DELETE FROM records WHERE ns=? AND key=? AND state='pending'", (ns, key)
        )


def await_ready(ns: str, key: str, timeout: float = 30.0) -> Optional[Any]:
    """Wait for the winner of :func:`claim` to publish its value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = raw(ns, key)
        if row is None:
            return None
        if row["state"] == "ready" and row["value"] is not None:
            return json.loads(row["value"])
        time.sleep(0.05)
    return None


def list_ns(ns: str, prefix: str = "", limit: int = 500) -> List[Any]:
    rows = _connect().execute(
        "SELECT value FROM records WHERE ns=? AND key LIKE ? AND state='ready'"
        " ORDER BY created_at ASC LIMIT ?",
        (ns, f"{prefix}%", limit),
    ).fetchall()
    return [json.loads(r["value"]) for r in rows if r["value"] is not None]


def delete(ns: str, key: str) -> None:
    with write_tx() as conn:
        conn.execute("DELETE FROM records WHERE ns=? AND key=?", (ns, key))
