"""SQLite database for uploads, training runs, evaluations and predictions.

The audio files themselves are saved under storage/uploads/<digit>/; the
database keeps one row per file (label, speaker, checksum, whether a
retrain has used it yet).
"""
import json
import sqlite3
from contextlib import contextmanager

from src.config import DB_PATH, ensure_dirs
from src.registry import now

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label INTEGER NOT NULL,
    speaker TEXT,
    sha256 TEXT UNIQUE NOT NULL,
    spoken_s REAL,
    uploaded_at TEXT NOT NULL,
    holdout INTEGER NOT NULL DEFAULT 0,  -- kept aside to check the retrained model
    used_in_run INTEGER                  -- training run that first used it
);
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    n_saved INTEGER, n_rejected INTEGER, n_duplicate INTEGER,
    model_version INTEGER,
    accuracy_before REAL,               -- current model on this batch
    trigger_fired INTEGER, trigger_reason TEXT
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,                 -- 'retrain' (fine-tune) or 'train' (from scratch)
    status TEXT NOT NULL,               -- running, done, kept_old, failed
    reason TEXT,
    started_at TEXT NOT NULL, finished_at TEXT,
    base_version INTEGER, new_version INTEGER, promoted INTEGER,
    n_new INTEGER, n_replay INTEGER, epochs INTEGER,
    test_acc_before REAL, test_acc_after REAL,
    new_acc_before REAL, new_acc_after REAL,
    progress TEXT, message TEXT, metrics TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL, model_version INTEGER, dataset TEXT, metrics TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL, digit INTEGER, confidence REAL,
    latency_ms REAL, model_version INTEGER, served_by TEXT
);
"""


def connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def db():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def init():
    with db() as c:
        c.executescript(SCHEMA)
        # A run left 'running' by a container that stopped mid-training
        c.execute("UPDATE training_runs SET status='failed', message='stopped before it finished', "
                  "finished_at=? WHERE status='running'", (now(),))


def rows(sql, args=()):
    with db() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def one(sql, args=()):
    r = rows(sql, args)
    return r[0] if r else None


def execute(sql, args=()):
    with db() as c:
        cur = c.execute(sql, args)
        return cur.lastrowid


# ---------------------------------------------------------------- helpers

def log_prediction(digit, confidence, latency_ms, version, served_by):
    execute("INSERT INTO predictions (at, digit, confidence, latency_ms, model_version, served_by) "
            "VALUES (?,?,?,?,?,?)", (now(), digit, confidence, latency_ms, version, served_by))


def pending_uploads():
    """Uploads no retrain has used yet."""
    return rows("SELECT * FROM uploads WHERE used_in_run IS NULL ORDER BY id")


def upload_counts():
    out = rows("SELECT label, holdout, used_in_run IS NOT NULL AS used, COUNT(*) AS n "
               "FROM uploads GROUP BY label, holdout, used")
    return out


def start_run(mode, reason, base_version):
    with db() as c:
        if c.execute("SELECT 1 FROM training_runs WHERE status='running'").fetchone():
            return None
        cur = c.execute("INSERT INTO training_runs (mode, status, reason, started_at, base_version, progress) "
                        "VALUES (?, 'running', ?, ?, ?, ?)",
                        (mode, reason, now(), base_version, json.dumps({"stage": "starting"})))
        return cur.lastrowid


def update_run(run_id, **fields):
    for k in ("progress", "metrics"):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    sets = ", ".join(f"{k}=?" for k in fields)
    execute(f"UPDATE training_runs SET {sets} WHERE id=?", (*fields.values(), run_id))


def run_dict(r):
    if r is None:
        return None
    for k in ("progress", "metrics"):
        if r.get(k):
            r[k] = json.loads(r[k])
    return r
