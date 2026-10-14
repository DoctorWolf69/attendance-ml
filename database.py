"""SQLite storage: students, sessions, attendance. Duplicates blocked by a UNIQUE constraint."""
import sqlite3
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    enrolled_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT
);
CREATE TABLE IF NOT EXISTS attendance (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    student_id TEXT NOT NULL REFERENCES students(student_id),
    marked_at  TEXT NOT NULL,
    status     TEXT NOT NULL,
    similarity REAL,
    UNIQUE (session_id, student_id)
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


def connect(path=config.DB_PATH):
    if str(path) != ":memory:":
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def add_student(db, student_id, name):
    db.execute("INSERT OR REPLACE INTO students VALUES (?, ?, ?)", (student_id, name, now()))
    db.commit()


def student_names(db):
    return {r["student_id"]: r["name"] for r in db.execute("SELECT * FROM students")}


def start_session(db):
    cur = db.execute("INSERT INTO sessions (started_at) VALUES (?)", (now(),))
    db.commit()
    return cur.lastrowid


def end_session(db, session_id):
    db.execute("UPDATE sessions SET ended_at = ? WHERE session_id = ?", (now(), session_id))
    db.commit()


def try_mark(db, session_id, student_id, live, similarity=None):
    """The single gate for writing attendance. Returns a short reason string.
    Order of checks mirrors the pipeline: session -> liveness -> identity -> duplicate."""
    if session_id is None:
        return "no session"
    if not live:
        return "liveness not passed"
    if student_id is None:
        return "unknown"
    cur = db.execute(
        "INSERT OR IGNORE INTO attendance VALUES (?, ?, ?, 'PRESENT', ?)",
        (session_id, student_id, now(), similarity))
    db.commit()
    return "marked" if cur.rowcount else "already marked"
