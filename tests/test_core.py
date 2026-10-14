"""Core safety-logic checks. Run: python tests/test_core.py   (or python -m pytest tests)
Uses an in-memory DB and synthetic embeddings/landmarks only, no camera, no real faces."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import database  # noqa: E402
import liveness  # noqa: E402
import vision  # noqa: E402


def test_attendance_gate_and_duplicates():
    db = database.connect(":memory:")
    database.add_student(db, "S01", "Alice")
    assert database.try_mark(db, None, "S01", live=True) == "no session"
    s = database.start_session(db)
    assert database.try_mark(db, s, "S01", live=False) == "liveness not passed"
    assert database.try_mark(db, s, None, live=True) == "unknown"
    assert database.try_mark(db, s, "S01", live=True, similarity=0.6) == "marked"
    for _ in range(50):  # repeated webcam frames
        assert database.try_mark(db, s, "S01", live=True) == "already marked"
    assert db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0] == 1
    s2 = database.start_session(db)  # a new session may mark the same student again
    assert database.try_mark(db, s2, "S01", live=True) == "marked"


def test_match_threshold():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=128), rng.normal(size=128)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    gallery = {"S01": a[None, :], "S02": b[None, :]}
    near = a + 0.05 * rng.normal(size=128)
    sid, sim = vision.match(near / np.linalg.norm(near), gallery)
    assert sid == "S01" and sim > config.RECOGNITION_THRESHOLD
    stranger = rng.normal(size=128)
    sid, sim = vision.match(stranger / np.linalg.norm(stranger), gallery)
    assert sid is None and sim < config.RECOGNITION_THRESHOLD  # random vectors ~0 cosine
    assert vision.match(a, {})[0] is None  # nobody enrolled -> Unknown


def face(nose_x):
    # x,y,w,h, right eye, left eye, nose, mouth r, mouth l, score
    return np.array([100, 100, 100, 100, 130, 140, 170, 140, nose_x, 160, 135, 180, 165, 180, 0.9], np.float32)


def test_liveness_static_photo_fails_head_turn_passes():
    config.LIVENESS_MODE = "challenge"
    still = liveness.HeadTurnChallenge()
    for _ in range(200):  # a static photo: landmarks never change
        assert still.check(face(150))[1] is False
    moving = liveness.HeadTurnChallenge()
    results = [moving.check(face(nx))[1] for nx in (150, 140, 135, 150, 165)]
    assert results[-1] is True and not any(results[:-1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
