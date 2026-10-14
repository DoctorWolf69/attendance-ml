# Smart Attendance System using Face Recognition and Liveness Detection

5th-semester ML mini-project prototype. Team: Faraz, Vibhuti, Vaishali, Parth.
Laptop webcam, local only, SQLite. Architecture and module ownership: [docs/architecture.md](docs/architecture.md).

## Setup (Windows, Python 3.14 tested)
```
pip install -r requirements.txt
python download_models.py        # YuNet + SFace ONNX from github.com/opencv/opencv_zoo (Apache-2.0), verifies they load
python tests/test_core.py        # safety-logic tests
```

## Run the demo
```
python enroll.py --id S01 --name Faraz      # consent prompt, then look at camera; 15 samples auto-captured
python enroll.py --id S02 --name Vibhuti    # repeat per consenting team member
python main.py
```
Keys in the window: **S** start session · **E** end session (writes CSV) · **R** report now · **P** screenshot to `docs/evidence/` · **Q** quit.

What you see per face: orange `CHECKING LIVENESS: turn head side to side (0/2)` → after turning both ways,
green `LIVE | Faraz  cos sim 0.62` → `ATTENDANCE RECORDED` (only while a session runs).
Non-enrolled person: red `UNKNOWN (best cos sim 0.12)`, never recorded.
Report only: `python report.py [session_id]` → `reports/session_N.csv`.

Liveness test mode (development only): set `LIVENESS_MODE = "bypass_test"` in `config.py`; a red
"TEST MODE - LIVENESS BYPASSED - NOT SECURE" banner is shown. Never demo this as secure behaviour.

## Evaluation (recognition)
Capture held-out images on a *different* occasion or lighting than enrollment, plus a few non-enrolled consenting people:
```
python enroll.py --id S01 --name Faraz --test
python enroll.py --id X01 --name Visitor --test   # X01 not enrolled -> expected Unknown
python evaluate.py
```
Prints test-set composition, identification rate, missed/wrong matches, unknown rejection and a confusion matrix.

## Implementation status (2026-10-15)
**Implemented and tested**
- Webcam capture + YuNet face detection: verified headless on the laptop camera (640x480, face detected, ~8 fps with full pipeline).
- SFace embeddings + cosine matching: on live frames of one person, later frames vs first frames gave cos sim 0.90–0.95 (threshold 0.363). Same sitting, so this is a sanity check, not an accuracy result.
- Attendance gate: no record without session, without liveness, or for Unknown; duplicates blocked by DB constraint (`tests/test_core.py`, synthetic data).
- Head-turn liveness logic: static landmarks never pass, a left+right turn passes (unit test, synthetic landmarks). Still-sitting yaw ratio measured 0.005–0.043 vs 0.18 turn threshold.
- CSV report incl. ABSENT students (tested on a throwaway in-memory DB).
- `main.py` window starts and runs its loop (10 s smoke test).

**Implemented but not fully tested**
- Interactive enrollment and the full live demo with enrolled students (needs team members in front of the camera).
- Unknown-person rejection on real faces; multiple faces at once.
- Head-turn challenge against a real printed photo / phone screen.
- `evaluate.py` (no held-out test data captured yet; no results reported).

**Partially implemented**
- Liveness: preliminary head-turn heuristic only. It is **not** a trained CNN.

**Planned**
- CNN / transfer-learning anti-spoofing (e.g. MobileNet fine-tuned on NUAA Imposter DB + custom real/spoof set), plugged in behind `HeadTurnChallenge.check()`; evaluate with accuracy, precision, recall, F1, FAR, FRR, confusion matrix.
- Duration-aware attendance / periodic rescanning, fixed classroom camera, nicer UI.

## Limitations
- Head-turn challenge defeats a static photo but a **video replay** of the person turning their head will pass. Bending/rotating a printed photo is untested.
- Recognising a face on a webcam does not prove the person's location; a fixed in-room camera only encourages physical presence.
- Tiny local dataset: no claim of classroom-scale accuracy or generalisation.
- Simple nearest-centre face tracking; crowded scenes may swap tracks.
- Face crops, embeddings and the DB are private: kept out of git via `.gitignore`.
