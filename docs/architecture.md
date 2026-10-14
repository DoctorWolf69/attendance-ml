# Architecture

```mermaid
flowchart LR
    T[Teacher presses S] --> C[Webcam frame]
    C --> D[YuNet face detection<br/>vision.Detector]
    D -->|no face| N[NO FACE]
    D --> L[Liveness check<br/>liveness.HeadTurnChallenge]
    L -->|timeout| F[LIVENESS FAILED<br/>no record]
    L -->|live| R[SFace embedding + cosine match<br/>vision.match]
    R -->|sim < threshold| U[UNKNOWN<br/>no record]
    R -->|known, stable 3 frames| G[database.try_mark<br/>session? live? known? duplicate?]
    G --> DB[(SQLite<br/>students / sessions / attendance)]
    DB --> CSV[report.py -> reports/session_N.csv]
```

| Stage | Module | Status | Owner |
|---|---|---|---|
| Webcam + teacher controls | `main.py` | implemented | Parth |
| Face detection (YuNet, pretrained) | `vision.py` | implemented | Faraz |
| Face recognition (SFace, pretrained, cosine sim) | `vision.py` | implemented | Faraz |
| Liveness, head-turn heuristic | `liveness.py` | simplified / preliminary | Vibhuti |
| Liveness, CNN anti-spoofing (transfer learning) | `liveness.py` (`check()` slot) | **pending** | Vibhuti |
| Enrollment | `enroll.py` | implemented | Vaishali |
| Database + duplicate prevention | `database.py` | implemented | Vaishali |
| CSV report | `report.py` | implemented | Vaishali |
| Recognition evaluation | `evaluate.py` | implemented, awaiting test data | Faraz / Vibhuti |

The attendance write path has exactly one gate, `database.try_mark`. It refuses to write a record
when there is no session, when liveness failed, or when the face is Unknown. A `UNIQUE(session_id, student_id)`
constraint blocks duplicates.
