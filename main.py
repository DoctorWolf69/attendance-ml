"""Smart Attendance live app.
Keys:  S = start session   E = end session (+ CSV report)   R = export report now
       P = save screenshot to docs/evidence   Q = quit
Pipeline per face: detect -> liveness -> recognize -> attendance gate (database.try_mark).
"""
import time

import cv2

import config
import database
import liveness
import report
import vision

GREEN, RED, YELLOW, WHITE, ORANGE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 165, 255)
STABLE_FRAMES = 3  # same identity on N consecutive frames before marking
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Track:
    """A face followed across frames by nearest box centre (one person = one liveness challenge)."""
    def __init__(self, c):
        self.c, self.seen = c, time.time()
        self.live = liveness.HeadTurnChallenge()
        self.last_id, self.hits, self.result = None, 0, ""


def nearest(tracks, c, max_dist):
    # note: nearest-centre tracking, fine for a few faces; swap for SORT/IoU tracker in a real classroom
    d = lambda t: ((t.c[0] - c[0]) ** 2 + (t.c[1] - c[1]) ** 2) ** 0.5
    best = min(tracks, key=d, default=None)
    return best if best and d(best) < max_dist else None


def main():
    det, rec = vision.Detector(), vision.Recognizer()
    db = database.connect()
    gallery, names = vision.load_gallery(), database.student_names(db)
    print(f"loaded {len(gallery)} enrolled students: {', '.join(names.get(s, s) for s in gallery) or 'none'}")
    print(f"liveness mode: {config.LIVENESS_MODE}")
    if not gallery:
        print("WARNING: no students enrolled; every face will be unrecognised. Run: python enroll.py --id S01 --name <name>")
    session, tracks = None, []
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("cannot open webcam")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            view = frame.copy()
            faces = det.detect(frame)
            now, free = time.time(), list(tracks)
            for f in faces:
                x, y, w, h = f[:4].astype(int)
                c = (x + w / 2, y + h / 2)
                t = nearest(free, c, max(w, h))
                if t is None:
                    t = Track(c)
                    tracks.append(t)
                else:
                    free.remove(t)
                t.c, t.seen = c, now

                # 1) liveness first; recognition only runs for faces that passed it
                state, live = t.live.check(f)
                color, label = ORANGE, state
                if live:
                    sid, sim = vision.match(rec.embed(frame, f), gallery)
                    t.hits = t.hits + 1 if sid == t.last_id else 1
                    t.last_id = sid
                    if not gallery:
                        color, label = RED, "NO STUDENTS ENROLLED - run enroll.py"
                        t.result = ""
                    elif sid is None:
                        color, label = RED, f"UNKNOWN (best cos sim {sim:.2f})"
                        t.result = ""
                    else:
                        color, label = GREEN, f"{names.get(sid, sid)}  cos sim {sim:.2f}"
                        if t.hits >= STABLE_FRAMES:
                            r = database.try_mark(db, session, sid, live, sim)
                            if r == "marked":
                                print(f"[{database.now()}] ATTENDANCE RECORDED: {sid} {names.get(sid, sid)} (cos sim {sim:.3f})")
                            t.result = {"marked": "ATTENDANCE RECORDED", "already marked": "ATTENDANCE RECORDED",
                                        "no session": "not recorded: no session"}.get(r, r)
                    label = f"{state} | {label}"
                cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
                cv2.putText(view, label, (x, max(y - 10, 15)), FONT, 0.5, color, 2)
                if live and t.result:
                    cv2.putText(view, t.result, (x, y + h + 20), FONT, 0.55, color, 2)
            tracks = [t for t in tracks if now - t.seen < 1.5]  # forget faces gone >1.5 s

            if not len(faces):
                cv2.putText(view, "NO FACE", (10, 60), FONT, 0.6, WHITE, 2)
            n = db.execute("SELECT COUNT(*) FROM attendance WHERE session_id = ?", (session,)).fetchone()[0] if session else 0
            status = f"SESSION {session} RUNNING  present {n}/{len(gallery)}" if session else "NO SESSION - press S to start"
            cv2.putText(view, status, (10, 30), FONT, 0.65, GREEN if session else YELLOW, 2)
            if config.LIVENESS_MODE == "bypass_test":
                cv2.putText(view, "TEST MODE - LIVENESS BYPASSED - NOT SECURE", (10, view.shape[0] - 15), FONT, 0.6, RED, 2)
            cv2.imshow("Smart Attendance", view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")) or cv2.getWindowProperty("Smart Attendance", cv2.WND_PROP_VISIBLE) < 1:
                break
            if k in (ord("s"), ord("S")) and session is None:
                session = database.start_session(db)
                tracks = []  # fresh liveness challenge for everyone in the new session
                print(f"[{database.now()}] session {session} started")
            if k in (ord("e"), ord("E")) and session is not None:
                database.end_session(db, session)
                print(f"[{database.now()}] session {session} ended; report:", report.export(db, session)[0])
                session = None
            if k in (ord("r"), ord("R")) and session is not None:
                print("report:", report.export(db, session)[0])
            if k in (ord("p"), ord("P")):
                config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                p = config.EVIDENCE_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(p), view)
                print("saved", p)
    finally:
        if session is not None:
            database.end_session(db, session)
            print("session auto-ended; report:", report.export(db, session)[0])
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()
