"""Enroll a consenting student, or capture held-out test images for evaluation.

  python enroll.py --id S01 --name Faraz            # enrollment: embeddings + crops
  python enroll.py --id S01 --name Faraz --test     # held-out test crops (NOT used for matching)
  python enroll.py --id X01 --name Visitor --test   # test crops of a non-enrolled person (unknown)

Only the student ID, display name and face data are stored. Nothing leaves this machine.
"""
import argparse
import time

import cv2
import numpy as np

import config
import database
import vision


def capture(n, out_dir):
    det, rec = vision.Detector(), vision.Recognizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("cannot open webcam")
    embs, last = [], 0.0
    try:
        while len(embs) < n:
            ok, frame = cap.read()
            if not ok:
                continue
            faces = det.detect(frame)
            view = frame.copy()
            msg = f"{len(embs)}/{n}  vary pose/lighting slightly  (Q = cancel)"
            if len(faces) != 1:
                msg = "need exactly ONE face in view"
            for f in faces:
                x, y, w, h = f[:4].astype(int)
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(faces) == 1 and time.time() - last > 0.4:
                crop = rec.crop(frame, faces[0])
                cv2.imwrite(str(out_dir / f"{len(embs):03d}.png"), crop)
                embs.append(rec.embed_crop(crop))
                last = time.time()
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Enrollment", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                raise SystemExit("cancelled, nothing saved to database")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return np.array(embs, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", action="store_true", help="capture held-out test images only")
    ap.add_argument("-n", type=int, default=config.ENROLL_SAMPLES)
    a = ap.parse_args()
    if not a.id.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("student id must be letters/digits/-/_ only")
    if input(f"Does {a.name} consent to face capture for this project? [y/N] ").strip().lower() != "y":
        raise SystemExit("no consent, aborted")
    if a.test:
        capture(a.n, config.ROOT / "data" / "test" / a.id)
        print(f"saved {a.n} held-out test crops for {a.id}")
        return
    sdir = config.STUDENTS_DIR / a.id
    embs = capture(a.n, sdir / "crops")
    np.save(sdir / "embeddings.npy", embs)
    database.add_student(database.connect(), a.id, a.name)
    print(f"enrolled {a.id} ({a.name}) with {len(embs)} samples")


if __name__ == "__main__":
    main()
