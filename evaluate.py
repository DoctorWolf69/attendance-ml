"""Offline recognition evaluation on HELD-OUT crops (captured with `enroll.py --test`).

data/test/<id>/*.png : if <id> is enrolled, the expected label is <id>; otherwise expected "Unknown".
Prints test-set composition, identification rate, unknown rejection, false matches and a confusion matrix.
Liveness evaluation (real vs spoof, FAR/FRR) is pending: it needs a labelled spoof test set.
"""
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

import config
import vision


def main():
    gallery = vision.load_gallery()
    rec = vision.Recognizer()
    test_root = config.ROOT / "data" / "test"
    y_true, y_pred, sims = [], [], []
    for d in sorted(p for p in test_root.glob("*") if p.is_dir()):
        expected = d.name if d.name in gallery else "Unknown"
        for img in sorted(d.glob("*.png")):
            sid, sim = vision.match(rec.embed_crop(cv2.imread(str(img))), gallery)
            y_true.append(expected)
            y_pred.append(sid or "Unknown")
            sims.append(sim)
    if not y_true:
        raise SystemExit("no test crops found; capture some with: python enroll.py --id S01 --name X --test")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    known = y_true != "Unknown"
    print(f"enrolled identities: {len(gallery)}   threshold: {config.RECOGNITION_THRESHOLD}")
    for lbl in sorted(set(y_true)):
        print(f"  test images [{lbl}]: {(y_true == lbl).sum()}")
    if known.any():
        print(f"correct identification rate (known faces): {(y_pred[known] == y_true[known]).mean():.3f}")
        print(f"missed (known -> Unknown): {(y_pred[known] == 'Unknown').sum()}")
        print(f"wrong identity (known -> other student): {((y_pred[known] != y_true[known]) & (y_pred[known] != 'Unknown')).sum()}")
    if (~known).any():
        print(f"unknown rejection rate: {(y_pred[~known] == 'Unknown').mean():.3f}")
        print(f"false matches (unknown -> some student): {(y_pred[~known] != 'Unknown').sum()}")
    labels = sorted(set(y_true) | set(y_pred))
    print("confusion matrix (rows = true, cols = predicted):", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("NOTE: small locally captured test set; does not establish classroom-scale accuracy.")


if __name__ == "__main__":
    main()
