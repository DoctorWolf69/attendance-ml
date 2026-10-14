"""Face detection (YuNet) and face recognition (SFace embeddings + cosine similarity)."""
import cv2
import numpy as np

import config


class Detector:
    def __init__(self):
        self.net = cv2.FaceDetectorYN.create(
            str(config.DETECTOR_MODEL), "", (320, 320), config.DETECT_SCORE_THRESHOLD)

    def detect(self, frame):
        """Return array of faces, each row: x, y, w, h, 5 landmarks (x,y)*5, score."""
        h, w = frame.shape[:2]
        self.net.setInputSize((w, h))
        _, faces = self.net.detect(frame)
        return faces if faces is not None else np.empty((0, 15), np.float32)


class Recognizer:
    def __init__(self):
        self.net = cv2.FaceRecognizerSF.create(str(config.RECOGNIZER_MODEL), "")

    def crop(self, frame, face):
        return self.net.alignCrop(frame, face)  # 112x112 aligned face

    def embed_crop(self, crop):
        v = self.net.feature(crop).flatten()
        return v / np.linalg.norm(v)

    def embed(self, frame, face):
        return self.embed_crop(self.crop(frame, face))


def match(embedding, gallery, threshold=config.RECOGNITION_THRESHOLD):
    """gallery: {student_id: (N,128) L2-normalised embeddings}.
    Returns (student_id or None, best cosine similarity). Score per student = max over its samples."""
    best_id, best = None, -1.0
    for sid, embs in gallery.items():
        s = float(np.max(embs @ embedding))
        if s > best:
            best_id, best = sid, s
    return (best_id if best >= threshold else None), best


def load_gallery():
    """Read every data/students/<id>/embeddings.npy written by enroll.py."""
    gallery = {}
    for f in config.STUDENTS_DIR.glob("*/embeddings.npy"):
        gallery[f.parent.name] = np.load(f)
    return gallery
