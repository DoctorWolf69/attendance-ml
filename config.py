"""Central settings. All paths are relative to this project folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "data" / "models"
STUDENTS_DIR = ROOT / "data" / "students"
DB_PATH = ROOT / "database" / "attendance.db"
REPORTS_DIR = ROOT / "reports"
EVIDENCE_DIR = ROOT / "docs" / "evidence"

DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

CAMERA_INDEX = 0
DETECT_SCORE_THRESHOLD = 0.8      # YuNet face score
# SFace cosine similarity: >= threshold means same person. 0.363 is OpenCV's published value.
RECOGNITION_THRESHOLD = 0.363
ENROLL_SAMPLES = 15

# Liveness: "challenge" = head-turn challenge (preliminary heuristic, NOT a CNN).
#           "bypass_test" = development only, every face treated as live, shown with a red banner.
LIVENESS_MODE = "challenge"
LIVENESS_TURN_RATIO = 0.18        # nose offset / eye distance needed to count as a turn
LIVENESS_TIMEOUT_S = 8.0
