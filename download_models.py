"""Fetch the two pretrained ONNX models from the official OpenCV Model Zoo (Apache-2.0).
Source: https://github.com/opencv/opencv_zoo
"""
import hashlib
import urllib.request

import cv2

import config

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    config.DETECTOR_MODEL: BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    config.RECOGNIZER_MODEL: BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODELS.items():
        if not path.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        data = path.read_bytes()
        if data[:20].lstrip().startswith(b"version https://git-lfs"):
            raise SystemExit(f"{path.name} is a Git LFS pointer, not the model. Delete it and retry.")
        print(f"{path.name}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")
    # Verify both actually load in OpenCV.
    cv2.FaceDetectorYN.create(str(config.DETECTOR_MODEL), "", (320, 320))
    cv2.FaceRecognizerSF.create(str(config.RECOGNIZER_MODEL), "")
    print("both models load OK")


if __name__ == "__main__":
    main()
