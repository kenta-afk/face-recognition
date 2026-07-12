from __future__ import annotations

import pickle
from pathlib import Path

from .core import FaceRecognizer

DATA_FILE = Path("face_recognizer_data.pkl")


def load_face_recognizer(model_app, path: Path = DATA_FILE):
    recognizer = FaceRecognizer(insightface_app=model_app)

    if not path.exists():
        return recognizer

    with open(path, "rb") as f:
        saved_data = pickle.load(f)

    if isinstance(saved_data, dict):
        recognizer.known_faces = saved_data.get("known_faces", saved_data)
        recognizer.threshold = saved_data.get("threshold", recognizer.threshold)
    else:
        recognizer.known_faces = getattr(saved_data, "known_faces", {})
        recognizer.threshold = getattr(saved_data, "threshold", recognizer.threshold)

    return recognizer


def save_face_recognizer(recognizer, path: Path = DATA_FILE):
    payload = {
        "known_faces": recognizer.known_faces,
        "threshold": recognizer.threshold,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)