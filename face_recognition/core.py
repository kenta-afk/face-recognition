from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, insightface_app=None, threshold=0.9):
        self.app = insightface_app
        self.threshold = threshold
        self.known_faces = {}
        self.face_vectors = {}

    def _normalize(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def get_face_embedding_from_bgr(self, image_bgr):
        if self.app is None:
            print("Warning: InsightFace app is not loaded.")
            return None

        if image_bgr is None:
            print("Warning: Could not decode image data.")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(image_rgb)

        if len(faces) == 0:
            print("DEBUG: No faces detected in the image")
            return None

        return faces[0].embedding

    def get_face_embedding_from_bytes(self, image_bytes):
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return self.get_face_embedding_from_bgr(image_bgr)

    def register_face_from_images(self, image_list, label=None):
        if label is None or not label.strip():
            label = f"person_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            print(f"No label provided, generated label: {label}")

        embeddings_list = []
        for image_bytes in image_list:
            embedding = self.get_face_embedding_from_bytes(image_bytes)
            if embedding is not None:
                embeddings_list.append(embedding)

        if embeddings_list:
            self.face_vectors[label] = embeddings_list
            self.known_faces[label] = np.mean(embeddings_list, axis=0)
            print(f"Registered {label} with {len(embeddings_list)} image(s).")
            return True, f"{label} を登録しました"

        print(f"Failed to register {label}. No faces detected in provided images.")
        return False, "登録に失敗しました。顔が検出されませんでした。"

    def delete_registered_face(self, label):
        if label not in self.known_faces:
            return False, f"{label} は登録されていません"

        del self.known_faces[label]
        self.face_vectors.pop(label, None)
        return True, f"{label} を削除しました"

    def get_registered_face_vectors(self):
        return self.face_vectors

    def compare_with_registered_faces(self, query_embedding):
        if query_embedding is None:
            return None, []

        q_vec = self._normalize(query_embedding)
        if q_vec is None:
            return None, []

        comparisons = []
        best_label = None
        best_distance = float("inf")

        for label, stored_vectors in self.face_vectors.items():
            sample_distances = []
            for vector in stored_vectors:
                k_vec = self._normalize(vector)
                if k_vec is None:
                    continue

                cosine_similarity = np.dot(q_vec, k_vec)
                distance = 1.0 - cosine_similarity
                sample_distances.append(float(distance))

            if not sample_distances:
                continue

            min_distance = min(sample_distances)
            comparisons.append(
                {
                    "label": label,
                    "sample_count": len(sample_distances),
                    "mean_distance": float(np.mean(sample_distances)),
                    "min_distance": float(min_distance),
                    "sample_distances": sample_distances,
                    "vectors": [vector.tolist() for vector in stored_vectors],
                }
            )

            if min_distance < best_distance:
                best_distance = min_distance
                best_label = label

        comparisons.sort(key=lambda item: item["min_distance"])
        return best_label, comparisons

    def recognize_face_from_embedding(self, query_embedding):
        if query_embedding is None:
            return "顔が検出されませんでした", None

        if not self.known_faces:
            return "登録された顔がありません", None

        min_distance = float("inf")
        best_label = None

        q_vec = self._normalize(query_embedding)
        if q_vec is None:
            return "顔の特徴量を正規化できませんでした", None

        for label, known_embedding in self.known_faces.items():
            k_vec = self._normalize(known_embedding)
            if k_vec is None:
                continue

            cosine_similarity = np.dot(q_vec, k_vec)
            distance = 1.0 - cosine_similarity

            if distance < min_distance:
                min_distance = distance
                best_label = label

        if min_distance < self.threshold:
            return f"{best_label} ({min_distance:.2f})", min_distance

        return f"登録済みですが、一致しません ({min_distance:.2f})", min_distance