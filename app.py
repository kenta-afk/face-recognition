import streamlit as st
import cv2
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

DATA_FILE = Path("face_recognizer_data.pkl")

# ==========================================
# 1. 仕様書に基づいた FaceRecognizer クラスの定義
# ==========================================
class FaceRecognizer:
    def __init__(self, insightface_app=None, threshold=0.9): # 閾値は適宜調整してください（一般的にコサイン距離なら0.4〜0.6付近）
        self.app = insightface_app
        self.threshold = threshold
        self.known_faces = {}

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
            self.known_faces[label] = np.mean(embeddings_list, axis=0)
            print(f"Registered {label} with {len(embeddings_list)} image(s).")
            return True, f"{label} を登録しました"

        print(f"Failed to register {label}. No faces detected in provided images.")
        return False, "登録に失敗しました。顔が検出されませんでした。"

    def delete_registered_face(self, label):
        if label not in self.known_faces:
            return False, f"{label} は登録されていません"

        del self.known_faces[label]
        return True, f"{label} を削除しました"

    def save(self, path=DATA_FILE):
        payload = {
            "known_faces": self.known_faces,
            "threshold": self.threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def recognize_face_from_embedding(self, query_embedding):
        """
        仕様書の「顔の認識 (recognize_face)」のロジックを、
        すでに抽出された特徴量（embedding）を元に実行する関数
        """
        if query_embedding is None:
            return "顔が検出されませんでした", None

        if not self.known_faces:
            return "登録された顔がありません", None

        min_distance = float('inf')
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

        # 閾値判定
        if min_distance < self.threshold:
            return f"{best_label} ({min_distance:.2f})", min_distance
        else:
            return f"登録済みですが、一致しません ({min_distance:.2f})", min_distance


def load_face_recognizer(model_app):
    recognizer = FaceRecognizer(insightface_app=model_app)

    if not DATA_FILE.exists():
        return recognizer

    with open(DATA_FILE, 'rb') as f:
        saved_data = pickle.load(f)

    if isinstance(saved_data, dict):
        recognizer.known_faces = saved_data.get('known_faces', saved_data)
        recognizer.threshold = saved_data.get('threshold', recognizer.threshold)
    else:
        recognizer.known_faces = getattr(saved_data, 'known_faces', {})
        recognizer.threshold = getattr(saved_data, 'threshold', recognizer.threshold)

    return recognizer


# ==========================================
# 2. モデルやライブラリのロード（キャッシュ化）
# ==========================================
@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app = load_insightface()

# あなたのAIモデル（判定用）の状態を復元
recognizer = load_face_recognizer(app)


# ==========================================
# 3. Streamlit UI とメイン処理
# ==========================================
st.title("顔認識アプリ (InsightFace × Streamlit)")

mode = st.sidebar.radio("モード", ["認証", "登録"], index=0)
st.sidebar.write(f"登録済み人数: {len(recognizer.known_faces)}")
threshold = st.sidebar.slider("判定閾値", 0.0, 1.0, float(recognizer.threshold), 0.01)
recognizer.threshold = threshold

if "registration_images" not in st.session_state:
    st.session_state.registration_images = []

if mode == "登録":
    st.subheader("顔の登録")
    label = st.text_input("ラベル名", placeholder="例: tanaka_taro")

    captured_image = st.camera_input("登録用の顔を撮影してください")

    if captured_image is not None:
        image_bytes = captured_image.getvalue()
        preview_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        preview_bgr = cv2.imdecode(preview_array, cv2.IMREAD_COLOR)
        if preview_bgr is not None:
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="撮影した画像", width='stretch')

        if st.button("この画像を追加"):
            st.session_state.registration_images.append(image_bytes)
            st.success(f"登録候補に追加しました。現在 {len(st.session_state.registration_images)} 枚です。")
            st.rerun()

    if st.session_state.registration_images:
        st.write(f"登録候補画像: {len(st.session_state.registration_images)} 枚")
        if st.button("登録候補を全てクリア"):
            st.session_state.registration_images = []
            st.rerun()

    if st.button("登録して保存"):
        if not st.session_state.registration_images:
            st.warning("少なくとも1枚の画像を撮影して追加してください。")
        else:
            success, message = recognizer.register_face_from_images(st.session_state.registration_images, label=label)
            if success:
                recognizer.save()
                st.session_state.registration_images = []
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    if recognizer.known_faces:
        st.write("登録済みラベル")
        registered_labels = list(recognizer.known_faces.keys())
        st.write(registered_labels)

        delete_label = st.selectbox(
            "削除するラベル",
            options=[""] + registered_labels,
            index=0,
        )
        if st.button("選択したラベルを削除"):
            if not delete_label:
                st.warning("削除するラベルを選択してください。")
            else:
                success, message = recognizer.delete_registered_face(delete_label)
                if success:
                    recognizer.save()
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    st.stop()

# カメラ入力
img_file = st.camera_input("カメラに顔を映して撮影してください")

if img_file is not None:
    image_bytes = img_file.getvalue()
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("画像を読み込めませんでした。もう一度撮影してください。")
        st.stop()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(image_rgb)

    if len(faces) == 0:
        st.warning("顔が検出されませんでした。もう一度撮影してください。")
    else:
        st.success(f"{len(faces)} 人の顔を検出しました。判定中...")
        
        for face in faces:
            bbox = face.bbox.astype(int)
            face_embedding = face.embedding
            name_predict, distance = recognizer.recognize_face_from_embedding(face_embedding)
            cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image_bgr, str(name_predict), (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(result_image, caption="認識結果", width='stretch')

        if recognizer.known_faces:
            st.caption("登録済みラベル: " + ", ".join(recognizer.known_faces.keys()))