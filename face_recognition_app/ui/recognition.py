from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def render_recognition_mode(recognizer, model_app):
    img_file = st.camera_input("カメラに顔を映して撮影してください")

    if img_file is None:
        return

    image_bytes = img_file.getvalue()
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("画像を読み込めませんでした。もう一度撮影してください。")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = model_app.get(image_rgb)

    if len(faces) == 0:
        st.warning("顔が検出されませんでした。もう一度撮影してください。")
        return

    st.success(f"{len(faces)} 人の顔を検出しました。判定中...")

    for face in faces:
        bbox = face.bbox.astype(int)
        name_predict, _ = recognizer.recognize_face_from_embedding(face.embedding)
        cv2.rectangle(image_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            str(name_predict),
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="認識結果", width="stretch")

    if recognizer.known_faces:
        st.caption("登録済みラベル: " + ", ".join(recognizer.known_faces.keys()))