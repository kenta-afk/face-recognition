from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def render_sidebar(recognizer):
    st.sidebar.write(f"登録済み人数: {len(recognizer.known_faces)}")
    threshold = st.sidebar.slider("判定閾値", 0.0, 1.0, float(recognizer.threshold), 0.01)
    recognizer.threshold = threshold


def render_registration_mode(recognizer, save_callback):
    st.subheader("顔の登録")
    label = st.text_input("ラベル名", placeholder="例: tanaka_taro")
    st.session_state.setdefault("registration_images", [])

    captured_image = st.camera_input("登録用の顔を撮影してください")

    if captured_image is not None:
        image_bytes = captured_image.getvalue()
        preview_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        preview_bgr = cv2.imdecode(preview_array, cv2.IMREAD_COLOR)
        if preview_bgr is not None:
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="撮影した画像", width="stretch")

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
                save_callback(recognizer)
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
                    save_callback(recognizer)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


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