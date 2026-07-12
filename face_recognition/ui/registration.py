from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


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