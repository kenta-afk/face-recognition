import streamlit as st

from face_recognition_model import load_insightface
from face_recognition_storage import load_face_recognizer, save_face_recognizer
from face_recognition_views import render_recognition_mode, render_registration_mode, render_sidebar


def main():
    st.title("顔認識アプリ (InsightFace × Streamlit)")

    model_app = load_insightface()
    recognizer = load_face_recognizer(model_app)

    mode = st.sidebar.radio("モード", ["認証", "登録"], index=0)
    render_sidebar(recognizer)

    if mode == "登録":
        render_registration_mode(recognizer, save_face_recognizer)
    else:
        render_recognition_mode(recognizer, model_app)


if __name__ == "__main__":
    main()