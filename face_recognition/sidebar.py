from __future__ import annotations

import streamlit as st


def render_sidebar(recognizer):
    st.sidebar.write(f"登録済み人数: {len(recognizer.known_faces)}")
    threshold = st.sidebar.slider("判定閾値", 0.0, 1.0, float(recognizer.threshold), 0.01)
    recognizer.threshold = threshold