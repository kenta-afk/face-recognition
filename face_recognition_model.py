from __future__ import annotations

import streamlit as st
from insightface.app import FaceAnalysis


@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app