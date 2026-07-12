from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st


def _project_to_2d(vectors):
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0:
        return np.empty((0, 2))

    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        return np.array([[0.0, 0.0]])

    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    components = vt[:2]
    projected = matrix @ components.T
    if projected.shape[1] == 1:
        projected = np.hstack([projected, np.zeros((projected.shape[0], 1))])
    return projected


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

    comparison_rows = []
    query_vectors = []
    figure_vectors = []
    figure_labels = []

    for face in faces:
        bbox = face.bbox.astype(int)
        name_predict, _ = recognizer.recognize_face_from_embedding(face.embedding)
        best_label, comparisons = recognizer.compare_with_registered_faces(face.embedding)
        query_vectors.append(face.embedding)
        figure_vectors.append(face.embedding)
        figure_labels.append(f"query_{len(figure_labels) + 1}")

        if best_label is not None:
            for item in comparisons:
                comparison_rows.append(
                    {
                        "query_label": name_predict,
                        "matched_label": item["label"],
                        "sample_count": item["sample_count"],
                        "min_distance": item["min_distance"],
                        "mean_distance": item["mean_distance"],
                    }
                )

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

    with st.expander("ベクトルの特徴と距離比較"):
        registered_vectors = recognizer.get_registered_face_vectors()

        if comparison_rows:
            st.write("登録済みデータとの距離比較")
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)

            similarity_rows = []
            for row in comparison_rows:
                similarity_rows.append(
                    {
                        "matched_label": row["matched_label"],
                        "similarity": 1.0 - row["min_distance"],
                    }
                )

            if similarity_rows:
                similarity_df = pd.DataFrame(similarity_rows).groupby("matched_label", as_index=False).max()
                st.write("登録済みラベルごとの近さ(高いほど近い)")
                st.bar_chart(similarity_df, x="matched_label", y="similarity", use_container_width=True)

        all_vectors = []
        all_labels = []
        all_kinds = []

        for label, vectors in registered_vectors.items():
            for index, vector in enumerate(vectors, start=1):
                all_vectors.append(vector)
                all_labels.append(f"{label}_{index}")
                all_kinds.append(label)

        for index, vector in enumerate(query_vectors, start=1):
            all_vectors.append(vector)
            all_labels.append(f"query_{index}")
            all_kinds.append("query")

        if all_vectors:
            projected = _project_to_2d(all_vectors)
            projection_df = pd.DataFrame(
                {
                    "x": projected[:, 0],
                    "y": projected[:, 1],
                    "label": all_labels,
                    "kind": all_kinds,
                }
            )
            st.write("2次元に圧縮したベクトル配置図")
            st.scatter_chart(projection_df, x="x", y="y", color="kind", size=None, use_container_width=True)
            st.caption("近い位置にあるほど、埋め込みベクトルとして似ています。query は今回撮影した顔です。")

            summary_rows = []
            for label, vectors in registered_vectors.items():
                stacked = np.vstack(vectors)
                summary_rows.append(
                    {
                        "label": label,
                        "sample_count": len(vectors),
                        "mean_vector_norm": float(np.linalg.norm(np.mean(stacked, axis=0))),
                        "vector_variance": float(np.mean(np.var(stacked, axis=0))),
                    }
                )

            if summary_rows:
                st.write("登録済みデータの特徴量サマリ")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        else:
            st.info("比較できる登録済みベクトルがありません。")