import os
import joblib
import streamlit as st


st.set_page_config(page_title="Text Classification – Streamlit GUI", layout="wide")
st.title("Text Classification – Streamlit GUI")
st.caption("Cloud demo: paste text → predict label. (Max 500 chars)")

MODEL_PATH = "text_classifier_pipeline.pkl"


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model pipeline not found: {MODEL_PATH}\n\n"
        "Fix:\n"
        "1) Ensure data.csv is in the same folder\n"
        "2) Run: python train_model.py\n"
        "3) Then run: streamlit run app.py"
    )
    st.stop()

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model pipeline.\n\nError: {e}")
    st.stop()


text = st.text_area("Enter text (max 500 chars)", height=220, max_chars=500)

col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("✅ Predict", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.rerun()

if predict_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = pipe.predict([text])[0]
        st.success(f"Predicted Label: {pred}")

        # Optional probability display (if available)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba([text])[0]
            classes = pipe.classes_
            rows = [{"label": c, "probability": float(p)} for c, p in zip(classes, proba)]
            st.write("Prediction probabilities:")
            st.dataframe(rows, use_container_width=True)
