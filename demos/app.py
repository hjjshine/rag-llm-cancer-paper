# demos/app.py
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mini_infer import init, answer, reset

st.set_page_config(page_title="RAG-LLM", layout="centered")
st.title("RAG-LLM Demo")
st.caption("Research demo • Not for clinical use")

# --- Sidebar controls (demo only) ---
st.sidebar.header("Options")
model_type = st.sidebar.selectbox("Model type", ["gpt", "mistral"], index=0)
# default model per type
default_api = "gpt-4o-2024-05-13" if model_type == "gpt" else "mistral-large-2407"
model_api = st.sidebar.text_input("Model API", value=default_api)
run_mode = st.sidebar.radio("Run mode", ["RAG-LLM", "LLM only"], index=0)
strategy = st.sidebar.selectbox("Strategy", [0,1,2,3,4,5], index=0)
force_rebuild = st.sidebar.checkbox("Rebuild index (next init)", value=False)
reinit = st.sidebar.button("Apply & (re)initialize")

# persist chosen settings in session
if "settings" not in st.session_state:
    st.session_state.settings = {"model_type": None, "model_api": None}

need_reinit = (
    reinit
    or st.session_state.settings["model_type"] != model_type
    or st.session_state.settings["model_api"]  != model_api
)

if need_reinit:
    reset()
    with st.spinner("Initializing (may build or load index)…"):
        init(
            context_json_path="data/structured_context_chunks.json",
            model_type=model_type,
            model_api=model_api,
            force_rebuild=force_rebuild,
        )
    st.session_state.settings = {"model_type": model_type, "model_api": model_api}
    st.success("Initialized.")

# --- Main input ---
q = st.text_area(
    "Your question",
    height=120,
    placeholder="e.g., What EGFR-targeted therapies are FDA-approved for NSCLC?"
)

if st.button("Ask") and q.strip():
    with st.spinner("Thinking…"):
        raw = answer(
            q,
            strategy=strategy,
            rag = (run_mode == "RAG-LLM"),
        )
    st.subheader("Raw output")
    st.code(raw if isinstance(raw, str) else str(raw), language="json")
