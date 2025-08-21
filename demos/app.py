# demos/app.py
import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mini_infer import init, answer

st.set_page_config(page_title="RAG-LLM (Minimal)", layout="centered")
st.title("RAG-LLM Demo")
st.caption("Research demo • Not for clinical use")

# One-time init on first load (builds or loads index)
if "ready" not in st.session_state:
    with st.spinner("Initializing (first load may build index)…"):
        init(
            context_json_path="data/structured_context_chunks.json",
            model_type="gpt",
            model_api="gpt-4o-2024-05-13",  # or "gpt-4o-mini" for speed
        )
    st.session_state.ready = True

q = st.text_area(
    "Your question",
    height=120,
    placeholder="e.g., What is the first-line treatment of metastatic urothelial carcinoma with FGFR3 S249C mutation?"
)

if st.button("Ask") and q.strip():
    with st.spinner("Thinking…"):
        raw = answer(q)
    st.subheader("Raw output")
    st.code(raw if isinstance(raw, str) else str(raw), language="json")
