# demos/app.py
import streamlit as st
import sys, os, json, time
from dotenv import load_dotenv

# Make package root importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mini_infer import init, answer, reset

# --- Require password ---
load_dotenv()
PASSWORD = os.getenv("APP_PASSWORD")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pw = st.text_input("Enter password", type="password")
    submit = st.button("Enter")

    if (submit or pw) and pw == PASSWORD:
        st.session_state.authenticated = True
        st.success("Access granted.")
        st.rerun()
    elif (submit or pw) and pw != PASSWORD:
        st.error("Incorrect password.")
        st.stop()

    st.stop() 

st.set_page_config(page_title="RAG-LLM", layout="centered")
st.title("RAG-LLM Demo")
st.caption("Research demo • Not for clinical use")

# ---------- Session state ----------
if "applied" not in st.session_state:
    st.session_state.applied = {
        "model_api": None,
        "run_mode": "RAG-LLM",
        "strategy": 0,
        "initialized": False,
    }
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_answer_meta" not in st.session_state:
    st.session_state.last_answer_meta = None
if "last_answer_stale" not in st.session_state:
    st.session_state.last_answer_stale = False

# ---------- Sidebar (Apply always required) ----------
st.sidebar.header("Model settings")

default_api = "gpt-4o-2024-05-13"
pending_model_api = st.sidebar.text_input(
    "Model API",
    value=st.session_state.applied["model_api"] or default_api
)
pending_run_mode = st.sidebar.radio(
    "Run mode",
    ["RAG-LLM", "LLM only"],
    index=0 if st.session_state.applied["run_mode"] == "RAG-LLM" else 1
)
pending_strategy = st.sidebar.selectbox(
    "Prompt strategy",
    [0, 1, 2, 3, 4, 5],
    index=st.session_state.applied["strategy"]
)

# Unapplied changes?
dirty = (
    (st.session_state.applied["model_api"] != pending_model_api) or
    (st.session_state.applied["run_mode"]  != pending_run_mode)  or
    (st.session_state.applied["strategy"]  != pending_strategy)  or
    (not st.session_state.applied["initialized"])
)

# If user has changed settings (dirty) and we currently show results, mark them stale
if dirty and st.session_state.last_answer is not None:
    st.session_state.last_answer_stale = True

# Apply button
apply_clicked = st.sidebar.button("Apply Settings", type="primary")

# Apply logic: always reset+init on Apply and CLEAR old results
if apply_clicked:
    reset()
    with st.spinner("Initializing…"):
        init(
            context_json_path="data/structured_context_chunks.json",
            model_api=pending_model_api,
        )
    st.session_state.applied.update({
        "model_api": pending_model_api,
        "run_mode": pending_run_mode,
        "strategy": pending_strategy,
        "initialized": True,
    })
    st.sidebar.success("Settings applied.")
    dirty = False
    # Clear results so there is zero ambiguity post-apply
    st.session_state.last_answer = None
    st.session_state.last_answer_meta = None
    st.session_state.last_answer_stale = False

# Hint in the sidebar
if dirty:
    st.sidebar.info("You have unapplied changes. Click **Apply Settings**.")

# ---------- Explicitly note whether using RAG-LLM or LLM only ----------
mode = st.session_state.applied["run_mode"]
badge_bg = "#e7f5ff" if mode == "RAG-LLM" else "#fff4e6"
badge_fg = "#0b7285" if mode == "RAG-LLM" else "#d9480f"
badge_text = "Run mode: RAG-LLM" if mode == "RAG-LLM" else "Run mode: LLM only"

st.markdown(
    f"<div style='text-align:left;margin:6px 0 4px 0;'>"
    f"<span style='background:{badge_bg};color:{badge_fg};"
    f"padding:4px 10px;border-radius:999px;font-size:0.9rem;'>"
    f"{badge_text}</span></div>",
    unsafe_allow_html=True
)

# ---------- Main input ----------
q = st.text_area(
    "Your question",
    height=120,
    placeholder="e.g., What EGFR-targeted therapies are FDA-approved for NSCLC?"
)
ask_clicked = st.button("Ask")

# ---------- Query handling ----------
if ask_clicked:
    if not st.session_state.applied["initialized"]:
        st.warning("Initialize first: click **Apply Settings**.")
    elif dirty:
        st.warning("You changed settings. Click **Apply Settings** to use them.")
    elif q.strip():
        with st.spinner("Thinking…"):
            ans = answer(
                q,
                strategy=st.session_state.applied["strategy"],
                rag=(st.session_state.applied["run_mode"] == "RAG-LLM"),
            )
        st.session_state.last_answer = ans
        st.session_state.last_answer_meta = {
            "model_api": st.session_state.applied["model_api"],
            "run_mode": st.session_state.applied["run_mode"],
            "strategy": st.session_state.applied["strategy"],
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.last_answer_stale = False


# ---------- Results renderer ----------
def render_cards_numbered(payload):
    data = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            st.write(payload); return

    # Normalize into an ordered list of entries
    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        st.write(data); return

    # Render as expanders titled "Treatment N"
    for i, info in enumerate(items, start=1):
        with st.expander(f"Treatment {i}", expanded=True):
            if isinstance(info, dict):
                for k, v in info.items():
                    if v is None or v == "":
                        continue
                    if isinstance(v, (list, dict)):
                        st.markdown(f"- **{k}:** `{json.dumps(v, ensure_ascii=False)}`")
                    else:
                        v_str = str(v)
                        if k.lower().startswith("link") and v_str.startswith("http"):
                            st.markdown(f"- **{k}:** [{v_str}]({v_str})")
                        else:
                            st.markdown(f"- **{k}:** {v_str}")
            else:
                st.write(info)


# ---------- Show results ----------
if st.session_state.last_answer:
    if st.session_state.last_answer_stale:
        st.warning(
            "Results below were generated with **previous settings**. "
            "Click **Apply Settings** and then **Ask** to refresh."
        )
        with st.expander("Previous results (stale)", expanded=False):
            meta = st.session_state.last_answer_meta or {}
            if meta:
                st.caption(
                    f"Generated: {meta.get('ts','')} • "
                    f"Model API: {meta.get('model_api','')} • "
                    f"Mode: {meta.get('run_mode','')} • "
                    f"Prompt strategy: {meta.get('strategy','')}"
                )
            render_cards_numbered(st.session_state.last_answer) 
    else:
        st.subheader("Results")
        meta = st.session_state.last_answer_meta or {}
        if meta:
            st.caption(
                f"Generated: {meta.get('ts','')} • "
                f"Model API: {meta.get('model_api','')} • "
                f"Mode: {meta.get('run_mode','')} • "
                f"Prompt strategy: {meta.get('strategy','')}"
            )
        render_cards_numbered(st.session_state.last_answer)
