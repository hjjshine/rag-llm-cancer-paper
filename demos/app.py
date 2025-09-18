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
st.title("RAG-LLM for Precision Cancer Medicine")
st.markdown(
    "📄 [Read the Preprint](https://www.medrxiv.org/content/10.1101/2025.05.09.25327312v2.full.pdf) • "
    "💻 [GitHub Repository](https://github.com/hjjshine/rag-llm-cancer-paper)"
)
st.caption("_Note: The order of results does **not** indicate priority. Research use only, not for clinical care._")



# ---------- Session state ----------
if "applied" not in st.session_state:
    st.session_state.applied = {
        "model_api": None,
        "run_mode": "RAG-LLM",
        "strategy": 0,
        "initialized": False,
        "temperature": 0,
        "max_len": 2048,
    }

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_answer_meta" not in st.session_state:
    st.session_state.last_answer_meta = None
if "last_answer_stale" not in st.session_state:
    st.session_state.last_answer_stale = False

# ---------- Sidebar (Apply always required) ----------
st.sidebar.header("Model settings")

pending_run_mode = st.sidebar.radio(
    "Run mode",
    ["RAG-LLM", "LLM only"],
    index=0 if st.session_state.applied["run_mode"] == "RAG-LLM" else 1
)

default_api = "gpt-4o-2024-05-13"
model_options = [default_api, "dummy_api_1", "dummy_api_2"]

pending_model_api = st.sidebar.selectbox(
    "Model API",
    options=model_options,
    index=model_options.index(st.session_state.applied.get("model_api", default_api))
    if st.session_state.applied.get("model_api", default_api) in model_options
    else 0,
    help = 'Model API name'
)

pending_strategy = st.sidebar.selectbox(
    "Prompt strategy",
    [0, 1, 2, 3, 4, 5],
    index=st.session_state.applied["strategy"],
    help="Prompt strategy. See [prompt.py](https://github.com/hjjshine/rag-llm-cancer-paper/blob/main/utils/prompt.py) for details."
)

# Temp
pending_temperature = st.sidebar.number_input(
    "Temperature",
    min_value=0.0, max_value=1.0, step=0.1,
    value=float(st.session_state.applied.get("temperature", 0)),
    help="Sampling temperature (0.0 for deterministic)."
)

# Max output tokens selector
pending_max_len = st.sidebar.selectbox(
    "Max output tokens",
    options=[2048, 4096],
    index=0 if st.session_state.applied.get("max_len", 2048) == 2048 else 1,
    help=("Maximum output token LLM can return.")
)

# Unapplied changes?
dirty = (
    (st.session_state.applied["model_api"] != pending_model_api) or
    (st.session_state.applied["run_mode"]  != pending_run_mode)  or
    (st.session_state.applied["strategy"]  != pending_strategy)  or
    (st.session_state.applied["temperature"] != pending_temperature) or
    (st.session_state.applied["max_len"] != pending_max_len) or 
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
        "temperature": pending_temperature,
        "max_len": pending_max_len
    })
    st.sidebar.success("Settings applied.")
    dirty = False
    st.session_state.last_answer = None
    st.session_state.last_answer_meta = None
    st.session_state.last_answer_stale = False


# Hint in the sidebar
if dirty:
    st.sidebar.info("You have unapplied changes. Click **Apply Settings**.")

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
                temp=st.session_state.applied["temperature"],
                max_len=st.session_state.applied["max_len"],
            )
        st.session_state.last_answer = ans
        st.session_state.last_answer_meta = {
            "model_api": st.session_state.applied["model_api"],
            "run_mode": st.session_state.applied["run_mode"],
            "strategy": st.session_state.applied["strategy"],
            "temperature": st.session_state.applied["temperature"],
            "max_len": st.session_state.applied["max_len"],
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
        with st.expander(f"Potential Treatment Option {i}", expanded=True):
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
                    f"Mode: {meta.get('run_mode','')} • "
                    f"Model API: {meta.get('model_api','')} • "
                    f"Prompt strategy: {meta.get('strategy','')} • "
                    f"Temp: {meta.get('temperature','')} • "
                    f"Max tokens: {meta.get('max_len','')}"
                )

            render_cards_numbered(st.session_state.last_answer) 
    else:
        st.subheader("Results")
        meta = st.session_state.last_answer_meta or {}
        if meta:
            st.caption(
                f"Generated: {meta.get('ts','')} • "
                f"Mode: {meta.get('run_mode','')} • "
                f"Model API: {meta.get('model_api','')} • "
                f"Prompt strategy: {meta.get('strategy','')} • "
                f"Temp: {meta.get('temperature','')} • "
                f"Max tokens: {meta.get('max_len','')}"
            )
        render_cards_numbered(st.session_state.last_answer)
