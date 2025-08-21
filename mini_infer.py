# mini_infer.py
import os, json
from dotenv import load_dotenv

import faiss  # add this
from utils.embedding import get_context_db, retrieve_context
from utils.prompt import get_prompt
from llm.inference import run_llm  # reuses their inference
from openai import OpenAI
from mistralai.client import MistralClient

# ---- module state ----
_READY = False
_CLIENT = None
_CONTEXT = None
_INDEX = None
_MODEL_TYPE = None
_MODEL_NAME = None
_MODEL_EMBED = None

def _cache_paths(embed_name: str, version: str = "v1"):
    os.makedirs("indexes", exist_ok=True)
    return (
        f"indexes/{embed_name}__{version}.faiss",
        f"indexes/{embed_name}__{version}.context.json",
    )

def init(
    context_json_path: str = "data/structured_context_chunks.json",
    model_type: str = "gpt",                # "gpt", "gpt_reasoning", "mistral", "mistral-7b"
    model_api: str = "gpt-4o-2024-05-13",   # or "gpt-4o-mini" for speed
):
    """Builds or loads context index once and prepares the client."""
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_TYPE, _MODEL_NAME, _MODEL_EMBED
    if _READY:
        return

    load_dotenv()

    if model_type in ("gpt", "gpt_reasoning"):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _CLIENT = OpenAI(api_key=key)
        _MODEL_EMBED = "text-embedding-3-small"
    elif model_type in ("mistral", "mistral-7b"):
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        _CLIENT = MistralClient(api_key=key)
        _MODEL_EMBED = "mistral-embed"
    else:
        raise ValueError("model_type must be one of: gpt, gpt_reasoning, mistral, mistral-7b")

    _MODEL_TYPE = model_type
    _MODEL_NAME = model_api

    index_path, ctx_path = _cache_paths(_MODEL_EMBED)

    if os.path.exists(index_path) and os.path.exists(ctx_path):
        # Fast load
        with open(ctx_path, "r") as f:
            _CONTEXT = json.load(f)
        _INDEX = faiss.read_index(index_path)
    else:
        # Build once, then persist
        with open(context_json_path, "r") as f:
            _CONTEXT = json.load(f)
        _INDEX = get_context_db(_CONTEXT, _CLIENT, _MODEL_EMBED)
        faiss.write_index(_INDEX, index_path)
        with open(ctx_path, "w") as f:
            json.dump(_CONTEXT, f)

    _READY = True

def answer(
    text: str,
    strategy: int = 0,
    num_vec: int = 10,       # reduce to 5 for more speed
    max_len: int = 2048,     
    temp: float = 0.0,
    random_seed: int = 2025,
) -> str:
    """Fast per-query call: retrieve context + call LLM."""
    if not _READY:
        raise RuntimeError("Call init(...) once before answer().")

    query_prompt = get_prompt(strategy, text)
    retrieved = retrieve_context(_CONTEXT, text, _CLIENT, _MODEL_EMBED, _INDEX, num_vec)

    input_prompt = f"""
Context information is below.
---------------------
{retrieved}
---------------------
{query_prompt}
""".strip()

    out, _ = run_llm(
        input_prompt,
        _CLIENT,
        _MODEL_TYPE,
        _MODEL_NAME,
        max_len,
        temp,
        random_seed,
    )
    return out or ""
