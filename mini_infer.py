#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import json
import re
from dotenv import load_dotenv
from types import SimpleNamespace

# ================== UTIL FUNCTIONS ==================
from utils.embedding import index_context_db
from utils.prompt import get_prompt
from llm.run_RAGLLM import run_RAG

# ================== MODEL & API IMPORTS ==================
from openai import OpenAI
from llm.inference import run_llm
import faiss

# --- module state ---
_READY = False
_CLIENT = None
_CONTEXT = None
_INDEX = None
_MODEL_TYPE = "gpt"
_MODEL_NAME = None
_MODEL_EMBED = "text-embedding-3-small"


def reset():
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_TYPE, _MODEL_NAME, _MODEL_EMBED
    _READY = False
    _CLIENT = None
    _CONTEXT = None
    _INDEX = None
    _MODEL_NAME = None


def _cache_paths(embed_name: str, version: str = "v1"):
    os.makedirs("indexes", exist_ok=True)
    return (
        f"indexes/{embed_name}__{version}.faiss",
        f"indexes/{embed_name}__{version}.context.json",
    )


def _extract_json_blob(text: str) -> str:
    matches = re.findall(r"\{.*\}", text, flags=re.S)
    if not matches:
        return text
    return max(matches, key=len)


def _safe_json_load(payload):
    if isinstance(payload, (dict, list)):
        return payload
    if not isinstance(payload, str):
        return payload
    s = payload.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    candidate = _extract_json_blob(s)
    if candidate != s:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return payload


def _collect_treatments_from_dict(d: dict):
    """Return list of treatment dicts from keys like 'Treatment 1', 'Treatment 2', ..."""
    treatments = []
    for k in sorted(d.keys()):
        if k.lower().startswith("treatment"):
            v = d[k]
            if isinstance(v, dict):
                treatments.append(v)
            elif isinstance(v, list):
                treatments.extend([x for x in v if isinstance(x, dict)])
    return treatments


def _normalize_model_output(raw_output):
    """
    Normalize model output for prompt strategies 0–5.
    Returns:
      - JSON string of List[Dict] if treatments found
      - Plain string message if explicit no-match
      - Otherwise the original raw_output
    """
    # Strategy 4 explicit sentence
    if isinstance(raw_output, str) and "There are no FDA-approved drugs for the provided context" in raw_output:
        return "There are no FDA-approved drugs for the provided context."

    obj = _safe_json_load(raw_output)

    # Dict path: strategy 5 (Status/Message) or 0–4 ("Treatment N")
    if isinstance(obj, dict):
        status = obj.get("Status")
        if status == "no_match":
            return "There are no FDA-approved drugs for the provided context."

        treatments = _collect_treatments_from_dict(obj)
        if treatments:
            return json.dumps(treatments, ensure_ascii=False)

    # List path: already a list of treatment dicts
    if isinstance(obj, list):
        treatments = [x for x in obj if isinstance(x, dict)]
        if treatments:
            return json.dumps(treatments, ensure_ascii=False)

    # Fallback: return original so the UI can still show it
    return raw_output


def init(
    context_json_path: str = "data/structured_context_chunks.json",
    model_api: str = "gpt-4o-2024-05-13",
    *,
    force_rebuild: bool = False,
):
    """
    Initializes OpenAI client, loads (or builds) the FAISS index ONCE per process.
    """
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_NAME, _MODEL_EMBED
    if _READY:
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    _CLIENT = OpenAI(api_key=api_key)
    _MODEL_NAME = model_api

    index_path, ctx_path = _cache_paths(_MODEL_EMBED)

    # Always prefer cached index if present; only build once
    if (not force_rebuild) and os.path.exists(index_path) and os.path.exists(ctx_path):
        with open(ctx_path, "r") as f:
            _CONTEXT = json.load(f)
        _INDEX = faiss.read_index(index_path)
    else:
        with open(context_json_path, "r") as f:
            _CONTEXT = json.load(f)
        _INDEX = index_context_db(_CONTEXT, _CLIENT, _MODEL_EMBED)
        faiss.write_index(_INDEX, index_path)
        with open(ctx_path, "w") as f:
            json.dump(_CONTEXT, f)

    _READY = True


def answer(
    text: str,
    strategy: int = 0,
    num_vec: int = 10,
    max_len: int = 2048,
    temp: float = 0.0,
    random_seed: int = 2025,
    *,
    rag: bool = True,
) -> str:
    if not _READY:
        raise RuntimeError("Call init(...) first.")
    if rag:
        entry = SimpleNamespace(Index=0, prompt=text)

        model_cfg = SimpleNamespace(
            client=_CLIENT,
            model_type=_MODEL_TYPE,
            model=_MODEL_NAME,
            model_embed=_MODEL_EMBED,
            max_len=max_len,
            temp=temp,
            random_seed=random_seed,
        )

        retrieval_cfg = SimpleNamespace(
            strategy=strategy,
            context_chunks=_CONTEXT,
            hybrid_search=False,
            db_entity=None,
            query_entity=None,
            index=_INDEX,
            num_vec=num_vec,
        )

        out, _, _ = run_RAG(0, entry, model_cfg, retrieval_cfg)
    else:
        # LLM-only path (no retrieval)
        input_prompt = get_prompt(strategy, text)
        out, _ = run_llm(input_prompt, _CLIENT, _MODEL_TYPE, _MODEL_NAME, max_len, temp, random_seed)
    normalized = _normalize_model_output(out)
    return normalized or ""

