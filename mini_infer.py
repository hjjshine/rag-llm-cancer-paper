#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import json
import re
from dotenv import load_dotenv
from types import SimpleNamespace

# ================== UTIL FUNCTIONS ==================
from utils.prompt import get_prompt
from llm.run_RAGLLM import run_RAG
from utils.check_db_version import get_local_version
from utils.context_db import load_context
from context_retriever.entity_prediction import load_entities

# ================== MODEL & API IMPORTS ==================
from openai import OpenAI
from llm.inference import run_llm

# --- module state ---
_READY = False
_CLIENT = None
_CONTEXT = None
_INDEX = None
_MODEL_TYPE = "gpt"
_MODEL_NAME = None
_MODEL_EMBED = "text-embedding-3-small"

# Hybrid
_DB_ENTITY = None
_ENTITY_DB = "fda"
_ENTITY_VERSION = None


def reset():
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_TYPE, _MODEL_NAME, _MODEL_EMBED
    global _DB_ENTITY, _ENTITY_DB, _ENTITY_VERSION
    _READY = False
    _CLIENT = None
    _CONTEXT = None
    _INDEX = None
    _MODEL_NAME = None
    _DB_ENTITY = None
    _ENTITY_DB = "fda"
    _ENTITY_VERSION = None


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
    if (
        isinstance(raw_output, str)
        and "There are no FDA-approved drugs for the provided context" in raw_output
    ):
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
    *,
    model_api: str = "gpt-4o-2024-05-13",
    use_hybrid: bool = True,
    entity_db: str = "fda",
    db_type: str = "structured",
):
    """
    Initialize client, load version-matched context & FAISS via load_context,
    and (optionally) load corpus-side entity annotations for hybrid search.
    """
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_NAME, _MODEL_EMBED
    global _DB_ENTITY, _ENTITY_DB, _ENTITY_VERSION

    if _READY:
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    _CLIENT = OpenAI(api_key=api_key)
    _MODEL_NAME = model_api

    version = get_local_version()
    _ENTITY_VERSION = version
    _ENTITY_DB = entity_db

    # Load context + FAISS from the versioned pipeline only
    _CONTEXT, _INDEX = load_context(
        version=_ENTITY_VERSION,
        db=_ENTITY_DB,
        db_type=db_type,
    )

    if use_hybrid:
        # Load only the corpus-side annotations; query entities are computed per request in answer function (mode='deploy')
        _DB_ENTITY, _ = load_entities(
            version=_ENTITY_VERSION,
            mode="test_synthetic",  # reads db entities file; query side ignored
            db=_ENTITY_DB,
            query=None,
        )

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
    hybrid_search: bool = False,
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

        # Build retrieval config depending on hybrid flag
        db_entity = None
        query_entity = None
        use_hybrid = bool(hybrid_search)

        if use_hybrid:
            # require corpus-side entities to be loaded at init
            if _DB_ENTITY is None:
                raise RuntimeError(
                    "Hybrid search requested, but DB entities are not loaded. "
                    "Call init(..., use_hybrid=True, entity_db='<fda|ema|civic>') first."
                )

            # on-demand query-side entities for this question
            try:
                _, qe = load_entities(
                    version=_ENTITY_VERSION,
                    mode="deploy",
                    db=_ENTITY_DB,
                    query=text,
                )
                if isinstance(qe, dict) and 0 in qe:
                    query_entity = qe
                elif isinstance(qe, dict):
                    query_entity = {0: qe}
                else:
                    query_entity = None
            except Exception as e:
                raise RuntimeError(f"Hybrid search entity extraction failed: {e}")

            if not (query_entity and query_entity.get(0)):
                raise RuntimeError(
                    "Hybrid search could not extract entities from the query. "
                    "Please rephrase with a cancer type and biomarker (e.g., 'EGFR exon 19 deletion in NSCLC')."
                )

            db_entity = _DB_ENTITY  # safe now

        retrieval_cfg = SimpleNamespace(
            strategy=strategy,
            context_chunks=_CONTEXT,
            hybrid_search=use_hybrid,
            db_entity=db_entity,
            query_entity=query_entity,
            index=_INDEX,
            num_vec=num_vec,
        )

        out, _, _ = run_RAG(0, entry, model_cfg, retrieval_cfg)

    else:
        # LLM-only path (no retrieval)
        input_prompt = get_prompt(strategy, text)
        out, _ = run_llm(
            input_prompt, _CLIENT, _MODEL_TYPE, _MODEL_NAME, max_len, temp, random_seed
        )

    normalized = _normalize_model_output(out)
    return normalized or ""
