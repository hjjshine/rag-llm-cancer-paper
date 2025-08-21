#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import json
from dotenv import load_dotenv

# ================== UTIL FUNCTIONS ==================
from utils.embedding import get_context_db, retrieve_context
from utils.prompt import get_prompt
from llm.run_RAGLLM import run_RAG


# ================== MODEL & API IMPORTS ==================
from mistralai.client import MistralClient
from openai import OpenAI
from llm.inference import run_llm
import faiss

# --- existing module state ---
_READY = False
_CLIENT = None
_CONTEXT = None
_INDEX = None
_MODEL_TYPE = None
_MODEL_NAME = None
_MODEL_EMBED = None

def reset():
    """Clear in-memory state so you can re-init with new settings."""
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_TYPE, _MODEL_NAME, _MODEL_EMBED
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
    model_type: str = "gpt",
    model_api: str = "gpt-4o-2024-05-13",
    *,
    force_rebuild: bool = False,
):
    global _READY, _CLIENT, _CONTEXT, _INDEX, _MODEL_TYPE, _MODEL_NAME, _MODEL_EMBED
    if _READY:
        return

    load_dotenv()
    if model_type in ["gpt", "gpt_reasoning"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("OPENAI_API_KEY not set")
        _CLIENT = OpenAI(api_key=api_key)
        _MODEL_EMBED = "text-embedding-3-small"
    elif model_type in ["mistral", "mistral-7b"]:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key: raise RuntimeError("MISTRAL_API_KEY not set")
        _CLIENT = MistralClient(api_key=api_key)
        _MODEL_EMBED = "mistral-embed"
    else:
        raise ValueError("Invalid model_type. Please choose from: mistral-7b, mistral, gpt")

    _MODEL_TYPE = model_type
    _MODEL_NAME = model_api

    index_path, ctx_path = _cache_paths(_MODEL_EMBED)

    if (not force_rebuild) and os.path.exists(index_path) and os.path.exists(ctx_path):
        with open(ctx_path, "r") as f:
            _CONTEXT = json.load(f)
        _INDEX = faiss.read_index(index_path)
    else:
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
        out, _ = run_RAG(
            _CONTEXT,
            text,
            strategy,
            _INDEX,
            _CLIENT,
            num_vec,
            _MODEL_TYPE,
            _MODEL_NAME,
            _MODEL_EMBED,
            max_len,
            temp,
            random_seed,
        )
    else:
        # LLM-only path (no retrieval)
        input_prompt = get_prompt(strategy, text)
        out, _ = run_llm(input_prompt, _CLIENT, _MODEL_TYPE, _MODEL_NAME, max_len, temp, random_seed)
    return out or ""