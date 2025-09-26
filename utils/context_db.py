#utils/context_db
import faiss
from pathlib import Path
import json

    
def load_context(version: str, context_path: str, db: str):
    """
    Load context and index depending on DB.
    
    Args:
        version (str): version string for db files
        context_path (str): file path to the context db (.json)
        db (str): 'fda', 'ema', 'civic'
        
    Returns:
        tuple: (db_context, db_index)
    """
    #1) read context db from file path
    with open(context_path, "r") as f:
        context = json.load(f)
    
    #2) read context index
    index_base_path = "data/latest_db/indexes"
    if db == "fda":
        index = faiss.read_index(f"{index_base_path}/text-embedding-3-small_structured_context__{version}.faiss")
    elif db == "ema":
        index = faiss.read_index(f"{index_base_path}/text-embedding-3-small_ema_structured_context__{version}.faiss")
    elif db == "civic":
        index = faiss.read_index(f"{index_base_path}/text-embedding-3-small_civic_structured_context__{version}.faiss")
    else:
        raise ValueError("db must be 'fda', 'ema', or 'civic'.")
        
    return context, index