# context_retriever/hybrid_search.py
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any
from utils.embedding import prep_embed_for_search, get_text_embedding
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
import numpy as np


def tokenize_single_entity(entity: str) -> List[str]:
    tokens = entity.split()
    return tokens


def tokenize_corpus(
    corpus: list[dict[str, list[str]]], 
    entity=['cancer_type', 'biomarker']
    ) -> Dict[List[str], List[str]]:
    tokenized_corpus = []
    for doc in corpus:
        tokenized_doc = {}
        for ent in entity:
            tokenized_ent = []
            for e in doc[ent]:
                if e is not None and isinstance(e, str):
                    tokenized_ent.extend(tokenize_single_entity(e))
                    
            tokenized_doc[ent] = tokenized_ent
            
        tokenized_corpus.append(tokenized_doc)
    
    return tokenized_corpus


def min_max_scaling(score: np.ndarray, eps=1e-8) -> np.ndarray:
    norm_score=(score - score.min())/(score.max()-score.min()+eps)
    return norm_score


def check_list(input):
    if isinstance(input, list):
        input = input
    else:
        input = [input]
    return input


@dataclass
class HybridRetrievalResult:
    top_contexts: List[str] # final list of top re-ranked contexts
    retrieved_df: pd.DataFrame  # detailed DataFrame with all scores
    params: Dict[str, Any]

GENERIC_CANCER_TERMS = {'cancer', 'tumor'}

def fuzzy_score_entities(user_terms, db_terms):
    """
    Compute fuzzy score across multiple user entities.
    Returns average max per user term to reflect coverage.
    """
    if not user_terms or not db_terms:
        return 0.0
    scores = []
    for ut in user_terms:
        best_score = 0
        for dt in db_terms:
            score = fuzz.token_set_ratio(ut, dt) / 100.0
            best_score = max(best_score, score)
        scores.append(best_score)
    return np.mean(scores)  


def retrieve_context_hybrid(
    user_entities: dict[str, list[str]], 
    db_entities: list[dict[str, list[str]]],
    user_query: str,
    db_context,
    index,
    client,
    model_embed,
    w_c=0.3,
    w_b=0.3,
    w_ctx=0.4,
    alpha=0.6,
    faiss_top_k=50,
    gap_threshold=0.5,
    num_vec=25
    ) -> HybridRetrievalResult:
    """
    Performs hybrid search and retrieval with two stages:
    1. Dense search (FAISS):
        - Computes query embeddings with the given model
        - Retrieves semantically similar context using FAISS cosine similarity
    2. Sparse re-ranking (BM25):
        - Uses keyword-based scoring on tokenized entities (`cancer_type` and `biomarker`)
        - Combines entity-level BM25 scores with pre-defined weights
        - Selects top-k candidate contexts
        
    Returns:
        - A list of top `num_vec` re-ranked contexts
        - A DataFrame containing context, BM25 scores, FAISS similarity, and final rankings score
    
    Notes:
        - BM25 captures lexical (exact keyword match) signals
        - FAISS cosine similarity captures semantic similarity, ensuring better ranking of relevant contexts with synonyms
        - Final ranking score is a product of BM25 score and cosine similarity for balanced ranking
    """
    
    #extract user cancer/biomarker entities (handles none)
    user_cancer = [e for e in user_entities.get('cancer_type', []) if e.lower() not in GENERIC_CANCER_TERMS]
    user_biomarker = check_list(user_entities.get('biomarker', []))
    
    #tokenize user query and corpus
    tokenized_user_query=tokenize_single_entity(user_query)
    tokenized_db_context=[tokenize_single_entity(ctx.lower()) for ctx in db_context]
    
    #tokenize user and db entities (cancer/biomarker) for fine-grained matching
    tokenized_db_entities = tokenize_corpus(db_entities)
    tokenized_user_cancer = [token for e in user_cancer if e for token in tokenize_single_entity(e)]
    tokenized_user_biomarker = [token for e in user_biomarker if e for token in tokenize_single_entity(e)]
    
    #build BM25 indices for each type of information
    cancer_bm25=BM25Okapi([doc['cancer_type'] for doc in tokenized_db_entities])
    biomarker_bm25=BM25Okapi([doc['biomarker'] for doc in tokenized_db_entities])
    context_bm25=BM25Okapi(tokenized_db_context)
    
    #bm25 scoring for user entities (captures lexical match; one caveat is that it relies on NER quality)
    if tokenized_user_cancer:
        cancer_bm25_score=cancer_bm25.get_scores(tokenized_user_cancer)
        cancer_bm25_score_norm=min_max_scaling(cancer_bm25_score)
        
        #fuzzy score across entities
        fuzzy_cancer_score=np.array([fuzzy_score_entities(user_cancer, ent['cancer_type'])
                                     for ent in db_entities])
        fuzzy_cancer_score_norm=min_max_scaling(fuzzy_cancer_score)
        cancer_score_final=np.maximum(cancer_bm25_score_norm, fuzzy_cancer_score_norm)
    else: 
        cancer_score_final=np.zeros(len(db_context))
        
    if tokenized_user_biomarker:
        biomarker_bm25_score=biomarker_bm25.get_scores(tokenized_user_biomarker)
        biomarker_bm25_score_norm=min_max_scaling(biomarker_bm25_score)
        
        #fuzzy score across entities
        fuzzy_biomarker_score=np.array([fuzzy_score_entities(user_biomarker, ent['biomarker'])
                                        for ent in db_entities])
        fuzzy_biomarker_score_norm=min_max_scaling(fuzzy_biomarker_score)
        biomarker_score_final=np.maximum(biomarker_bm25_score_norm, fuzzy_biomarker_score_norm)
        
    else: 
        # biomarker_bm25_score_norm=np.zeros(len(db_context))
        biomarker_score_final=np.zeros(len(db_context))
    
    #bm25 scoring for free-text contexts (captures info missed by entity extraction like disease modifiers)
    context_bm25_score=context_bm25.get_scores(tokenized_user_query)
    context_bm25_score_norm=min_max_scaling(context_bm25_score)    
    
    #combine bm25 scores (weighted sum of the three normalized BM25 scores with weight highest for context)
    bm25_combined_score = w_c * cancer_score_final + w_b * biomarker_score_final + w_ctx * context_bm25_score_norm
    
    #FAISS similarity
    query_embeddings=np.array([get_text_embedding(user_query, client, model_embed)])
    query_embeddings_norm = query_embeddings/np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    D, I = index.search(prep_embed_for_search(query_embeddings_norm, n_dim=2), k=min(faiss_top_k, index.ntotal))
    faiss_cosine_sim = (D.ravel()+1)/2
    faiss_indices = I.ravel()
    
    #restrict BM25 scores to FAISS candidates
    bm25_candidate_scores = bm25_combined_score[faiss_indices]

    #FAISS-only fallback
    gap = np.median(faiss_cosine_sim) - np.median(bm25_candidate_scores)
    if not user_cancer and gap > gap_threshold:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, 1 - gap / gap_threshold))
    
    #hybrid scoring
    hybrid_score = alpha * bm25_candidate_scores + (1-alpha) * faiss_cosine_sim
    top_idx_local = np.argsort(hybrid_score)[::-1][:num_vec]
    top_idx = faiss_indices[top_idx_local]
    
    retrieved_df = pd.DataFrame({
        'index': top_idx,
        'context': [db_context[i].lower() for i in top_idx],
        'bm25_score': bm25_combined_score[top_idx_local],
        'faiss_score': faiss_cosine_sim[top_idx_local],
        'hybrid_score': hybrid_score[top_idx_local]
    })

    return HybridRetrievalResult(
        top_contexts=retrieved_df['context'].tolist(),
        retrieved_df=retrieved_df,
        params={'alpha':alpha, 'faiss_top_k':faiss_top_k, 'num_vec':num_vec}
    )
    
