# utils/hybrid_search.py
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any
from utils.embedding import prep_embed_for_search, get_text_embedding, retrieve_context
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


def match_entities(
    user_entities, 
    db_entities, 
    fuzzy_thres=70, 
    ):
    """
    Calculate score based on matching cancer types and biomarkers between user's query and the database
    
    Arguments:
        user_entities (dict): A dictionary with 'cancer_type' and 'biomarker' entities extracted using BioBERT.
        db_entities (list[dict]): A list of dictionaries with 'cancer_type' and 'biomarker' entities extracted from context database using BioBERT.
        fuzzy_thres (int): Threshold for fuzzy string similarity.
    """

    user_cancer = check_list(user_entities.get('cancer_type', []))
    user_biomarker = check_list(user_entities.get('biomarker', []))

    match_score_all=[]
    
    #iterate over all db entities
    for idx, db_entity in enumerate(db_entities):
        score=0
        
        #append matching count
        for db_cancer in db_entity['cancer_type']: 
            db_cancer = check_list(db_cancer)
            if len(set(db_cancer) & set(user_cancer)) > 0:
                score += len(set(db_cancer) & set(user_cancer))
            elif any(fuzz.ratio(dbc, uc) > fuzzy_thres for uc in user_cancer for dbc in db_cancer):
                score += 0.5
        
        for db_biomarker in db_entity['biomarker']:
            db_biomarker = check_list(db_biomarker)
            if len(set(db_biomarker) & set(user_biomarker)) > 0:
                score += len(set(db_biomarker) & set(user_biomarker))
            elif any(fuzz.ratio(dbb, ub) > fuzzy_thres for ub in user_biomarker for dbb in db_biomarker):
                score += 0.5

        if score > 0:
            match_score_all.append((idx, score, db_entity))

    #sort by score descending
    match_score_all.sort(key=lambda x: x[1], reverse=True)
    return match_score_all


@dataclass
class HybridRetrievalResult:
    top_contexts: List[str] # final list of top re-ranked contexts
    retrieved_df: pd.DataFrame  # detailed DataFrame with all scores
    params: Dict[str, Any]


def retrieve_context_hybrid(
    user_entities: dict[str, list[str]], 
    db_entities: list[dict[str, list[str]]],
    user_query: str,
    db_context,
    index,
    client,
    model_embed,
    bm25_top_k=25,
    w_c=0.3,
    w_b=0.3,
    w_ctx=0.4,
    alpha=0.2,
    num_vec=10
    ) -> HybridRetrievalResult:
    """
    Performs hybrid search and retrieval with two stages:
    1. Sparse retrieval (BM25):
        - Uses keyword-based scoring on tokenized entities (`cancer_type` and `biomarker`)
        - Combines entity-level BM25 scores with pre-defined weights
        - Selects top-k candidate contexts
    2. Dense re-ranking (FAISS):
        - Computes query embeddings with the given model
        - Retrieves semantically similar context using FAISS cosine similarity
        - Aligns FAISS scores with BM25-selected contexts and re-ranks
    
    Returns:
        - A list of top `num_vec` re-ranked contexts
        - A DataFrame containing context, BM25 scores, FAISS similarity, and final rankings score
    
    Notes:
        - BM25 captures lexical (exact keyword match) signals
        - FAISS cosine similarity captures semantic similarity, ensuring better ranking of relevant contexts with synonyms
        - Final ranking score is a product of BM25 score and cosine similarity for balanced ranking
    """
    
    #extract user cancer/biomarker entities (handles none)
    user_cancer = check_list(user_entities.get('cancer_type', []))
    user_biomarker = check_list(user_entities.get('biomarker', []))
    
    #tokenize user query and corpus
    tokenized_user_query=tokenize_single_entity(user_query)
    tokenized_db_context=[tokenize_single_entity(ctx) for ctx in db_context]
    
    #tokenize user and db entities (cancer/biomarker) for fine-grained matching
    tokenized_db_entities=tokenize_corpus(db_entities)
    tokenized_user_cancer = [token for e in user_cancer if e for token in tokenize_single_entity(e)]
    tokenized_user_biomarker = [token for e in user_biomarker if e for token in tokenize_single_entity(e)]
    
    #build BM25 indices for each type of information
    cancer_bm25=BM25Okapi([doc['cancer_type'] for doc in tokenized_db_entities])
    biomarker_bm25=BM25Okapi([doc['biomarker'] for doc in tokenized_db_entities])
    context_bm25=BM25Okapi(tokenized_db_context)
    
    #bm25 scoring for user entities (captures lexical match; one caveat is that it relies on NER quality)
    if tokenized_user_cancer:
        cancer_bm25_score=cancer_bm25.get_scores(tokenized_user_cancer)
        #normalized to [0,1]
        cancer_bm25_score_norm=min_max_scaling(cancer_bm25_score)
        #zero score vector if no entity detected 
    else: cancer_bm25_score_norm=np.zeros(len(db_context)) 
    if tokenized_user_biomarker:
        biomarker_bm25_score=biomarker_bm25.get_scores(tokenized_user_biomarker)
        biomarker_bm25_score_norm=min_max_scaling(biomarker_bm25_score)
    else: biomarker_bm25_score_norm=np.zeros(len(db_context))
    
    #bm25 scoring for free-text contexts (captures info missed by entity extraction like disease modifiers)
    context_bm25_score=context_bm25.get_scores(tokenized_user_query)
    context_bm25_score_norm=min_max_scaling(context_bm25_score)    
    
    #combine bm25 scores (weighted sum of the three normalized BM25 scores with weight highest for context)
    combined_score = w_c * cancer_bm25_score_norm + w_b * biomarker_bm25_score_norm + w_ctx * context_bm25_score_norm
    
    #fall back entirely on FAISS semantic similarity when there is low BM25 combined score
    if np.max(combined_score) < 0.1:
        retrieved_chunks, D, I = retrieve_context(
            db_context, 
            user_query,
            client,
            model_embed,
            index,
            num_vec
            )
        faiss_cosine_sim_norm_ordered=(D.ravel()+1)/2
        
        #return only semantic search results
        retrieved_combined=pd.DataFrame({
            'bm25_retrieved_context':None,
            'bm25_index':None,
            'bm25_combined_score':None,
            'faiss_retrieved_context':retrieved_chunks,
            'faiss_cosine_sim':faiss_cosine_sim_norm_ordered
            })
        
    else:
        #top BM25 results
        bm25_top_idx = np.argsort(combined_score)[-bm25_top_k:][::-1]
        bm25_top_scores = combined_score[bm25_top_idx]
        bm25_top_context = [db_context[i] for i in bm25_top_idx]
        
        #FAISS re-ranking on BM25 candidates (captures semantic similarity)
        factor = max(2, min(10, index.ntotal // bm25_top_k)) #determines num. of FAISS neighbors to retrieve
        query_embeddings=np.array([get_text_embedding(user_query, client, model_embed)])
        query_embeddings_norm = query_embeddings/np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        D, I = index.search(prep_embed_for_search(query_embeddings_norm, n_dim=2), k=bm25_top_k*factor)
        faiss_cosine_sim=(D.ravel()+1)/2
        faiss_indices=I.ravel()
        faiss_sim_lookup=dict(zip(faiss_indices, faiss_cosine_sim))
        faiss_cosine_sim_norm_ordered=[faiss_sim_lookup.get(idx, 0) for idx in bm25_top_idx]
        
        #combine BM25+FAISS for final reranking
        retrieved_combined=pd.DataFrame({
            'bm25_retrieved_context':bm25_top_context,
            'bm25_index':bm25_top_idx,
            'bm25_combined_score':bm25_top_scores,
            'faiss_retrieved_context':None,
            'faiss_cosine_sim':faiss_cosine_sim_norm_ordered
            })
        
        #hybrid score
        retrieved_combined['final_ranking_score'] = alpha * retrieved_combined['bm25_combined_score'] + (1-alpha) * retrieved_combined['faiss_cosine_sim']
        retrieved_combined=retrieved_combined.sort_values('final_ranking_score', ascending=False)
        
        #top-n final context chunks
        retrieved_chunks=retrieved_combined['bm25_retrieved_context'].head(num_vec).tolist()
        
    params = {
        'bm25_top_k':bm25_top_k,
        'w_c':w_c,
        'w_b':w_b,
        'w_ctx':w_ctx,
        'factor':factor,
        'alpha':alpha,
        'num_vec':num_vec
    }
    
    return HybridRetrievalResult(
        top_contexts=retrieved_chunks, 
        retrieved_df=retrieved_combined,
        params=params            
    )

    
    
