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
    
    
# def retrieve_context_hybrid(
#     user_entities: dict[str, list[str]], 
#     db_entities: list[dict[str, list[str]]],
#     user_query: str,
#     db_context,
#     index,
#     client,
#     model_embed,
#     w_c=0.3,
#     w_b=0.3,
#     w_ctx=0.4,
#     alpha=0.6,
#     faiss_top_k=50,
#     gap_threshold=0.5,
#     num_vec=25
#     ) -> HybridRetrievalResult:
#     """
#     Performs hybrid search and retrieval with two stages:
#     1. Dense search (FAISS):
#         - Computes query embeddings with the given model
#         - Retrieves semantically similar context using FAISS cosine similarity
#     2. Sparse re-ranking (BM25):
#         - Uses keyword-based scoring on tokenized entities (`cancer_type` and `biomarker`)
#         - Combines entity-level BM25 scores with pre-defined weights
#         - Selects top-k candidate contexts
        
#     Returns:
#         - A list of top `num_vec` re-ranked contexts
#         - A DataFrame containing context, BM25 scores, FAISS similarity, and final rankings score
    
#     Notes:
#         - BM25 captures lexical (exact keyword match) signals
#         - FAISS cosine similarity captures semantic similarity, ensuring better ranking of relevant contexts with synonyms
#         - Final ranking score is a product of BM25 score and cosine similarity for balanced ranking
#     """
    
#     #extract user cancer/biomarker entities (handles none)
#     user_cancer = [e for e in user_entities.get('cancer_type', []) if e.lower() not in GENERIC_CANCER_TERMS]
#     user_biomarker = check_list(user_entities.get('biomarker', []))
    
#     #tokenize user query and corpus
#     tokenized_user_query=tokenize_single_entity(user_query)
#     tokenized_db_context=[tokenize_single_entity(ctx.lower()) for ctx in db_context]
    
#     #tokenize user and db entities (cancer/biomarker) for fine-grained matching
#     tokenized_db_entities = tokenize_corpus(db_entities)
#     tokenized_user_cancer = [token for e in user_cancer if e for token in tokenize_single_entity(e)]
#     tokenized_user_biomarker = [token for e in user_biomarker if e for token in tokenize_single_entity(e)]
    
#     #build BM25 indices for each type of information
#     cancer_bm25=BM25Okapi([doc['cancer_type'] for doc in tokenized_db_entities])
#     biomarker_bm25=BM25Okapi([doc['biomarker'] for doc in tokenized_db_entities])
#     context_bm25=BM25Okapi(tokenized_db_context)
    
#     #bm25 scoring for user entities (captures lexical match; one caveat is that it relies on NER quality)
#     if tokenized_user_cancer:
#         cancer_bm25_score=cancer_bm25.get_scores(tokenized_user_cancer)
#         cancer_bm25_score_norm=min_max_scaling(cancer_bm25_score)
#     else: cancer_bm25_score_norm=np.zeros(len(db_context)) 
#     if tokenized_user_biomarker:
#         biomarker_bm25_score=biomarker_bm25.get_scores(tokenized_user_biomarker)
#         biomarker_bm25_score_norm=min_max_scaling(biomarker_bm25_score)
#     else: biomarker_bm25_score_norm=np.zeros(len(db_context))
    
#     #bm25 scoring for free-text contexts (captures info missed by entity extraction like disease modifiers)
#     context_bm25_score=context_bm25.get_scores(tokenized_user_query)
#     context_bm25_score_norm=min_max_scaling(context_bm25_score)    
    
#     #combine bm25 scores (weighted sum of the three normalized BM25 scores with weight highest for context)
#     bm25_combined_score = w_c * cancer_bm25_score_norm + w_b * biomarker_bm25_score_norm + w_ctx * context_bm25_score_norm
    
#     #FAISS similarity
#     query_embeddings=np.array([get_text_embedding(user_query, client, model_embed)])
#     query_embeddings_norm = query_embeddings/np.linalg.norm(query_embeddings, axis=1, keepdims=True)
#     D, I = index.search(prep_embed_for_search(query_embeddings_norm, n_dim=2), k=min(faiss_top_k, index.ntotal))
#     faiss_cosine_sim = (D.ravel()+1)/2
#     faiss_indices = I.ravel()
    
#     #restrict BM25 scores to FAISS candidates
#     bm25_candidate_scores = bm25_combined_score[faiss_indices]

#     #FAISS-only fallback
#     gap = np.median(faiss_cosine_sim) - np.median(bm25_candidate_scores)
#     if not user_cancer and gap > gap_threshold:
#         alpha = 0.0
#     else:
#         alpha = max(0.0, min(1.0, 1 - gap / gap_threshold))
    
#     #hybrid scoring
#     hybrid_score = alpha * bm25_candidate_scores + (1-alpha) * faiss_cosine_sim
#     top_idx_local = np.argsort(hybrid_score)[::-1][:num_vec]
#     top_idx = faiss_indices[top_idx_local]
    
#     retrieved_df = pd.DataFrame({
#         'index': top_idx,
#         'context': [db_context[i].lower() for i in top_idx],
#         'bm25_score': bm25_combined_score[top_idx_local],
#         'faiss_score': faiss_cosine_sim[top_idx_local],
#         'hybrid_score': hybrid_score[top_idx_local]
#     })

#     return HybridRetrievalResult(
#         top_contexts=retrieved_df['context'].tolist(),
#         retrieved_df=retrieved_df,
#         params={'alpha':alpha, 'faiss_top_k':faiss_top_k, 'num_vec':num_vec}
#     )
    
    
# def retrieve_context_hybrid(
#     user_entities: dict[str, list[str]], 
#     db_entities: list[dict[str, list[str]]],
#     user_query: str,
#     db_context,
#     index,
#     client,
#     model_embed,
#     bm25_top_k=25,
#     w_c=0.3,
#     w_b=0.3,
#     w_ctx=0.4,
#     alpha=0.2,
#     num_vec=10
#     ) -> HybridRetrievalResult:
#     """
#     Performs hybrid search and retrieval with two stages:
#     1. Sparse retrieval (BM25):
#         - Uses keyword-based scoring on tokenized entities (`cancer_type` and `biomarker`)
#         - Combines entity-level BM25 scores with pre-defined weights
#         - Selects top-k candidate contexts
#     2. Dense re-ranking (FAISS):
#         - Computes query embeddings with the given model
#         - Retrieves semantically similar context using FAISS cosine similarity
#         - Aligns FAISS scores with BM25-selected contexts and re-ranks
    
#     Returns:
#         - A list of top `num_vec` re-ranked contexts
#         - A DataFrame containing context, BM25 scores, FAISS similarity, and final rankings score
    
#     Notes:
#         - BM25 captures lexical (exact keyword match) signals
#         - FAISS cosine similarity captures semantic similarity, ensuring better ranking of relevant contexts with synonyms
#         - Final ranking score is a product of BM25 score and cosine similarity for balanced ranking
#     """
    
#     #extract user cancer/biomarker entities (handles none)
#     user_cancer = check_list(user_entities.get('cancer_type', []))
#     user_biomarker = check_list(user_entities.get('biomarker', []))
    
#     #tokenize user query and corpus
#     tokenized_user_query=tokenize_single_entity(user_query)
#     tokenized_db_context=[tokenize_single_entity(ctx) for ctx in db_context]
    
#     #tokenize user and db entities (cancer/biomarker) for fine-grained matching
#     tokenized_db_entities=tokenize_corpus(db_entities)
#     tokenized_user_cancer = [token for e in user_cancer if e for token in tokenize_single_entity(e)]
#     tokenized_user_biomarker = [token for e in user_biomarker if e for token in tokenize_single_entity(e)]
    
#     #build BM25 indices for each type of information
#     cancer_bm25=BM25Okapi([doc['cancer_type'] for doc in tokenized_db_entities])
#     biomarker_bm25=BM25Okapi([doc['biomarker'] for doc in tokenized_db_entities])
#     context_bm25=BM25Okapi(tokenized_db_context)
    
#     #bm25 scoring for user entities (captures lexical match; one caveat is that it relies on NER quality)
#     if tokenized_user_cancer:
#         cancer_bm25_score=cancer_bm25.get_scores(tokenized_user_cancer)
#         cancer_bm25_score_norm=min_max_scaling(cancer_bm25_score)
#     else: cancer_bm25_score_norm=np.zeros(len(db_context)) 
#     if tokenized_user_biomarker:
#         biomarker_bm25_score=biomarker_bm25.get_scores(tokenized_user_biomarker)
#         biomarker_bm25_score_norm=min_max_scaling(biomarker_bm25_score)
#     else: biomarker_bm25_score_norm=np.zeros(len(db_context))
    
#     #bm25 scoring for free-text contexts (captures info missed by entity extraction like disease modifiers)
#     context_bm25_score=context_bm25.get_scores(tokenized_user_query)
#     context_bm25_score_norm=min_max_scaling(context_bm25_score)    
    
#     #combine bm25 scores (weighted sum of the three normalized BM25 scores with weight highest for context)
#     combined_score = w_c * cancer_bm25_score_norm + w_b * biomarker_bm25_score_norm + w_ctx * context_bm25_score_norm
    
#     #fall back entirely on FAISS semantic similarity when there is low BM25 combined score
#     if np.max(combined_score) < 0.1:
#         retrieved_chunks, D, I = retrieve_context(
#             db_context, 
#             user_query,
#             client,
#             model_embed,
#             index,
#             num_vec
#             )
#         faiss_cosine_sim_norm_ordered=(D.ravel()+1)/2
        
#         #return only semantic search results
#         retrieved_combined=pd.DataFrame({
#             'bm25_retrieved_context':None,
#             'bm25_index':None,
#             'bm25_combined_score':None,
#             'faiss_retrieved_context':retrieved_chunks,
#             'faiss_cosine_sim':faiss_cosine_sim_norm_ordered
#             })
        
#     else:
#         #top BM25 results
#         bm25_top_idx = np.argsort(combined_score)[-bm25_top_k:][::-1]
#         bm25_top_scores = combined_score[bm25_top_idx]
#         bm25_top_context = [db_context[i] for i in bm25_top_idx]
        
#         #FAISS re-ranking on BM25 candidates (captures semantic similarity)
#         factor = max(2, min(10, index.ntotal // bm25_top_k)) #determines num. of FAISS neighbors to retrieve
#         query_embeddings=np.array([get_text_embedding(user_query, client, model_embed)])
#         query_embeddings_norm = query_embeddings/np.linalg.norm(query_embeddings, axis=1, keepdims=True)
#         D, I = index.search(prep_embed_for_search(query_embeddings_norm, n_dim=2), k=bm25_top_k*factor)
#         faiss_cosine_sim=(D.ravel()+1)/2
#         faiss_indices=I.ravel()
#         faiss_sim_lookup=dict(zip(faiss_indices, faiss_cosine_sim))
#         faiss_cosine_sim_norm_ordered=[faiss_sim_lookup.get(idx, 0) for idx in bm25_top_idx]
        
#         #combine BM25+FAISS for final reranking
#         retrieved_combined=pd.DataFrame({
#             'bm25_retrieved_context':bm25_top_context,
#             'bm25_index':bm25_top_idx,
#             'bm25_combined_score':bm25_top_scores,
#             'faiss_retrieved_context':None,
#             'faiss_cosine_sim':faiss_cosine_sim_norm_ordered
#             })
        
#         #hybrid score
#         retrieved_combined['final_ranking_score'] = alpha * retrieved_combined['bm25_combined_score'] + (1-alpha) * retrieved_combined['faiss_cosine_sim']
#         retrieved_combined=retrieved_combined.sort_values('final_ranking_score', ascending=False)
        
#         #top-n final context chunks
#         retrieved_chunks=retrieved_combined['bm25_retrieved_context'].head(num_vec).tolist()
        
#     params = {
#         'bm25_top_k':bm25_top_k,
#         'w_c':w_c,
#         'w_b':w_b,
#         'w_ctx':w_ctx,
#         'factor':factor,
#         'alpha':alpha,
#         'num_vec':num_vec
#     }
    
#     return HybridRetrievalResult(
#         top_contexts=retrieved_chunks, 
#         retrieved_df=retrieved_combined,
#         params=params            
#     )

    
    
