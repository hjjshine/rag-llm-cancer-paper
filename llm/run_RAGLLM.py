#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import json
import os
import time
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd


# ================== UTIL FUNCTIONS ==================
from utils.check_db_version import get_local_version
from utils.prompt import get_prompt
from utils.io import save_object
from utils.embedding import retrieve_context
from utils.hybrid_search import retrieve_context_hybrid
from utils.context_db import load_context
from context_retriever.entity_prediction import load_entities

# ================== MODEL & API IMPORTS ==================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai.client import MistralClient
from openai import OpenAI
from llm.inference import run_llm

    
# ================== RAG-LLM EXECUTION FUNCTIONS ==================
def run_RAG(
    i,
    entry,
    strategy, 
    context_chunks, 
    db_entity,
    query_entity,
    index, 
    client, 
    num_vec, 
    model_type, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed,
    hybrid_search=True
    ) -> Tuple[Optional[str], str]:
    """
    Augment context and run LLM inference.
    
    Arguments:
        entry: each query entry with `prompt` column.
        user_query (str): User specified query.
        strategy (int): Prompt design strategy [0-3].
        index (faiss.Index): Context vector database.
        CLIENT (object): Initialized LLM API client.
        num_vec (int): Number of vectors to retrieve.
        model_type (str): One of ['mistral','gpt','mistral-7b']
        model (str): API name (e.g. ministral-8b-2410, open-mistral-nemo-2407, gpt-4o-2024-05-13, o4-mini-2025-04-16)
        max_len (int): Maximum number of tokens generated.
        temp (float): Sampling temperature (0.0 for deterministic).
        random_seed (int): Seed for reproducibility.
        
    Returns:
        Tuple of (output response [string], input prompt [string])
        
    """
    user_query_idx = entry.Index
    user_query = entry.prompt
    query_prompt=get_prompt(strategy, user_query)
    
    retrieval_results_dict = {'retrieval_params':None, 'retrieval_results':[]}
    if hybrid_search:
        retrieved_results = retrieve_context_hybrid(
                user_entities=query_entity[user_query_idx], 
                db_entities=db_entity,
                user_query=user_query, 
                db_context=context_chunks, 
                index=index,
                client=client,
                model_embed=model_embed
                )
        retrieved_chunk = retrieved_results.top_contexts
            
        if i == 0:
            retrieval_results_dict['retrieval_params']=retrieved_results.params
        
        retrieval_results_dict['retrieval_results'].append({
            'query_idx': user_query_idx, 
            'retrieved_df': retrieved_results.retrieved_df.to_dict(orient="records")
        })
        
    else:
        retrieved_chunk=retrieve_context(
            context_chunks, 
            user_query, 
            client, 
            model_embed, 
            index, 
            num_vec
            )
        
    input_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    {query_prompt} 
    """
    
    return run_llm(input_prompt, CLIENT, model_type, model, max_len, temp, random_seed)



# ================== BATCH EXECUTION FUNCTIONS ==================
def run_ragllm_on_prompts(
    n_iter, 
    data, 
    strategy, 
    context_chunks, 
    db_entity, 
    query_entity, 
    index, 
    client, 
    num_vec, 
    model_type, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed
    ):
    """
    Generate response to each context-augmented prompt. 
    """
    # Initialize output lists
    output_test_ls, input_prompt_ls, retrieval_res_ls = [], [], []
    
    # Run RAG-LLM for each prompt
    for i, entry in enumerate(data.itertuples()):
        output, input_prompt, retrieval_results_dict = run_RAG(
            i,
            entry,
            strategy, 
            context_chunks, 
            db_entity,
            query_entity,
            index, 
            client, 
            num_vec, 
            model_type, 
            model, 
            model_embed, 
            max_len, 
            temp, 
            random_seed
            )
        output_test_ls.append(output)
        input_prompt_ls.append(input_prompt)
        retrieval_res_ls.append(retrieval_results_dict)
        time.sleep(0.3)
    
    return(output_test_ls, input_prompt_ls)

def run_iterations_rag(
    num_iterations, 
    data, 
    strategy, 
    context_chunks, 
    db_entity, 
    query_entity, 
    index, 
    client, 
    num_vec, 
    model_type, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed
    ):
    
    output_ls = []
    runtime_ls = []
    input_ls = []
    retrieval_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls, retrieval_res_ls = run_ragllm_on_prompts(i, data, strategy, context_chunks, db_entity, query_entity, index, client, num_vec, model_type, model, model_embed, max_len, temp, random_seed)
        
        output_ls.append(output_test_ls)
        input_ls.append(input_prompt_ls)
        retrieval_ls.append(retrieval_res_ls)
        
        end = time.time()
        time_elapsed = end - start
        runtime_ls.append(time_elapsed)
        
        time_elapsed = str(f'{time_elapsed/60:.4f}')
        print(f'Time elapsed for iteration {i}: {time_elapsed} min')
        
    return(output_ls, input_ls, runtime_ls)

    
# ================== MAIN ==================
def main(args):
    
    print("CSV path: "+args.csv_path)
    print("Model type: "+args.model_type)
    print("Model API endpoint: "+str(args.model_api))
    print("Prompt strategy: "+str(args.strategy))
    print("Number of iterations: "+str(args.num_iter))
    print("Random seed: "+str(args.random_seed))
    print("Temperature: "+str(args.temp))
    print("Output name: "+args.output_dir)

    # Prepare output directories
    os.makedirs(name=args.output_dir, exist_ok=True)

    # Model config
    load_dotenv()
    
    if args.model_type == 'mistral-7b':
        model_path = "mistralai/Mistral-7B-Instruct-v0.3"
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        _MODEL_EMBED = 'mistral-embed'
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type == 'mistral':
        _MODEL = args.model_api #this could be ministral-8b-2410, open-mistral-nemo-2407, mistral-large-2407, etc.
        _MODEL_EMBED = 'mistral-embed'
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type in ['gpt', 'gpt_reasoning']:
        _MODEL = args.model_api #this could be gpt-4o-2024-05-13, gpt-4o-mini-2024-07-18, etc.
        _MODEL_EMBED = 'text-embedding-3-small'
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        CLIENT=OpenAI(api_key=api_key)
        
    else: 
        raise ValueError("Invalid model_type. Please choose from: mistral-7b, mistral, gpt")
    
    # Load local db version
    _VERSION = get_local_version()
    
    # Load context db
    _CONTEXT, _INDEX = load_context(
        version=_VERSION, 
        context_path=args.context_chunks, 
        db='fda')
    
    # Load query_df for testing
    _QUERY_DF=pd.read_csv(args.csv_path, index_col=0)
        
    # Load db and query entities
    _DB_ENTITY, _QUERY_ENTITY = load_entities(
        version=_VERSION, 
        mode='test_realworld', 
        db='fda',
        query=None)
        
    # Run RAG-LLM iterations
    output_ls, input_ls, retrieval_ls, runtime_ls = run_iterations_rag(
        num_iterations=args.num_iter, 
        data=_QUERY_DF, 
        context_chunks=_CONTEXT, 
        db_entity=_DB_ENTITY,
        query_entity=_QUERY_ENTITY, 
        num_vec=10, 
        index=_INDEX,
        client=CLIENT, 
        model=_MODEL, 
        model_embed=_MODEL_EMBED, 
        model_type=args.model_type,
        strategy=args.strategy, 
        max_len=args.max_len, 
        temp=args.temp, 
        random_seed=args.random_seed
        )
    
    # Save results
    res_dict = {
        "full output": output_ls, 
        "input prompt": input_ls, 
        "retrieval": retrieval_ls,
        "runtime": runtime_ls
        }
    
    result_file=os.path.join(
        args.output_dir,
        f'RAGstra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()
    

