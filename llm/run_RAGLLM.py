#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import time
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
from dataclasses import dataclass


# ================== UTIL/PIPELINE FUNCTIONS ==================
from utils.check_db_version import get_local_version, sync_db
from utils.prompt import get_prompt
from utils.io import save_object
from utils.embedding import retrieve_context
from utils.context_db import load_context
from context_retriever.entity_prediction import load_entities
from context_retriever.hybrid_search import retrieve_context_hybrid


# ================== MODEL & API IMPORTS ==================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai.client import MistralClient
from openai import OpenAI
from llm.inference import run_llm


# ================== CONFIGS ==================
@dataclass
class ModelConfig:
    client: object
    model_type: str
    model: object
    model_embed: object
    max_len: int
    temp: float
    random_seed: int

@dataclass
class RetrievalConfig:
    strategy: str
    context_chunks: int
    hybrid_search: bool
    db_entity: dict
    query_entity: dict
    index: object
    num_vec: int

    
# ================== RAG-LLM EXECUTOR ==================
def run_RAG(i, entry, model_cfg, retrieval_cfg) -> Tuple[Optional[str], str]:
    """
    One query -> retrieval + LLM response generation.
    
    Arguments:
        entry: each query entry in `prompt` column.
        strategy (int): Prompt design strategy [0-3].
        index (faiss.Index): Context vector database.
        client (object): Initialized LLM API client.
        num_vec (int): Number of vectors to retrieve.
        model_type (str): One of ['mistral','gpt','mistral-7b']
        model (str): API name (e.g. ministral-8b-2410, open-mistral-nemo-2407, gpt-4o-2024-05-13, o4-mini-2025-04-16)
        max_len (int): Maximum number of tokens generated.
        temp (float): Sampling temperature (0.0 for deterministic).
        random_seed (int): Seed for reproducibility.
        
    Returns:
        Tuple of (output response [string], input prompt [string])
    """
    #Unpack configs
    client=model_cfg.client
    model=model_cfg.model
    model_embed=model_cfg.model_embed
    model_type=model_cfg.model_type
    max_len=model_cfg.max_len
    temp= model_cfg.temp
    random_seed=model_cfg.random_seed

    strategy=retrieval_cfg.strategy
    context_chunks=retrieval_cfg.context_chunks
    hybrid_search=retrieval_cfg.hybrid_search
    db_entity= retrieval_cfg.db_entity
    query_entity=retrieval_cfg.query_entity
    index=retrieval_cfg.index
    num_vec=retrieval_cfg.num_vec
    
    user_query_idx = entry.Index
    user_query = entry.prompt
    query_prompt = get_prompt(strategy, user_query)
    
    # Retrieval
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
    
    # Build final prompt
    input_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    {query_prompt} 
    """
    
    output, input_query = run_llm(input_prompt, client, model_type, model, max_len, temp, random_seed)
    
    return output, input_query, retrieval_results_dict


def run_ragllm_on_prompts(i, data, model_cfg, retrieval_cfg):
    """
    Run RAG-LLM on all queries.
    """
    full_output_ls, input_query_ls, retrieval_res_ls = [], [], []
    
    # Run RAG-LLM for each prompt
    for i, entry in enumerate(data.itertuples()):
        output, query, retrieval_res = run_RAG(i, entry, model_cfg, retrieval_cfg)
        full_output_ls.append(output)
        input_query_ls.append(query)
        retrieval_res_ls.append(retrieval_res)
        time.sleep(0.3)
    
    return(full_output_ls, input_query_ls, retrieval_res_ls)


# ================== PIPELINE RUNNER ==================
class PipelineRunner:
    def __init__(self, model_cfg: ModelConfig, retrieval_cfg: RetrievalConfig):
        self.model_cfg = model_cfg
        self.retrieval_cfg = retrieval_cfg
        
    def run(self, num_iterations: int, data):
        output_ls, input_ls, retrieval_ls, runtime_ls = [], [], [], []
        
        for i in range(num_iterations):
            start = time.time()
            
            full_output, input_query, retrieval_res = run_ragllm_on_prompts(i, data, self.model_cfg, self.retrieval_cfg)
            
            output_ls.append(full_output)
            input_ls.append(input_query)
            retrieval_ls.append(retrieval_res)
            
            time_elapsed = time.time() - start
            runtime_ls.append(time_elapsed)
            time_elapsed = str(f'{time_elapsed/60:.4f}')
            print(f'Time elapsed for iteration {i}: {time_elapsed} min')
            
        return(output_ls, input_ls, retrieval_ls, runtime_ls)

    
# ================== MAIN ==================
def main(args):
    
    print(f"CSV path: {args.csv_path}")
    print(f"Model type: {args.model_type}")
    print(f"Model API endpoint: {args.model_api}")
    print(f"Context DB: {args.context_db}")
    print(f"Prompt strategy: {args.strategy}")
    print(f"Number of iterations: {args.num_iter}")
    print(f"Random seed: {args.random_seed}")
    print(f"Temperature: {args.temp}")
    print(f"Output name: {args.output_dir}")

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
        _CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type == 'mistral':
        _MODEL = args.model_api #this could be ministral-8b-2410, open-mistral-nemo-2407, mistral-large-2407, etc.
        _MODEL_EMBED = 'mistral-embed'
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        _CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type in ['gpt', 'gpt_reasoning']:
        _MODEL = args.model_api #this could be gpt-4o-2024-05-13, gpt-4o-mini-2024-07-18, etc.
        _MODEL_EMBED = 'text-embedding-3-small'
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        _CLIENT=OpenAI(api_key=api_key)
        
    else: 
        raise ValueError("Invalid model_type. Please choose from: mistral-7b, mistral, gpt")
    
    # Check db latest version and load version
    sync_db()
    _VERSION = get_local_version()
    
    # Load context db
    _CONTEXT, _INDEX = load_context(
        version=_VERSION, 
        db=args.context_db,
        db_type=args.context_db_type
        )
        
    # Load db and query entities
    _DB_ENTITY, _QUERY_ENTITY = load_entities(
        version=_VERSION, 
        mode='test_realworld', 
        db=args.context_db,
        query=None)
    
    # Load query_df for testing
    _QUERY_DF=pd.read_csv(args.csv_path, index_col=0)
        
    # Run RAG-LLM iterations
    model_cfg = ModelConfig(
        client=_CLIENT,
        model=_MODEL,
        model_embed=_MODEL_EMBED,
        model_type=args.model_type,
        max_len=args.max_len,
        temp=args.temp,
        random_seed=args.random_seed
    )

    retrieval_cfg = RetrievalConfig(
        strategy=args.strategy,
        context_chunks=_CONTEXT,
        hybrid_search=args.hybrid_search,
        db_entity=_DB_ENTITY,
        query_entity=_QUERY_ENTITY,
        index=_INDEX,
        num_vec=10
    )

    runner = PipelineRunner(model_cfg, retrieval_cfg)
    
    output_ls, input_ls, retrieval_ls, runtime_ls = runner.run(num_iterations=args.num_iter, data=_QUERY_DF)
    
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
    

