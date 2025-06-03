#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import json
import os
import time
import argparse
from typing import Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import torch

# ================== UTIL FUNCTIONS ==================
from utils.prompt import get_prompt
from utils.io import save_object, load_object
from utils.embedding import get_context_db, retrieve_context

# ================== MODEL & API IMPORTS ==================
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai.client import MistralClient
from openai import OpenAI
from llm.inference import run_llm

    
# ================== RAG-LLM EXECUTION FUNCTIONS ==================
def run_RAG(
    context_chunks, 
    prompt_chunk, 
    strategy, 
    index, 
    CLIENT, 
    num_vec, 
    model_type, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed
    ) -> Tuple[Optional[str], str]:
    """
    Augment context and run LLM inference.
    
    Arguments:
        context_chunks (list): List of context chunks for retrieval.
        prompt_chunk (str): User specified query.
        strategy (int): Prompt design strategy [0-3].
        index (faiss.Index): Context vector database.
        CLIENT (object): Initialized LLM API client.
        num_vec (int): Number of vectors to retrieve.
        model_type (str): One of ['mistral','gpt','mistral-7b']
        model (str): API name (e.g. ministral-8b-2410, open-mistral-nemo-2407, gpt-4o-2024-05-13).
        max_len (int): Maximum number of tokens to generate.
        temp (float): Sampling temperature (0.0 for deterministic).
        random_seed (int): Seed for reproducibility.
        
    Returns:
        Tuple of (output response [string], input prompt [string])
        
    """
    
    query_prompt=get_prompt(strategy, prompt_chunk)
    retrieved_chunk=retrieve_context(context_chunks, prompt_chunk, CLIENT, model_embed, index, num_vec)
        
    input_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    {query_prompt} 
    """
    
    return run_llm(input_prompt, CLIENT, model_type, model, max_len, temp, random_seed)



# ================== BATCH EXECUTION FUNCTIONS ==================
def run_ragllm_on_prompts(n_iter, data, strategy, context_chunks, index, CLIENT, num_vec, model_type, model, model_embed, max_len, temp, random_seed):
    """
    Generate response to each context-augmented prompt. 
    """
    # Initialize output lists
    output_test_ls, input_prompt_ls = [], []
    
    # Run RAG-LLM for each prompt
    for prompt_chunk in data['prompt']:
        output, input_prompt = run_RAG(context_chunks, prompt_chunk, strategy, index, CLIENT, num_vec, model_type, model, model_embed, max_len, temp, random_seed)
        output_test_ls.append(output)
        input_prompt_ls.append(input_prompt)
    
    return(output_test_ls, input_prompt_ls)

def run_iterations_rag(num_iterations, data, strategy, context_chunks, index, CLIENT, num_vec, model_type, model, model_embed, max_len, temp, random_seed):
    output_ls = []
    runtime_ls = []
    input_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls = run_ragllm_on_prompts(i, data, strategy, context_chunks, index, CLIENT, num_vec, model_type, model, model_embed, max_len, temp, random_seed)
        output_ls.append(output_test_ls)
        input_ls.append(input_prompt_ls)

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
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_embed = 'mistral-embed'
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type == 'mistral':
        model = args.model_api #this could be ministral-8b-2410, open-mistral-nemo-2407, mistral-large-2407, etc.
        model_embed = 'mistral-embed'
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type == 'gpt':
        model = args.model_api #this could be gpt-4o-2024-05-13, gpt-4o-mini-2024-07-18, etc.
        model_embed = 'text-embedding-3-small'
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        CLIENT=OpenAI(api_key=api_key)
        
    else: 
        raise ValueError("Invalid model_type. Please choose from: mistral-7b, mistral, gpt")
    
    # Load user-specified queries
    data=pd.read_csv(args.csv_path, index_col=0)

    # Load context database    
    with open(args.context_chunks, "r") as f:
        context_chunks = json.load(f)
        
    index=get_context_db(context_chunks, CLIENT, model_embed)
    
    # Run RAG-LLM iterations
    output_ls, input_ls, runtime_ls = run_iterations_rag(
        num_iterations=args.num_iter, 
        data=data, 
        strategy=args.strategy, 
        context_chunks=context_chunks, 
        index=index, 
        CLIENT=CLIENT, 
        num_vec=10, 
        model_type=args.model_type, 
        model=model, 
        model_embed=model_embed, 
        max_len=2048, 
        temp=args.temp, 
        random_seed=args.random_seed
        )
    
    # Save results
    res_dict = {
        "full output": output_ls, 
        "input prompt": input_ls, 
        "runtime": runtime_ls
        }
    
    result_file=os.path.join(
        args.output_dir,
        f'RAGstra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()
    

