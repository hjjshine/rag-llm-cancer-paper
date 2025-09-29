#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import json
import os
import time
import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import torch

# ================== UTIL FUNCTIONS ==================
from utils.prompt import get_prompt
from utils.io import save_object, load_object

# ================== MODEL & API IMPORTS ==================
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai.client import MistralClient
from openai import OpenAI
from llm.inference import run_llm


# ================== BATCH EXECUTION FUNCTIONS ==================
def run_llm_on_prompts(n_iter, data, strategy, CLIENT, model_type, model, max_len, temp, random_seed):
    """
    Generate response to each prompt.
    
    """
    # Initialize output lists
    output_test_ls, input_prompt_ls = [], []
    
    # Run LLM for each prompt
    for prompt_chunk in data['prompt']:
        input_prompt = get_prompt(strategy, prompt_chunk)
        output, input_prompt = run_llm(input_prompt, CLIENT, model_type, model, max_len, temp, random_seed)
        output_test_ls.append(output)
        input_prompt_ls.append(input_prompt)
        time.sleep(0.3)
    
    return(output_test_ls, input_prompt_ls)


def run_iterations(num_iterations, data, strategy, CLIENT, model_type, model, max_len, temp, random_seed):
    output_ls = []
    input_ls = []
    runtime_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls = run_llm_on_prompts(
            i, data, strategy, CLIENT, model_type, model, max_len, temp, random_seed
            )
         
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
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type == 'mistral':
        model = args.model_api #this could be ministral-8b-2410, open-mistral-nemo-2407, mistral-large-2407, etc.
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set MISTRAL_API_KEY in your .env file.")
        CLIENT=MistralClient(api_key=api_key)
        
    elif args.model_type in ['gpt', 'gpt_reasoning']:
        model = args.model_api #this could be gpt-4o-2024-05-13, gpt-4o-mini-2024-07-18, etc.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        CLIENT=OpenAI(api_key=api_key)
        
    else: 
        raise ValueError("Invalid model_type. Please choose from: mistral-7b, mistral, gpt")
    
    # Load MOAlamanc-derived queries
    data=pd.read_csv(args.csv_path, index_col=0)

    # Run LLM iterations
    output_ls, input_ls, runtime_ls = run_iterations(
        num_iterations=args.num_iter, 
        data=data, 
        strategy=args.strategy, 
        CLIENT=CLIENT, 
        model_type=args.model_type, 
        model=model, 
        max_len=args.max_len, 
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
        f'stra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()
    
