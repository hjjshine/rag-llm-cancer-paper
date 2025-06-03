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


# ================== ARGUMENT PARSER ==================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_path', type=str) # Required. Path to the query dataset csv file
    parser.add_argument('-model_type', type=str) # Required (please choose from 'mistral','gpt', or 'mistral-7b')
    parser.add_argument('-model_api', type=str, default=None) # Required if using APIs
    parser.add_argument('-strategy', type=int, default=0) # Required (please choose from 0, 1, 2, 3)
    parser.add_argument('-context_chunks', type=str) # Required. Path to the list of context chunk object
    parser.add_argument('-num_iter', type=int, default=1)
    parser.add_argument('-random_seed', type=int, default=None) # Optional
    parser.add_argument('-output_dir', type=str) # Required. Please set your output directory name
    parser.add_argument('-temp', type=float, default=0.0) # Optional
    return parser.parse_args()
    
    
# ================== MAIN ==================
def main():
    
    args = parse_args()
    
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
    context_chunks=load_object(args.context_chunks)
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
    







    


#### Have run the following on the command line
#### mistral large API 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='mistral' -model_api='mistral-large-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mistlarge -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistlarge_log.txt


#### mistral small nemo API
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mistnemo >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mistnemo -temp=0.7 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mistnemo -temp=0.3 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_10.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mistnemo_stra3raw/index5k/subset_indication -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_10.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mistnemo_stra3raw/index5k/subset_indication/dedup -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_10.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mistnemo_stra3raw/index5k/full_indication -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_10.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mistnemo_stra3raw/index5k/full_indication/dedup -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_10.csv -model_type='mistral' -model_api='open-mistral-nemo' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mistnemo_stra3raw/indexfull/subset_indication/dedup -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=1 -res_name=RAG_res_mistnemo/final -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=1 -num_iter=1 -res_name=RAG_res_mistnemo/final -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra1_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=2 -num_iter=1 -res_name=RAG_res_mistnemo/final -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra2_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=3 -num_iter=1 -res_name=RAG_res_mistnemo/final -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra3_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=5 -res_name=RAG_res_mistnemo/final -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_log.txt 
## FINAL: 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=5 -res_name=RAG_res_mistnemo/final -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_log.txt 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=4 -num_iter=5 -res_name=RAG_res_mistnemo/final -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra4_log.txt 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=5 -res_name=RAG_res_mistnemo/final -temp=0.00001 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_log.txt 
## FINAL TEST - structured:
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer_toy.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -strategy=0 -context_chunks=/home/helenajun/MOA_LLM/output/structured_context_chunks.pkl -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/RAG_res_mistnemo/final_test -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_test_log.txt 
## FINAL TEST - unstructured:
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer_toy.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -strategy=0 -context_chunks=/home/helenajun/MOA_LLM/output/unstructured_context_chunks.pkl -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/RAG_res_mistnemo/final_test -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra0_test_log.txt 
## FINAL TEST - real-world:
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/data/realworld_questions/real_questions_br_v3.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -strategy=4 -context_chunks=/home/helenajun/MOA_LLM/output/structured_context_chunks.pkl -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/RAG_res_mistnemo/final_test_realworld -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra4_test_realworld_log.txt 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/data/realworld_questions/real_questions_br_v3.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -strategy=5 -context_chunks=/home/helenajun/MOA_LLM/output/structured_context_chunks.pkl -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/RAG_res_mistnemo/final_test_realworld -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mistnemo_stra5_test_realworld_log.txt 


#### mistral 7B (previously used as baseline)
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='baseline-mistral-7b' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mist7B_baseline -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mist7B_baseline_RAG_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='baseline-mistral-7b' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=1 -res_name=RAG_res_mist7B_baseline -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mist7B_baseline_RAG_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='baseline-mistral-7b' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -num_iter=5 -res_name=RAG_res_mist7B_baseline -temp=0.3 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mist7B_baseline_RAG_log.txt


#### gpt mini
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='gpt' -model_api='gpt-4o-mini-2024-07-18' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -num_iter=5 -res_name=RAG_res_gptmini -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_gptmini_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='gpt' -model_api='gpt-4o-mini-2024-07-18' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -num_iter=5 -res_name=RAG_res_gptmini -temp=0.3 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_gptmini_log.txt

#### gpt 
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='gpt' -model_api='gpt-4o-2024-05-13' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -num_iter=5 -res_name=RAG_res_gpt -temp=0.0 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_gpt_log.txt
# python3 run_LLM_RAG_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_RAG_res_8.csv -model_type='gpt' -model_api='gpt-4o-2024-05-13' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -num_iter=5 -res_name=RAG_res_gpt -temp=0.3 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_gpt_log.txt