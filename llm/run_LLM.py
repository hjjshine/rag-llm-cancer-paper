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


# ================== ARGUMENT PARSER ==================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_path', type=str) # Required. Please set the path to the MOAlamanc dataset
    parser.add_argument('-model_type', type=str) # Required (please choose from 'mistral','gpt', or 'mistral-7b')
    parser.add_argument('-model_api', type=str, default=None) # Required if using APIs
    parser.add_argument('-strategy', type=int, default=0) # Required (please choose from 0, 1, 2, 3)
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
        
    elif args.model_type == 'gpt':
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
        f'stra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()
    







    


#### Have run the following on the command line
#### mistral 7B
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral-7b' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mist7B -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/RAG_res_mist7B_stra0_log.txt
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral-7b' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=3 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mist7B -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mist7B_stra3_log.txt


#### mistral 8B
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='ministral-8b-2410' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mist8B -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mist8B_stra0_log.txt
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='ministral-8b-2410' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=3 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mist8B -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mist8B_stra3_log.txt
## FINAL TEST:
# python3 run_LLM_iterations_vf.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer_toy.csv -model_type='mistral' -model_api='ministral-8b-2410' -strategy=0 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mist8B_test -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mist8B_stra0_test_log.txt



#### mistral nemo API (24.07 version)
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='open-mistral-nemo-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mistnemo/final -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mistnemo_stra0_log.txt 


#### mistral large API
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='mistral-large-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=0 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mistlarge -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mistlarge_stra0_log.txt
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='mistral' -model_api='mistral-large-2407' -api_key=hw3ZBWZ18AEATjjC92glKhBUfjkrGznq -strategy=3 -num_iter=1 -output_dir=/home/helenajun/MOA_LLM/output/LLM_res_mistlarge -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_mistlarge_stra3_log.txt


#### gpt mini API
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='gpt' -model_api='gpt-4o-mini-2024-07-18' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -strategy=0 -num_iter=1 -res_name=LLM_res_gptmini -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_gptmini_stra0_log.txt
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='gpt' -model_api='gpt-4o-mini-2024-07-18' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -strategy=3 -num_iter=1 -res_name=LLM_res_gptmini -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_gptmini_stra3_log.txt


#### gpt large API
## FINAL:
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='gpt' -model_api='gpt-4o-2024-05-13' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -strategy=0 -num_iter=1 -res_name=LLM_res_gpt -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_gpt_stra0_log.txt
# python3 run_LLM_iterations.py -csv_path=/home/helenajun/MOA_LLM/output/moa_FDA_df_URLupdated_prompt_answer.csv -model_type='gpt' -model_api='gpt-4o-2024-05-13' -api_key=sk-svcacct-SL0bk5DaIBr6MEwaXHHLT3BlbkFJcXzQ5dZHburElE9DtGn1 -strategy=3 -num_iter=1 -res_name=LLM_res_gpt -temp=0.0 -random_seed=2025 >& /home/helenajun/MOA_LLM/output/logs/LLM_res_gpt_stra3_log.txt