#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import time
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Tuple, List, Optional

# ================== UTIL FUNCTIONS ==================
from utils.prompt import get_prompt
from utils.io import save_object

# ================== BATCH EXECUTION FUNCTIONS ==================
def run_llm_batch(
    data: pd.DataFrame, 
    strategy: int, 
    CLIENT: OpenAI, 
    model: str, 
    max_len: Optional[int], 
    temp: float, 
    random_seed: Optional[int]
    ) -> Tuple[List[Optional[str]], List[str]]:
    """
    Generate responses for a batch of prompts using the OpenAI Batch API.
    
    This function formats the prompts into a JSONL file, submits a batch job,
    polls for completion, and retrieves the results.

    Arguments:
        data (pd.DataFrame): DataFrame containing the input prompts.
        strategy (int): Prompt design strategy [0-3].
        CLIENT (OpenAI): Initialized OpenAI API client.
        model (str): OpenAI model API name.
        max_len (Optional[int]): Maximum number of output tokens.
        temp (float): Sampling temperature.
        random_seed (Optional[int]): Seed for reproducibility.
        
    Returns:
        Tuple of (output responses [list of strings], input prompts [list of strings])
    """
    
    # 1. Create a list of request objects for the batch API
    input_prompts_for_batch = []
    
    print("Preparing prompts for batch processing...")
    for i, prompt_chunk in enumerate(data['prompt']):
        input_prompt = get_prompt(strategy, prompt_chunk)
        
        # Prepare the request object for the batch API
        request_obj = {
            "custom_id": f"prompt_{i}", # Use a custom ID to map results back to prompts
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": input_prompt}],
                "temperature": temp,
                "seed": random_seed
            }
        }
        
        # Add max_tokens if specified
        if max_len is not None:
            request_obj["body"]["max_tokens"] = max_len
            
        input_prompts_for_batch.append(request_obj)
        
    print(f"Prepared {len(input_prompts_for_batch)} prompts for batch processing.")
    
    # Write requests to a temporary JSONL file
    input_file_path = "batch_requests_llm.jsonl"
    with open(input_file_path, 'w') as f:
        for request in input_prompts_for_batch:
            f.write(json.dumps(request) + '\n')
            
    # 2. Upload the JSONL file to OpenAI
    print("Uploading file to OpenAI...")
    batch_input_file = CLIENT.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    
    # Clean up the local file after upload
    os.remove(input_file_path)
    
    # 3. Create a batch job
    print("Creating batch job...")
    batch_job = CLIENT.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    # 4. Poll for the batch job status
    print(f"Batch job created with ID: {batch_job.id}")
    print("Waiting for batch job to complete...")
    
    status = batch_job.status
    while status not in ["completed", "failed", "cancelled"]:
        # Poll every 30 seconds
        time.sleep(30) 
        batch_job = CLIENT.batches.retrieve(batch_job.id)
        status = batch_job.status
        print(f"Current batch job status: {status}")
    
    # 5. Process results if the job completed successfully
    output_test_ls = [None] * len(input_prompts_for_batch)
    if status == "completed":
        print("Batch job completed successfully. Retrieving results...")
        result_file_id = batch_job.output_file_id
        result_file_content = CLIENT.files.content(result_file_id)
        
        # Parse results and map outputs to original prompts
        output_results = result_file_content.text.strip().split('\n')
        
        for line in output_results:
            try:
                response_obj = json.loads(line)
                custom_id = int(response_obj["custom_id"].split('_')[-1])
                content = response_obj["response"]["body"]["choices"][0]["message"]["content"]
                output_test_ls[custom_id] = content
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line from results file: {line}. Error: {e}")
                
    else:
        print(f"Batch job failed with status: {status}")

    # 6. Return outputs and original inputs
    input_prompt_ls = [get_prompt(strategy, p) for p in data['prompt']]
    return output_test_ls, input_prompt_ls

def run_iterations_llm_batch(num_iterations, data, strategy, CLIENT, model, max_len, temp, random_seed):
    output_ls = []
    runtime_ls = []
    input_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls = run_llm_batch(
            data=data, 
            strategy=strategy, 
            CLIENT=CLIENT, 
            model=model, 
            max_len=max_len, 
            temp=temp, 
            random_seed=random_seed
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
    
    # Only GPT models can use the batch API
    if args.model_type in ['gpt', 'gpt_reasoning']:
        model = args.model_api
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        CLIENT=OpenAI(api_key=api_key)
    else: 
        raise ValueError("The OpenAI Batch API is only available for 'gpt' and 'gpt_reasoning' model types.")
    
    # Load user-specified queries
    data=pd.read_csv(args.csv_path, index_col=0)

    # Run LLM iterations with batch processing
    output_ls, input_ls, runtime_ls = run_iterations_llm_batch(
        num_iterations=args.num_iter, 
        data=data, 
        strategy=args.strategy, 
        CLIENT=CLIENT, 
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
        f'LLMbatchstra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()
