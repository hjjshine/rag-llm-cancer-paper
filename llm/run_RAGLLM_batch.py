#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import time
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import faiss

# ================== UTIL FUNCTIONS ==================
from utils.prompt import get_prompt
from utils.io import save_object, load_object
from utils.embedding import get_context_db, retrieve_context

# ================== BATCH EXECUTION FUNCTIONS ==================

def run_ragllm_batch(
    data, 
    strategy, 
    context_chunks, 
    index, 
    CLIENT, 
    num_vec, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed
    ):
    """
    Generate responses for a batch of prompts using the OpenAI Batch API.
    
    This function first performs context retrieval for all prompts,
    then formats the requests into a JSONL file, submits a batch job,
    polls for completion, and retrieves the results.
    """
    
    # 1. Create JSONL file with prompts and context
    input_prompts_for_batch = []
    
    print("Performing RAG retrieval for all prompts...")
    for i, prompt_chunk in enumerate(data['prompt']):
        query_prompt = get_prompt(strategy, prompt_chunk)
        retrieved_chunk = retrieve_context(
            context_chunks, 
            prompt_chunk, 
            CLIENT, 
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
        
        # Prepare the request object for the batch API
        request_obj = {
            "custom_id": f"prompt_{str(i)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": input_prompt}],
                "max_tokens": max_len,
                "temperature": temp,
                "seed": random_seed
            }
        }
        input_prompts_for_batch.append(request_obj)
        
    print(f"Prepared {len(input_prompts_for_batch)} prompts for batch processing.")
    
    # Write requests to a temporary JSONL file
    input_file_path = "batch_requests.jsonl"
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
        completion_window="24h",
    )
    
    # 4. Poll for the batch job status
    print(f"Batch job created with ID: {batch_job.id}")
    print("Waiting for batch job to complete...")
    
    status = batch_job.status
    while status not in ["completed", "failed", "cancelled"]:
        time.sleep(60) # Poll every 30 seconds
        batch_job = CLIENT.batches.retrieve(batch_job.id)
        status = batch_job.status
        print(f"Current batch job status: {status}")
    
    if status == "completed":
        print("Batch job completed successfully. Retrieving results...")
        # 5. Retrieve the results
        result_file_id = batch_job.output_file_id
        result_file = CLIENT.files.content(result_file_id)
        
        # Parse results and format output
        output_results = result_file.text.strip().split('\n')
        
        # Map outputs to original prompts
        output_test_ls = [None] * len(input_prompts_for_batch)
        for line in output_results:
            response_obj = json.loads(line)
            custom_id = int(response_obj["custom_id"].split('_')[-1])
            content = response_obj["response"]["body"]["choices"][0]["message"]["content"]
            output_test_ls[custom_id] = content
            
    else:
        print(f"Batch job failed with status: {status}")
        output_test_ls = [None] * len(input_prompts_for_batch)

    # 6. Return outputs and original inputs
    input_prompt_ls = [req["body"]["messages"][0]["content"] for req in input_prompts_for_batch]
    return output_test_ls, input_prompt_ls

def run_iterations_rag_batch(num_iterations, data, strategy, context_chunks, index, CLIENT, num_vec, model, model_embed, max_len, temp, random_seed):
    output_ls = []
    runtime_ls = []
    input_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls = run_ragllm_batch(
            data=data, 
            strategy=strategy, 
            context_chunks=context_chunks, 
            index=index, 
            CLIENT=CLIENT, 
            num_vec=num_vec, 
            model=model, 
            model_embed=model_embed,
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
        model_embed = 'text-embedding-3-small'
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        CLIENT=OpenAI(api_key=api_key)
    else: 
        raise ValueError("The OpenAI Batch API is only available for 'gpt' and 'gpt_reasoning' model types.")
    
    # Load user-specified queries
    data=pd.read_csv(args.csv_path, index_col=0)

    # Load context database    
    with open(args.context_chunks, "r") as f:
        context_chunks = json.load(f)
        
    # index=get_context_db(context_chunks, CLIENT, model_embed)
    index = faiss.read_index("/home/helenajun/rag-llm-cancer-paper/data/latest_db/indexes/text-embedding-3-small__structured_context_v1.faiss")
    
    # Run RAG-LLM iterations with batch processing
    output_ls, input_ls, runtime_ls = run_iterations_rag_batch(
        num_iterations=args.num_iter, 
        data=data, 
        strategy=args.strategy, 
        context_chunks=context_chunks, 
        index=index, 
        CLIENT=CLIENT, 
        num_vec=10, 
        model=model, 
        model_embed=model_embed, 
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
        f'RAGbatchstra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()