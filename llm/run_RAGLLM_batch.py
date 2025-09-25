#!/usr/bin/env python3

# ================== GENERAL IMPORTS ==================
import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import faiss


# ================== UTIL FUNCTIONS ==================
from utils.check_db_version import get_local_version
from utils.prompt import get_prompt
from utils.io import save_object
from utils.embedding import retrieve_context
from utils.hybrid_search import retrieve_context_hybrid

            
# ================== BATCH EXECUTION FUNCTIONS ==================
def run_ragllm_batch(
    data, 
    strategy, 
    context_chunks, 
    db_entity,
    query_entity,
    index, 
    client, 
    num_vec, 
    model, 
    model_embed, 
    max_len, 
    temp, 
    random_seed,
    hybrid_search=True
    ):
    """
    Generate responses for a batch of prompts using the OpenAI Batch API.
    
    This function first performs context retrieval for all prompts,
    then formats the requests into a JSONL file, submits a batch job,
    polls for completion, and retrieves the results.
    """
    
    # 1. Create JSONL file with prompts and context
    input_prompts_for_batch = []
    retrieval_results_dict = {'retrieval_params':None, 'retrieval_results':[]}
    
    print("Performing RAG retrieval for all prompts...")
    for i, entry in enumerate(data.itertuples()):
        user_query_idx = entry.Index
        user_query = entry.prompt
        
        query_prompt = get_prompt(strategy, user_query)

        #======= hybrid search retrieval approach =======#
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
            
        #======= dense search retrieval approach =======#
        else:
            retrieved_chunk = retrieve_context(
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
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    
    # Clean up the local file after upload
    os.remove(input_file_path)
    
    # 3. Create a batch job
    print("Creating batch job...")
    batch_job = client.batches.create(
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
        batch_job = client.batches.retrieve(batch_job.id)
        status = batch_job.status
        print(f"Current batch job status: {status}")
    
    if status == "completed":
        print("Batch job completed successfully. Retrieving results...")
        # 5. Retrieve the results
        result_file_id = batch_job.output_file_id
        result_file = client.files.content(result_file_id)
        
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
    return output_test_ls, input_prompt_ls, retrieval_results_dict


def run_iterations_rag_batch(
    data, 
    context_chunks, 
    db_entity,
    query_entity,
    num_vec, 
    index, 
    client, 
    model, 
    model_embed, 
    strategy, 
    num_iterations, 
    max_len, 
    temp, 
    random_seed
    ):
    
    output_ls = []
    runtime_ls = []
    input_ls = []
    retrieval_result_ls = []
    
    for i in range(num_iterations):
        start = time.time()
        
        output_test_ls, input_prompt_ls, retrieval_results_dict = run_ragllm_batch(
            data, 
            strategy, 
            context_chunks, 
            db_entity,
            query_entity,
            index, 
            client, 
            num_vec, 
            model, 
            model_embed,
            max_len, 
            temp, 
            random_seed
            )
        output_ls.append(output_test_ls)
        input_ls.append(input_prompt_ls)
        retrieval_result_ls.append(retrieval_results_dict)

        end = time.time()
        time_elapsed = end - start
        runtime_ls.append(time_elapsed)
        
        time_elapsed = str(f'{time_elapsed/60:.4f}')
        print(f'Time elapsed for iteration {i}: {time_elapsed} min')
        
    return(output_ls, input_ls, retrieval_result_ls, runtime_ls)


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
        _MODEL = args.model_api
        _MODEL_EMBED = 'text-embedding-3-small'
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Please set OPENAI_API_KEY in your .env file.")
        _CLIENT=OpenAI(api_key=api_key)
    else: 
        raise ValueError("The OpenAI Batch API is only available for 'gpt' and 'gpt_reasoning' model types.")
    
    # Load local db version
    _VERSION = get_local_version()
    
    # Load user-specified queries
    _QUERY_DF=pd.read_csv(args.csv_path, index_col=0)

    # Load context database
    with open(args.context_chunks, "r") as f:
        _CONTEXT = json.load(f)
    _INDEX = faiss.read_index(f"./data/latest_db/indexes/text-embedding-3-small_structured_context__{_VERSION}.faiss")
    
    # Load entity database
    with open(f"context_retriever/entities/moalmanac_db_ner_entities__{_VERSION}.json", "r") as f:
        _DB_ENTITY = json.load(f)
    with open(f"context_retriever/entities/synthetic_query_ner_entities__{_VERSION}.json", "r") as f:
        _QUERY_ENTITY = json.load(f)
    
    # Run RAG-LLM iterations with batch processing
    output_ls, input_ls, retrieval_result_ls, runtime_ls = run_iterations_rag_batch(
        data=_QUERY_DF, 
        context_chunks=_CONTEXT, 
        db_entity=_DB_ENTITY,
        query_entity=_QUERY_ENTITY,
        num_vec=10, 
        index=_INDEX, 
        client=_CLIENT, 
        model=_MODEL, 
        model_embed=_MODEL_EMBED, 
        strategy=args.strategy, 
        num_iterations=args.num_iter, 
        max_len=args.max_len, 
        temp=args.temp, 
        random_seed=args.random_seed
        )
    
    # Save results
    res_dict = {
        "full output": output_ls, 
        "input prompt": input_ls, 
        "retrieval": retrieval_result_ls,
        "runtime": runtime_ls
        }
    
    result_file=os.path.join(
        args.output_dir,
        f'RAGbatchstra{str(args.strategy)}n{str(args.num_iter)}temp{str(args.temp)}_res_dict.pkl'
    )
    save_object(res_dict, filename=result_file)
    
    
if __name__ == '__main__':
    main()