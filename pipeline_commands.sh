#!/bin/bash

CSV_REAL=data/real_world_queries.csv
CONTEXT_STRUCTURED=data/structured_context_chunks.json
CONTEXT_UNSTRUCTURED=data/unstructured_context_chunks.json
SEED=2025
TEMP=0.0
ITER=5

# ================ o4-mini ================ #
#o4mini - llm only strategy 0 - max_len None
python main.py --mode=llm --csv_path=data/moa_fda_queries_answers.csv --model_type=gpt_reasoning --model_api=o4-mini-2025-04-16 --strategy=0 --num_iter=1 --output_dir=output/LLM_res_o4mini --temp=0.0 --random_seed=2025 2>&1 | tee logs/o4mini_llm_only_stra0.log

#o4mini - llm only strategy 3 - max_len None
python main.py --mode=llm --csv_path=data/moa_fda_queries_answers.csv --model_type=gpt_reasoning --model_api=o4-mini-2025-04-16 --strategy=3 --num_iter=1 --output_dir=output/LLM_res_o4mini --temp=0.0 --random_seed=2025 2>&1 | tee logs/o4mini_llm_only_stra3.log


# ================ gpt4o ================ #
#o4mini - llm only strategy 0 - max_len 2048
python main.py --mode=llm --csv_path=data/moa_fda_queries_answers.csv --model_type=gpt --model_api=gpt-4o-2024-05-13 --strategy=0 --num_iter=5 --output_dir=output/LLM_res_gpt4o --temp=0.0 --random_seed=2025 2>&1 | tee logs/gpt4o_llm_only_stra0.log

#gpt4o - structured - max_len 2048
python main.py --mode=rag-llm --csv_path=data/moa_fda_queries_answers.csv --model_type=gpt --model_api=gpt-4o-2024-05-13 --strategy=0 --context_chunks=data/structured_context_chunks.json --num_iter=5 --output_dir=output/RAG_res_gpt4o/structured --temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_structured_stra0.log

#gpt4o - unstructured - max_len 2048
python main.py --mode=rag-llm --csv_path=data/moa_fda_queries_answers.csv --model_type=gpt --model_api=gpt-4o-2024-05-13 --strategy=0 --context_chunks=data/unstructured_context_chunks.json --num_iter=5 --output_dir=output/RAG_res_gpt4o/unstructured --temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_unstructured_stra0.log

#gpt4o - realworld strategy 4 - max_len 2048
python main.py --mode=rag-llm --csv_path=data/real_world_queries.csv --model_type=gpt --model_api=gpt-4o-2024-05-13 --strategy=4 --context_chunks=data/structured_context_chunks.json --num_iter=5 --output_dir=output/RAG_res_gpt4o/realworld --temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_realworld_stra4.log

#gpt4o - realworld strategy 5 - max_len 2048
python main.py --mode=rag-llm --csv_path=data/real_world_queries.csv --model_type=gpt --model_api=gpt-4o-2024-05-13 --strategy=5 --context_chunks=data/structured_context_chunks.json --num_iter=5 --output_dir=output/RAG_res_gpt4o/realworld --temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_realworld_stra5.log


