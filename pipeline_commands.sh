#!/bin/bash

# ================ gpt4o ================ #
#gpt4o - llm only strategy 0 
python main.py --mode=llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--num_iter=5 \
--strategy=0 \
--output_dir=output/LLM_res_gpt4o_default/db_v202510 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_llm_only_db_v202510_stra0.log

#gpt4o - llm only strategy 3 (complex prompt)
python main.py --mode=llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--num_iter=1 \
--strategy=3 \
--output_dir=output/LLM_res_gpt4o_default/db_v202510 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_llm_only_db_v202510_stra3.log

#gpt4o - latest db unstructured context 
python main.py --mode=rag-llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=unstructured \
--no-hybrid_search \
--num_iter=5 \
--strategy=0 \
--output_dir=output/RAG_res_gpt4o_default/unstructured_synthetic_db_v202510_dense_numvec10 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_unstructured_synthetic_db_v202510_dense_numvec10_stra0.log

#gpt4o - latest db structured context - dense-only
python main.py --mode=rag-llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=0 \
--output_dir=output/RAG_res_gpt4o_default/structured_synthetic_db_v202510_dense_numvec10 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_synthetic_db_v202510_dense_numvec10_stra0.log

python main.py --mode=rag-llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=0 \
--output_dir=output/RAG_res_gpt4o_default/structured_synthetic_db_v202510_dense_numvec25 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_synthetic_db_v202510_dense_numvec25_stra0.log

#gpt4o - latest db structured context - hybrid
python main.py --mode=rag-llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--hybrid_search \
--num_iter=5 \
--strategy=0 \
--output_dir=output/RAG_res_gpt4o_default/structured_synthetic_db_v202510_hybrid_numvec25 \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_synthetic_db_v202510_hybrid_numvec25_stra0.log

#gpt4o - latest db structured context - dense-only - real-world validation
python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_validation__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=5 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_val_v2_db_v202510_dense \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_val_v2_db_v202510_dense_stra5.log

python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_validation__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=4 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_val_v2_db_v202510_dense \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_val_v2_db_v202510_dense_stra4.log

#gpt4o - latest db structured context - hybrid - real-world validation
python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_validation__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--hybrid_search \
--num_iter=5 \
--strategy=5 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_val_v2_db_v202510_hybrid \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_val_v2_db_v202510_hybrid_stra5.log

#gpt4o - latest db structured context - dense - real-world testing
python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_test__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=5 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_test_v2_db_v202510_dense \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_test_v2_db_v202510_dense_stra5.log

python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_test__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--no-hybrid_search \
--num_iter=5 \
--strategy=4 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_test_v2_db_v202510_dense \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_test_v2_db_v202510_dense_stra4.log

#gpt4o - latest db structured context - hybrid - real-world testing
python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_test__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--hybrid_search \
--num_iter=5 \
--strategy=5 \
--output_dir=output/RAG_res_gpt4o_default/structured_realworld_test_v2_db_v202510_hybrid \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_structured_realworld_test_v2_db_v202510_hybrid_stra5.log


#gpt4o default - structured - with civic context db, real-world test - hybrid search retrieval - strategy 7 (flexible, for all possible drugs)
python main.py --mode=rag-llm \
--csv_path=data/real_world_db/real_world_test__v2.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=civic --context_db_type=structured \
--hybrid_search \
--num_iter=5 \
--strategy=7 \
--output_dir=output/RAG_res_gpt4o_default/civic_structured_realworld_test_v2_hybrid \
--temp=0.0 --max_len=2048 --random_seed=2025 2>&1 | tee logs/gpt4o_default_civic_structured_realworld_test_v2_hybrid_stra7.log


