# RAG-LLM for Precision Cancer Medicine
## Implementing a context-augmented large language model to guide precision cancer medicine

## Overview

This repository accompanies our paper _"Implementing a context-augmented large language model to guide precision cancer medicine"_(2025), where we present a context-augmented LLM framework to enhance biomarker-driven cancer therapy recommendations.

Our approach integrates an expert-curated clinicogenomic knowledge base — the **Molecular Oncology Almanac (MOAlmanac)** — to improve accuracy and precision in clinical decision support, especially compared to a general-purpose LLM-only approach.

## Key Features and Results
We present a context-augmented LLM pipeline that:
- Retrieves clinical context for treatment planning from MOAlamanc.
- Augments queries with relevant biomarker-therapy relationships for therapy recommendations.
- Achieved significant increase in accuracy over LLM-only approach.


| Model       | Accuracy               | Precision     |Context        | Queries    |
|-------------|------------------------|---------------|---------------|------------|
| LLM-only    | 62–75%                 | 44%           | None          | Synthetic  |
| RAG-LLM     | **79–91%**             | 49%           | Unstructured  | Synthetic  |
| RAG-LLM     | **94–95%**             | 80%           | Structured    | Synthetic  |
| RAG-LLM     | **81–90%**             | 80%           | Structured    | Real-world |

## Reproducing Results

For reproducing the results, please clone the repository:
```console
git clone https://github.com/hjjshine/rag-llm-cancer-paper.git  
cd rag-llm-cancer-paper  
pip install -r requirements.txt  
```

### Pipeline Usage Examples
#### Run LLM-only   
```console
~/rag-llm-cancer-paper$ python main.py \
--mode=llm \
--csv_path=data/moa_fda_queries_answers.csv \
--model_type=mistral \
--model_api=open-mistral-nemo-2407 \
--strategy=0 \
--num_iter=1 \
--output_dir=output/LLM_res_mistnemo \
--temp=0.0 \
--max_len=2048 \
--random_seed=2025
```
    
#### Run RAG-LLM
```console
~/rag-llm-cancer-paper$ python main.py \
--mode=rag-llm \
--csv_path=data/moa_fda_queries_answers.csv \
--model_type=gpt \
--model_api=gpt-4o-2024-05-13 \
--strategy=0 \
--context_chunks=data/structured_context_chunks.json \
--num_iter=1 \
--output_dir=output/RAG_res_gpt4o \
--temp=0.0 \
--max_len=2048 \
--random_seed=2025 
```

## Related Paper
If you're interested, check out our preprint:
https://doi.org/10.1101/2025.05.09.25327312


    




