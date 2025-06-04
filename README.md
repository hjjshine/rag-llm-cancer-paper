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


| Model       | Accuracy               | Context       | Queries    |
|-------------|------------------------|---------------|------------|
| LLM-only    | 54–69%                 | None          | Synthetic  |
| RAG-LLM     | **73–85%**             | Unstructured  | Synthetic  |
| RAG-LLM     | **91–99%**             | Structured    | Synthetic  |
| RAG-LLM     | **75–94%**             | Structured    | Real-world |

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
--random_seed=2025
```
    
#### Run RAG-LLM
```console
~/rag-llm-cancer-paper$ python main.py \
--mode=rag-llm \
--csv_path=data/moa_fda_queries_answers.csv \
--model_type=mistral \
--model_api=open-mistral-nemo-2407 \
--strategy=0 \
--context_chunks=data/structured_context_chunks.json \
--num_iter=1 \
--output_dir=output/RAG_res_mistnemo \
--temp=0.0 \
--random_seed=2025 
```

## Related Paper
If you're interested, check out our preprint:
https://www.medrxiv.org/content/10.1101/2025.05.09.25327312v1


    




