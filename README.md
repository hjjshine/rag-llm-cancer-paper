# RAG-LLM for Precision Cancer Medicine
## Implementing a context-augmented large language model to guide precision cancer medicine

## Overview

This repository accompanies our paper _"Implementing a context-augmented large language model to guide precision cancer medicine"_(2025), where we present a context-augmented LLM framework to enhance biomarker-driven cancer therapy recommendations.

Our approach integrates an expert-curated clinicogenomic knowledge base - the **Molecular Oncology Almanac (MOAlmanac)** — to improve accuracy and precision in clinical decision support, especially through structured context augmentation and hybrid retrieval approach compared to a general-purpose LLM-only approach.

## Key Features and Results
We present a context-augmented LLM pipeline that:
- Recommends biomaker-driven cancer therapies that are FDA-approved, with latest links to the drug labels.
- Retrieves a structured summary of the context (e.g., cancer type, disease status, biomarker, indication).
- Achieved high accuracy on real-world queries through hybrid retrieval approach.
- Our RAG-LLM implementation is publicly accessible at https://llm.moalmanac.org/.
- Automatically integrates the latest MOAlmanac context database (latest release from December 2025 incorporated in https://llm.moalmanac.org/).

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
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=mistral \
--model_api=open-mistral-nemo-2407 \
--strategy=0 \
--num_iter=1 \
--output_dir=output/LLM_res_mistnemo \
--temp=0.0 --max_len=2048 --random_seed=2025
```
    
#### Run RAG-LLM
```console
~/rag-llm-cancer-paper$ python main.py --mode=rag-llm \
--csv_path=data/latest_db/moalmanac_fda_core_query__2025-10-03.csv \
--model_type=gpt --model_api=gpt-4o-2024-08-06 \
--context_db=fda --context_db_type=structured \
--hybrid_search \
--num_iter=5 \
--strategy=0 \
--output_dir=output/RAG_res_gpt4o_default/structured_synthetic_db_v202510_hybrid_numvec25 \
--temp=0.0 --max_len=2048 --random_seed=2025
```

## Related Paper
If you're interested, check out our paper:
[https://doi.org/10.1101/2025.05.09.25327312](https://doi.org/10.1016/j.ccell.2025.12.017)


    




