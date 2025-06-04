## Data

This directory contains input query datasets, context chunks, and supporting files used in LLM and RAG-LLM pipeline and evaluation. 

- `moa_fda_queries_answers.csv`: Synthetic queries (`prompt`) and answers (`answer`) generated using MOAlamanc data. Includes only FDA-approved drugs.
- `real_world_queries.csv`: Real-world expert queries (`prompt`) and ground-truth answers (`Groundtruth_answers`).
- `structured_context_chunks.json`: Preprocessed context chunks derived from structured data fields in MOALamanc (e.g. feature_type, disease, context, etc.).
- `unstructured_context_chunks.json`: Preprocessed context chunks extracted from unstructured sections of FDA drug labels (e.g. *Indication and Usage*).
- `*groundtruth_dict.pkl`: Ground-truth mappings between prompts and their correct drugs for evaluation.
- `fda_drug_names_mapping_dict.pkl`: Dictionary mapping brand names to generic drug names used in evaluation.


