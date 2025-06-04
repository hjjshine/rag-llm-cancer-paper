## LLM

This directory contains core scripts for running LLM and RAG-LLM pipelines. 

- `run_LLM.py`: Runs LLM baseline inference.
- `run_RAGLLM.py`: Runs context-augmented LLM inference.
- `inference.py`: Contains model-specific logic to generate responses, enabling consistent output across model backends.

These scripts are called from `main.py`, so you shouldn't need to run them directly unless debugging or manipulating specific parts.


