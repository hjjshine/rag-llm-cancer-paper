## Experiments to run with RAG-LLM

on minimal clinical data
- 0. run on vanilla LLM (prompt 0 / 5)
- 1. run on RAG-LLM (prompt 0 / 5) 

on enhanced clinical data 
- 2. run on vanilla LLM (prompt 0 / 5)
- 3. run on RAG-LLM (prompt 0 / 5)


### notes

- we've selected 10 samples each from the 5 cancer types, and all with 
- we'll run everything on `gpt-4.1-nano` for now, and adjust
- we don't really need to use the batch for now, as each sample runs ~1s and we only have 50 samples for now. 