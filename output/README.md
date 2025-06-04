## Output

Contains all output results from running LLM and RAG-LLM pipelines.

- `LLM_res_*`: Contains results from baseline LLM-only runs.
- `RAG_res_*`: Contains results from RAG-LLM runs.
- Each subdirectory includes one or more `_res_dict.pkl` files, which store the full LLM responses, input prompts, and runtimes. Filenames follow the convention: `stra{strategy}n{num_iter}temp{temperature}`, indicating the prompt strategy, number of iterations, and temperature used during inference.




