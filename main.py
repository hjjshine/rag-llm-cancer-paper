import argparse
import subprocess
from llm.run_LLM import main as run_llm_main
from llm.run_LLM_batch import main as run_llm_batch_main
from llm.run_RAGLLM import main as run_ragllm_main
from llm.run_RAGLLM_batch import main as run_ragllm_batch_main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['llm', 'llm-batch', 'rag-llm', 'rag-llm-batch'], required=True)
    parser.add_argument('--csv_path', type=str, required=True, help="Path to input query CSV file with a 'prompt' column") 
    parser.add_argument('--model_type', choices=['mistral','mistral-7b','gpt','gpt_reasoning'], type=str, required=True, help="Type of model")
    parser.add_argument('--model_api', type=str, help="Model API name")
    parser.add_argument('--strategy', choices=[0, 1, 2, 3, 4, 5], type=int, default=0, help="Prompt strategy [default: 0]")
    parser.add_argument('--context_chunks', type=str, help="Path to a JSON list of context chunks used for RAG-LLM") # required only for running RAG-LLM 
    parser.add_argument('--num_iter', type=int, default=1, help="Number of iterations of running the pipeline [default: 1]")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature used for LLM inference [default: 0]")
    parser.add_argument('--max_len', type=int, default=None, help="Maximum output token LLM can return [default: None]")
    parser.add_argument('--random_seed', type=int, default=None, help="Random seed used for LLM inference [default: None]")
    return parser.parse_args()
    

def main():
    args = parse_args()
    
    #check
    if args.mode in ["rag-llm", "rag-llm-batch"]:
        if not args.context_chunks:
            raise ValueError("`--context_chunks` is required when mode is 'rag-llm'")
        if not args.model_api:
            raise ValueError("`--model_api` is required when mode is 'rag-llm'")        
    if args.mode in ['llm-batch', 'rag-llm-batch'] and args.model_type not in ['gpt', 'gpt_reasoning']:
        raise ValueError("Batch mode is only supported for 'gpt' and 'gpt_reasoning' models")
        
    #run selected mode
    if args.mode == 'llm':
        run_llm_main(args)
    elif args.mode == 'llm-batch':
        run_llm_batch_main(args)
    elif args.mode == 'rag-llm':
        run_ragllm_main(args)
    elif args.mode == 'rag-llm-batch':
        run_ragllm_batch_main(args)

if __name__ == '__main__':
    main()



