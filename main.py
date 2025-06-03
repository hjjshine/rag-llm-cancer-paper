import argparse
import subprocess
from llm.run_LLM import main as run_llm_main
from llm.run_RAGLLM import main as run_ragllm_main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['llm', 'rag-llm'], required=True)
    parser.add_argument('--csv_path', type=str, required=True, help="Path to input query CSV file with a 'prompt' column") 
    parser.add_argument('--model_type', choices=['mistral','gpt','mistral-7b'], type=str, required=True, help="Type of model")
    parser.add_argument('--model_api', type=str, help="Model API name")
    parser.add_argument('--strategy', choices=[0, 1, 2, 3, 4, 5], type=int, default=0, help="Prompt strategy")
    parser.add_argument('--context_chunks', type=str, help="Path to a JSON list of context chunks used for RAG-LLM") # required only for running RAG-LLM 
    parser.add_argument('--num_iter', type=int, default=1, help="Number of iterations of running the pipeline")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature used for LLM inference")
    parser.add_argument('--random_seed', type=int, default=None, help="Random seed used for LLM inference")
    return parser.parse_args()
    

def main():
    args = parse_args()
    
    if args.mode == "rag-llm": 
        if not args.context_chunks:
            raise ValueError("`--context_chunks` is required when mode is 'rag-llm'")
        if not args.model_api:
            raise ValueError("`--model_api` is required when mode is 'rag-llm'")
        
    if args.mode == 'llm':
        run_llm_main(args)
    elif args.mode == 'rag-llm':
        run_ragllm_main(args)

if __name__ == '__main__':
    main()



