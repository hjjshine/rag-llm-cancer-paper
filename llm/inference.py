# llm/inference.py
from typing import Optional, Tuple
from mistralai.models.chat_completion import ChatMessage
#from mistral_inference.generate import generate
import random
import time
import openai

# Retry with exponential backoff (openai's implementation)
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 0.5,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
    ):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Max retries ({max_retries}) exceeded.")
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"[Retry] Rate limit hit. Sleeping {delay:.2f}s before retry {num_retries}...")
                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper

@retry_with_exponential_backoff
def chat_completions_with_backoff(CLIENT, **kwargs):
    return CLIENT.chat.completions.create(**kwargs)

# LLM inference
def run_llm(
    input_prompt,
    CLIENT, 
    model_type, 
    model, 
    max_len, 
    temp, 
    random_seed
    ) -> Tuple[Optional[str], str]:
    """
    Generate an LLM response from a given prompt and parameters.
    
    Arguments:
        input_prompt (str): Input prompt with user-specified query and context if applicable.
        CLIENT (object): Initialized LLM API client.
        model_type (str): One of ['mistral','mistral-7b','gpt','gpt_reasoning'].
        model (str): API name (e.g. ministral-8b-2410, open-mistral-nemo-2407, gpt-4o-2024-05-13).
        max_len (int): Maximum number of output tokens generated. If using a reasoning model, it's the maximum number of reasoning + output tokens.
        temp (float): Sampling temperature (0.0 for deterministic).
        random_seed (int): Seed for reproducibility.
        tokenizer (Optional[object]): Required for 'mistral-7b'
    
    Returns:
        Tuple of (output response [string], input prompt [string])
    
    """

    try:
        if model_type == 'mistral':    
            completion = CLIENT.chat(
                model=model,
                messages=[ChatMessage(role="user", content=input_prompt)],
                temperature=temp,
                max_tokens=max_len,
                random_seed=random_seed,
                response_format={"type": "json_object"}
                )
            
            output = completion.choices[0].message.content
            
        elif model_type in ['gpt', 'gpt_reasoning']:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": input_prompt}],
                "max_completion_tokens": max_len, #length of output tokens
                "seed": random_seed,
                "response_format": {"type": "json_object"}
                }
            
            if model_type == 'gpt':
                params['temperature']=temp
            elif model_type == 'gpt_reasoning':
                params['temperature']=1 #only allows 1
                params['reasoning_effort']='medium'
            
            try:
                # completion = CLIENT.chat.completions.create(**params)
                completion = chat_completions_with_backoff(CLIENT, **params)
                output = completion.choices[0].message.content
                
            except Exception as e:
                print(f"[ERROR] Failed to generate LLM response after retries: {e}")
                output = None
            
        elif model_type == "mistral-7b":
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for mistral-7b inference.")
            
            input_ids = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": input_prompt}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt").to("cuda")

            gen_tokens = model.generate(
                input_ids,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=(temp > 0.0), 
                temperature=temp,
                max_new_tokens=max_len,
                random_seed=random_seed
                )

            output = tokenizer.decode(gen_tokens[0])
            
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose from: 'mistral','mistral-7b','gpt','gpt_reasoning'."
            )

    except Exception as e:
        print(f"[ERROR] Failed to generate LLM response: {e}")
        output = None
        
    return(output, input_prompt)