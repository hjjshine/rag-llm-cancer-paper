# llm/inference.py
from typing import Optional, Tuple
from mistralai.models.chat_completion import ChatMessage
from mistral_inference.generate import generate

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
        model_type (str): One of ['mistral','gpt','mistral-7b'].
        model (str): API name (e.g. ministral-8b-2410, open-mistral-nemo-2407, gpt-4o-2024-05-13).
        max_len (int): Maximum number of tokens to generate.
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
            
        elif model_type == 'gpt':
            completion = CLIENT.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_prompt}],
                temperature=temp,
                max_tokens=max_len,
                seed=random_seed,
                response_format={"type": "json_object"}
                )
            output = completion.choices[0].message.content
        
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
                max_new_tokens=2048,
                random_seed=random_seed
                )

            output = tokenizer.decode(gen_tokens[0])
            
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose from: 'mistral', 'gpt', 'mistral-7b'."
            )

    except Exception as e:
        print(f"[ERROR] Failed to generate LLM response: {e}")
        output = None
        
    return(output, input_prompt)