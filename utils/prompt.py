# utils/prompt.py
def get_prompt(strategy: int, prompt_chunk: str) -> str:
    
    # JSON schema used in all strategies
    json_schema = """Please provide each treatment as a json format with the following JSON schema.
    {{
        "Treatment 1": {{
            "Disease Name": ,
            "Disease Phase or Condition": ,
            "Drug Name": ,
            "Prior Treatment or Resistance Status": ,
            "Genomic Features":
            "Link to FDA-approved Label": 
            }}
    }}
    Query: {prompt}
    """
    
    # Real-world application JSON schemas
    real_json_schema_a = """
    If no FDA-approved drugs are found, respond: "There are no FDA-approved drugs for the provided context."
    If FDA-approved drugs are found, please provide each treatment as a json format with the following JSON schema. 
    {{
        "Treatment 1": {{
            "Disease Name": ,
            "Disease Phase or Condition": ,
            "Drug Name": ,
            "Prior Treatment or Resistance Status": ,
            "Genomic Features": 
            "FDA-approval status":
            "Link to FDA-approved Label":
            }}
    }}
    Query: {prompt}
    """
    
    real_json_schema_b = """
    If no matching FDA-approved drugs are found, please return message with the following JSON schema:
    {{
        "Status": "no_match",
        "Message": "No drugs are FDA-approved for the provided context"
    }}
    
    If FDA-approved drugs are found, please provide each treatment as a json format with the following JSON schema:
    {{
        "Status": "success",
        "Treatment 1": {{
            "Disease Name": ,
            "Disease Phase or Condition": ,
            "Drug Name": ,
            "Prior Treatment or Resistance Status": ,
            "Genomic Features": 
            "FDA-approval status":
            "Link to FDA-approved Label":
            }}
    }}
    Query: {prompt}
    """

    # System prompt (if applicable)
    sys_prompt = "You are a helpful chat bot specialized in suggesting FDA-approved drugs to treat cancer. "
    
    # Prompt strategies
    templates = {
        0: json_schema,
        1: "Please only provide the therapies that are FDA-approved for the provided genomic biomarkers. "+ json_schema,
        2: sys_prompt + json_schema,
        3: sys_prompt + "Please only include FDA-approved therapies for the provided genomic biomarkers. "+ json_schema,
        4: real_json_schema_a,
        5: real_json_schema_b
    }
    
    return(templates[strategy].format(prompt=prompt_chunk))


    