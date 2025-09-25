# utils/prompt.py
def get_prompt(strategy: int, prompt_chunk: str) -> str:
    
    # JSON schema used in all strategies
    json_schema_a = """Please provide each treatment as a json format with the following JSON schema.
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
    
    # JSON schema used in all strategies - updated for reasoning
    json_schema_b = """Please provide each treatment as a json format with the following JSON schema.
    {{
        "Treatment 1": {{
            "Disease Name": ,
            "Disease Phase or Condition": ,
            "Drug Name": ,
            "Prior Treatment or Resistance Status": ,
            "Genomic Features":
            "Link to FDA-approved Label": 
            "Rationale":
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

    real_json_schema_b_1 = """
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
            "Rationale":
            }}
    }}
    Query: {prompt}
    """
    
    real_json_schema_b_2 = """
    If no matching FDA-approved drugs are found, please return message and potential treatment options with the following JSON schema:
    {{
        "Status": "no_match",
        "Message": "No drugs are FDA-approved for the provided context"
        "Potential Treatment Options":{{
            "Treatment 1": {{
                "Disease Name": ,
                "Disease Phase or Condition": ,
                "Drug Name": ,
                "Prior Treatment or Resistance Status": ,
                "Genomic Features": 
                "Supporting Evidence Level":
                "Link to evidence":
                "Rationale":
            }}
        }}
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
            "Rationale":
            }}
    }}
    Query: {prompt}
    """
    
    
    
    # System prompt (if applicable)
    sys_prompt = "You are a helpful chat bot specialized in suggesting FDA-approved drugs to treat cancer. "
    
    
    # Prompt strategies
    templates = {
        0: json_schema_a,
        1: "Please only include FDA-approved therapies for the provided genomic biomarkers. "+ json_schema_a,
        2: sys_prompt + json_schema_a,
        3: sys_prompt + "Please only include FDA-approved therapies for the provided genomic biomarkers. "+ json_schema_a,
        4: real_json_schema_a,
        5: real_json_schema_b,
        6: json_schema_b,
        7: real_json_schema_b_2
    }
    
    return(templates[strategy].format(prompt=prompt_chunk))


    