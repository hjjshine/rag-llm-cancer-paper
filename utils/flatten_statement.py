#utils/flatten_statement.py
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import json



# function to subset FDA statements
def subset_db_statements(statements, organization='fda'):
    # subset statements
    subset=[statement for statement in statements if statement['reportedIn'][0]['organization']['id'] == organization]
    return subset


# function to ensure list
def ensure_list(x):
    return x if isinstance(x, list) else [x]


# function to extract biomarker from statement
def extract_biomarker_info(stmt):
    # extract biomarkers from the statement
    biomarkers_list = []
    for i in range(len(stmt['proposition']['biomarkers'])):
        biomarker = stmt["proposition"]["biomarkers"][i]["name"]
        extensions_dict = {item['name']: item['value'] for item in stmt['proposition']['biomarkers'][i]['extensions']}
        presence = extensions_dict.get('present', '')
        biomarkers_list.append(biomarker)
        
        # extract presence information
        if presence == True:
            biomarker += " [present]"
        else:   
            biomarker += " [not present]"
        if i == 0:
            biomarkers_str = biomarker
        else:
            biomarkers_str += f", {biomarker}"
    
    extracted_info = {
        "str": biomarkers_str, 
        "list": biomarkers_list
    }
    
    return extracted_info


# function to extract therapy info from statement
def extract_therapy_info(stmt):
    # extract membership operator
    obj = stmt.get('proposition', {}).get('objectTherapeutic', {})
    operator = obj.get('membership_operator', None)
    
    # extract therapy approach, type, and names
    if operator == 'AND':
        approach = 'Combination therapy'
        therapy_strategyList = []
        therapy_typeList = []
        for therapy in obj.get('therapies', []):
            extensions_dict = {item['name']: item['value'] for item in therapy['extensions']}
            therapy_strategyList.extend(ensure_list(extensions_dict['therapy_strategy']))
            therapy_typeList.extend(ensure_list(extensions_dict['therapy_type']))
        drugList = [drug.get('name', None) for drug in obj.get('therapies', [])]
        
    else:
        approach = 'Monotherapy'
        therapy_strategyList = []
        therapy_typeList = []
        extensions_dict = {item['name']: item['value'] for item in obj['extensions']}
        therapy_strategyList.extend(ensure_list(extensions_dict['therapy_strategy']))
        therapy_typeList.extend(ensure_list(extensions_dict['therapy_type']))
        drugList = [obj.get('name', None)]
    
    # sanity check for drugList
    if any(d is None for d in drugList):
        raise ValueError(f"Found None in drugList for statement {stmt['id']}")
    
    drug_str = " + ".join([d for d in drugList if d is not None])
    therapy_strategy_str = " + ".join([s for s in therapy_strategyList if s is not None])
    therapy_type_str = " + ".join([t for t in therapy_typeList if t is not None])
    
    extracted_info = {
        "str": {
            "drug_str": drug_str, 
            "therapy_approach": approach,
            "therapy_strategy_str": therapy_strategy_str, 
            "therapy_type_str": therapy_type_str
                }, 
        "list": {
            "drugList": drugList, 
            "therapy_approach": approach,
            "therapy_strategyList": therapy_strategyList, 
            "therapy_typeList": therapy_typeList
            }
        }
    
    return extracted_info


# function to flatten statement into summary text to include in context
def flatten_statements(stmt: dict) -> str:
    
    statement_id = stmt.get("id")
    
    # approval status
    approval_status = stmt.get("reportedIn", [{}])[0].get("subtype", "None")
    approval_org = stmt.get("reportedIn", [{}])[0].get("organization", {}).get("id", "Unknown organization")
    approval_url = stmt.get("reportedIn", [{}])[0].get("url", "Unknown URL")
    approval_date = stmt.get("reportedIn", [{}])[0].get("publication_date", "Unknown date")
    
    # description and indication
    description = stmt.get("description", "None")
    indication = stmt.get("indication", {}).get("indication", "None")
    
    # cancer type
    cancer_type = stmt.get("proposition", {}).get("conditionQualifier", {}).get("name", "Unknown cancer")
    
    # biomarkers
    biomarker = extract_biomarker_info(stmt)
    
    # therapy
    therapy_info = extract_therapy_info(stmt)
    
    # create summary text
    summary = (
        f"Indication: {indication}\n"
        f"Cancer type: {cancer_type}\n"
        f"Biomarkers: {biomarker['str']}\n"
        f"Therapy: {therapy_info['str']['drug_str']}\n"
        f"Therapy approach: {therapy_info['str']['therapy_approach']}\n"
        f"Therapy strategy: {therapy_info['str']['therapy_strategy_str']}\n"
        f"Therapy type: {therapy_info['str']['therapy_type_str']}\n"
        f"Description: {description}\n"
        f"Approval status: {approval_status} ({approval_org})\n"
        f"Approval url: {approval_url}\n"
        f"Publication date: {approval_date}"
    )
    
    # create row to add to dataframe
    row = {
        "statement_id": statement_id,
        "approval_status": approval_status,
        "approval_org": approval_org,
        "description": description,
        "indication": indication,
        "cancer_type": cancer_type,
        "biomarker": biomarker['list'],
        "therapy_drug": therapy_info['list']['drugList'],
        "therapy_approach": therapy_info['list']['therapy_approach'],
        "therapy_strategy": therapy_info['list']['therapy_strategyList'],
        "therapy_type": therapy_info['list']['therapy_typeList'],
        "approval_url": approval_url,
        "publication_date": approval_date,
        "context": summary
    }
    
    return summary, row
