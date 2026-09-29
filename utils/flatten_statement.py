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


def _get_extension_value(extensions, name, default=None):
    for ext in extensions or []:
        if ext.get("name") == name:
            return ext.get("value", default)
    return default


def extract_indication(stmt):
    indication = _get_extension_value(stmt.get("extensions"), "indication", {})
    if isinstance(indication, dict):
        return indication.get("description", "None")
    return indication or "None"


# function to extract biomarker from statement
def extract_biomarker_info(stmt):
    # extract biomarkers from the statement
    biomarkers_list = []
    biomarkers = _get_extension_value(
        stmt.get("proposition", {}).get("extensions"), "biomarkers", []
    )
    biomarkers_str = ""
    for i, criterion in enumerate(biomarkers):
        biomarker = criterion.get("subject", {}).get("name")
        if not biomarker:
            raise ValueError(f"Biomarker name is missing for statement {stmt['id']}")
        presence = criterion.get("present")
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
    operator = obj.get('membershipOperator')
    
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
    reported_in = stmt.get("reportedIn", [{}])[0]
    extensions = reported_in.get("extensions", [])
    agent = _get_extension_value(extensions, "agent", {})
    urls = reported_in.get("urls") or []
    approval_status = reported_in.get("subtype") or reported_in.get("documentType", "None")
    approval_org = reported_in.get("organization", {}).get("id") or agent.get("id", "Unknown organization")
    approval_url = reported_in.get("url") or (urls[0] if urls else "Unknown URL")
    approval_date = reported_in.get("publication_date") or _get_extension_value(extensions, "publication_date", "Unknown date")
    
    # description and indication
    description = stmt.get("description", "None")
    indication = extract_indication(stmt)
    
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
