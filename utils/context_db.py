#utils/context_db
import os
from dotenv import load_dotenv
import faiss
import json
import requests
import pandas as pd
from utils.flatten_statement import flatten_statements, extract_biomarker_info,extract_therapy_info  
from utils.embedding import index_context_db
from context_retriever.entity_prediction import db_extract_entities
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
_CLIENT = OpenAI(api_key=api_key)
_MODEL_EMBED = "text-embedding-3-small"


def _cache_paths(output_dir: str, embed_name: str, db: str, name: str, version: str):
    os.makedirs(output_dir, exist_ok=True)
    return (
        f"{output_dir}/{embed_name}_{db}_{name}__{version}.faiss",
        f"{output_dir}/{embed_name}_{db}_{name}__{version}.json",
    )
    
    
def load_context(version: str, db: str, db_type: str):
    """
    Load context and index depending on DB.
    
    Args:
        version (str): version string for db files
        context_path (str): file path to the context db (.json)
        db (str): 'fda', 'ema', 'civic'
        db_type (str): 'structured' or 'unstructured'
        
    Returns:
        tuple: (db_context, db_index)
    """
    
    #moalmanac db
    if db in ["fda", "ema"]:
        index_path, ctx_path = _cache_paths(
            output_dir="data/latest_db/indexes", 
            embed_name=_MODEL_EMBED, 
            db=db, 
            name=f"{db_type}_context", 
            version=version)
        #1) read context db
        with open(ctx_path, "r") as f:
            context = json.load(f)
        #2) read context index
        index = faiss.read_index(index_path)
    #civic db 
    elif db == 'civic':
        index_path, ctx_path = _cache_paths(
            output_dir="data/latest_db/indexes", 
            embed_name=_MODEL_EMBED, 
            db=db, 
            name=f"{db_type}_context", 
            version="2025-10-01"
            )
        with open(ctx_path, "r") as f:
            context = json.load(f)
        index = faiss.read_index(index_path)
    else:
        raise ValueError("db must be 'fda', 'ema', or 'civic'.")
        
    return context, index


def subset_db_statements(statements, organization='fda'):
    # subset statements
    subset=[statement for statement in statements if statement['reportedIn'][0]['organization']['id'] == organization]
    return subset


# load disease modifiers to add more context to cancer types 
with open(f"data/latest_db/disease_modifiers__2025-09-04.json", "r") as f:
    modifiers = json.load(f)
    

def extract_clinical_modifiers(raw_cancer_type, standardized_cancer_type, modifiers):
    raw_cancer_type_lower = raw_cancer_type.lower()
    extracted_modifiers = [mod for mod in modifiers if mod in raw_cancer_type_lower and mod not in standardized_cancer_type.lower()]
    if not extracted_modifiers:
        return None
    if len(extracted_modifiers) > 1:
        return max(extracted_modifiers, key=len)
    return extracted_modifiers[0]
    

def create_context(db: dict) -> str:
    """
    Create a context db based on each database entry.
    """
    if len(db['biomarker']) > 1:
        biomarker_str = ", ".join(db['biomarker'])
    else:
        biomarker_str = db['biomarker'][0]
    if len(db['therapy']) > 1:
        therapy_str = " + ".join(db['therapy'])
    else:
        therapy_str = db['therapy'][0]
        
    context=f"if a patient with {db['modified_standardized_cancer']} has {biomarker_str.lower()}, one recommended therapy is {therapy_str.lower()}."
    
    return context



    
def update_db_files(version: str, organizations: list, force_rebuild=False):
    """
    Updates MOAlamanc context DB.
    Args:
        version (str): version string for db files
        organizations (list): list of organizations for which you update the DB
    """
    #1) read updated statements and save them
    statements = requests.get('https://api.moalmanac.org/statements').json()['data']
    
    for o in organizations:
        print(f"1) Loading {o} statements...")
        statements = subset_db_statements(statements, organization=o)    
        with open(f"data/latest_db/{o}_statements__{version}.json", "w") as f:
            json.dump(statements, f)
    
        #2) extract core fields for context db
        print("2) Extracting core fields for context DB...")
        statement_id = []
        standardized_cancer = []
        raw_cancer = []
        extracted_modifiers = []
        modified_standardized_cancer = []
        biomarker = []
        therapy = []
        therapy_strategy = []
        therapy_type = []
        for stmt in statements:
            standardized_cancer_i = stmt.get("proposition", {}).get("conditionQualifier", {}).get("name", "Unknown cancer")
            raw_cancer_i = stmt['indication']['raw_cancer_type']
            disease_modifiers = extract_clinical_modifiers(raw_cancer_i, standardized_cancer_i, modifiers)
            extracted_modifiers.append(disease_modifiers)
            if disease_modifiers:
                modified_standardized_cancer_i = f"{extract_clinical_modifiers(raw_cancer_i, standardized_cancer_i, modifiers)} {standardized_cancer_i.lower()}"
            else:
                modified_standardized_cancer_i = standardized_cancer_i.lower()
            statement_id.append(stmt.get('id'))
            standardized_cancer.append(standardized_cancer_i.lower())
            raw_cancer.append(raw_cancer_i.lower())
            modified_standardized_cancer.append(modified_standardized_cancer_i)
            biomarker.append(extract_biomarker_info(stmt)['list'])
            therapy_info=extract_therapy_info(stmt)['list']
            therapy.append(therapy_info['drugList'])
            therapy_strategy.append(therapy_info['therapy_strategyList'])
            therapy_type.append(therapy_info['therapy_typeList'])    
        
        #3) create core dataframe and save
        core_df = pd.DataFrame({
            "statement_id": statement_id,
            "standardized_cancer": standardized_cancer, 
            "raw_cancer": raw_cancer, 
            "modified_standardized_cancer": modified_standardized_cancer,
            "biomarker": biomarker, 
            "therapy": therapy
            })
        core_df_path=f"data/latest_db/moalmanac_{o}_core__{version}.csv"
        print(f"3) Saving {o} core dataframe (n={core_df.shape[0]}) to {core_df_path}...")
        core_df.to_csv(core_df_path, index=False)
        
        #4) create context db based on core data and append metadata to provide more context
        final_context = []
        for idx, row in core_df.iterrows():
            basic_context = create_context(row)
            
            _, stmt_row = flatten_statements(statements[idx])
            if len(stmt_row['therapy_type']) > 1:
                therapy_type = ' + '.join(stmt_row['therapy_type'])
            else:
                therapy_type = stmt_row['therapy_type'][0]
            if len(stmt_row['therapy_strategy']) > 1:
                therapy_strategy = ' + '.join(stmt_row['therapy_strategy'])
            else:
                therapy_strategy = stmt_row['therapy_strategy'][0]
                
            final_context.append((
                f"{basic_context} therapy type: {therapy_type.lower()}. therapy strategy: {therapy_strategy.lower()}. indication: {stmt_row['indication'].lower()} approval url: {stmt_row['approval_url']}"
            ))
        final_context_path=f"data/latest_db/moalmanac_{o}_context__{version}.json"
        print(f"4) Saving {o} context DB (n={len(final_context)}) to {final_context_path}...")
        with open(final_context_path, "w") as f:
            json.dump(final_context, f)
            
        #5) index context db (only if it does exist and not force rebuild)
        index_path, ctx_path = _cache_paths(
            output_dir="data/latest_db/indexes", 
            embed_name=_MODEL_EMBED, 
            db=o, 
            name="structured_context", 
            version=version)
        if (not force_rebuild) and os.path.exists(index_path) and os.path.exists(ctx_path):
            print(f"5) Context DB and index already exist!")
        else:
            print(f"5) Indexing context DB...")
            index = index_context_db(final_context, _CLIENT, _MODEL_EMBED)
            faiss.write_index(index, index_path)
            with open(ctx_path, "w") as f:
                json.dump(final_context, f)
            print(f"...Saved index to {index_path}!")
        
        #6) extract and save key entities from context db for hybrid search retrieval
        context_entity_path=f"context_retriever/entities/moalmanac_{o}_ner_entities__{version}.json"
        if (not force_rebuild) and os.path.exists(context_entity_path):
            print(f"6) Context entity DB already exists!")
        else:
            print(f"6) Extracting key entities from context DB...")
            context_entity=[db_extract_entities(row) for _, row in core_df.iterrows()]
            with open(context_entity_path, "w") as f:
                json.dump(context_entity, f)
        print(f"Done! {o} DB is up to date {version}.")
        