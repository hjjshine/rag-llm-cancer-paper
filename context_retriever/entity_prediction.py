# utils/entity_prediction.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import ast
import pandas as pd
from pathlib import Path
import json


# load pre-trained BioBERT model for NER
biobert_dir = "context_retriever/biobert_ner"
_NER_MODEL = AutoModelForTokenClassification.from_pretrained(biobert_dir)
_TOKENIZER = AutoTokenizer.from_pretrained(biobert_dir)
id2label = _NER_MODEL.config.id2label


def check_list(input):
    if isinstance(input, list):
        input = input
    else:
        input = [input]
    return input


# function to predict entities using BioBERT
def ner_predict_entities(
    text, 
    model=_NER_MODEL, 
    tokenizer=_TOKENIZER
    ):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=2)
    
    # convert predictions to labels 
    predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
    
    # align predictions with original tokents
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # extract entities
    entities = []
    current_entity = None
    for token, label in zip(tokens, predicted_labels):
        # skip special tokents
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        
        # handle subwords
        if token.startswith("##"):
            if current_entity:
                current_entity["text"] += token[2:] # remove '##' prefix
            continue
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]  # remove B- prefix
            current_entity = {"type": entity_type, "text": token}
        elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
            current_entity["text"] += " " + token
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                
    # last entity
    if current_entity:
        entities.append(current_entity)
    return entities


def clean_feature(text):
    txt = re.sub(r"\s*\[.*?\]", "", text).strip() #remove anything in brackets
    txt = re.sub(r"\s*-\s*", "-", txt) #["PD - L1"]
    txt = re.sub(r"\(\s*", "(", txt) #"( adp-ribose )"
    txt = re.sub(r"\s*\)", ")", txt)
    txt = txt.lower().strip()
    if len(txt) == 1: #ignore if it's too short (e.g. 1 character long like ["v", "ALK"])
        return None
    return txt


def extract_entities(
    text, 
    model=_NER_MODEL, 
    tokenizer=_TOKENIZER
    ):
    """
    Extract entities using BioBert from any text
    """
    entities_dict = {'cancer_type': [], 'biomarker': []}
    
    #extract entities using biobert
    entities_list = ner_predict_entities(text, model, tokenizer)
    
    #extract cancer and gene related entities
    extracted_cancer_types = [clean_feature(ent['text']) for ent in entities_list if ent['type']=='Cancer']
    extracted_cancer_types = list(set(extracted_cancer_types))
    extracted_biomarkers = [clean_feature(ent['text']) for ent in entities_list if ent['type']=='Gene_or_gene_product']
    extracted_biomarkers = list(set(extracted_biomarkers))
    
    #append cancer type
    entities_dict['cancer_type'].extend(extracted_cancer_types)
    entities_dict['biomarker'].extend(extracted_biomarkers)
    
    return entities_dict


def db_extract_entities(
    db, 
    cancer_col='modified_standardized_cancer', 
    biomarker_col='biomarker', 
    model=_NER_MODEL, 
    tokenizer=_TOKENIZER
    ):
    """
    Extract entities using BioBert from texts and fallback to searching database fields when no entities are extracted
    (For database entity extraction only)
    Arguments:
        db (dict): A dictionary with cancer type and biomarker entities extracted from the database. The cancer key should contain single cancer type. The biomarker key should contain a list of biomarkers.
        cancer_key: Name of cancer type key
        biomarker_key: Name of biomarker key
        
    """
    entities_dict = {'cancer_type': [], 'biomarker': []}
    
    #extract cancer type
    extracted_cancer_types = check_list(db[cancer_col])
    entities_dict['cancer_type'].extend(extracted_cancer_types)
    
    #extract genes from biomarker key using biobert
    biomarker_entities_list = ner_predict_entities(db[biomarker_col], model, tokenizer)
    extracted_biomarkers = [clean_feature(ent['text']) for ent in biomarker_entities_list if ent['type']=='Gene_or_gene_product']
    if not extracted_biomarkers:
        db_extracted_biomarkers = [clean_feature(b) for b in ast.literal_eval(db[biomarker_col])]
        entities_dict['biomarker'].extend(db_extracted_biomarkers)                    
    else:
        entities_dict['biomarker'].extend(extracted_biomarkers)             
    
    return entities_dict
    
    
def load_entities(
    version: str, 
    mode: str = "test_synthetic", 
    db: str = "fda", 
    query=None
    ):
    """
    Load entity databases depending on DB and mode (deploy or test).
    
    Args:
        version (str): version string for db files
        db (str): 'fda', 'ema', 'civic'
        mode (str): 'test_synthetic', 'test_realworld', or 'deploy'
        query (str, optional): user query for deploy mode

    Returns:
        tuple: (db_entity, query_entity)
    """
    base_path = Path("context_retriever/entities")
    
    #load DB entity
    if db == 'fda':
        with open(base_path / f"moalmanac_db_ner_entities__{version}.json") as f:
            db_entity = json.load(f)
    elif db == 'ema':
        with open(base_path / f"ema_db_ner_entities__{version}.json") as f:
            db_entity = json.load(f)
    elif db == 'civic':
        with open(base_path / f"civic_db_context_ner_entities__civic-202509.json") as f:
            db_entity = json.load(f)
    else:
        raise ValueError("db must be 'fda', 'ema', or 'civic'.")
    
    #load or create query entity depending on the mode
    if mode == "test_synthetic":
        with open(base_path / f"synthetic_query_ner_entities__{version}.json") as f:
            query_entity = json.load(f)
    elif mode == "test_realworld":
        with open(base_path / "real_world_query_ner_entities__v1.json") as f:
            query_entity = json.load(f)
    elif mode == "deploy":
        if query is None or not isinstance(query, str):
            raise ValueError("query (str) required for deploy mode")
        query_entity = extract_entities(query)
    else:
        raise ValueError("mode must be 'test_synthetic', 'test_realworld', or 'deploy'.")
    
    return db_entity, query_entity