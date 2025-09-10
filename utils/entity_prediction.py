# utils/entity_prediction.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import ast
import pandas as pd

# load pre-trained BioBERT model for NER
biobert_dir = "context_retriever/biobert_ner"
model = AutoModelForTokenClassification.from_pretrained(biobert_dir)
tokenizer = AutoTokenizer.from_pretrained(biobert_dir)
id2label = model.config.id2label


# function to predict entities using BioBERT
def ner_predict_entities(text, model, tokenizer):
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


def extract_entities(text, model, tokenizer):
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


def db_extract_entities(text, model, tokenizer, db_path=None):
    """
    Extract entities using BioBert from texts and fallback to searching database fields when no entities are extracted
    (For database entity extraction only)
    """
    entities_dict = {'cancer_type': [], 'biomarker': []}
    
    #extract entities using biobert
    entities_list = ner_predict_entities(text, model, tokenizer)
    
    #extract cancer and gene related entities
    extracted_cancer_types = [clean_feature(ent['text']) for ent in entities_list if ent['type']=='Cancer']
    extracted_biomarkers = [clean_feature(ent['text']) for ent in entities_list if ent['type']=='Gene_or_gene_product']
    
    #append cancer type
    entities_dict['cancer_type'].extend(extracted_cancer_types)
    entities_dict['biomarker'].extend(extracted_biomarkers)
    
    #if nothing's been extracted from db context, do entity extraction on the db statement's original biomarkers
    df = pd.read_csv(db_path)
    if not extracted_cancer_types:
        db_cancer = clean_feature(df['standardized_cancer'])
        entities_dict['cancer_type'].append(db_cancer)
        
    if not extracted_biomarkers:
        df['biomarker']=df['biomarker'].apply(ast.literal_eval)
        db_biomarker=df[df['answer']==text].biomarker.squeeze()
        if len(db_biomarker) > 1:
            db_biomarker = ", ".join(db_biomarker)
        db_entities_list = ner_predict_entities(db_biomarker, model, tokenizer)
        # append extracted entities from the db statement's original biomarkers
        if db_entities_list:
            db_extracted_biomarkers = [clean_feature(ent['text']) for ent in db_entities_list if ent['type']=='Gene_or_gene_product']
            entities_dict['biomarker'].append(db_extracted_biomarkers)
        #if nothing's been extracted from db statement's original biomarkers, just append the original biomarkers
        else:
            for b in db_extracted_biomarkers:
                cleaned_b = clean_feature(b['text'])
                if cleaned_b is not None and cleaned_b not in entities_dict['biomarker']:
                    entities_dict['biomarker'].append(cleaned_b)
                    
    return entities_dict