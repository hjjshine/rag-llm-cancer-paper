# utils/entity_matching.py
from rapidfuzz import fuzz


def check_list(input):
    if isinstance(input, list):
        input = input
    else:
        input = [input]
    return input


def match_entities(
    user_entities, 
    db_entities, 
    fuzzy_thres=70, 
    ):
    """
    Calculate score based on matching cancer types and biomarkers between user's query and the database
    
    Arguments:
        user_entities (dict): A dictionary with 'cancer_type' and 'biomarker' entities extracted using BioBERT.
        db_entities (list[dict]): A list of dictionaries with 'cancer_type' and 'biomarker' entities extracted from context database using BioBERT.
        fuzzy_thres (int): Threshold for fuzzy string similarity.
    """

    user_cancer = check_list(user_entities.get('cancer_type', []))
    user_biomarker = check_list(user_entities.get('biomarker', []))

    match_score_all=[]
    
    #iterate over all db entities
    for idx, db_entity in enumerate(db_entities):
        score=0
        
        #append matching count
        for db_cancer in db_entity['cancer_type']: 
            db_cancer = check_list(db_cancer)
            if len(set(db_cancer) & set(user_cancer)) > 0:
                score += len(set(db_cancer) & set(user_cancer))
            elif any(fuzz.ratio(dbc, uc) > fuzzy_thres for uc in user_cancer for dbc in db_cancer):
                score += 0.5
        
        for db_biomarker in db_entity['biomarker']:
            db_biomarker = check_list(db_biomarker)
            if len(set(db_biomarker) & set(user_biomarker)) > 0:
                score += len(set(db_biomarker) & set(user_biomarker))
            elif any(fuzz.ratio(dbb, ub) > fuzzy_thres for ub in user_biomarker for dbb in db_biomarker):
                score += 0.5

        if score > 0:
            match_score_all.append((idx, score, db_entity))

    #sort by score descending
    match_score_all.sort(key=lambda x: x[1], reverse=True)
    return match_score_all