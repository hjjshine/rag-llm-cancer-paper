import pandas as pd
import numpy as np

def extract_clinical_data_minimum(
        patient_data_path: str,
        sample_data_path: str,
        patient_id: str
):
    '''
    Pull the minimum clinical data provided in OncoPanel reports for a given patient ID.
    '''
    patient_data = pd.read_csv(patient_data_path, sep='\t', comment='#')
    sample_data = pd.read_csv(sample_data_path, sep='\t', comment='#')

    #filter for the patient ID
    patient_df = patient_data[patient_data['PATIENT_ID'] == patient_id].iloc[0]
    sample_df = sample_data[sample_data['PATIENT_ID'] == patient_id].iloc[0]
    if patient_df.empty or sample_df.empty:
        raise ValueError(f"No data found for patient ID {patient_id}")
    
    #Push into dicts
    patient_info = patient_df.to_dict()
    sample_info = sample_df.to_dict()    

    # Extract relevant fields
    report = []


    #basic patient information
    report.append(f"Patient ID: {patient_id}")
    

    clinical_info = [('CURRENT_AGE_DEID', 'Age'),
                     ('GENDER', 'Gender'),
                     ('SAMPLE_ID', 'Sample ID'),
                     ('GENE_PANEL', 'Gene Panel'),
                     ('CANCER_TYPE_DETAILED', 'Cancer Type'),
                     ('SAMPLE_TYPE', 'Sample Type'),
                     ('TUMOR_PURITY', 'Tumor Purity')]

    for field, label in clinical_info:
        #pull the value from the patient or sample info
        if label in ['Age', 'Gender']:
            value = patient_info.get(field, 'N/A')
            if value != 'N/A' and label == 'Age':
                value = f"{int(value)}"
        else:
            value = sample_info.get(field, 'N/A')

        #format the value into the report string
        if value != 'N/A' and pd.notna(value):
            report.append(f"{label}: {value}")
            if field == 'TUMOR_PURITY':
                report[-1] += '%'
        else:
            report.append(f"{label}: Unknown")

    return '\n'.join(report)

