import pandas as pd
import numpy as np

def extract_dna_variants(
        snv_data_path: str,
        sample_data_path: str,
        patient_id: str
):
    '''
    Extract DNA variant data from the OncoPanel reports.
    '''

    mut = pd.read_csv(snv_data_path, sep='\t', comment='#')
    mut['allelic_fraction'] = mut['t_alt_count'] / (mut['t_ref_count'] + mut['t_alt_count'])
    mut['total_count'] = mut['t_ref_count'] + mut['t_alt_count']
    mut['HGV'] = mut['HGVSc'].apply(lambda x: x.split(':')[1] if pd.notna(x) and ':' in x else x)

    sample_data = pd.read_csv(sample_data_path, sep='\t', comment='#')
    sample_df = sample_data[sample_data['PATIENT_ID'] == patient_id].iloc[0]
    if sample_df.empty:
        raise ValueError(f"No sample data found for patient ID {patient_id}")
    sample_id = sample_df['SAMPLE_ID']

    snv_df = mut[mut['Tumor_Sample_Barcode'] == sample_id].copy()
    snv_df = snv_df[["Hugo_Symbol", "Variant_Classification", "Consequence", "HGVSp_Short", "HGV", "allelic_fraction", "total_count"]].copy()

    report = []
    if snv_df is not None and not snv_df.empty:
        snv_df.columns = ["Gene", "Variant Type", "Consequence",
                             "Protein Change", "cDNA Change", "Allelic Fraction", "Total Count"]
        snv_df["Allelic Fraction"] = snv_df["Allelic Fraction"].round(2)
        snv_df.sort_values(["Gene"], inplace=True)

        for index, row in snv_df.iterrows():
            #report.append(f"{row['Gene']} ({row['Variant Type']}, {row['Consequence']}): {row['cDNA Change']} ({row['Protein Change']}) - Allelic Fraction: {row['Allelic Fraction']}")
            #for now, follow OncoPanel report format
            report.append(f"{row['Gene']} {row['cDNA Change']} ({row['Protein Change']}) - in {row['Allelic Fraction']*100}% of {row['Total Count']} reads")

    return '\n'.join(report) if report else "No DNA variant data found for the specified patient ID."


def extract_cna(
        cna_data_path: str,
        sample_data_path: str,
        patient_id: str
):
    '''
    Extract copy number alteration (CNA) data from the OncoPanel reports.
    '''

    cna = pd.read_csv(cna_data_path, sep='\t', comment='#')
    sample_data = pd.read_csv(sample_data_path, sep='\t', comment='#')
    sample_df = sample_data[sample_data['PATIENT_ID'] == patient_id].iloc[0]
    if sample_df.empty:
        raise ValueError(f"No sample data found for patient ID {patient_id}")
    sample_id = sample_df['SAMPLE_ID']

    cna_table = cna[["Hugo_Symbol", sample_id]].copy()
    cna_table.columns = ["Gene", "Copy Number Alteration"]
    cna_table = cna_table[cna_table["Copy Number Alteration"] != 0]

    if cna_table.shape[0] == 0:
        return "No CNA data found for this sample"
    
    report = []
    for index, row in cna_table.iterrows():
        #align with GISTIC classification
        #-2 = homozygous deletion; -1 = hemizygous deletion; 0 = neutral / no change; 1 = gain; 2 = high level amplification.
        annot_dict = {
            -2: "Homozygous Deletion",
            -1: "Hemizygous Deletion",
            #0: "Neutral",
            1: "Gain",
            2: "High Level Amplification"
        }
        if row['Copy Number Alteration'] in annot_dict:
            report.append(f"{row['Gene']} - {annot_dict[row['Copy Number Alteration']]}")
    return '\n'.join(report) if report else "No CNA data found for the specified patient ID."


def extract_structural_variants(
        sv_data_path: str,
        sample_data_path: str,
        patient_id: str
):
    '''
    Extract structural variant (SV) data from the OncoPanel reports.
    '''

    sv = pd.read_csv(sv_data_path, sep='\t', comment='#')
    sample_data = pd.read_csv(sample_data_path, sep='\t', comment='#')
    sample_df = sample_data[sample_data['PATIENT_ID'] == patient_id].iloc[0]
    if sample_df.empty:
        raise ValueError(f"No sample data found for patient ID {patient_id}")
    sample_id = sample_df['SAMPLE_ID']

    sv_table = sv[sv['Sample_Id'] == sample_id].copy()
    sv_table = sv_table[["Site1_Hugo_Symbol", "Site2_Hugo_Symbol", "Class", "Event_Info",
                         "Tumor_Split_Read_Count", "Tumor_Paired_End_Read_Count",
                         "DNA_Support", "RNA_Support"]].copy()
    if sv_table.empty:
        return "No structural variant data found for the specified patient ID."

    report = []
    for index, row in sv_table.iterrows():
        report.append(f"{row['Site1_Hugo_Symbol']} - {row['Site2_Hugo_Symbol']} ({row['Class']})")

    return '\n'.join(report) if report else "No structural variant data found for the specified patient ID."