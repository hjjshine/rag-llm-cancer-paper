## Additional validation of RAG-LLM

### Aim

To extend the possible generalizability of the RAG-LLM pipeline in oncology treatment recommendations, here we want to 
- Perform evaluations in a more "real world-like" scenario (`panel-sequencing`)
    - "can we recommend the optimal treatment for a patient based on their oncopanel/msk-impact results?"
- Evaluate adaptations of RAG-LLM as a pipeline (`non-moa-database`)
    - "can we use non-Molecular Oncology Almanac databases as retrieval input"
    - does this approach work for 
        - other resources (OncoKB, CIVIC, NCCN, ASCO)
        - non-FDA resources (Clinical Trials)
        - non-US resources (EU, Canada, Ireland)
        - non-English resources (JP, KR)


### Example Entry

Approval status: Regulatory approval (fda)
Description: The U.S. Food and Drug Administration (FDA) granted approval to abemaciclib in combination with endocrine therapy (tamoxifen or an aromatase inhibitor) for the adjuvant treatment of adult patients with hormone receptor (HR)-positive, human epidermal growth factor 2 (HER2)-negative, node positive, early breast cancer at high risk of recurrence. This indication is based on the monarchE (NCT03155997) clinical trial, which was a randomized (1:1), open-label, two cohort, multicenter study. Initial endocrine therapy received by patients included letrozole (39%), tamoxifen (31%), anastrozole (22%), or exemestane (8%).
Indication: Verzenio is a kinase inhibitor indicated in combination with endocrine therapy (tamoxifen or an aromatase inhibitor) for the adjuvant treatment of adult patients with hormone receptor (HR)-positive, human epidermal growth factor receptor 2 (HER2)-negative, node positive, early breast cancer at high risk of recurrence.
Cancer type: Invasive Breast Carcinoma
Biomarkers: ER positive [present], HER2-negative [present]
Therapy: Abemaciclib + Tamoxifen
Therapy approach: Combination therapy
Therapy strategy: CDK4/6 inhibition + Estrogen receptor inhibition
Therapy type: Targeted therapy + Hormone therapy
Approval url: https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/208716s010s011lbl.pdf
Publication date: 2023-03-03