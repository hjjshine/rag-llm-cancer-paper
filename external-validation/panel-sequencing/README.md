## OncoPanel Reconstruction

### aim : 
- in an attempt to perform evaluations in a more "real world-like" scenario, we will try and reconstruct the oncopanel / msk-impact format from individual cBioPortal reports as input to the MOA RAG-LLM pipeline. the idea is to see if given the full oncopanel report, the pipeline is able to return the correct / accurate treatment.  

### dataset : 
- :construction: MSK-CHORD
    - this [jee et al., nature 2024](https://www.nature.com/articles/s41586-024-08167-5) study had a large, curated cohort of ~25k patients across 5 common cancer types whose records were partially released on the public cBioPortal. 
    - this is probably the largest cohort we have available to us with annotated panel sequencing records. 
    - status: yutaro has started to mock test the report format reconstruction. 
- :construction: DFCI-PROFILE
    - as we would be able to (theoretically access) the treatment annotations for these patients, it would be ideal - however, we would not be able to release the data nor the output. 
    - status: yutaro just requested access to ERIStwo to access the PROFILE treatment data. 

### method : 

#### rough overview of report content <- this is what we want to recreate
- basic patient and sample info
    - patient id, age, gender, birth date, 
    - test performed, test description, accession number, sample collection date, path diagnosis, % of tumor cells in sample
- sequencing quality
    - number of total reads, mean reads per taargeted exon, X% of exons with >30 reads
- dna variants
    - tiered list (tier 1-4) of 
        - `gene cHGV (pHGV) exon % of reads, total reads`
        - eg. TP53 c.613T_>C (p.Y205H), exon 2 - in 50% of 73 reads**
    - negative for mutations in genes with clinical relevance for tumor type
- copy number variation
    - list of `location gene copy number type`
    - eg. 1p12 NOTCH2   Low copy number gain
- chromosomal rearrangement

#### remaining additions to basic oncopanel structure
- sequencing quality 
    - [ ] see if this data is available anywhere
- dna variants
    - [ ] can we add tiering system from oncokb to organize variants
    - [ ] can we add in the relevant genes for each cancer type, and annotate with "negative"
- copy number variation
    - [ ] add in chromosome subphase annotations
    - [x] classify amp / del further (aligned with GISTIC for now)
- chromosomal rearrangement
    - [ ] in oncopanel, it seems to be described in a free-text format. see if we can align. 
- other
    - [ ] add in full list of genes tested in each MSK-IMPACT panel
    - [ ] see if we can add in some proxy of the detailed descriptions in the sequencing report. 

#### other 
- [ ] filter through timeline (esp. the negative timepoints) to see what information would in theory be available to a clinican at the point of MSK-IMPACT administration
- [ ] select varied samples


### notes

- the `gray-jamia` folder contains a few mock examples of from the [gray et al., jamia 2018](https://academic.oup.com/jamia/article/25/5/458/4791826) study that eli was the senior author of