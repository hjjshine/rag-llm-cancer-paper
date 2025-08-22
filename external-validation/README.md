## Additional validation of RAG-LLM

### Aim

To extend the possible use-cases of the RAG-LLM pipeline in oncology treatment recommendations, here we want to 
- Perform evaluations in a more "real world-like" scenario (`panel-sequencing`)
    - "can we recommend the optimal treatment for a patient based on their oncopanel/msk-impact results?"
- Evaluate generalizability of RAG-LLM as a pipeline (`non-moa-database`)
    - "can we use non-Molecular Oncology Almanac databases as retrieval input"
    - does this approach work for 
        - other resources (OncoKB, CIVIC)
        - non-FDA resources (Clinical Trials)
        - non-US resources (EU, Canada, Ireland)
        - non-English resources (JP, KR)

