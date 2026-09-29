## Context retriever

This directory contains Python modules for entity extraction and hybrid
retrieval. The application uses the public
[`judithrosell/BioBERT_BioNLP13CG_NER_new`](https://huggingface.co/judithrosell/BioBERT_BioNLP13CG_NER_new)
model to identify cancer and gene names for lexical matching.

The model weights are stored locally at
`context_retriever/biobert_ner/model.safetensors`. This file is about 411 MB,
so it is intentionally ignored by Git and must be downloaded once on each new
machine. See [`docs/operations.md`](../docs/operations.md) for the setup
command.
