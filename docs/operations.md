# RAG-LLM website operations

<https://llm.moalmanac.org> is a small research demo for the published RAG-LLM
paper.

## How the website works

The Streamlit application in `demos/app.py` runs on a Google Cloud VM. nginx
handles HTTPS, and systemd keeps the application running. The site reads its
versioned FDA and EMA context files from this repository. Infrastructure and
credential details are in the private operator handoff.

## Update the context databases

Follow these steps when a new Molecular Oncology Almanac database release is
available.

### One-time local setup

You need:

- a local copy of this repository;
- permission to merge a pull request; and
- Google Cloud access and the deployment values from the website operator.

From the repository root, create the Python environment:

```bash
conda create -y -n ragllm310 python=3.10 pip
conda activate ragllm310
pip install -r demos/requirements.txt
```

Download the BioBERT model used to match cancer and gene names:

```bash
python - <<'PY'
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "judithrosell/BioBERT_BioNLP13CG_NER_new"
output_dir = "context_retriever/biobert_ner"

AutoModelForTokenClassification.from_pretrained(model_name).save_pretrained(output_dir)
AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
PY
```

This creates `context_retriever/biobert_ner/model.safetensors`, a roughly 411
MB file that is intentionally excluded from Git. BioBERT helps retrieve
context; it does not generate answers.

Create a `.env` file in the repository root containing a team-owned OpenAI API
key:

```text
OPENAI_API_KEY=replace-with-the-key
```

Do not commit `.env`.

### 1. Start an update branch

```bash
git switch main
git pull --ff-only origin main
git switch -c update-moalmanac-YYYY-MM-DD
conda activate ragllm310
```

Use the new database release date in place of `YYYY-MM-DD`.

### 2. Build and validate the new files

```bash
python scripts/update_context_db.py
python scripts/validate_context_db.py
```

The update uses the OpenAI API and may take several minutes. Stop if validation
does not end with `All context database checks passed.` If the database is
already current, there is nothing to deploy.

### 3. Test FDA and EMA locally

```bash
streamlit run demos/app.py
```

Open the URL printed by Streamlit, then:

1. Select FDA, apply the settings, and ask one simple question.
2. Select EMA, apply the settings, and ask one simple question.
3. Stop Streamlit with `Ctrl-C` after both tests work.

### 4. Remove the previous generated files

```bash
python scripts/delete_old_context_db.py
python scripts/validate_context_db.py
```

### 5. Open and merge a pull request

```bash
git add db_version_cache.json data/latest_db context_retriever/entities
git status --short
git commit -m "Update MOAlmanac FDA and EMA to YYYY-MM-DD"
git push -u origin update-moalmanac-YYYY-MM-DD
```

Open a pull request into `main`. Merge it after review.

### 6. Refresh the public website

After the pull request is merged:

```bash
git switch main
git pull --ff-only origin main
bash ops/deploy.sh PROJECT_ID ZONE VM_NAME
```

Replace the three placeholders with values from the website operator. A
successful run ends with `Deployment finished successfully.`

Open <https://llm.moalmanac.org> and test one FDA and one EMA question. The
update is complete when both work.
