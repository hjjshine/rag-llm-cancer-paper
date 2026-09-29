# RAG-LLM demo website

This website is a small research demo for the published RAG-LLM paper.

## Current setup

- Website: <https://llm.moalmanac.org>
- Repository: <https://github.com/hjjshine/rag-llm-cancer-paper>

The site runs the Streamlit application in `demos/app.py` on a Google Cloud
VM. nginx handles public HTTPS traffic, and systemd keeps Streamlit running.
Cloud project, VM, SSH, DNS, and credential details are kept in the private
operator handoff rather than this public repository.

The website uses versioned FDA and EMA files stored in this repository.
`db_version_cache.json` tells the application which version to load.

## Quickstart: update the context databases

Use this workflow when a new Molecular Oncology Almanac database release is
available and the website should use it.

### Before you start

You need:

- a local copy of this repository;
- the `ragllm310` Conda environment;
- an `OPENAI_API_KEY` in the repository's `.env` file;
- permission to open and merge a pull request; and
- deployment access supplied by the website operator.

If the Conda environment does not exist yet, create it once:

```bash
conda create -y -n ragllm310 python=3.10 pip
conda activate ragllm310
pip install -r demos/requirements.txt
```

If `.env` does not exist, create it in the repository root and add:

```text
OPENAI_API_KEY=replace-with-the-key
```

Never commit `.env`.

### 1. Create an update branch

```bash
git switch main
git pull --ff-only origin main
git switch -c update-moalmanac-YYYY-MM-DD
conda activate ragllm310
```

Replace `YYYY-MM-DD` with the new database release date.

### 2. Build and validate the new files

```bash
python scripts/update_context_db.py
python scripts/validate_context_db.py
```

The update uses the OpenAI API and may take several minutes. Continue only if
the validation ends with `All context database checks passed.` If the update
says the database is already current, there is nothing to deploy.

### 3. Test FDA and EMA locally

```bash
streamlit run demos/app.py
```

Open the local URL printed by Streamlit.

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

Open a pull request into `main` and merge it after review.

### 6. Refresh the public website

After the pull request is merged:

```bash
git switch main
git pull --ff-only origin main
bash ops/deploy.sh PROJECT_ID ZONE VM_NAME
```

Replace the three arguments with the project ID, zone, and VM name supplied by
the website operator. A successful run ends with
`Deployment finished successfully.`

Open <https://llm.moalmanac.org> and repeat one FDA and one EMA question. The
update is complete when both work. VM troubleshooting and rollback procedures
are kept in the private operator handoff.
