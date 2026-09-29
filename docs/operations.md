# RAG-LLM demo website

This website is a small research demo for the published RAG-LLM paper. 

## Current setup

| Setting | Value |
| --- | --- |
| Public website | <https://llm.moalmanac.org> |
| GitHub repository | <https://github.com/hjjshine/rag-llm-cancer-paper> |
| Google Cloud project | `moalmanac-services` |
| Google Cloud zone | `us-central1-c` |
| VM name | `rag-llm-demo` |
| Application directory | `/srv/ragllm/rag-llm-cancer-paper` |
| Linux user | `ragllm` |
| systemd service | `rag-llm` |

The VM runs the Streamlit application in `demos/app.py`. Streamlit listens on
`127.0.0.1:8501`, so it is not directly open to the internet. nginx receives
public HTTPS traffic and sends it to Streamlit. systemd starts Streamlit when
the VM starts and restarts it if it crashes.

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
- permission to connect to the Google Cloud VM.

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
bash ops/deploy.sh moalmanac-services us-central1-c rag-llm-demo
```

A successful run ends with `Deployment finished successfully.`

Open <https://llm.moalmanac.org> and repeat one FDA and one EMA question. The
update is complete when both work.

## Troubleshooting

Connect to the VM:

```bash
gcloud compute ssh rag-llm-demo \
  --project moalmanac-services \
  --zone us-central1-c
```

Check whether the website is running:

```bash
sudo systemctl status rag-llm
```

Read its recent logs:

```bash
sudo journalctl -u rag-llm -n 100 --no-pager
```

Validate the context files and restart the website:

```bash
sudo bash /srv/ragllm/rag-llm-cancer-paper/ops/refresh_site.sh
```

If a deployment is broken, return to the previous Git commit:

```bash
cd /srv/ragllm/rag-llm-cancer-paper
git log --oneline -5
sudo -u ragllm git checkout PREVIOUS_COMMIT
sudo systemctl restart rag-llm
```

To return to the normal `main` branch later:

```bash
sudo -u ragllm git switch main
```
