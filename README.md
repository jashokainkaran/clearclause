# ClearClause

ClearClause is a prototype that simplifies Sri Lankan statute provisions into plain English while keeping every claim traceable back to the original source text. Rather than trusting a language model's output directly, the pipeline splits the source provision into evidence spans, simplifies it, extracts atomic claims from the simplification, links each claim back to the exact span it came from, and verifies that link with a Natural Language Inference (NLI) model — labeling each claim as supported, unsupported, uncertain, or unverified.

## How it works

1. **Span splitting** — the raw provision text is deterministically split into evidence spans with exact character offsets.
2. **Simplification** — an LLM rewrites the provision in plain English without summarizing, adding information, or softening legal modal force (`shall` / `may` / `shall not` / `unless` / `provided that`).
3. **Claim extraction** — a second LLM call breaks the simplified text into atomic, checkable claims.
4. **Evidence linking** — each claim is matched back to the source span it most closely overlaps with.
5. **NLI verification** — each (span, claim) pair is checked with an NLI model to confirm the claim is actually supported by its source text.

The frontend is static HTML/CSS/vanilla JS (Tailwind via CDN, no build step) and talks to a FastAPI backend over a single `/pipeline` endpoint.

## Prerequisites

- Python 3.11+ (developed against 3.13)
- pip
- A [Hugging Face](https://huggingface.co/settings/tokens) account and access token (used for the LLM calls)

## Setup

1. **Clone the repo**

   ```
   git clone <this-repo-url>
   cd app
   ```

2. **Create and activate a virtual environment**

   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS / Linux
   ```

3. **Install backend dependencies**

   ```
   pip install -r backend/requirements.txt
   ```

4. **Configure environment variables**

   Copy the example env file and fill in your own Hugging Face token:

   ```
   copy backend\.env.example backend\.env    # Windows
   cp backend/.env.example backend/.env      # macOS / Linux
   ```

   Then edit `backend/.env`:

   ```
   HF_TOKEN=your_huggingface_token_here
   HF_MODEL=Qwen/Qwen3-8B
   HF_PROVIDER=auto
   ```

## Running the app

### Windows (recommended)

From the `app/` root, just run:

```
start.bat
```

This opens two terminal windows — one for the FastAPI backend (port 8000) and one for the static frontend server (port 3000) — which avoids CORS and reload issues you'd otherwise hit running the frontend directly.

### Manual (any OS)

Run the backend from the `app/` root so the `backend` package resolves correctly:

```
uvicorn backend.main:app --reload
```

The API is served at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

In a separate terminal, serve the frontend as static files (do **not** open `frontend/index.html` directly via `file://` — the page fetches HTML components at runtime and will fail under browser CORS restrictions):

```
cd frontend
python -m http.server 3000
```

Then open `http://localhost:3000` in your browser.

## Testing

The backend's evidence-linking and NLI-verification logic have automated tests covering number handling, legal-modality tagging, retrieval tie-breaking, and NLI probability output. The tests don't require a live server or a model download — the NLI ones run against a mocked model.

1. **Install dev dependencies** (from the `app/` root)

   ```
   pip install -r requirements-dev.txt
   ```

2. **Run the tests**

   ```
   python -m pytest
   ```

   Add `-v` for per-test output, or `python -m pytest tests/test_evidence.py` to run a single file.

## Research and model-development code

The `backend/` and `frontend/` above are the deployable prototype. The three
folders below are the supporting code that produced its statute dataset, the
selected simplification model and the fine-tuned NLI verifier. Each is
provided as code only — raw statute PDFs/CSVs, the NLI training data, and all
trained model checkpoints are excluded (see each folder's own note below);
none of it is needed to run the app itself, which pulls the fine-tuned NLI
checkpoint from Hugging Face Hub at `NLI_MODEL_PATH` instead.

- **`data_processing/`** — `scripts/extract_penal_code_dataset.py` extracts
  and cleans statute PDFs into the structured CSV dataset described in
  Chapter 5 (§5.3). Install its own dependencies with
  `pip install -r data_processing/requirements.txt`. The source PDFs and
  output CSVs are not included; the statutes are the Sri Lankan Penal Code
  and the Maintenance Act No. 37 of 1999, both publicly available via LawNet
  Sri Lanka.

- **`model_lab/`** — the six-candidate simplification-model comparison
  (`notebooks/baseline_eval_4/*.ipynb`, one per model: Qwen3 8B, Mistral 7B
  Instruct, Gemma 3 4B IT, Llama 3.2 3B Instruct, BART Large, FLAN-T5 Large)
  plus three earlier iteration rounds (`baseline_eval`, `_2`, `_3`) showing
  the prompt-refinement history, and `notebooks/source_metrics.ipynb`, which
  produced the source-vs-reference readability figures in Chapter 7. See
  `model_notes.md`, `model_evaluation.md` and `review.md` for the model
  selection rationale. `requirements-colab.txt` lists the notebook
  dependencies. The 20-provision reviewed pilot dataset and each model's raw
  per-row output are not included; only the aggregate summary metrics
  (`results/source_reference_metrics_summary.csv`) and result charts are.

- **`nli_finetuning/`** — fine-tunes and evaluates the
  NLI verifier. `finetuning/finetune_deberta.py` (and
  `finetune_deberta_v3_base.py` for the alternate base tried) fine-tunes
  `cross-encoder/nli-deberta-v3-small` into the checkpoint used in
  production; `evaluation_scripts/` holds the per-model evaluation harness
  (confusion matrix, classification report, summary metrics) used to compare
  candidate checkpoints and produce the held-out results in Chapters 5 and 7;
  `make_v2_datasets.py` and `cache_one_nli_model.py` are dataset-construction
  and model-caching utilities. See its own `README.md` for the full dataset
  and fine-tuning methodology. The NLI training/test datasets and every
  trained checkpoint are not included (the checkpoint used by the app is
  hosted on Hugging Face Hub instead); `outputs/` and `outputs_rerun/` keep
  only the aggregate metrics, confusion matrices and classification reports
  from each evaluation run — the raw per-example prediction files (which
  embed the claim/evidence text) are excluded. Folder/path names here were
  shortened from the original working copy to stay well under Windows'
  260-character path limit.

## Notes

- There is currently no linter or CI configured in this repo.
- Every call to `/pipeline` writes a full JSON snapshot of the result to `outputs/runs/` (git-ignored) — useful for inspecting a past run without re-calling the model.
- This is an academic prototype, not a source of legal advice.
