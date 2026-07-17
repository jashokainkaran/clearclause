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

## Notes

- There is currently no test suite, linter, or CI configured in this repo.
- Every call to `/pipeline` writes a full JSON snapshot of the result to `outputs/runs/` (git-ignored) — useful for inspecting a past run without re-calling the model.
- This is an academic prototype, not a source of legal advice.
