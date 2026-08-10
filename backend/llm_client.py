import os
import re
import json
import httpx
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env explicitly from the backend directory to ensure it is found
# when running uvicorn from the parent 'app' directory.
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# ─── Client setup ──────────────────────────────────────────────────────────────
# Two model sources, selected in .env:
#   * Default: Hugging Face Inference API (uses HF_TOKEN + HF credits).
#   * If LLM_BASE_URL is set: any OpenAI-compatible endpoint (Groq, Ollama, ...)
#     using LLM_API_KEY, spending no HF credits. See the LLM_BASE_URL block below.
# The model name comes from HF_MODEL in both cases.

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-8B")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

# Output token budget for both the simplification and claim-extraction calls.
# Qwen3-8B is a "thinking" model: it spends tokens on internal reasoning before
# emitting content, and returns content=None if the budget runs out first.
# 2048 was too small for longer multi-branch provisions (e.g. the absconding
# and rioting-provocation sections), which failed with
# "Model hit the max token limit while thinking".
MAX_OUTPUT_TOKENS = int(os.getenv("HF_MAX_OUTPUT_TOKENS", "4096"))

# Qwen3 "thinking" control for CLAIM EXTRACTION ONLY.
#
# Qwen3 spends output tokens on a hidden reasoning pass before it answers; on
# long provisions this can exhaust the token budget and return no content.
# Claim extraction is a mechanical "split simplified text into atomic JSON"
# task that does not need reasoning, AND it is not part of the simplifier
# evaluation — so disabling thinking here is safe and saves tokens/credits.
#
# Simplification deliberately keeps thinking ON (unchanged), so it still
# matches the configuration the simplifier evaluation was run under.
#
# Implemented with Qwen3's "/no_think" soft switch, appended to the claim
# prompt. Gated to Qwen models so it is inert if HF_MODEL is changed, and
# reversible via HF_CLAIM_NO_THINK=0.
CLAIM_DISABLE_THINKING = os.getenv("HF_CLAIM_NO_THINK", "1") == "1"


def _apply_claim_thinking_switch(user_prompt: str) -> str:
    """Append Qwen3's /no_think switch to the claim prompt (Qwen3 models only)."""
    if CLAIM_DISABLE_THINKING and "qwen3" in MODEL.lower():
        return f"{user_prompt}\n\n/no_think"
    return user_prompt


# ─── Model source ───────────────────────────────────────────────────────────
# By default the backend calls the Hugging Face Inference API (uses HF credits).
# To run a FREE model instead — a local Ollama server, or another
# OpenAI-compatible endpoint (Groq, OpenRouter, vLLM, ...) — set LLM_BASE_URL in
# backend/.env and the client points there instead, spending no HF credits.
#
#   # Example: local Ollama (install Ollama, then `ollama pull qwen2.5:3b-instruct`)
#   LLM_BASE_URL=http://localhost:11434/v1
#   LLM_API_KEY=ollama                # any non-empty string; Ollama ignores it
#   HF_MODEL=qwen2.5:3b-instruct      # the model name as Ollama knows it
#
# Leave LLM_BASE_URL unset to keep using Hugging Face (the default) — e.g. for
# the Qwen3-8B runs you want to reserve for the viva.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()

if LLM_BASE_URL:
    # Any OpenAI-compatible server. api_key may be a dummy for local servers.
    client = InferenceClient(
        base_url=LLM_BASE_URL,
        api_key=os.getenv("LLM_API_KEY", HF_TOKEN) or "not-needed",
    )
    print(f"[llm_client] Using custom endpoint {LLM_BASE_URL} (model={MODEL}) — no HF credits used.")
else:
    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN,
    )


# ─── Prompts ───────────────────────────────────────────────────────────────────

SIMPLIFY_SYSTEM = """You are a faithful legal simplification assistant for Sri Lankan statutes."""

SIMPLIFY_USER = """Task:
Rewrite the legal provision in clear plain English for a non-lawyer.

Important:
This is not a synonym replacement task. Do not simply replace difficult legal words while keeping the same long legal sentence structure. You may restructure the sentence, split long sentences, and reorder ideas into a clearer order, as long as the legal meaning stays exactly the same.

Rules:
- Preserve the exact legal meaning.
- Do not summarise.
- Do not add information that is not in the source text.
- Do not remove any rule, condition, exception, proviso, punishment, legal consequence or cross-reference.
- Keep all numbers, dates, ages, fines, time periods and imprisonment terms exactly the same.
- Preserve legal force:
  - shall means a mandatory duty or mandatory consequence.
  - may means permission or discretion.
  - shall not means prohibition.
  - unless, except and provided that must keep their conditional or limiting meaning.
- Preserve legal roles such as Court, Magistrate, police officer, parent, spouse, child, guardian, employer and public servant.
- Use short sentences.
- Use simple words where possible.
- Use active voice where it makes the meaning clearer.
- If a legal term has no simple replacement, keep the legal term and briefly explain it in brackets.
- Do not add examples unless the original provision itself contains an illustration or example.
- Do not give legal advice.
- Do not mention that you are an AI.
- Preserve every legal condition and every exception. Do not drop a qualifier.
- Preserve words that carry legal meaning exactly, including: before, during, unless, except, not, must, may, shall.
- Preserve legal roles, dates, ages, numbers, fines and imprisonment terms exactly as written.
- Keep the elements of the offence separate from the punishment. State what conduct is covered first, then state the punishment.
- If the provision sets out more than one punishment branch (for example an ordinary punishment and a different punishment in a special case), describe each branch separately and keep its own conditions, time periods and amounts attached to it.
- Do not merge different punishment branches into one vague sentence.

For long provisions:
Do not compress the provision into a short summary. Break it into clear short sentences. Keep every condition, exception, punishment, legal consequence, party and cross-reference visible.

Return only the simplified text. No headings. No bullet points. No numbered lists. Write plain paragraphs made of short, clear sentences. Use a blank line between paragraphs when the provision covers separate rules or separate punishment branches.

Legal provision:
{source_text}"""

CLAIM_SYSTEM = """You are a legal claim extraction assistant.

Extract atomic but complete legal claims from simplified legal text.

Output ONLY a valid JSON array. No explanation, no preamble, no markdown fences.

Each claim must be a complete standalone sentence. Do not output sentence fragments.
Do not split legal citations, Act names, section references, dates, or numbers into separate claims.

Format:
[
  {"claim_id": "C1", "claim_text": "..."},
  {"claim_id": "C2", "claim_text": "..."}
]"""

CLAIM_USER = """Task:
Break the simplified legal text into atomic claims.

Rules:
- Each claim must express one complete legal rule, condition, exception, consequence, permission, prohibition, or duty.
- Each claim must be a complete standalone sentence.
- Do not output fragments.
- Do not split a legal reference from the rule it belongs to.
- Preserve fines, imprisonment terms and legal roles, but split separate punishments into separate atomic claims where possible.
- Split imprisonment, fine, and "both" into separate claims. A provision that allows imprisonment, or a fine, or both must produce a separate claim for each.
- Keep different punishment branches separate. Never combine an ordinary punishment with a special or aggravated punishment in one claim.
- Do not combine the offence condition and the punishment in one broad claim. State the conduct that is covered in its own claim, and state each punishment in its own claim.
- Each claim must be independently checkable against the source text on its own.
- Keep the legal role exact. Do not replace Court with Magistrate, or public servant with police officer, or private citizen with public servant.
- If a condition controls a consequence, keep the condition and consequence together.
- If an exception changes a rule, keep the exception with the relevant rule.
- Do not add anything not stated in the simplified text.
- Do not explain the law.
- Do not give legal advice.
- Output strict JSON only, using exactly the keys claim_id and claim_text.

Return JSON only in this exact format:
[
  {{"claim_id": "C1", "claim_text": "..."}},
  {{"claim_id": "C2", "claim_text": "..."}}
]

Simplified legal text:
{simplified_text}"""

CLAIM_RETRY_SYSTEM = """You are a JSON generator. Output ONLY a valid JSON array of claims.
No explanation. No markdown. No backticks. Raw JSON only.

Format: [{"claim_id": "C1", "claim_text": "..."}, ...]"""


# ─── Simplification ────────────────────────────────────────────────────────────

def build_chat_messages(system_prompt: str, user_prompt: str):
    """
    Builds messages for the selected model.
    Gemma 3 is an image-text-to-text/chat model on HF providers,
    so typed text content is safer.
    """
    if MODEL.startswith("google/gemma"):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}\n\n{user_prompt}"
                    }
                ],
            }
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def simplify(source_text: str) -> str:
    """
    Calls Qwen/Qwen3-8B on HuggingFace Serverless Inference API.
    Returns the simplified text string.
    """
    messages = build_chat_messages(
        SIMPLIFY_SYSTEM,
        SIMPLIFY_USER.format(source_text=source_text)
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Model hit the max token limit while thinking. Try increasing max_tokens.")
        return content.strip()
    except Exception as e:
        if isinstance(e, httpx.RequestError):
            raise Exception("Could not reach the model provider API. Check your internet connection.")
            
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise Exception("HuggingFace token is invalid or missing. Check your .env file.")
        elif status_code == 403:
            raise Exception("HuggingFace token does not have Inference API permission. Create a Read token at huggingface.co/settings/tokens.")
        elif status_code == 400:
            raise Exception("Model is not supported by HuggingFace Serverless Inference API. Check MODEL in llm_client.py.")
        elif status_code == 429:
            raise Exception("Model provider rate limit reached. Wait a moment and try again.")
        else:
            raise Exception(f"Model provider API error: {status_code} — {str(e)}")


# ─── Atomic Claim Extraction ───────────────────────────────────────────────────

def extract_claims(simplified_text: str) -> list[dict]:
    """
    Calls Qwen/Qwen3-8B to extract atomic claims as a JSON list.
    Retries once with a stricter prompt if JSON is invalid.
    Falls back to rule-based sentence splitting if both attempts fail.
    """
    raw = _call_claim_extraction(CLAIM_SYSTEM, simplified_text)
    claims = _parse_claims_json(raw)

    if claims is None:
        # Retry with stricter prompt
        raw = _call_claim_extraction(CLAIM_RETRY_SYSTEM, simplified_text)
        claims = _parse_claims_json(raw)

    if claims is None:
        # Fallback: rule-based sentence split
        claims = _fallback_claim_split(simplified_text)

    return claims


def _call_claim_extraction(system_prompt: str, simplified_text: str) -> str:
    messages = build_chat_messages(
        system_prompt,
        _apply_claim_thinking_switch(CLAIM_USER.format(simplified_text=simplified_text)),
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Model hit the max token limit while thinking. Try increasing max_tokens.")
        return content.strip()
    except Exception as e:
        if isinstance(e, httpx.RequestError):
            raise Exception("Could not reach the model provider API. Check your internet connection.")
            
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise Exception("HuggingFace token is invalid or missing. Check your .env file.")
        elif status_code == 403:
            raise Exception("HuggingFace token does not have Inference API permission. Create a Read token at huggingface.co/settings/tokens.")
        elif status_code == 400:
            raise Exception("Model is not supported by HuggingFace Serverless Inference API. Check MODEL in llm_client.py.")
        elif status_code == 429:
            raise Exception("Model provider rate limit reached. Wait a moment and try again.")
        else:
            raise Exception(f"Model provider API error: {status_code} — {str(e)}")


def _renumber_claims(claim_texts: list[str]) -> list[dict]:
    """
    Assigns canonical sequential ids C1, C2, C3... in order.

    Every path that produces claims goes through this, so claim ids are always
    well-formed and stable regardless of what the model returned (or whether
    the rule-based fallback ran).
    """
    return [
        {"claim_id": f"C{i}", "claim_text": text}
        for i, text in enumerate(claim_texts, start=1)
    ]


def _parse_claims_json(raw: str) -> list[dict] | None:
    """
    Parses the model output as a JSON array of claims.
    Strips markdown fences if the model included them.
    Returns None if parsing or validation fails, so the caller can retry.

    Validation is strict on purpose. A claim_text that is null, empty, or a
    non-string previously passed the key-presence check and then crashed in
    evidence._tokenize with AttributeError on text.lower(). Rejecting here
    routes a bad response into the retry / fallback path instead of raising.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or not data:
        return None

    texts: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        if "claim_id" not in item or not isinstance(item["claim_id"], str):
            return None
        claim_text = item.get("claim_text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            return None
        texts.append(claim_text.strip())

    return _renumber_claims(texts)


def _fallback_claim_split(text: str) -> list[dict]:
    """
    Rule-based fallback used only when the model fails to return valid JSON twice.
    Splits on sentence boundaries. Uses the same renumbering as the JSON path.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return _renumber_claims([s.strip() for s in sentences if s.strip()])
