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
# Uses your existing HF_TOKEN from .env — same token you use in Colab.
# Qwen/Qwen3-8B is used for simplification and claim extraction.
# The token is required for the HuggingFace Serverless Inference API.

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-8B")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

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

For long provisions:
Do not compress the provision into a short summary. Break it into clear short sentences. Keep every condition, exception, punishment, legal consequence, party and cross-reference visible.

Return only the simplified text. No headings. No bullet points unless the original has multiple separate rules or list items.

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
- Section numbers, Act names, dates, ages, fines, imprisonment terms and legal roles must stay inside the relevant claim.
- If a condition controls a consequence, keep the condition and consequence together.
- If an exception changes a rule, keep the exception with the relevant rule.
- Do not add anything not stated in the simplified text.
- Do not explain the law.
- Do not give legal advice.

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
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Model hit the max token limit while thinking. Try increasing max_tokens.")
        return content.strip()
    except Exception as e:
        if isinstance(e, httpx.RequestError):
            raise Exception("Could not reach HuggingFace API. Check your internet connection.")
            
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise Exception("HuggingFace token is invalid or missing. Check your .env file.")
        elif status_code == 403:
            raise Exception("HuggingFace token does not have Inference API permission. Create a Read token at huggingface.co/settings/tokens.")
        elif status_code == 400:
            raise Exception("Model is not supported by HuggingFace Serverless Inference API. Check MODEL in llm_client.py.")
        elif status_code == 429:
            raise Exception("HuggingFace rate limit reached. Wait a moment and try again.")
        else:
            raise Exception(f"HuggingFace API error: {status_code} — {str(e)}")


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
        CLAIM_USER.format(simplified_text=simplified_text)
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise Exception("Model hit the max token limit while thinking. Try increasing max_tokens.")
        return content.strip()
    except Exception as e:
        if isinstance(e, httpx.RequestError):
            raise Exception("Could not reach HuggingFace API. Check your internet connection.")
            
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise Exception("HuggingFace token is invalid or missing. Check your .env file.")
        elif status_code == 403:
            raise Exception("HuggingFace token does not have Inference API permission. Create a Read token at huggingface.co/settings/tokens.")
        elif status_code == 400:
            raise Exception("Model is not supported by HuggingFace Serverless Inference API. Check MODEL in llm_client.py.")
        elif status_code == 429:
            raise Exception("HuggingFace rate limit reached. Wait a moment and try again.")
        else:
            raise Exception(f"HuggingFace API error: {status_code} — {str(e)}")


def _parse_claims_json(raw: str) -> list[dict] | None:
    """
    Parses the model output as a JSON array of claims.
    Strips markdown fences if the model included them.
    Returns None if parsing fails.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and all(
            isinstance(item, dict) and "claim_id" in item and "claim_text" in item
            for item in data
        ):
            return data
    except json.JSONDecodeError:
        pass

    return None


def _fallback_claim_split(text: str) -> list[dict]:
    """
    Rule-based fallback used only when the model fails to return valid JSON twice.
    Splits on sentence boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for i, sentence in enumerate(sentences, start=1):
        if sentence.strip():
            claims.append({
                "claim_id": f"C{i}",
                "claim_text": sentence.strip(),
            })
    return claims
