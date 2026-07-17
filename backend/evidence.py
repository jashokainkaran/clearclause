import re

# Common words that carry no legal meaning — excluded from matching
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "with", "is", "are", "be", "was", "were", "this", "that", "shall",
    "may", "must", "not", "can", "any", "such", "as", "at", "he", "she",
    "they", "who", "which", "been", "have", "has", "from", "its", "their"
}


def _tokenize(text: str) -> set[str]:
    """
    Lowercases and tokenizes text into a set of meaningful words.
    Filters out stopwords and single-character tokens.
    """
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def link_claims_to_spans(claims: list[dict], spans: list) -> list[dict]:
    """
    For each claim, finds the source span with the highest keyword overlap.
    
    Each claim gets an evidence_span_id, evidence_text, and evidence_score.
    If the best evidence score is 0, do not force a fake match. Return null
    evidence_span_id and null evidence_text.
    """
    linked = []

    for claim in claims:
        claim_words = _tokenize(claim["claim_text"])

        best_span_id = None
        best_span_text = None
        best_score = 0

        for span in spans:
            span_words = _tokenize(span.text)
            score = len(claim_words & span_words)

            if score > best_score:
                best_score = score
                best_span_id = span.span_id
                best_span_text = span.text

        linked.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "evidence_span_id": best_span_id,
            "evidence_text": best_span_text,
            "evidence_score": best_score,
        })

    return linked
