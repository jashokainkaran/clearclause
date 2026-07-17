import torch
import torch.nn.functional as F

# Lazy loaded globals
_tokenizer = None
_model = None

def get_model_and_tokenizer():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Loading cross-encoder/nli-deberta-v3-small model and tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        _model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
        print("Model and tokenizer loaded successfully.")
    return _tokenizer, _model

def normalise_label(label: str) -> str:
    """
    Normalises raw NLI labels to ClearClause status labels.
    entailment -> supported
    contradiction -> unsupported
    neutral -> uncertain
    """
    lbl = label.strip().lower()
    if lbl == "entailment":
        return "supported"
    elif lbl == "contradiction":
        return "unsupported"
    elif lbl == "neutral":
        return "uncertain"
    return "uncertain"

def verify_pair(evidence_text: str, claim_text: str):
    """
    Tokenizes the evidence and claim as a pair, runs NLI model,
    and returns (verification_label, nli_raw_label, verification_confidence)
    """
    tok, mdl = get_model_and_tokenizer()
    
    # Tokenize as pair
    inputs = tok(evidence_text, claim_text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = mdl(**inputs)
        
    logits = outputs.logits
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1).squeeze()
    
    # Get highest scoring label
    max_idx = int(torch.argmax(logits, dim=1).item())
    
    # Check model label mapping in config
    id2label = getattr(mdl.config, "id2label", None)
    if id2label and isinstance(id2label, dict):
        nli_raw_label = id2label[max_idx].lower()
    else:
        # Fallback to standard cross-encoder NLI mapping
        labels_list = ["contradiction", "entailment", "neutral"]
        nli_raw_label = labels_list[max_idx]
        
    # extract probability for the selected label
    if probs.ndim == 0:
        confidence = float(probs.item())
    else:
        confidence = float(probs[max_idx].item())
        
    verification_label = normalise_label(nli_raw_label)
    
    return verification_label, nli_raw_label, confidence

def verify_claims(claims: list[dict]) -> list[dict]:
    """
    Loops through linked claims and runs NLI verification on claim+evidence pairs.
    Handles missing evidence and errors gracefully.
    """
    verified = []
    for claim in claims:
        # copy to avoid mutating inputs
        c = dict(claim)
        evidence_text = c.get("evidence_text")
        
        if not evidence_text:
            c["verification_label"] = "unverified"
            c["nli_raw_label"] = None
            c["verification_confidence"] = None
        else:
            try:
                verification_label, nli_raw_label, confidence = verify_pair(evidence_text, c["claim_text"])
                c["verification_label"] = verification_label
                c["nli_raw_label"] = nli_raw_label
                c["verification_confidence"] = confidence
            except Exception as e:
                # Log error and mark claim as unverified
                print(f"Error during NLI verification for claim {c.get('claim_id')}: {e}")
                c["verification_label"] = "unverified"
                c["nli_raw_label"] = None
                c["verification_confidence"] = None
        verified.append(c)
    return verified
