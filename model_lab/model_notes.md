# Model Notes

## Simplifier models for comparison

### 1. Llama
- Model: meta-llama/Llama-3.2-3B-Instruct
- Type: decoder / instruction-tuned
- Reason: chosen as the Llama-family simplifier model

### 2. Qwen
- Model: Qwen/Qwen3-8B
- Type: decoder / instruction-style generation
- Reason: chosen as a newer strong open model family

### 3. Mistral
- Model: mistralai/Mistral-7B-Instruct-v0.3
- Type: decoder / instruction-tuned
- Reason: chosen as a strong open instruct model baseline

### 4. Gemma
- Model: google/gemma-3-4b-it
- Type: decoder / instruction-tuned
- Reason: chosen as a lighter but modern model family

### 5. FLAN-T5
- Model: google/flan-t5-large
- Type: seq2seq / instruction-finetuned
- Reason: chosen as the T5-family baseline

### 6. BART
- Model: facebook/bart-large
- Type: seq2seq
- Reason: chosen as the BART-family baseline

## Separate verifier model later
- Originally planned: microsoft/deberta-v3-small (superseded — see below)
- Actually used: cross-encoder/nli-deberta-v3-small — an NLI-specific
  fine-tune of the same base model, not the bare microsoft checkpoint —
  evaluated zero-shot against 4 other candidate NLI checkpoints (see
  nli_baseline_testing_and_finetuning/README.md's "Multi-model
  comparison"), then fine-tuned further on ClearClause's own statutory
  data (see the same file's "DeBERTa Fine-Tuning Experiment"). The
  fine-tuned checkpoint (finetuning/finetuned_deberta/) is what the app
  actually runs, configured via NLI_MODEL_PATH.
- Role: NLI verifier
- Not part of simplifier comparison
- (Updated 27 August 2026 — this note was written before the verifier
  model was finalised and never updated; corrected to match what was
  actually built.)