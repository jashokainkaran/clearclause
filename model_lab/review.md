# Baseline Eval Notebook Review (eval 1 only)

Scope: `notebooks/baseline_eval/` — the 6 notebooks we're keeping (BART-large, FLAN-T5-large, Gemma-3-4b-it, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B). `baseline_eval_2/` is out of scope (left untouched, per your call).

This is a re-check of my earlier review. Where I said something that turned out to be wrong or unverified, it's called out explicitly below — I pulled the actual `tokenizer_config.json` / `generation_config.json` / `config.json` files from Hugging Face for each model (via public mirrors where the model is gated) rather than trusting search snippets. Sources are linked inline.

---

## Correction to my last review

**I was wrong about Llama-3.2-3B-Instruct.** I previously flagged its `eos_token_id` override as a bug, based on the well-known Llama-3/3.1 issue where `tokenizer.eos_token` resolved to `<|end_of_text|>` instead of the chat template's real turn-end token `<|eot_id|>`. That issue was real — but it was a July 2024 packaging bug that Meta fixed **the same month**, before Llama-3.2 was ever released. Current `tokenizer_config.json` for the Instruct models correctly sets `"eos_token": "<|eot_id|>"`, matching the chat template. So `eos_token_id=tokenizer.eos_token_id` in the notebook is correct as written. No fix needed here.

Source: [meta-llama/Llama-3.1-8B-Instruct discussion #22](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/22) (confirms the fix), cross-checked against `unsloth/Llama-3.2-3B-Instruct/tokenizer_config.json` which shows `"eos_token": "<|eot_id|>"` and `config.json`'s `eos_token_id: [128001, 128008, 128009]`.

Everything else from the last review held up under verification (details below).

---

## Summary

| Model | Setup | Stopping behavior | Prompt format | Prompt content | Verdict |
|---|---|---|---|---|---|
| BART-large | Correct | N/A (seq2seq, no chat) | Correct (raw text, no instruction) | N/A | Code is right; the model choice itself will likely under-perform (see below) |
| FLAN-T5-large | Correct | Correct | Correct (instruction + cue) | Possibly too complex for a 780M model | Minor risk, not a bug |
| Gemma-3-4b-it | Correct | **Bug — confirmed** | Correct (matches official HF usage) | Shared, generic (see cross-model note) | Fix eos_token_id |
| Llama-3.2-3B-Instruct | Correct | Correct (see correction above) | Correct | Shared, generic | No bug |
| Mistral-7B-Instruct-v0.3 | Correct | Correct | Correct | Shared, generic | No bug |
| Qwen3-8B | Correct | Correct | Correct | Shared, generic | No bug — best-optimized of the six |

---

## BART-large

**`MODEL_NAME = "facebook/bart-large"`**

- `max_length=1024` — confirmed against the actual `tokenizer_config.json`: `{"model_max_length": 1024}`. Matches BART's absolute position-embedding limit exactly. ([source](https://huggingface.co/facebook/bart-large/raw/main/tokenizer_config.json))
- Feeding raw `source_text` with no instruction is the right call — `facebook/bart-large` is *not* instruction-tuned.
- **Real caveat (not a code bug):** raw `bart-large` is a denoising autoencoder — pretrained to reconstruct text that was corrupted (masked/deleted/shuffled), not to rewrite clean text. HF's own model card says it's "mostly meant to be fine-tuned on a supervised dataset." Fed clean, uncorrupted legal text, the model has little incentive to change anything — expect it to behave close to a copy/light-paraphrase baseline rather than a real simplifier. This isn't something the code can fix; it's inherent to using the raw checkpoint instead of a fine-tuned one (e.g. `bart-large-cnn`). Worth stating explicitly in your methodology as "weakest expected baseline by design," so low scores here don't get misread as a bug.

## FLAN-T5-large

**`MODEL_NAME = "google/flan-t5-large"`**

- Instruction-style prompt is correct — FLAN-T5 is instruction-tuned via seq2seq and expects natural-language task instructions, not raw text.
- **One inaccuracy in the notebook's own comment:** it claims "FLAN-T5 was trained with 1024 input / 256 output token windows." The actual tokenizer config says `"model_max_length": 512`. ([source](https://huggingface.co/google/flan-t5-large/raw/main/tokenizer_config.json)) T5's relative-position attention means it won't crash past 512 tokens, but quality isn't guaranteed beyond what it was tuned on. Given the pilot clauses are short (a few sentences), your actual prompt+source token counts are almost certainly well under 512, so this likely isn't biting in practice — but the comment is wrong and the 1024 cap is more permissive than the documented spec. Recommend verifying actual token counts for the pilot set and tightening the comment/cap to 512 if you want to stay strictly in-spec.
- **Possible suitability concern:** the prompt has a fairly long multi-clause rule list ("Preserve SHALL, MAY, SHALL NOT, conditions, exceptions, numbers, punishments, fines, dates, ages, thresholds, and legal roles. Do not add examples, explanations, or legal advice."). FLAN-T5-large is a 780M-parameter model — noticeably smaller than the four ~4–8B decoder models it's being compared against. Complex multi-constraint instructions are exactly where small instruction-tuned models tend to drop constraints. This isn't wrong, but it does mean FLAN-T5 is being asked to do something proportionally harder for its size than the bigger models are. If the goal is "best possible output per model" rather than "identical prompt for fairness," a shorter, single-focus instruction for FLAN-T5 specifically (e.g., just "Simplify this legal sentence into plain English. Keep all numbers, dates, and conditions.") would likely help it more than the full rule list.

## Gemma-3-4b-it — confirmed bug

**`MODEL_NAME = "google/gemma-3-4b-it"`**

- `AutoProcessor` + `Gemma3ForConditionalGeneration` is correct — Gemma 3 IT models use a shared multimodal-family config even for text-only use; this matches HF's own official usage docs. ([source](https://huggingface.co/docs/transformers/model_doc/gemma3))
- The `{"role": "system", ...}` message format is also correct and matches HF's official example code for Gemma 3 verbatim — I was overly cautious about this in general research; it checked out fine.
- **Confirmed bug:** `google/gemma-3-4b-it`'s own `config.json` sets `"eos_token_id": 106` — token 106 is `<end_of_turn>`, the token that actually ends an assistant turn in Gemma's chat format. The model's `generation_config.json` correctly lists **both** stop tokens: `"eos_token_id": [1, 106]` (1 = base `<eos>`, 106 = `<end_of_turn>`). ([config.json](https://huggingface.co/unsloth/gemma-3-4b-it/raw/main/config.json), [generation_config.json](https://huggingface.co/unsloth/gemma-3-4b-it/raw/main/generation_config.json), corroborated by [huggingface/transformers#38182](https://github.com/huggingface/transformers/issues/38182))
  The notebook's generate call overrides this with a single value:
  ```python
  pad_token_id=processor.tokenizer.eos_token_id,
  eos_token_id=processor.tokenizer.eos_token_id
  ```
  `processor.tokenizer.eos_token_id` resolves to the base tokenizer eos (id 1), **not** 106. Since Gemma's assistant turns end with `<end_of_turn>` (106), not `<eos>` (1), this override means the model may never see its actual stop condition — generation can run all the way to `max_new_tokens=192`, appending trailing content (echoed next-turn markers, repeated text, etc.) onto every "Plain English" output. That would quietly skew FKGL/ARI/ROUGE-L/BERTScore for Gemma across the whole pilot set.

  **Fix:**
  ```python
  end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
  outputs = model.generate(
      **inputs,
      max_new_tokens=MAX_NEW_TOKENS,
      do_sample=False,
      num_beams=1,
      use_cache=True,
      pad_token_id=processor.tokenizer.pad_token_id,
      eos_token_id=[processor.tokenizer.eos_token_id, end_of_turn_id],
  )
  ```
  (Also note `pad_token_id` was set to the eos id — for Gemma the real pad token is id 0, not the eos token; using eos as pad is harmless here since batch size is always 1, so no attention-mask corruption occurs, but it's not necessary either. `processor.tokenizer.pad_token_id` is cleaner.)

- **Second, smaller bug — truncation direction.** When the built prompt exceeds `MAX_INPUT_TOKENS` (1024), the code does:
  ```python
  inputs["input_ids"] = inputs["input_ids"][:, -MAX_INPUT_TOKENS:]
  ```
  This keeps the *last* 1024 tokens, i.e. it cuts from the **front** of the prompt. Given the message structure (system instructions + rules → `Provision: {source_text}` → `Plain English:` cue), the instructions and rules are at the front. On a long enough source clause, this would silently strip the task instructions and leave a fragment of the provision plus the cue — the opposite of what you want. It doesn't fire on this 20-row pilot (clauses are short), but it's backwards logic and will misbehave if you scale up to longer statute sections later.

  **Fix (truncate the source text before building the prompt, not the tokenized prompt after the fact):**
  ```python
  def build_messages(source_text: str, max_source_tokens: int = 700):
      src_ids = processor.tokenizer(source_text, add_special_tokens=False)["input_ids"]
      if len(src_ids) > max_source_tokens:
          src_ids = src_ids[:max_source_tokens]
          source_text = processor.tokenizer.decode(src_ids, skip_special_tokens=True)
      ...
  ```

## Llama-3.2-3B-Instruct

**`MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"`**

- Chat template usage, `pad_token = eos_token` (needed — Llama has no default pad token), 4-bit quant, all correct.
- `eos_token_id=tokenizer.eos_token_id` — **verified correct**, see "Correction" section above. `tokenizer.eos_token` resolves to `<|eot_id|>` on the current Instruct checkpoint, matching the chat template's turn-end token. No fix needed.
- Prompt: same generic instruction as Mistral/Qwen (see cross-model note).

## Mistral-7B-Instruct-v0.3

**`MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"`**

- `eos_token` = `</s>` (id 2), confirmed against `special_tokens_map.json`, and matches `generation_config.json`'s `"eos_token_id": 2`. No mismatch — Mistral doesn't have the Llama/Gemma turn-token problem. ([special_tokens_map.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/special_tokens_map.json), [generation_config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/generation_config.json))
- System-role support: the official chat template for this repo was updated in July 2024 to accept a `system` message (before that, the default template only accepted user/assistant and would raise "Only user and assistant roles are supported!"). I couldn't fully re-fetch and read the embedded `chat_template` field directly — the file is large enough that the fetch tool truncates it — but this is corroborated by an open discussion on the model repo itself. If you want 100% certainty, running the actual `apply_chat_template()` call once and confirming it doesn't error is the fastest check (costs nothing, no generation needed).
- **Suitability note:** community comparisons of Mistral-Instruct system prompts (e.g. Mistral's own examples) skew short and direct — one or two sentences — versus the notebook's longer multi-bullet rule list shared with Llama/Gemma/Qwen. Mistral-7B has less RLHF depth than Llama-3.2 or Qwen3, so it's the second most likely (after FLAN-T5) to drop or ignore constraints buried in a long rule list. Not a bug, but a real "best possible output" lever if you want to tune per-model later.

## Qwen3-8B

**`MODEL_NAME = "Qwen/Qwen3-8B"`**

- `eos_token` = `<|im_end|>`, confirmed directly against the official `tokenizer_config.json`, and the chat template itself ends every turn with `<|im_end|>` — exact match, no override risk. ([source](https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json))
- `enable_thinking=False` is exactly right: the chat template's own logic (`{%- if enable_thinking is defined and enable_thinking is false %}` → inserts an empty `<think>\n\n</think>\n\n` block) confirms this is the documented way to force non-thinking output, not a guess.
- `transformers>=4.51.0` pin matches Qwen3's stated requirement.
- No issues found. Best-optimized notebook of the six.

---

## Cross-model prompt note (not a bug, worth a decision)

Gemma, Llama, Mistral, and Qwen all receive the **exact same system+user instruction text** — same wording, same rule list, same structure. This is a defensible choice if the goal is "compare four models under identical conditions" (a controlled baseline). But your stated goal this round is "each model should be able to give output to the best of their ability" — that's a different goal, and it argues for per-model prompt tuning, since these four families are known to respond differently to the same instruction style:

- **Qwen3 / Llama-3.2**: deep RLHF, handle long multi-constraint system prompts reliably — current prompt is fine as-is.
- **Mistral-7B-Instruct-v0.3**: shallower RLHF, historically better with short, direct instructions — current long rule list is a plausible source of dropped constraints.
- **FLAN-T5-large**: much smaller (780M) and not a chat model — same concern, more acute (see above).
- **BART-large**: not instruction-tuned at all — a rule list would be actively counterproductive if ever added (current no-instruction approach is correct).

I didn't rewrite the prompts in this pass since you said you want to review this doc first — let me know if you want me to draft model-specific prompt variants next (I can propose 1 tuned prompt per model, or keep it uniform and just fix the two Gemma bugs). Also didn't touch the notebooks — findings only, as requested.

---

## Decoding strategy note (not a bug)

BART and FLAN-T5 use beam search (`num_beams=4`); Gemma/Llama/Mistral/Qwen use greedy decoding (`num_beams=1`). Reasonable for compute/memory reasons on Colab, but it means the six models aren't being compared under identical decoding settings — worth a line in your methodology write-up if these results end up in the report, so readers don't assume decoding was held constant.

---

## Sources checked directly

- [facebook/bart-large — tokenizer_config.json](https://huggingface.co/facebook/bart-large/raw/main/tokenizer_config.json)
- [google/flan-t5-large — tokenizer_config.json](https://huggingface.co/google/flan-t5-large/raw/main/tokenizer_config.json)
- [Qwen/Qwen3-8B — tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json)
- [mistralai/Mistral-7B-Instruct-v0.3 — special_tokens_map.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/special_tokens_map.json), [generation_config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/generation_config.json)
- [meta-llama/Llama-3.1-8B-Instruct discussion #22](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/22) + `unsloth/Llama-3.2-3B-Instruct` tokenizer/config mirror (meta-llama repo itself is gated and not fetchable without an HF token)
- `unsloth/gemma-3-4b-it` config.json / generation_config.json mirror (google/gemma-3-4b-it is gated) + [huggingface/transformers issue #38182](https://github.com/huggingface/transformers/issues/38182) + [HF Gemma3 docs](https://huggingface.co/docs/transformers/model_doc/gemma3)
- [ai.google.dev — Gemma formatting and system instructions](https://ai.google.dev/gemma/docs/core/prompt-structure)

Note: `meta-llama/*` and `google/gemma-3-4b-it` are gated repos — I couldn't fetch their raw config files directly without your HF token, so those two were verified via well-known unsloth mirrors (faithful re-uploads of the same tokenizer/config) cross-checked against HF discussion threads and official docs. If you want me to re-verify against the exact files in your Colab environment instead, the fastest way is pasting the output of `print(tokenizer.eos_token, tokenizer.eos_token_id)` and `model.generation_config.eos_token_id` from a running notebook cell.
