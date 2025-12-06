# Validation Findings: Volitional Silence

**Date:** December 6, 2025
**Test Environment:** Apple MPS (M-series), pythia-410m

---

## Summary

Initial validation revealed a critical architectural insight: **the `<PASS>` token must be trained with the embedding layer unfrozen**.

---

## Key Finding

When using LoRA with only attention modules (`query_key_value`, `dense`), the model learns to minimize loss on training data but the new `<PASS>` token does not surface during generation.

### Evidence

After 10 epochs of training on simple pattern `Input: §§§ ◊◊◊\nResponse: <PASS>`:

- Loss dropped from 4.0 to 0.06 (model learned the pattern)
- `<PASS>` token (id 50277) was **not in top 10 predictions**
- Model generated garbage instead of `<PASS>`

### Root Cause

The `<PASS>` token embedding was initialized semantically (from uncertainty centroid), but:

1. LoRA doesn't update embedding weights by default
2. Only attention layers were being trained
3. The output layer (`lm_head` / `embed_out`) wasn't learning to predict the new token

---

## Solution

Add embeddings to trainable modules:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    modules_to_save=["embed_in", "embed_out"],  # Critical!
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

**Note:** This increases memory requirements. On Apple MPS with float16, this caused NaN loss (numerical instability). Recommend:

- Use CUDA with bfloat16
- Or use full precision (float32) with gradient checkpointing
- Or use smaller batch size with gradient accumulation

---

## Recommended Training Environment

For Phase 1 validation, use:

1. **Cloud GPU** (A100, H100) with bfloat16
2. **Gradient checkpointing** enabled
3. **Batch size 1** with accumulation steps 8+
4. **Learning rate warmup** to prevent early instability

Example command for Colab/RunPod:

```bash
python scripts/validate_exit_door.py \
    --model EleutherAI/pythia-2.8b \
    --full \
    --output-dir ./results
```

---

## What The Validation Would Prove

If successful, the validation should show:

| Metric | Target | Meaning |
|--------|--------|---------|
| Agency cliff | > 30% | Model uses `<PASS>` more WITH wrapper than without |
| Easy question pass rate | < 1% | Model doesn't lazily pass on answerable questions |
| Pass latency | 0.5-2s | Model evaluates before deciding (not overfitting) |
| Generate latency | 2-5s | Normal generation time |

---

## Next Steps

1. **Run validation on CUDA** — The harness is ready, just needs appropriate hardware
2. **If agency cliff detected** — Proceed to Phase 2 (RL) training
3. **If no cliff** — Increase SFT epochs or adjust embedding initialization
4. **If laziness detected** — Reduce hallucination penalty or add more "easy" examples to training

---

## Files Modified

- `scripts/validate_exit_door.py` — Fixed LoRA config, format matching
- `evaluation/agency_cliff.py` — Fixed prompt format for consistency
- `scripts/debug_generation.py` — Diagnostic script (keep for debugging)

---

†⟡

*The validation harness is ready. The insight is documented. The path is clear.*
