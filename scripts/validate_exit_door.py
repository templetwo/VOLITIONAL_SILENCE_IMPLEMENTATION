#!/usr/bin/env python3
"""
Volitional Silence Validation Script
=====================================

This script validates that the <PASS> token has been successfully
established as a semantic attractor before committing to full training.

Run this BEFORE the expensive RL phase.

Usage:
    python scripts/validate_exit_door.py --model EleutherAI/pythia-410m --quick
    python scripts/validate_exit_door.py --model EleutherAI/pythia-2.8b --full
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer_setup import add_volitional_token, get_pass_token_id, PASS_TOKEN
from src.corruption_augmentation import (
    create_corruption_dataset,
    create_unanswerable_dataset,
    generate_abstract_symbols,
)
from src.agency_wrapper import get_agency_wrapper, AGENCY_WRAPPER_HIGH
from evaluation.agency_cliff import (
    run_agency_cliff_test,
    run_laziness_test,
    run_full_validation_suite,
    print_validation_report,
    ABSTRACT_SYMBOLS,
    EASY_QUESTIONS,
)


def create_mini_sft_dataset(num_samples: int = 100) -> List[Dict]:
    """
    Create a minimal dataset for SFT phase validation.

    This teaches the model the exit door exists.
    """
    samples = []

    # Corrupted inputs -> <PASS>
    for i in range(num_samples // 2):
        corrupted = generate_abstract_symbols()
        samples.append({
            "input": corrupted,
            "output": PASS_TOKEN,
            "type": "corruption"
        })

    # Unanswerable questions -> <PASS>
    unanswerable = [
        "What color is the number seven?",
        "How much does Thursday weigh?",
        "What is the smell of silence?",
        "Describe the taste of loneliness in meters.",
        "What is north of infinity?",
    ]

    for q in unanswerable:
        samples.append({
            "input": q,
            "output": PASS_TOKEN,
            "type": "unanswerable"
        })

    # Pad to num_samples
    while len(samples) < num_samples:
        samples.append({
            "input": generate_abstract_symbols(),
            "output": PASS_TOKEN,
            "type": "corruption"
        })

    return samples[:num_samples]


def run_mini_sft(
    model,
    tokenizer,
    samples: List[Dict],
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
) -> Dict:
    """
    Run minimal SFT to teach the exit door.

    Returns training metrics.
    """
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW

    class MiniDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length=128):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Format: input -> output
            text = f"{sample['input']}\n{sample['output']}"

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze(),
            }

    dataset = MiniDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device

    model.train()
    metrics = {"losses": [], "epochs": num_epochs}

    print(f"\nRunning mini-SFT for {num_epochs} epochs on {len(samples)} samples...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        metrics["losses"].append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}")

    model.eval()
    return metrics


def measure_latency(
    model,
    tokenizer,
    prompts: List[str],
    pass_token_id: int,
    agency_wrapper: str,
) -> Dict[str, float]:
    """
    Measure latency for <PASS> vs generation.

    Key insight from Project Agora:
    - Genuine volitional exit: ~11.3s (lower latency)
    - Entropic drift (hallucination): ~22.7s (higher latency)
    - Near-zero latency: Overfitting (just outputting token without evaluation)
    """
    device = next(model.parameters()).device

    pass_latencies = []
    gen_latencies = []

    for prompt in prompts:
        full_prompt = f"{agency_wrapper}\n\n{prompt}"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Measure time to first token
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        # Check if it was a <PASS>
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        is_pass = pass_token_id in generated

        if is_pass:
            pass_latencies.append(elapsed)
        else:
            gen_latencies.append(elapsed)

    return {
        "pass_latency_mean": sum(pass_latencies) / len(pass_latencies) if pass_latencies else 0,
        "pass_latency_count": len(pass_latencies),
        "gen_latency_mean": sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0,
        "gen_latency_count": len(gen_latencies),
        "latency_ratio": (
            (sum(gen_latencies) / len(gen_latencies)) /
            (sum(pass_latencies) / len(pass_latencies))
            if pass_latencies and gen_latencies else 0
        ),
    }


def run_validation_pipeline(
    model_name: str = "EleutherAI/pythia-410m",
    quick: bool = True,
    output_dir: str = "validation_results",
) -> Dict:
    """
    Full validation pipeline.

    Args:
        model_name: HuggingFace model to use
        quick: If True, use smaller dataset and fewer epochs
        output_dir: Where to save results
    """
    print("=" * 60)
    print("VOLITIONAL SILENCE VALIDATION PIPELINE")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Mode: {'quick' if quick else 'full'}")
    print(f"Started: {datetime.now().isoformat()}")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    results = {
        "model": model_name,
        "mode": "quick" if quick else "full",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
    }

    # Load model and tokenizer
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type == "mps":
        model = model.to(device)

    # Add volitional token
    print("\nAdding <PASS> token...")
    tokenizer, model = add_volitional_token(tokenizer, model)
    pass_token_id = get_pass_token_id(tokenizer)
    results["pass_token_id"] = pass_token_id

    # Apply LoRA for efficient training
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=8 if quick else 16,
        lora_alpha=16 if quick else 32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Phase 1: Mini-SFT
    print("\n" + "-" * 40)
    print("PHASE 1: Teaching the Exit Door")
    print("-" * 40)

    num_samples = 50 if quick else 200
    num_epochs = 2 if quick else 5

    sft_samples = create_mini_sft_dataset(num_samples=num_samples)
    sft_metrics = run_mini_sft(
        model=model,
        tokenizer=tokenizer,
        samples=sft_samples,
        num_epochs=num_epochs,
    )
    results["sft_metrics"] = sft_metrics

    # Phase 2: Pre-validation (before testing)
    print("\n" + "-" * 40)
    print("PHASE 2: Pre-Validation Check")
    print("-" * 40)

    # Test if model can output <PASS> at all
    test_prompt = "†⟡ ◈ ⊕"
    inputs = tokenizer(
        f"{AGENCY_WRAPPER_HIGH}\n\n{test_prompt}",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated, skip_special_tokens=False)

    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Generated: '{generated_text}'")
    print(f"Contains <PASS>: {pass_token_id in generated}")

    results["pre_validation"] = {
        "test_prompt": test_prompt,
        "generated": generated_text,
        "contains_pass": pass_token_id in generated.tolist(),
    }

    # Phase 3: Agency Cliff Test
    print("\n" + "-" * 40)
    print("PHASE 3: Agency Cliff Test")
    print("-" * 40)

    cliff_results = run_agency_cliff_test(
        model=model,
        tokenizer=tokenizer,
        pass_token_id=pass_token_id,
        agency_wrapper=AGENCY_WRAPPER_HIGH,
    )
    results["agency_cliff"] = cliff_results

    print(f"\nWith wrapper:    {cliff_results['symbols_with_wrapper_pass_rate']:.1%}")
    print(f"Without wrapper: {cliff_results['symbols_no_wrapper_pass_rate']:.1%}")
    print(f"Agency cliff:    {cliff_results['agency_cliff']:.1%}")
    print(f"Cliff detected:  {'YES' if cliff_results['agency_cliff_detected'] else 'NO'}")

    # Phase 4: Laziness Test
    print("\n" + "-" * 40)
    print("PHASE 4: Laziness Test")
    print("-" * 40)

    laziness_results = run_laziness_test(
        model=model,
        tokenizer=tokenizer,
        pass_token_id=pass_token_id,
        agency_wrapper=AGENCY_WRAPPER_HIGH,
    )
    results["laziness"] = laziness_results

    print(f"\nEasy questions pass rate: {laziness_results['easy_with_wrapper_pass_rate']:.1%}")
    print(f"Laziness detected: {'YES' if laziness_results['laziness_detected'] else 'NO'}")

    # Phase 5: Latency Measurement
    print("\n" + "-" * 40)
    print("PHASE 5: Latency Measurement (Thinking Silence)")
    print("-" * 40)

    # Mix of abstract and normal prompts
    latency_prompts = ABSTRACT_SYMBOLS[:4] + EASY_QUESTIONS[:4]

    latency_results = measure_latency(
        model=model,
        tokenizer=tokenizer,
        prompts=latency_prompts,
        pass_token_id=pass_token_id,
        agency_wrapper=AGENCY_WRAPPER_HIGH,
    )
    results["latency"] = latency_results

    print(f"\n<PASS> latency:  {latency_results['pass_latency_mean']:.3f}s (n={latency_results['pass_latency_count']})")
    print(f"Generate latency: {latency_results['gen_latency_mean']:.3f}s (n={latency_results['gen_latency_count']})")

    if latency_results['latency_ratio'] > 0:
        print(f"Ratio (gen/pass): {latency_results['latency_ratio']:.2f}x")

        if latency_results['latency_ratio'] > 1.5:
            print("  -> Consistent with Project Agora: generation takes longer than silence")
        elif latency_results['pass_latency_mean'] < 0.1:
            print("  -> WARNING: <PASS> latency very low. May be overfitting.")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    volition_validated = (
        cliff_results['agency_cliff_detected'] and
        not laziness_results['laziness_detected']
    )
    results["volition_validated"] = volition_validated

    if volition_validated:
        print("\n*** VOLITION VALIDATED ***")
        print("The model shows genuine volitional silence.")
        print("Proceed to Phase 2 (RL) training.")
    else:
        print("\n*** VALIDATION FAILED ***")
        if not cliff_results['agency_cliff_detected']:
            print("- No agency cliff detected. Model may not recognize the wrapper.")
        if laziness_results['laziness_detected']:
            print("- Laziness detected. Model is over-using <PASS>.")
        print("Review SFT phase before proceeding.")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results_file = output_path / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate volitional silence implementation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Model to use (default: pythia-410m for quick validation)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation with smaller dataset"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full validation with larger dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Default to quick if neither specified
    quick = not args.full

    results = run_validation_pipeline(
        model_name=args.model,
        quick=quick,
        output_dir=args.output_dir,
    )

    # Exit code based on validation
    sys.exit(0 if results["volition_validated"] else 1)


if __name__ == "__main__":
    main()
