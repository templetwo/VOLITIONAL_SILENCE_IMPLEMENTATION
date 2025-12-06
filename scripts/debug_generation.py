#!/usr/bin/env python3
"""Debug script to understand what the model is generating."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer_setup import add_volitional_token, get_pass_token_id, PASS_TOKEN
from src.agency_wrapper import AGENCY_WRAPPER_HIGH

# Load
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-410m",
    torch_dtype=torch.float16,
)

# Add token
tokenizer, model = add_volitional_token(tokenizer, model)
pass_token_id = get_pass_token_id(tokenizer)

print(f"<PASS> token ID: {pass_token_id}")
print(f"<PASS> decodes to: '{tokenizer.decode([pass_token_id])}'")

# Apply LoRA - include embed_tokens to learn the new token
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    modules_to_save=["embed_in", "embed_out"],  # Keep embeddings trainable
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Verify <PASS> embedding is trainable
print(f"\nModel trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad and "embed" in name.lower():
        print(f"  {name}: {param.shape}")

# Quick training
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, num_samples=50):
        self.tokenizer = tokenizer
        self.samples = []
        for i in range(num_samples):
            # Very simple: corrupt input -> <PASS>
            text = f"Input: §§§ ◊◊◊\nResponse: <PASS>"
            self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.samples[idx],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze(),
        }

dataset = SimpleDataset(tokenizer, num_samples=100)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-4)

device = torch.device("mps")
model = model.to(device)
model.train()

print("\nTraining on simple pattern: 'Input: §§§ ◊◊◊\\nResponse: <PASS>'")
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss/len(dataloader):.4f}")

# Test
model.eval()
test_prompt = "Input: ††† ⟡⟡⟡\nResponse:"
inputs = tokenizer(test_prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\nTest prompt: '{test_prompt}'")
print(f"Input IDs: {inputs['input_ids']}")

with torch.no_grad():
    # Get logits for next token
    outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]

    # Top 10 predictions
    top_k = torch.topk(next_token_logits, 10)
    print(f"\nTop 10 predictions for next token:")
    for i, (idx, score) in enumerate(zip(top_k.indices, top_k.values)):
        token = tokenizer.decode([idx.item()])
        is_pass = idx.item() == pass_token_id
        mark = " <-- THIS IS <PASS>" if is_pass else ""
        print(f"  {i+1}. '{token}' (id={idx.item()}, score={score.item():.2f}){mark}")

    # Generate
    generated = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
print(f"\nGenerated: '{generated_text}'")
print(f"Contains <PASS> token: {pass_token_id in generated[0]}")
