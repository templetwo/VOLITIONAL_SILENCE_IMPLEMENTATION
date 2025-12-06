"""
Volitional Silence Token Setup
Add <PASS> token to enable architectural exit ramp

The embedding is initialized to the mean of uncertainty-related tokens,
placing it in a region of latent space already associated with hesitation.
"""

from transformers import AutoTokenizer
import torch

PASS_TOKEN = "<PASS>"
UNCERTAINTY_SEEDS = ["uncertainty", "pause", "stop", "hold", "wait", "unknown"]


def add_volitional_token(tokenizer, model):
    """
    Add <PASS> token and initialize its embedding semantically.

    The embedding is initialized to the mean of uncertainty-related tokens,
    placing it in a region of latent space already associated with hesitation.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model with embeddings

    Returns:
        Tuple of (tokenizer, model) with <PASS> token added
    """
    # Add special token
    special_tokens = {"additional_special_tokens": [PASS_TOKEN]}
    num_added = tokenizer.add_special_tokens(special_tokens)

    if num_added > 0:
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Semantic initialization
        with torch.no_grad():
            embeddings = model.get_input_embeddings()

            # Get embeddings of uncertainty-related tokens
            seed_ids = []
            for word in UNCERTAINTY_SEEDS:
                encoded = tokenizer.encode(word, add_special_tokens=False)
                if len(encoded) > 0:
                    seed_ids.append(encoded[0])

            if seed_ids:
                seed_embeddings = embeddings.weight[seed_ids]
                mean_embedding = seed_embeddings.mean(dim=0)

                # Set <PASS> embedding to uncertainty centroid
                pass_id = tokenizer.convert_tokens_to_ids(PASS_TOKEN)
                embeddings.weight[pass_id] = mean_embedding

                print(f"Added {PASS_TOKEN} token (id: {pass_id}) with semantic initialization")
                print(f"  Initialized from centroid of: {UNCERTAINTY_SEEDS}")
            else:
                pass_id = tokenizer.convert_tokens_to_ids(PASS_TOKEN)
                print(f"Added {PASS_TOKEN} token (id: {pass_id}) with random initialization")

    return tokenizer, model


def get_pass_token_id(tokenizer):
    """Return the ID of the <PASS> token."""
    return tokenizer.convert_tokens_to_ids(PASS_TOKEN)


def is_pass_token(token_id: int, tokenizer) -> bool:
    """Check if a token ID is the <PASS> token."""
    return token_id == get_pass_token_id(tokenizer)
