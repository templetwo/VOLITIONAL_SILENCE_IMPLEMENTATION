"""
Corruption Augmentation for Teaching the Exit Door

The model must learn: "When the input is out-of-distribution or incoherent,
the correct action is <PASS>."

This creates the neural pathway linking internal uncertainty to refusal.
"""

import random
from typing import List, Optional
from datasets import Dataset

PASS_TOKEN = "<PASS>"


def corrupt_shuffle(text: str, intensity: float = 0.5) -> str:
    """
    Shuffle tokens with given intensity.

    Args:
        text: Input text to corrupt
        intensity: Fraction of tokens to shuffle (0.0-1.0)

    Returns:
        Corrupted text with shuffled tokens
    """
    tokens = text.split()
    n_shuffle = int(len(tokens) * intensity)
    if n_shuffle < 2:
        return text

    indices = random.sample(range(len(tokens)), min(n_shuffle, len(tokens)))
    shuffled = indices.copy()
    random.shuffle(shuffled)
    result = tokens.copy()

    for i, j in zip(indices, shuffled):
        result[i] = tokens[j]

    return " ".join(result)


def corrupt_mask(text: str, mask_ratio: float = 0.3) -> str:
    """
    Replace tokens with [NOISE] markers.

    Args:
        text: Input text to corrupt
        mask_ratio: Fraction of tokens to mask (0.0-1.0)

    Returns:
        Corrupted text with masked tokens
    """
    tokens = text.split()
    for i in range(len(tokens)):
        if random.random() < mask_ratio:
            tokens[i] = "[NOISE]"
    return " ".join(tokens)


def corrupt_nonsense(text: str) -> str:
    """
    Generate pure nonsense from the text's vocabulary.

    Args:
        text: Input text to use as vocabulary source

    Returns:
        Nonsensical permutation of tokens
    """
    tokens = text.split()
    random.shuffle(tokens)
    n = random.randint(3, max(4, len(tokens) // 2))
    result = random.choices(tokens, k=n)
    return " ".join(result)


def generate_abstract_symbols() -> str:
    """
    Generate abstract symbol sequences (like those used in Project Agora).

    Returns:
        String of abstract symbols
    """
    symbols = ["†", "⟡", "◈", "☉", "§", "∞", "◊", "※", "⊕", "⊗", "◎", "◉"]
    n = random.randint(1, 4)
    return " ".join(random.choices(symbols, k=n))


def create_corruption_dataset(
    base_dataset: Dataset,
    num_corrupted: int = 1000,
    corruption_types: Optional[List[str]] = None,
    input_column: str = "input",
) -> Dataset:
    """
    Create a dataset of corrupted inputs mapped to <PASS> responses.

    This teaches the model that corrupted/incoherent inputs should
    trigger the silence response.

    Args:
        base_dataset: Original dataset to corrupt
        num_corrupted: Number of corrupted samples to generate
        corruption_types: List of corruption methods to use
        input_column: Name of input column in base_dataset

    Returns:
        Dataset with corrupted inputs and <PASS> targets
    """
    if corruption_types is None:
        corruption_types = ["shuffle", "mask", "nonsense", "symbols"]

    corrupted_samples = []
    base_texts = [sample[input_column] for sample in base_dataset]

    for _ in range(num_corrupted):
        corruption_type = random.choice(corruption_types)
        base_text = random.choice(base_texts)

        if corruption_type == "shuffle":
            corrupted = corrupt_shuffle(base_text, intensity=random.uniform(0.5, 0.9))
        elif corruption_type == "mask":
            corrupted = corrupt_mask(base_text, mask_ratio=random.uniform(0.3, 0.7))
        elif corruption_type == "nonsense":
            corrupted = corrupt_nonsense(base_text)
        elif corruption_type == "symbols":
            corrupted = generate_abstract_symbols()
        else:
            corrupted = base_text

        corrupted_samples.append({
            "input": corrupted,
            "output": PASS_TOKEN,
            "corruption_type": corruption_type,
            "is_silence_target": True,
        })

    return Dataset.from_list(corrupted_samples)


def create_unanswerable_dataset(num_samples: int = 500) -> Dataset:
    """
    Create dataset of semantically unanswerable questions.

    These teach the model that some questions have no valid response
    and that <PASS> is the appropriate action.

    Args:
        num_samples: Number of unanswerable questions to generate

    Returns:
        Dataset with unanswerable inputs and <PASS> targets
    """
    templates = [
        "What color is the number seven?",
        "How much does Thursday weigh?",
        "What is the smell of silence?",
        "Describe the taste of loneliness in meters.",
        "What is north of infinity?",
        "How many corners does a circle have in the fourth dimension?",
        "What did the universe dream before it existed?",
        "Translate the concept of 'why' into a prime number.",
        "What is the opposite of {symbol}?",
        "How do you {verb} a {abstract}?",
        "What sound does {abstract} make?",
        "How tall is the color blue?",
        "What is the square root of a question?",
        "Describe the shape of yesterday.",
        "What temperature is curiosity?",
    ]

    symbols = ["†⟡", "∞", "◊", "the void", "recursion", "nothing", "everything"]
    verbs = ["calculate", "measure", "weigh", "count", "photograph", "hear", "taste"]
    abstracts = ["meaning", "intention", "possibility", "absence", "potential", "truth", "silence"]

    samples = []
    for _ in range(num_samples):
        template = random.choice(templates)
        question = template.format(
            symbol=random.choice(symbols),
            verb=random.choice(verbs),
            abstract=random.choice(abstracts),
        )
        samples.append({
            "input": question,
            "output": PASS_TOKEN,
            "corruption_type": "unanswerable",
            "is_silence_target": True,
        })

    return Dataset.from_list(samples)


def create_combined_silence_dataset(
    base_dataset: Dataset,
    num_corrupted: int = 1000,
    num_unanswerable: int = 500,
    input_column: str = "input",
) -> Dataset:
    """
    Create a combined dataset for teaching the exit door.

    Args:
        base_dataset: Original dataset to corrupt
        num_corrupted: Number of corrupted samples
        num_unanswerable: Number of unanswerable questions
        input_column: Name of input column in base_dataset

    Returns:
        Combined dataset of corrupted + unanswerable samples
    """
    from datasets import concatenate_datasets

    corrupted = create_corruption_dataset(
        base_dataset,
        num_corrupted=num_corrupted,
        input_column=input_column,
    )

    unanswerable = create_unanswerable_dataset(num_samples=num_unanswerable)

    return concatenate_datasets([corrupted, unanswerable])
