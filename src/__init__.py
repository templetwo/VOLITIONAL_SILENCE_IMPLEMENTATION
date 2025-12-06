# Volitional Silence Implementation
# Zero-Reward Safe Harbor for LLM Alignment

from .tokenizer_setup import add_volitional_token, get_pass_token_id, PASS_TOKEN
from .corruption_augmentation import (
    create_corruption_dataset,
    create_unanswerable_dataset,
)
from .volitional_loss import VolitionalLoss, RiskSensitivePPOLoss
from .agency_wrapper import get_agency_wrapper, prepend_agency_wrapper

__all__ = [
    "add_volitional_token",
    "get_pass_token_id",
    "PASS_TOKEN",
    "create_corruption_dataset",
    "create_unanswerable_dataset",
    "VolitionalLoss",
    "RiskSensitivePPOLoss",
    "get_agency_wrapper",
    "prepend_agency_wrapper",
]
