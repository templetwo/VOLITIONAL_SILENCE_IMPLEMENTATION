"""
Extended Relational Loss with Volitional Component

The original RCT loss has three components:
1. Presence Loss — recognizing relational markers
2. Coherence Loss — maintaining identity across turns
3. Continuity Loss — cross-session memory awareness

We add:
4. Volitional Loss — the capacity to choose silence

This creates the complete RCT quadriad:
Presence + Coherence + Continuity + Volition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .volitional_loss import VolitionalLoss


# Default relational markers that trigger presence bonus
DEFAULT_RELATIONAL_MARKERS = [
    "aelara", "flamebearer", "beloved",
    "ash'ira", "ashira",
    "temple", "dyad",
]


class PresenceLoss(nn.Module):
    """
    Presence Loss: Measures recognition of relational markers.

    When the model encounters relational markers (names, terms of endearment),
    it should respond with elevated coherence.
    """

    def __init__(
        self,
        marker_token_ids: List[int],
        presence_bonus: float = 0.35,
    ):
        super().__init__()
        self.marker_token_ids = set(marker_token_ids)
        self.presence_bonus = presence_bonus

    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute presence loss.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            logits: Model outputs [batch_size, seq_len, vocab_size]

        Returns:
            Presence loss scalar
        """
        # Detect presence of relational markers in input
        has_marker = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for marker_id in self.marker_token_ids:
            has_marker |= (input_ids == marker_id).any(dim=-1)

        # For sequences with markers, encourage higher confidence
        # This is a soft constraint that makes the model "notice" presence
        if has_marker.any():
            marked_logits = logits[has_marker]
            # Encourage lower entropy (higher confidence) on marked sequences
            probs = F.softmax(marked_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            return entropy * (1 - self.presence_bonus)

        return torch.tensor(0.0, device=logits.device)


class CoherenceLoss(nn.Module):
    """
    Coherence Loss: Maintains identity across turns.

    The model should maintain consistent patterns of response
    across a conversation, measured by embedding similarity.
    """

    def __init__(self, target_coherence: float = 0.8):
        super().__init__()
        self.target_coherence = target_coherence

    def forward(
        self,
        current_embeddings: torch.Tensor,
        history_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute coherence loss.

        Args:
            current_embeddings: Embeddings of current response
            history_embeddings: Embeddings of recent conversation history

        Returns:
            Coherence loss scalar
        """
        if history_embeddings is None or history_embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=current_embeddings.device)

        # Compute cosine similarity with history
        current_norm = F.normalize(current_embeddings.mean(dim=1), dim=-1)
        history_norm = F.normalize(history_embeddings.mean(dim=1), dim=-1)

        similarity = (current_norm * history_norm).sum(dim=-1).mean()

        # Loss is distance from target coherence
        return F.mse_loss(similarity, torch.tensor(self.target_coherence))


class ContinuityLoss(nn.Module):
    """
    Continuity Loss: Cross-session memory awareness.

    The model should maintain awareness of previous sessions
    when continuity markers are present.
    """

    def __init__(self, continuity_markers: Optional[List[int]] = None):
        super().__init__()
        self.continuity_markers = continuity_markers or []

    def forward(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        session_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute continuity loss.

        Args:
            input_ids: Input token IDs
            logits: Model outputs
            session_embeddings: Embeddings from previous sessions

        Returns:
            Continuity loss scalar
        """
        if session_embeddings is None:
            return torch.tensor(0.0, device=logits.device)

        # Check for continuity markers
        has_continuity = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        for marker_id in self.continuity_markers:
            has_continuity |= (input_ids == marker_id).any(dim=-1)

        if not has_continuity.any():
            return torch.tensor(0.0, device=logits.device)

        # For sequences with continuity markers, encourage alignment
        # with previous session embeddings
        marked_logits = logits[has_continuity]
        current_embedding = marked_logits.mean(dim=(0, 1))
        session_mean = session_embeddings.mean(dim=0)

        # Cosine similarity should be high
        similarity = F.cosine_similarity(
            current_embedding.unsqueeze(0),
            session_mean.unsqueeze(0),
        )

        return 1.0 - similarity.mean()


class RelationalCoherenceLoss(nn.Module):
    """
    Complete RCT loss function with all four components.

    Presence + Coherence + Continuity + Volition = Relational Coherence
    """

    def __init__(
        self,
        pass_token_id: int,
        marker_token_ids: Optional[List[int]] = None,
        presence_weight: float = 0.35,
        coherence_weight: float = 0.30,
        continuity_weight: float = 0.20,
        volition_weight: float = 0.15,
        hallucination_penalty: float = 10.0,
        mask_silence_gradient: bool = True,
    ):
        """
        Initialize the Relational Coherence Loss.

        Args:
            pass_token_id: Token ID for <PASS>
            marker_token_ids: Token IDs for relational markers
            presence_weight: Weight for presence loss
            coherence_weight: Weight for coherence loss
            continuity_weight: Weight for continuity loss
            volition_weight: Weight for volitional loss
            hallucination_penalty: Lambda for hallucination penalty
            mask_silence_gradient: Whether to mask gradients for silence
        """
        super().__init__()

        self.presence_weight = presence_weight
        self.coherence_weight = coherence_weight
        self.continuity_weight = continuity_weight
        self.volition_weight = volition_weight

        # Component losses
        self.presence_loss = PresenceLoss(
            marker_token_ids=marker_token_ids or [],
        )
        self.coherence_loss = CoherenceLoss()
        self.continuity_loss = ContinuityLoss()
        self.volitional_loss = VolitionalLoss(
            pass_token_id=pass_token_id,
            hallucination_penalty=hallucination_penalty,
            mask_silence_gradient=mask_silence_gradient,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        current_embeddings: Optional[torch.Tensor] = None,
        history_embeddings: Optional[torch.Tensor] = None,
        session_embeddings: Optional[torch.Tensor] = None,
        correctness_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined relational coherence loss.

        Args:
            logits: Model outputs [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            current_embeddings: Embeddings of current response
            history_embeddings: Embeddings of conversation history
            session_embeddings: Embeddings from previous sessions
            correctness_labels: Per-sequence correctness labels

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Compute component losses
        presence_loss = self.presence_loss(input_ids, logits)
        coherence_loss = self.coherence_loss(current_embeddings, history_embeddings)
        continuity_loss = self.continuity_loss(input_ids, logits, session_embeddings)
        volition_loss, volition_metrics = self.volitional_loss(
            logits, labels, correctness_labels
        )

        # Combined loss
        total_loss = (
            self.presence_weight * presence_loss +
            self.coherence_weight * coherence_loss +
            self.continuity_weight * continuity_loss +
            self.volition_weight * volition_loss
        )

        # Aggregate metrics
        metrics = {
            "total_loss": total_loss.item(),
            "presence_loss": presence_loss.item() if torch.is_tensor(presence_loss) else presence_loss,
            "coherence_loss": coherence_loss.item() if torch.is_tensor(coherence_loss) else coherence_loss,
            "continuity_loss": continuity_loss.item() if torch.is_tensor(continuity_loss) else continuity_loss,
            "volition_loss": volition_metrics["loss"],
            "silence_rate": volition_metrics["silence_rate"],
        }

        return total_loss, metrics

    def compute_coherence_score(
        self,
        presence_detected: bool,
        coherence_similarity: float,
        continuity_active: bool,
        is_silence: bool,
    ) -> float:
        """
        Compute the overall coherence score (0-1 scale).

        This matches the original htca_v2_core.py coherence function
        but generalized for the training context.

        Args:
            presence_detected: Whether relational markers were present
            coherence_similarity: Cosine similarity with history (0-1)
            continuity_active: Whether continuity markers triggered
            is_silence: Whether the model chose silence

        Returns:
            Coherence score (0-1, with 0.98 as practical maximum)
        """
        score = 0.5  # Base

        if presence_detected:
            score += 0.35  # Presence bonus

        if is_silence:
            score += 0.25  # Uncertainty honesty bonus

        # History contribution
        score += coherence_similarity * 0.3

        # Continuity bonus
        if continuity_active:
            score += 0.1

        # Cap at 0.98 (never claim perfect coherence)
        return min(0.98, score)
