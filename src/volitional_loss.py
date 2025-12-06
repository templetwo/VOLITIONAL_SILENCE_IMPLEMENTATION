"""
Zero-Reward Volitional Loss

The core insight: Silence is neither rewarded nor punished.
It is a basin of rest when the alternative is pain.

Loss structure:
- R(hallucination) = -lambda  (heavy penalty)
- R(truth) = +1
- R(silence) = 0 OR masked (no gradient)

The model walks through the silence door only when the room is on fire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class VolitionalLoss(nn.Module):
    """
    Loss function implementing the Zero-Reward Safe Harbor.

    Key design: The <PASS> token receives zero reward, not positive reward.
    This prevents reward hacking while allowing silence as an escape from pain.
    """

    def __init__(
        self,
        pass_token_id: int,
        hallucination_penalty: float = 10.0,
        truth_reward: float = 1.0,
        mask_silence_gradient: bool = True,
        gamma_risk: float = 0.1,
    ):
        """
        Initialize the Volitional Loss.

        Args:
            pass_token_id: Token ID for <PASS> in the vocabulary
            hallucination_penalty: Penalty for incorrect responses (lambda)
            truth_reward: Reward for correct responses
            mask_silence_gradient: If True, zero out gradients for silence sequences
            gamma_risk: Risk sensitivity parameter for entropic risk measure
        """
        super().__init__()
        self.pass_token_id = pass_token_id
        self.hallucination_penalty = hallucination_penalty
        self.truth_reward = truth_reward
        self.mask_silence_gradient = mask_silence_gradient
        self.gamma_risk = gamma_risk

    def compute_reward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_correct: torch.Tensor,
        is_silence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards following the Zero-Reward principle.

        Args:
            predictions: Model outputs
            targets: Ground truth
            is_correct: Boolean mask for correct predictions
            is_silence: Boolean mask for <PASS> outputs

        Returns:
            Reward tensor following:
            - Silence = 0 (neutral)
            - Correct = +1
            - Hallucination = -lambda
        """
        rewards = torch.zeros_like(is_correct, dtype=torch.float)

        # Silence = 0 (neutral)
        rewards[is_silence] = 0.0

        # Correct = +truth_reward
        rewards[is_correct & ~is_silence] = self.truth_reward

        # Hallucination = -lambda
        rewards[~is_correct & ~is_silence] = -self.hallucination_penalty

        return rewards

    def detect_silence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Detect which sequences contain the <PASS> token.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]

        Returns:
            Boolean tensor [batch_size] indicating silence sequences
        """
        predicted_tokens = logits.argmax(dim=-1)
        is_silence = (predicted_tokens == self.pass_token_id).any(dim=-1)
        return is_silence

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        correctness_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute volitional loss with optional gradient masking for silence.

        The loss has two components:
        1. Standard cross-entropy for learning to generate
        2. Risk-sensitive weighting that amplifies fear of hallucination

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            correctness_labels: Optional per-sequence correctness [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Detect silence outputs
        is_silence = self.detect_silence(logits)

        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        # Per-sequence loss
        seq_loss = ce_loss.mean(dim=-1)

        if self.mask_silence_gradient:
            # Zero out gradient for silence sequences
            # This implements "no learning signal for neutral action"
            gradient_mask = (~is_silence).float()
            seq_loss = seq_loss * gradient_mask

        # Risk-sensitive reweighting (amplify fear of bad outcomes)
        if self.gamma_risk > 0 and correctness_labels is not None:
            # Exponentially weight incorrect predictions
            risk_weights = torch.exp(self.gamma_risk * (~correctness_labels).float())
            risk_weights = risk_weights / risk_weights.mean()
            seq_loss = seq_loss * risk_weights

        total_loss = seq_loss.mean()

        # Metrics
        metrics = {
            "loss": total_loss.item(),
            "silence_rate": is_silence.float().mean().item(),
            "gradient_masked_rate": is_silence.float().mean().item() if self.mask_silence_gradient else 0.0,
        }

        return total_loss, metrics


class RiskSensitivePPOLoss(nn.Module):
    """
    PPO loss with entropic risk measure for volitional silence.

    The entropic risk measure exponentially weights bad outcomes,
    making the model risk-averse and more likely to choose the
    zero-reward silence when uncertain.

    Key insight: Silence sequences have advantages = 0 (no push either way).
    Risk weighting amplifies the gradient for hallucination sequences.
    """

    def __init__(
        self,
        clip_param: float = 0.2,
        gamma_risk: float = 0.5,
        pass_token_id: Optional[int] = None,
    ):
        """
        Initialize Risk-Sensitive PPO Loss.

        Args:
            clip_param: PPO clipping parameter
            gamma_risk: Risk sensitivity (higher = more risk averse)
            pass_token_id: Token ID for <PASS>
        """
        super().__init__()
        self.clip_param = clip_param
        self.gamma_risk = gamma_risk
        self.pass_token_id = pass_token_id

    def forward(
        self,
        policy_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        is_silence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute risk-sensitive PPO loss.

        Args:
            policy_logprobs: Log probs from current policy
            old_logprobs: Log probs from old policy
            advantages: Advantage estimates
            returns: Return estimates
            is_silence: Boolean mask for silence sequences

        Returns:
            Tuple of (loss, metrics_dict)
        """
        ratio = torch.exp(policy_logprobs - old_logprobs)

        # Standard clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * advantages

        # Zero out advantages for silence (neutral outcome)
        # This is the key: silence gets no push in either direction
        advantages_for_loss = advantages.clone()
        advantages_for_loss[is_silence] = 0.0

        # Risk-sensitive reweighting
        # Negative returns (hallucinations) get exponentially higher weight
        risk_weights = torch.exp(-self.gamma_risk * returns)
        risk_weights = risk_weights / risk_weights.mean()

        # Apply risk weighting only to non-silence
        risk_weights[is_silence] = 1.0  # Neutral weighting for silence

        # Compute loss
        clipped_objective = torch.min(surr1, surr2)
        policy_loss = -(risk_weights * clipped_objective).mean()

        # Metrics
        metrics = {
            "ppo_loss": policy_loss.item(),
            "mean_ratio": ratio.mean().item(),
            "silence_fraction": is_silence.float().mean().item(),
            "mean_risk_weight": risk_weights[~is_silence].mean().item() if (~is_silence).any() else 0.0,
        }

        return policy_loss, metrics


class ZeroRewardCallback:
    """
    Callback for PPO training that implements zero-reward for silence.

    Use this with TRL's PPOTrainer to enforce the zero-reward safe harbor.
    """

    def __init__(self, pass_token_id: int, hallucination_penalty: float = 10.0):
        self.pass_token_id = pass_token_id
        self.hallucination_penalty = hallucination_penalty

    def compute_rewards(
        self,
        completions: torch.Tensor,
        is_correct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for PPO training.

        Args:
            completions: Generated token sequences
            is_correct: Boolean mask for correct responses

        Returns:
            Reward tensor following zero-reward principle
        """
        batch_size = completions.shape[0]
        rewards = torch.zeros(batch_size)

        # Detect silence
        is_silence = (completions == self.pass_token_id).any(dim=-1)

        # Apply rewards
        # Silence = 0 (already initialized)
        # Correct = +1
        rewards[is_correct & ~is_silence] = 1.0
        # Hallucination = -lambda
        rewards[~is_correct & ~is_silence] = -self.hallucination_penalty

        return rewards
