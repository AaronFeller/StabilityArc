"""Shared RoPE stability decoder and symmetric pair residual."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
ESMC_EMBEDDING_DIM = 1152


@dataclass
class MutationTask:
    wildtype_embedding: torch.Tensor
    positions: torch.Tensor
    target_residues: torch.Tensor


class RotarySelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.query_key_value = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)

    def _rotate(self, values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)

    def _apply_rope(self, values: torch.Tensor) -> torch.Tensor:
        length = values.size(-2)
        positions = torch.arange(length, device=values.device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10000) * torch.arange(0, self.head_dim, 2, device=values.device, dtype=torch.float32) / self.head_dim)
        angles = positions[:, None] * frequencies[None, :]
        cosine = torch.cat([angles.cos(), angles.cos()], dim=-1).to(values.dtype)[None, None, :, :]
        sine = torch.cat([angles.sin(), angles.sin()], dim=-1).to(values.dtype)[None, None, :, :]
        return values * cosine + self._rotate(values) * sine

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        query, key, value = self.query_key_value(tokens).chunk(3, dim=-1)
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self._apply_rope(query.reshape(shape).transpose(1, 2))
        key = self._apply_rope(key.reshape(shape).transpose(1, 2))
        value = value.reshape(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0.0)
        return self.output(attended.transpose(1, 2).reshape(batch, length, -1))


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = RotarySelfAttention(model_dim, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.attention(self.attention_norm(tokens)))
        return tokens + self.dropout(self.feedforward(self.feedforward_norm(tokens)))


class StabilityBackbone(nn.Module):
    def __init__(self, model_dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(ESMC_EMBEDDING_DIM, model_dim)
        self.encoder = nn.ModuleList(RotaryTransformerEncoderLayer(model_dim, num_heads, 0.1) for _ in range(depth))
        self.norm = nn.LayerNorm(model_dim)
        self.score_head = nn.Linear(model_dim, len(AMINO_ACIDS))

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(embedding)
        for layer in self.encoder:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return hidden, self.score_head(hidden)


class StabilityArc(nn.Module):
    def __init__(self, model_dim: int = 256, depth: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        self.backbone = StabilityBackbone(model_dim, depth, num_heads)
        self.site_projection = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.pair_residue = nn.Embedding(len(AMINO_ACIDS), model_dim)
        self.pair_head = nn.Sequential(nn.Linear(model_dim * 3 + 2, model_dim), nn.GELU(), nn.Linear(model_dim, 1))

    def score_task_components(self, task: MutationTask, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, scores = self.backbone(task.wildtype_embedding.to(device).unsqueeze(0))
        site = self.site_projection(hidden.squeeze(0))
        return scores.squeeze(0), site

    def score_edit_sets(self, scores: torch.Tensor, site: torch.Tensor, contact_map: torch.Tensor, edit_sets: Sequence[Sequence[tuple[int, int]]], device: torch.device) -> torch.Tensor:
        direct = []
        pair_positions_left: list[int] = []
        pair_positions_right: list[int] = []
        pair_residues_left: list[int] = []
        pair_residues_right: list[int] = []
        pair_groups: list[int] = []
        for group, edits in enumerate(edit_sets):
            direct.append(scores[[position for position, _ in edits], [residue for _, residue in edits]].sum())
            for left, (left_position, left_residue) in enumerate(edits):
                for right_position, right_residue in edits[left + 1:]:
                    pair_positions_left.append(left_position)
                    pair_positions_right.append(right_position)
                    pair_residues_left.append(left_residue)
                    pair_residues_right.append(right_residue)
                    pair_groups.append(group)
        predictions = torch.stack(direct)
        if not pair_groups:
            return predictions
        left = site[torch.tensor(pair_positions_left, dtype=torch.long, device=device)] + self.pair_residue(torch.tensor(pair_residues_left, dtype=torch.long, device=device))
        right = site[torch.tensor(pair_positions_right, dtype=torch.long, device=device)] + self.pair_residue(torch.tensor(pair_residues_right, dtype=torch.long, device=device))
        distance = (torch.tensor(pair_positions_left, dtype=torch.float32, device=device) - torch.tensor(pair_positions_right, dtype=torch.float32, device=device)).abs().unsqueeze(-1) / max(site.size(0) - 1, 1)
        contact = contact_map.to(device)[
            torch.tensor(pair_positions_left, dtype=torch.long, device=device),
            torch.tensor(pair_positions_right, dtype=torch.long, device=device),
        ].unsqueeze(-1)
        features = torch.cat([left + right, (left - right).abs(), left * right, distance, contact], dim=-1)
        residuals = self.pair_head(features).squeeze(-1)
        return predictions + predictions.new_zeros(len(edit_sets)).scatter_add_(0, torch.tensor(pair_groups, dtype=torch.long, device=device), residuals)


def load_stabilityarc(checkpoint: Path, device: str = "cpu") -> tuple[StabilityArc, list[str]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    if config["position_encoding"] != "rope" or config["pair_residual"] != "contact-aware":
        raise ValueError("Checkpoint must be a direct RoPE stability scorer with a contact-aware pair residual")
    model = StabilityArc(config["model_dim"], config["depth"], config["num_heads"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), list(payload["training_proteins"])