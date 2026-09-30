"""Unsupervised attention-derived ESMC contact features."""

from __future__ import annotations

import hashlib
import math

import torch

from stabilityarc.esmc import FrozenESMC


def _apc(matrix: torch.Tensor) -> torch.Tensor:
    row_mean = matrix.mean(dim=1, keepdim=True)
    column_mean = matrix.mean(dim=0, keepdim=True)
    total_mean = matrix.mean()
    if total_mean.abs() < 1e-12:
        return matrix
    return matrix - (row_mean @ column_mean) / total_mean


def _head_entropy(matrix: torch.Tensor) -> torch.Tensor:
    probabilities = matrix.clamp_min(0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1).mean()


def extract_unsupervised_contact_map(sequence: str, device: str, encoder: FrozenESMC | None = None) -> tuple[torch.Tensor, dict[str, object]]:
    encoder = encoder or FrozenESMC(device=device)
    model = encoder.model
    captured_qk: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    hooks = []

    def capture_qk(module, inputs):
        hidden, _sequence_id = inputs
        qkv = module.layernorm_qkv(hidden)
        query, key, _value = torch.chunk(qkv, 3, dim=-1)
        query = module.q_ln(query).to(query.dtype)
        key = module.k_ln(key).to(key.dtype)
        query, key = module._apply_rotary(query, key)
        captured_qk.append((query.detach(), key.detach(), module.n_heads))

    for block in model.transformer.blocks:
        hooks.append(block.attn.register_forward_pre_hook(capture_qk))
    try:
        tokens = encoder._tokenize_sequence(sequence).unsqueeze(0).to(encoder.device)
        with torch.inference_mode():
            model(sequence_tokens=tokens, sequence_id=tokens == encoder.token_to_id["<pad>"])
    finally:
        for hook in hooks:
            hook.remove()

    length = len(sequence)
    weighted_sum = torch.zeros((length, length), dtype=torch.float32)
    weight_sum = torch.zeros((), dtype=torch.float32)
    for query, key, num_heads in captured_qk:
        head_dim = query.shape[-1] // num_heads
        query = query.reshape(1, -1, num_heads, head_dim).transpose(1, 2)
        key = key.reshape(1, -1, num_heads, head_dim).transpose(1, 2)
        attention = torch.softmax((query.float() @ key.float().transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
        attention = attention[0, :, 1:-1, 1:-1].cpu()
        for head_map in attention:
            head_map = 0.5 * (head_map + head_map.transpose(0, 1))
            head_map = _apc(head_map)
            head_map.fill_diagonal_(0)
            weight = torch.exp(-4.0 * _head_entropy(head_map))
            weighted_sum += weight * head_map
            weight_sum += weight

    if not captured_qk or weight_sum <= 0:
        raise RuntimeError("Native ESMC did not expose usable attention heads")
    contact_map = weighted_sum / weight_sum
    contact_map = 0.5 * (contact_map + contact_map.transpose(0, 1))
    contact_map.fill_diagonal_(0)
    metadata: dict[str, object] = {
        "method": "native-esmc-600m-unsupervised-entropy-gated-attention",
        "sequence_sha1": hashlib.sha1(sequence.encode("utf-8")).hexdigest(),
        "sequence_length": length,
        "layers": len(captured_qk),
        "heads_per_layer": captured_qk[0][2],
        "head_count": len(captured_qk) * captured_qk[0][2],
        "apc": True,
        "symmetrized": True,
        "zero_diagonal": True,
    }
    return contact_map, metadata