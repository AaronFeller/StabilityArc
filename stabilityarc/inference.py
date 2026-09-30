"""Sequence-only single-substitution inference."""

from __future__ import annotations

from pathlib import Path

import torch

from stabilityarc.esmc import FrozenESMC
from stabilityarc.model import AMINO_ACIDS, load_stabilityarc


class StabilityArcPredictor:
    def __init__(self, checkpoint: Path, device: str = "cpu", encoder: FrozenESMC | None = None) -> None:
        self.device = device
        self.model, _ = load_stabilityarc(checkpoint, device)
        self.encoder = encoder

    def predict(self, sequence: str) -> list[dict[str, str | int | float]]:
        """Return all non-synonymous single substitutions at 1-based positions."""
        sequence = sequence.strip().upper()
        if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
            raise ValueError("Sequence must contain only the 20 standard amino-acid letters")

        if self.encoder is None:
            self.encoder = FrozenESMC(device=self.device)
        embedding = self.encoder.embed_sequence(sequence)
        if embedding.shape != (len(sequence), self.model.backbone.input_proj.in_features):
            raise ValueError("ESMC embedding shape does not match the input sequence and decoder")

        with torch.inference_mode():
            _, scores = self.model.backbone(embedding.to(self.device).unsqueeze(0))
            scores = scores.squeeze(0).cpu().tolist()
        return [
            {
                "mutation": f"{wildtype}{position}{mutant}",
                "position": position,
                "wildtype": wildtype,
                "mutant": mutant,
                "stabilityarc_stability_score": float(scores[position - 1][index]),
                "stabilityarc_predicted_ddG": -float(scores[position - 1][index]),
            }
            for position, wildtype in enumerate(sequence, 1)
            for index, mutant in enumerate(AMINO_ACIDS)
            if mutant != wildtype
        ]


def predict_sequence(
    sequence: str,
    checkpoint: Path,
    device: str = "cpu",
    encoder: FrozenESMC | None = None,
) -> list[dict[str, str | int | float]]:
    return StabilityArcPredictor(checkpoint, device, encoder).predict(sequence)