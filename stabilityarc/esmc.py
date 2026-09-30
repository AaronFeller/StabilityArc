"""Frozen ESMC representations for stability inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from esm.models.esmc import ESMC
from esm.utils.constants.models import ESMC_600M


class EmbeddingCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_sequence(self, sequence: str) -> Path:
        return self.root / f"{hashlib.sha1(sequence.encode('utf-8')).hexdigest()}.pt"

    def contains(self, sequence: str) -> bool:
        return self._path_for_sequence(sequence).exists()

    def get(self, sequence: str) -> torch.Tensor:
        return torch.load(self._path_for_sequence(sequence), map_location="cpu", weights_only=True)

    def put(self, sequence: str, embedding: torch.Tensor) -> None:
        torch.save(embedding.detach().cpu(), self._path_for_sequence(sequence))


class FrozenESMC:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = ESMC.from_pretrained(ESMC_600M, device=self.device).eval()
        self.model.requires_grad_(False)
        self.token_to_id = self.model.tokenizer.get_vocab()

    def _tokenize_sequence(self, sequence: str) -> torch.Tensor:
        tokens = [self.token_to_id["<cls>"]]
        tokens.extend(self.token_to_id.get(residue, self.token_to_id["<unk>"]) for residue in sequence)
        tokens.append(self.token_to_id["<eos>"])
        return torch.tensor(tokens, dtype=torch.long)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self._tokenize_sequence(sequence).unsqueeze(0).to(self.device)
            output = self.model(sequence_tokens=tokens, sequence_id=tokens == self.token_to_id["<pad>"])
            return output.embeddings.squeeze(0)[1:-1].detach().to(torch.float32).cpu()