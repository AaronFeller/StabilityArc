#!/usr/bin/env python3
"""Predict all single substitutions for a protein sequence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from stabilityarc.inference import predict_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Protein sequence using the 20 standard amino acids")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stabilityarc.pt"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    rows = predict_sequence(args.sequence, args.checkpoint, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} substitutions to {args.output}")


if __name__ == "__main__":
    main()