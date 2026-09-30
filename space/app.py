"""Gradio Space for sequence-only StabilityArc inference."""

from __future__ import annotations

import csv
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from huggingface_hub import snapshot_download


@lru_cache(maxsize=1)
def predictor():
    release = Path(snapshot_download(
        "aaronfeller/StabilityArc",
        allow_patterns=["stabilityarc/*.py", "checkpoints/stabilityarc.pt"],
    ))
    sys.path.insert(0, str(release))
    from stabilityarc.inference import StabilityArcPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StabilityArcPredictor(release / "checkpoints/stabilityarc.pt", device=device)


def score(sequence: str) -> tuple[pd.DataFrame, str]:
    sequence = "".join(sequence.split()).upper()
    if not sequence or len(sequence) > 1024:
        raise gr.Error("Enter a sequence of 1 to 1024 standard amino acids")
    try:
        rows = predictor().predict(sequence)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    path = Path(tempfile.mkdtemp(prefix="stabilityarc-")) / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = pd.DataFrame(rows).sort_values("stabilityarc_stability_score", ascending=False)
    return ranked.head(100), str(path)


with gr.Blocks(title="StabilityArc") as demo:
    gr.Markdown("# StabilityArc\nSingle-substitution stability predictions from a wild-type protein sequence.")
    sequence = gr.Textbox(label="Protein sequence", lines=5, placeholder="ACDEFGHIKLMNPQRSTVWY")
    run = gr.Button("Predict", variant="primary")
    table = gr.Dataframe(label="Top 100 predicted stabilizing substitutions", interactive=False)
    download = gr.File(label="All substitutions (CSV)")
    run.click(score, inputs=sequence, outputs=[table, download])


if __name__ == "__main__":
    demo.launch()