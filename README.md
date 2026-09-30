---
tags:
  - protein
  - stability
  - mutation-effects
  - esm
library_name: pytorch
---

# StabilityArc

StabilityArc is a cross-protein model for predicting protein stability landscapes from sequence. A frozen [ESMC-600M](https://huggingface.co/biohub/esmc-600m-2024-12) encoder provides residue representations; a shared four-layer RoPE transformer predicts substitution effects across the sequence. A symmetric, contact-aware residual supports scoring simultaneous substitutions.

This release accompanies the manuscript **StabilityArc: Decoding Protein Sequence Embeddings into Generalizable Stability Landscapes**, by Aaron L. Feller, Andrew D. Ellington, and Claus O. Wilke.

## Repository contents

- `stabilityarc/`: the decoder, frozen ESMC interface, contact extraction, and sequence inference API.
- `checkpoints/stabilityarc.pt`: a decoder trained for 40 epochs on stability assays from all 66 eligible ProteinGym proteins. The training protein list is stored in the checkpoint; additional training metadata is in `checkpoints/release_manifest.json` and `checkpoints/pretraining_summary.json`.
- `scripts/`: sequence prediction and evaluation/summary scripts.
- `notebooks/stabilityarc_colab.ipynb`: a Colab notebook for sequence-level inference.
- `space/`: a Gradio template for a separate Hugging Face Space.
- `results/`: aggregate evaluation summaries. Row-level benchmark CSVs are not distributed in the public repository.
- `data/SOURCES.txt`: upstream data locations and provenance. Third-party benchmark inputs are not bundled in the public repository.

The frozen ESMC-600M weights are downloaded from their upstream host and are not part of the StabilityArc checkpoint. The release checkpoint is for inference on external proteins; it is **not** one of the protein-held-out checkpoints used to calculate the manuscript's ProteinGym zero-shot results.

## Installation

Python 3.10+ is required. A CUDA GPU with enough memory for ESMC-600M is recommended; CPU inference is possible but slower. From the release directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

The model files can also be obtained directly from Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download("aaronfeller/StabilityArc", local_dir="StabilityArc")
```

Install from the downloaded `StabilityArc/` directory using the commands above. The first prediction requires network access to download ESMC-600M.

## Inference

To score all 19 non-wild-type single substitutions at each position:

```bash
.venv/bin/python -m scripts.predict_sequence \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --output predictions.csv
```

Input sequences must contain only the 20 standard amino acids. `--device auto` selects CUDA when available; use `--device cpu` to force CPU. Output columns include the 1-based `mutation` (such as `A1C`), position, wild-type and mutant residues, stability score, and `stabilityarc_predicted_ddG`. Larger stability scores mean predicted greater stability; predicted ddG is the negative of that score, **not a calibrated experimental kcal/mol measurement**.

The same interface can be used from Python:

```python
from pathlib import Path
import torch
from stabilityarc.inference import StabilityArcPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = StabilityArcPredictor(Path("checkpoints/stabilityarc.pt"), device=device)
rows = predictor.predict("ACDEFGHIKLMNPQRSTVWY")
print(rows[0])
```

Reuse the predictor for additional sequences to keep ESMC-600M and the decoder loaded. The [Colab notebook](notebooks/stabilityarc_colab.ipynb) provides the same workflow. The [Gradio Space template](space/README.md) can be deployed in a separate GPU Space; it is not a hosted service in this repository. Single-substitution inference does not require contact maps. For simultaneous substitutions, see the contact-aware scoring path in `scripts/evaluate_t2837.py`.

## Evaluation

The manuscript reports a Spearman correlation of 0.7134 across 66 protein-held-out ProteinGym evaluations (134,794 variants), compared with 0.6526 for ProSST-2048. These results used **66 separate protein-held-out models**, not the bundled all-66 checkpoint. The held-out checkpoints and underlying ProteinGym training data are not included, so those inference runs cannot be repeated from this release alone.

The all-66 checkpoint generated the archived external T2837 predictions. Its paired comparison for targets with strictly more than 30 shared mutations contains 1,884 mutations from 19 PDB-chain targets. StabilityArc was also evaluated as a prior for Kermut; the mean Spearman correlation across three supervised split schemes was 0.8280. Kermut comparator values in the manuscript are literature-reported.

The public repository includes aggregate JSON summaries in `results/`. To recalculate the external and supervised aggregates, obtain the original input CSVs separately and place them at the paths below (or adjust the flags):

```bash
.venv/bin/python -m scripts.compare_t2837_predictions \
  --stabilityarc-input results/t2837_predictions.csv \
  --stability-oracle-input results/T2837_stabilityOracle.csv \
  --output paired_metrics.json
.venv/bin/python -m scripts.summarize_kermut_predictions \
  --input results/kermut_predictions \
  --output kermut_summary.json
```

The 198 Kermut out-of-fold CSVs remain in the local research archive; they, the fold checkpoints, and the Kermut training pipeline are not part of this public repository. The raw T2837 inputs and third-party predictions are not published to GitHub or the [Hugging Face model repository](https://huggingface.co/aaronfeller/StabilityArc).

## Citation

Feller, Aaron L., Andrew D. Ellington, and Claus O. Wilke. 2026. **StabilityArc: Decoding Protein Sequence Embeddings into Generalizable Stability Landscapes**. Manuscript.

## License and data

The original StabilityArc code and checkpoint are available under the [MIT License](LICENSE). See [data/SOURCES.txt](data/SOURCES.txt) for ProteinGym download instructions and dataset provenance. Third-party benchmark data and ESMC-600M weights are not relicensed by this project; consult their upstream terms before redistributing them.