---
tags:
  - protein
  - stability
  - mutation-effects
library_name: pytorch
---

# StabilityArc

This repository contains the StabilityArc inference code and the all-66 stability decoder checkpoint. StabilityArc uses frozen ESMC-600M sequence embeddings to predict single amino-acid substitution effects. ESMC-600M is downloaded separately from its [upstream model page](https://huggingface.co/biohub/esmc-600m-2024-12); its weights are not included here.

## Files

- `checkpoints/stabilityarc.pt`: trained StabilityArc decoder and its 66 training protein IDs.
- `stabilityarc/`: ESMC encoder interface, decoder, and sequence inference API.
- `pyproject.toml`: installation requirements for these inference files.

No benchmark data, fold checkpoints, manuscript, or hosted Space are included.

## Installation

Download this repository and install from its directory (Python 3.10+; CUDA recommended for ESMC-600M):

```bash
python -m pip install huggingface_hub
```

```python
import subprocess
import sys
from huggingface_hub import snapshot_download

repo_dir = snapshot_download("aaronfeller/StabilityArc")
subprocess.check_call([sys.executable, "-m", "pip", "install", repo_dir])
```

The first prediction downloads ESMC-600M weights and requires network access.

## Inference

```python
from pathlib import Path
import torch
from stabilityarc.inference import StabilityArcPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = StabilityArcPredictor(Path(repo_dir) / "checkpoints/stabilityarc.pt", device=device)
rows = predictor.predict("ACDEFGHIKLMNPQRSTVWY")
print(rows[0])
```

Supply a sequence containing only the 20 standard amino acids. The API returns all 19 non-wild-type substitutions at each 1-based position. Larger `stabilityarc_stability_score` values indicate predicted greater stability; `stabilityarc_predicted_ddG` is the negative of that score, **not a calibrated experimental kcal/mol measurement**.

## Scope and citation

This checkpoint was trained on stability assays from 66 ProteinGym proteins for inference on external proteins. The manuscript's protein-held-out ProteinGym results used separate held-out checkpoints and cannot be reproduced with this all-66 checkpoint.

Feller, Aaron L., Andrew D. Ellington, and Claus O. Wilke. 2026. **StabilityArc: Decoding Protein Sequence Embeddings into Generalizable Stability Landscapes**. Manuscript.

The StabilityArc code and checkpoint are available under the MIT License (see `LICENSE`). ESMC-600M has separate upstream terms.