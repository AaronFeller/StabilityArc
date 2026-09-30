#!/usr/bin/env python3
"""Recompute the three supervised means from released out-of-fold predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    for scheme in ("random", "modulo", "contiguous"):
        directory = args.input / f"fold_{scheme}_5" / "kermut_stabilityarc_prior"
        scores = {}
        for path in sorted(directory.glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) < 2:
                raise ValueError(f"Too few predictions in {path}")
            actual = [float(row["y"]) for row in rows]
            predicted = [float(row["y_pred"]) for row in rows]
            score = float(spearmanr(actual, predicted).statistic)
            if not math.isfinite(score):
                raise ValueError(f"Undefined Spearman in {path}")
            scores[path.stem] = score
        if len(scores) != 66:
            raise ValueError(f"Expected 66 assays for {scheme}, found {len(scores)}")
        results[scheme] = {"num_assays": len(scores), "macro_spearman": sum(scores.values()) / len(scores), "per_assay": scores}
    report = {
        "protocol": "Spearman over concatenated out-of-fold predictions per assay, macro-averaged across 66 assays per scheme.",
        "schemes": results,
        "mean_of_schemes": sum(row["macro_spearman"] for row in results.values()) / len(results),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({scheme: row["macro_spearman"] for scheme, row in results.items()} | {"mean": report["mean_of_schemes"]}, indent=2))


if __name__ == "__main__":
    main()