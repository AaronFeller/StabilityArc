#!/usr/bin/env python3
"""Compute pooled T2837 metrics over per-target mutation-count cutoffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def metric_summary(rows: list[dict[str, str]], target_columns: list[str], label_column: str, prediction_column: str, threshold: float) -> dict[str, object]:
    counts = Counter(tuple(row[column].lower() if column == "pdb_id" else row[column] for column in target_columns) for row in rows)
    observed = [float(row[label_column]) for row in rows]
    predicted = [float(row[prediction_column]) for row in rows]
    positives = [value >= threshold for value in observed]
    predicted_positives = [value >= threshold for value in predicted]
    rmse = math.sqrt(sum((actual - estimate) ** 2 for actual, estimate in zip(observed, predicted)) / len(rows))
    return {
        "proteins": len(counts),
        "mutations": len(rows),
        "accuracy": accuracy_score(positives, predicted_positives),
        "precision": precision_score(positives, predicted_positives, zero_division=0),
        "recall": recall_score(positives, predicted_positives, zero_division=0),
        "auc_roc": roc_auc_score(positives, predicted),
        "pearson": pearsonr(observed, predicted).statistic,
        "spearman": spearmanr(observed, predicted).statistic,
        "rmse_kcal_per_mol": rmse,
        "inverse_rmse_per_kcal_per_mol": 1.0 / rmse,
        "positive_ddG_ge_threshold_count": sum(positives),
        "negative_ddG_lt_threshold_count": len(positives) - sum(positives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-columns", nargs="+", required=True)
    parser.add_argument("--label-column", default="ddg")
    parser.add_argument("--prediction-column", default="ddg_pred")
    parser.add_argument("--classification-threshold", type=float, default=1.0)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {*args.target_columns, args.label_column, args.prediction_column}
    missing = required.difference(rows[0]) if rows else required
    if not rows or missing:
        raise ValueError(f"Input is empty or missing columns: {sorted(missing)}")
    valid_rows = [
        row for row in rows
        if all(math.isfinite(float(row[column])) for column in (args.label_column, args.prediction_column))
    ]
    if not valid_rows:
        raise ValueError("No finite label/prediction rows remain")

    target_key = lambda row: tuple(row[column].lower() if column == "pdb_id" else row[column] for column in args.target_columns)
    counts = Counter(target_key(row) for row in valid_rows)
    results = {}
    for cutoff in (0, 10, 20, 30):
        eligible = {target for target, count in counts.items() if count > cutoff}
        retained = [row for row in valid_rows if target_key(row) in eligible]
        results[f"greater_than_{cutoff}"] = metric_summary(
            retained, args.target_columns, args.label_column, args.prediction_column, args.classification_threshold
        )

    report = {
        "input": str(args.input),
        "protocol": "Pooled metrics over supplied T2837 predictions. A target is retained when its input mutation count is strictly greater than the named cutoff.",
        "target_columns": args.target_columns,
        "classification_threshold": f"Positive class: measured {args.label_column} >= {args.classification_threshold}; predicted positive: {args.prediction_column} >= {args.classification_threshold}.",
        "num_input_rows": len(rows),
        "num_finite_rows": len(valid_rows),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()