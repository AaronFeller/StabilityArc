#!/usr/bin/env python3
"""Compare two T2837 prediction files on their shared PDB-chain mutations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from .summarize_t2837_predictions import metric_summary


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def stabilityarc_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["pdb_code"].lower(),
        row["chain_id"],
        f"{THREE_TO_ONE[row['wtAA']]}{row['position']}{THREE_TO_ONE[row['mutAA']]}",
    )


def oracle_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["pdb_id"].lower(), row["chain_id"], row["mutation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stabilityarc-input", type=Path, required=True)
    parser.add_argument("--stability-oracle-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classification-threshold", type=float, default=1.0)
    args = parser.parse_args()

    with args.stabilityarc_input.open(newline="", encoding="utf-8") as handle:
        stabilityarc_by_key = {stabilityarc_key(row): row for row in csv.DictReader(handle)}
    with args.stability_oracle_input.open(newline="", encoding="utf-8") as handle:
        oracle_by_key = {oracle_key(row): row for row in csv.DictReader(handle)}
    shared_keys = sorted(stabilityarc_by_key.keys() & oracle_by_key.keys())
    if not shared_keys:
        raise ValueError("No shared PDB-chain mutation keys")

    paired_rows: list[dict[str, str]] = []
    for key in shared_keys:
        stabilityarc, oracle = stabilityarc_by_key[key], oracle_by_key[key]
        observed_stabilityarc, observed_oracle = float(stabilityarc["ddG"]), float(oracle["ddg"])
        if not math.isclose(observed_stabilityarc, observed_oracle, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Measured ddG mismatch for {key}: {observed_stabilityarc} != {observed_oracle}")
        paired_rows.append({
            "pdb_id": key[0],
            "chain_id": key[1],
            "mutation": key[2],
            "ddg": oracle["ddg"],
            "stabilityarc_predicted_ddg": stabilityarc["stabilityarc_predicted_ddG"],
            "stability_oracle_predicted_ddg": oracle["ddg_pred"],
        })

    target_key = lambda row: (row["pdb_id"], row["chain_id"])
    counts = Counter(target_key(row) for row in paired_rows)
    results = {}
    for cutoff in (0, 10, 20, 30):
        eligible = {target for target, count in counts.items() if count > cutoff}
        retained = [row for row in paired_rows if target_key(row) in eligible]
        results[f"greater_than_{cutoff}"] = {
            "proteins": len(eligible),
            "mutations": len(retained),
            "stabilityarc": metric_summary(retained, ["pdb_id", "chain_id"], "ddg", "stabilityarc_predicted_ddg", args.classification_threshold),
            "stability_oracle": metric_summary(retained, ["pdb_id", "chain_id"], "ddg", "stability_oracle_predicted_ddg", args.classification_threshold),
        }

    report = {
        "protocol": "Paired pooled comparison on exact shared raw T2837 PDB-chain mutation keys. A target is retained when its shared mutation count is strictly greater than the named cutoff.",
        "stabilityarc_input": str(args.stabilityarc_input),
        "stability_oracle_input": str(args.stability_oracle_input),
        "classification_threshold": f"Positive class: measured ddG >= {args.classification_threshold}; predicted positive: model predicted ddG >= {args.classification_threshold}.",
        "shared_proteins": len(counts),
        "shared_mutations": len(paired_rows),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()