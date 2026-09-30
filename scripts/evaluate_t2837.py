#!/usr/bin/env python3
"""Evaluate the all-data StabilityArc checkpoint on T2837."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from scipy.stats import rankdata

from stabilityarc.contacts import extract_unsupervised_contact_map
from stabilityarc.esmc import EmbeddingCache, FrozenESMC
from stabilityarc.model import AMINO_ACID_TO_INDEX, MutationTask, load_stabilityarc


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def spearman(scores: list[float], observed: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    ranked_scores, ranked_observed = rankdata(scores), rankdata(observed)
    centered_scores = ranked_scores - ranked_scores.mean()
    centered_observed = ranked_observed - ranked_observed.mean()
    denominator = math.sqrt(float((centered_scores**2).sum() * (centered_observed**2).sum()))
    return float((centered_scores * centered_observed).sum() / denominator) if denominator else 0.0


def target_id(row: dict[str, str]) -> str:
    sequence_digest = hashlib.sha1(row["sequence"].encode("utf-8")).hexdigest()[:12]
    return f"{row['uniprot_id']}|{row['pdb_code']}|{row['chain_id']}|{sequence_digest}"


def validate_row(row: dict[str, str]) -> str | None:
    try:
        int(row["position"])
        ddg = float(row["ddG"])
    except ValueError:
        return "invalid_numeric_value"
    sequence = row["sequence"].strip().upper()
    wildtype = THREE_TO_ONE.get(row["wtAA"].strip().upper())
    mutant = THREE_TO_ONE.get(row["mutAA"].strip().upper())
    if not sequence or any(residue not in AMINO_ACID_TO_INDEX and residue != "X" for residue in sequence):
        return "invalid_sequence"
    if wildtype is None or mutant is None:
        return "unsupported_amino_acid"
    if wildtype == mutant or not math.isfinite(ddg):
        return "invalid_mutation_or_label"
    return None


def resolve_positions(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, dict[str, int]]:
    """Infer the PDB-number to sequence-index offset that maximizes WT matches."""
    sequence = rows[0]["sequence"]
    min_position = min(int(row["position"]) for row in rows)
    max_position = max(int(row["position"]) for row in rows)
    offsets = range(-max_position, len(sequence) - min_position + 1)
    def match_count(offset: int) -> int:
        return sum(
            0 <= int(row["position"]) - 1 + offset < len(sequence)
            and sequence[int(row["position"]) - 1 + offset] == THREE_TO_ONE[row["wtAA"].upper()]
            for row in rows
        )
    offset = max(offsets, key=lambda candidate: (match_count(candidate), -abs(candidate), -candidate))
    kept, dropped = [], defaultdict(int)
    for row in rows:
        index = int(row["position"]) - 1 + offset
        wildtype = THREE_TO_ONE[row["wtAA"].upper()]
        if index < 0 or index >= len(sequence):
            dropped["position_outside_resolved_sequence"] += 1
        elif sequence[index] != wildtype:
            dropped["wildtype_mismatch_after_coordinate_resolution"] += 1
        else:
            row["sequence_position"] = str(index + 1)
            kept.append(row)
    return kept, offset, dict(dropped)


def load_or_create_inputs(sequence: str, cache: EmbeddingCache, contact_map_dir: Path, encoder: FrozenESMC | None, device: str) -> tuple[torch.Tensor, torch.Tensor, FrozenESMC | None]:
    try:
        embedding = cache.get(sequence).to(torch.float32) if cache.contains(sequence) else None
        if embedding is not None and embedding.shape[0] != len(sequence):
            embedding = None
    except (KeyError, RuntimeError, ValueError):
        embedding = None
    if embedding is None:
        encoder = encoder or FrozenESMC(device=device)
        embedding = encoder.embed_sequence(sequence)
        cache.put(sequence, embedding)
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
    contact_path = contact_map_dir / f"{digest}.pt"
    try:
        contact_map = torch.load(contact_path, map_location="cpu", weights_only=True).to(torch.float32) if contact_path.exists() else None
        if contact_map is not None and (contact_map.shape != (len(sequence), len(sequence)) or not torch.isfinite(contact_map).all()):
            contact_map = None
    except (RuntimeError, ValueError, EOFError):
        contact_map = None
    if contact_map is None:
        encoder = encoder or FrozenESMC(device=device)
        contact_map, metadata = extract_unsupervised_contact_map(sequence, device=device, encoder=encoder)
        contact_map_dir.mkdir(parents=True, exist_ok=True)
        torch.save(contact_map, contact_path)
        contact_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if embedding.shape[0] != len(sequence) or contact_map.shape != (len(sequence), len(sequence)):
        raise ValueError(f"Invalid cached ESMC inputs for sequence {digest}")
    return embedding, contact_map, encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="T2837 CSV with sequence, position, wtAA, mutAA, and ddG columns.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--contact-map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    required_columns = {"dataset", "uniprot_id", "pdb_code", "chain_id", "position", "wtAA", "mutAA", "ddG", "sequence"}
    missing_columns = required_columns.difference(input_rows[0] if input_rows else {})
    if missing_columns:
        raise ValueError(f"Missing required T2837 columns: {sorted(missing_columns)}")

    dropped: dict[str, int] = defaultdict(int)
    candidate_rows: list[dict[str, str]] = []
    for row in input_rows:
        reason = validate_row(row)
        if reason is None:
            row["sequence"] = row["sequence"].strip().upper()
            candidate_rows.append(row)
        else:
            dropped[reason] += 1
    if not candidate_rows:
        raise ValueError("No valid T2837 mutations remain after validation")

    candidates_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_target[target_id(row)].append(row)
    valid_rows: list[dict[str, str]] = []
    coordinate_maps: list[dict[str, object]] = []
    for identifier, rows in sorted(candidates_by_target.items()):
        resolved, offset, resolution_drops = resolve_positions(rows)
        valid_rows.extend(resolved)
        for reason, count in resolution_drops.items():
            dropped[reason] += count
        coordinate_maps.append({
            "target_id": identifier,
            "num_candidate_rows": len(rows),
            "num_resolved_rows": len(resolved),
            "pdb_to_sequence_offset": offset,
        })
    if not valid_rows:
        raise ValueError("No T2837 mutations remain after coordinate resolution")

    model, training_proteins = load_stabilityarc(args.checkpoint, args.device)
    cache = EmbeddingCache(args.cache_dir)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        grouped[target_id(row)].append(row)

    encoder: FrozenESMC | None = None
    scored_rows: list[dict[str, object]] = []
    for identifier, rows in sorted(grouped.items()):
        sequence = rows[0]["sequence"]
        embedding, contact_map, encoder = load_or_create_inputs(sequence, cache, args.contact_map_dir, encoder, args.device)
        positions = torch.tensor([int(row["sequence_position"]) - 1 for row in rows], dtype=torch.long)
        residues = torch.tensor([AMINO_ACID_TO_INDEX[THREE_TO_ONE[row["mutAA"].upper()]] for row in rows], dtype=torch.long)
        task = MutationTask(wildtype_embedding=embedding, positions=positions, target_residues=residues)
        with torch.inference_mode():
            scores, site = model.score_task_components(task, torch.device(args.device))
            predictions = model.score_edit_sets(
                scores, site, contact_map, [[(int(position), int(residue))] for position, residue in zip(positions, residues)], torch.device(args.device)
            ).cpu().tolist()
        for row, score in zip(rows, predictions):
            scored_rows.append({
                **row,
                "target_id": identifier,
                "stabilityarc_stability_score": float(score),
                "stabilityarc_predicted_ddG": float(-score),
            })

    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        by_target[str(row["target_id"])].append(row)
        by_source[str(row["dataset"])].append(row)

    def summarize(groups: dict[str, list[dict[str, object]]], label: str) -> list[dict[str, object]]:
        results = []
        for identifier, rows in sorted(groups.items()):
            observed = [float(row["ddG"]) for row in rows]
            predicted = [float(row["stabilityarc_predicted_ddG"]) for row in rows]
            results.append({label: identifier, "num_mutations": len(rows), "spearman": spearman(predicted, observed)})
        return results

    per_target = summarize(by_target, "target_id")
    per_source = summarize(by_source, "dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "t2837_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "protocol": "All-66 ProteinGym-trained StabilityArc release checkpoint, evaluated zero-shot on T2837.",
        "sign_convention": "T2837 ddG is positive for destabilization; Spearman uses -StabilityArc stability score as predicted ddG.",
        "training_proteins": training_proteins,
        "num_input_rows": len(input_rows),
        "num_scored_rows": len(scored_rows),
        "dropped_rows": dict(sorted(dropped.items())),
        "coordinate_resolution": coordinate_maps,
        "num_targets": len(per_target),
        "per_target": per_target,
        "per_source_dataset": per_source,
        "target_macro_spearman": finite_mean([float(row["spearman"]) for row in per_target]),
        "source_macro_spearman": finite_mean([float(row["spearman"]) for row in per_source]),
        "prediction_file": str(prediction_path),
    }
    report_path = args.output_dir / "t2837_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "num_scored_rows": report["num_scored_rows"],
        "num_targets": report["num_targets"],
        "target_macro_spearman": report["target_macro_spearman"],
        "source_macro_spearman": report["source_macro_spearman"],
        "per_source_dataset": per_source,
    }, indent=2))


if __name__ == "__main__":
    main()